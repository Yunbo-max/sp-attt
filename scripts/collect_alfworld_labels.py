"""Resumable, replay-based ALFWorld KEEP/LEARN label collection.

The default CLI values are deliberately small for a local correctness run. For paper labels use
`--horizon remaining` and a fixed decoding seed; do not mix short-horizon pilot rows with paper
rows.
"""

import argparse
import json
from pathlib import Path

from sp_attt.alfworld_runner import (
    QwenTextPolicy,
    load_alfworld_config,
    make_alfworld_game_env,
    seed_everything,
)
from sp_attt.types import CandidateExperience


def replay(env, actions):
    observation, info = env.reset()
    observation = observation[0]
    for action in actions:
        next_obs, _, dones, infos = env.step([action])
        if dones[0]:
            raise RuntimeError("candidate prefix terminated during counterfactual replay")
        observation = next_obs[0]
        info = infos if isinstance(infos, dict) else infos[0]
    return observation, info


def commands(info):
    return list(info["admissible_commands"][0])


def rollout(policy, env, observation, info, history, horizon, max_new_tokens):
    total = 0.0
    done = False
    success = False
    for _ in range(horizon):
        current = observation
        action, _generation = policy.act(observation, history, commands(info), max_new_tokens)
        next_obs, scores, dones, infos = env.step([action])
        total += float(scores[0])
        done = bool(dones[0])
        info_won = infos.get("won", [False])[0] if isinstance(infos, dict) else infos[0].get("won", False)
        success = bool(success or info_won or scores[0] > 0)
        history.append((current, action))
        observation = next_obs[0]
        info = infos if isinstance(infos, dict) else infos[0]
        if done:
            break
    return total, success, done


def paired_label(policy, config, gamefile, actions, history, candidate, snapshot,
                 horizon, max_new_tokens):
    returns = {}
    for mode in ("keep", "learn"):
        policy.restore_adapter(snapshot)
        _wrapper, env = make_alfworld_game_env(config, gamefile)
        observation, info = replay(env, actions)
        if mode == "learn":
            policy.learner.update(candidate, "attt")
        total, success, done = rollout(policy, env, observation, info, list(history), horizon,
                                       max_new_tokens)
        env.close()
        returns[mode] = {"return": total, "success": success, "done": done}
    policy.restore_adapter(snapshot)
    return returns


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
parser.add_argument("--config", default="configs/alfworld.yaml")
parser.add_argument("--data-dir", default="/root/.cache/alfworld")
parser.add_argument("--split", default="valid_seen")
parser.add_argument("--target-labels", type=int, default=10)
parser.add_argument("--games", type=int, default=10)
parser.add_argument("--start-game", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=50)
parser.add_argument("--candidate-every", type=int, default=5)
parser.add_argument("--min-checkpoint", type=int, default=5)
parser.add_argument("--horizon", default="1")
parser.add_argument("--max-new-tokens", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.horizon == "remaining" and args.max_steps <= args.min_checkpoint:
    raise ValueError("remaining horizon must be positive: set max_steps > min_checkpoint")

seed_everything(args.seed)
output_path = Path(args.output)
existing = {}
if output_path.exists():
    with output_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                existing[row["label_id"]] = row
config = load_alfworld_config(args.config, data_dir=args.data_dir, split=args.split, games=args.games)
listing_wrapper, listing_env = make_alfworld_game_env(config)
gamefiles = listing_wrapper.game_files[args.start_game : args.start_game + args.games]
listing_env.close()
policy = QwenTextPolicy(args.model, use_lora=True)

with output_path.open("a", encoding="utf-8") as stream:
    for local_index, gamefile in enumerate(gamefiles):
        game_index = args.start_game + local_index
        if len(existing) >= args.target_labels:
            break
        policy.reset_episode()
        wrapper, env = make_alfworld_game_env(config, gamefile)
        observation, info = env.reset(); observation = observation[0]
        history = []; actions = []; generations = []
        for step in range(1, args.max_steps + 1):
            current = observation
            action, generation = policy.act(observation, history, commands(info), args.max_new_tokens)
            next_obs, scores, dones, infos = env.step([action])
            if dones[0]:
                break
            history.append((current, action)); actions.append(action); generations.append(generation)
            observation = next_obs[0]
            info = infos if isinstance(infos, dict) else infos[0]
            if step % args.candidate_every != 0:
                continue
            if step < args.min_checkpoint:
                # Keep the online aTTT trajectory evolving, but only materialize
                # labels after the requested late-checkpoint warmup.
                policy.learner.update(
                    CandidateExperience(str(game_index), step // args.candidate_every,
                                         f"Generation: {generation}\nAction: {action}", action,
                                         current, step, args.max_steps), "attt")
                continue
            label_id = f"{game_index}:{step}"
            snapshot = policy.snapshot_adapter()
            text = f"Generation: {generations[-1]}\nAction: {actions[-1]}"
            candidate = CandidateExperience(str(game_index), step // args.candidate_every, text,
                                             action, current, step, args.max_steps)
            if label_id not in existing:
                horizon = args.max_steps - step if args.horizon == "remaining" else int(args.horizon)
                if horizon <= 0:
                    raise ValueError(f"non-positive counterfactual horizon at checkpoint {step}")
                returns = paired_label(policy, config, gamefile, actions, history, candidate,
                                       snapshot, horizon, args.max_new_tokens)
                row = {
                    "label_id": label_id, "episode_id": str(game_index), "checkpoint": step,
                    "gamefile": gamefile, "horizon": horizon,
                    "keep_return": returns["keep"]["return"],
                    "learn_return": returns["learn"]["return"],
                    "utility": returns["learn"]["return"] - returns["keep"]["return"],
                    "keep_success": returns["keep"]["success"],
                    "learn_success": returns["learn"]["success"],
                    "seed": args.seed,
                }
                stream.write(json.dumps(row) + "\n"); stream.flush()
                existing[label_id] = row
                print(json.dumps(row), flush=True)
            policy.restore_adapter(snapshot)
            policy.learner.update(candidate, "attt")
            if len(existing) >= args.target_labels:
                break
        env.close()

print(json.dumps({"labels": len(existing), "target": args.target_labels,
                  "output": str(output_path)}, indent=2))
