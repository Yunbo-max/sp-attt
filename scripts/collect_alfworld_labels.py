"""Resumable, replay-based ALFWorld KEEP/LEARN label collection.

The default CLI values are deliberately small for a local correctness run. For paper labels use
`--horizon remaining` and a fixed decoding seed; do not mix short-horizon pilot rows with paper
rows.
"""

import argparse
import json
import random
import traceback
from pathlib import Path

import torch

from sp_attt.alfworld_runner import (
    QwenTextPolicy,
    load_alfworld_config,
    make_alfworld_game_env,
    seed_everything,
)
from sp_attt.counterfactual import alfworld_return
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


def rollout(policy, env, observation, info, history, horizon, max_new_tokens,
            start_step, max_steps):
    done = False
    success = False
    finish_step = max_steps
    for offset in range(horizon):
        current = observation
        action, _generation = policy.act(observation, history, commands(info), max_new_tokens)
        next_obs, scores, dones, infos = env.step([action])
        done = bool(dones[0])
        info_won = infos.get("won", [False])[0] if isinstance(infos, dict) else infos[0].get("won", False)
        success = bool(success or info_won or scores[0] > 0)
        if success and finish_step == max_steps:
            finish_step = start_step + offset + 1
        history.append((current, action))
        observation = next_obs[0]
        info = infos if isinstance(infos, dict) else infos[0]
        if done:
            break
    return alfworld_return(success, finish_step, max_steps), success, done


def paired_label(policy, config, gamefile, actions, history, candidate, snapshot,
                 horizon, max_new_tokens):
    returns = {}
    try:
        for mode in ("keep", "learn"):
            policy.restore_adapter(snapshot)
            env = None
            try:
                _wrapper, env = make_alfworld_game_env(config, gamefile)
                observation, info = replay(env, actions)
                # ``history`` stops before the candidate action.  Replay has advanced the
                # environment through that action, so include its observation/action pair
                # when rebuilding the official ReAct prompt for the future rollout.
                branch_history = list(history)
                branch_history.append((candidate.observation, candidate.action))
                if mode == "learn":
                    policy.learner.update(candidate, "attt")
                total, success, done = rollout(policy, env, observation, info, branch_history,
                                               horizon, max_new_tokens, candidate.step,
                                               candidate.max_steps)
                returns[mode] = {"return": total, "success": success, "done": done}
            finally:
                if env is not None:
                    env.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
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
parser.add_argument("--max-checkpoint", type=int, default=None,
                    help="Optional cap on candidate step for resource-bounded runs")
parser.add_argument("--max-checkpoints-per-game", type=int, default=3)
parser.add_argument("--horizon", default="1")
parser.add_argument("--max-new-tokens", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.horizon == "remaining" and args.max_steps <= args.min_checkpoint:
    raise ValueError("remaining horizon must be positive: set max_steps > min_checkpoint")

seed_everything(args.seed)
output_path = Path(args.output)
errors_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
existing = {}
failed = set()
if output_path.exists():
    with output_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                existing[row["label_id"]] = row
if errors_path.exists():
    with errors_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                failed.add(json.loads(line)["label_id"])
config = load_alfworld_config(args.config, data_dir=args.data_dir, split=args.split, games=args.games)
listing_wrapper, listing_env = make_alfworld_game_env(config)
gamefiles = list(listing_wrapper.game_files)
# Directory enumeration is deterministic but highly clustered by task/object.
# Shuffle once with the experiment seed so checkpoints cover the train split
# rather than a contiguous family of nearly identical trials.  The resulting
# order is reproducible and is part of the label manifest's episode identity.
random.Random(args.seed).shuffle(gamefiles)
gamefiles = gamefiles[args.start_game : args.start_game + args.games]
listing_env.close()
policy = QwenTextPolicy(args.model, use_lora=True)

with output_path.open("a", encoding="utf-8") as stream, errors_path.open("a", encoding="utf-8") as error_stream:
    for local_index, gamefile in enumerate(gamefiles):
        game_index = args.start_game + local_index
        if len(existing) >= args.target_labels:
            break
        policy.reset_episode()
        policy.set_gamefile(gamefile)
        wrapper, env = make_alfworld_game_env(config, gamefile)
        observation, info = env.reset(); observation = observation[0]
        history = []; actions = []; generations = []
        eligible = [step for step in range(args.candidate_every, args.max_steps + 1,
                                           args.candidate_every)
                    if step >= args.min_checkpoint and
                    (args.max_checkpoint is None or step <= args.max_checkpoint) and
                    (args.horizon != "remaining" or step < args.max_steps)]
        # Match the protocol's three relative-position strata while keeping
        # collection tractable and avoiding highly correlated adjacent labels.
        rng = random.Random(args.seed + game_index)
        bins = ([], [], [])
        for step in eligible:
            bins[min(2, int(3 * (step / args.max_steps)))].append(step)
        candidate_steps = []
        for bucket in bins:
            if bucket:
                candidate_steps.append(rng.choice(bucket))
        # When collecting fewer than three opportunities, sample among the
        # strata rather than always taking the earliest bucket.  This preserves
        # coverage of early/middle/late checkpoints while keeping at most the
        # requested number per episode.
        if len(candidate_steps) > args.max_checkpoints_per_game:
            candidate_steps = rng.sample(candidate_steps, args.max_checkpoints_per_game)
        if len(candidate_steps) < args.max_checkpoints_per_game:
            remaining = [step for step in eligible if step not in candidate_steps]
            rng.shuffle(remaining)
            candidate_steps.extend(remaining[:args.max_checkpoints_per_game - len(candidate_steps)])
        candidate_steps = set(candidate_steps[:args.max_checkpoints_per_game])
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
            if step not in candidate_steps:
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
            if label_id not in existing and label_id not in failed:
                horizon = args.max_steps - step if args.horizon == "remaining" else int(args.horizon)
                if horizon <= 0:
                    raise ValueError(f"non-positive counterfactual horizon at checkpoint {step}")
                try:
                    returns = paired_label(policy, config, gamefile, actions, history, candidate,
                                           snapshot, horizon, args.max_new_tokens)
                except Exception as exc:
                    error_row = {
                        "label_id": label_id, "episode_id": str(game_index),
                        "checkpoint": step, "gamefile": gamefile,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    error_stream.write(json.dumps(error_row) + "\n")
                    error_stream.flush()
                    failed.add(label_id)
                    print(json.dumps({"skipped_label": label_id,
                                      "error_type": type(exc).__name__}), flush=True)
                    policy.restore_adapter(snapshot)
                    continue
                row = {
                    "label_id": label_id, "episode_id": str(game_index), "checkpoint": step,
                    "gamefile": gamefile, "horizon": horizon,
                    # Keep the predictor-visible checkpoint payload beside the
                    # counterfactual target.  Without these fields a label file
                    # can be audited, but it cannot train/replay a gate because
                    # the candidate representation and history are gone.
                    "candidate_text": text,
                    "candidate_action": action,
                    "candidate_observation": current,
                    "history_update_texts": [
                        f"Generation: {generation}\nAction: {past_action}"
                        for generation, past_action in zip(generations[:-1], actions[:-1])
                    ],
                    "relative_position": step / args.max_steps,
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
            # With one checkpoint per episode there is no downstream online
            # trajectory needed for another label; stop after this paired branch.
            if args.max_checkpoints_per_game == 1:
                break
            if len(existing) >= args.target_labels:
                break
        env.close()

print(json.dumps({"labels": len(existing), "target": args.target_labels,
                  "output": str(output_path)}, indent=2))
