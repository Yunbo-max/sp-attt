"""Generate one paired ALFWorld KEEP/LEARN label by replaying one gamefile."""

import argparse
import json

from sp_attt.alfworld_runner import QwenTextPolicy, load_alfworld_config, seed_everything
from sp_attt.types import CandidateExperience


def make_game_env(config, gamefile):
    from alfworld.agents.environment import get_environment
    wrapper = get_environment(config["env"]["type"])(config, train_eval="eval_in_distribution")
    if gamefile is not None:
        wrapper.game_files = [gamefile]
        wrapper.num_games = 1
    return wrapper, wrapper.init_env(batch_size=1)


def info_commands(info):
    return list(info["admissible_commands"][0])


def step_env(env, observation, info, policy, history, max_new_tokens):
    action, generation = policy.act(observation, history, info_commands(info), max_new_tokens)
    next_obs, scores, dones, infos = env.step([action])
    next_info = {key: value for key, value in infos.items()} if isinstance(infos, dict) else infos[0]
    return next_obs[0], next_info, action, generation, float(scores[0]), bool(dones[0])


def replay(env, actions):
    observation, info = env.reset()
    observation = observation[0]
    for action in actions:
        next_obs, _scores, _dones, infos = env.step([action])
        observation = next_obs[0]
        info = {key: value for key, value in infos.items()} if isinstance(infos, dict) else infos[0]
    return observation, info


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
parser.add_argument("--config", default="configs/alfworld.yaml")
parser.add_argument("--data-dir", default="/root/.cache/alfworld")
parser.add_argument("--split", default="valid_seen")
parser.add_argument("--checkpoint", type=int, default=5)
parser.add_argument("--horizon", type=int, default=2)
parser.add_argument("--max-new-tokens", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", required=True)
args = parser.parse_args()

seed_everything(args.seed)
config = load_alfworld_config(args.config, data_dir=args.data_dir, split=args.split, games=1)
policy = QwenTextPolicy(args.model, use_lora=True)
base_wrapper, base_env = make_game_env(config, None)
gamefile = base_env.gamefiles[0]
observation, info = base_env.reset(); observation = observation[0]
history = []; actions = []; generations = []
for step in range(args.checkpoint):
    current_observation = observation
    observation, info, action, generation, _, done = step_env(
        base_env, observation, info, policy, history, args.max_new_tokens)
    history.append((current_observation, action)); actions.append(action); generations.append(generation)
    if done:
        raise RuntimeError("episode terminated before checkpoint")
candidate_text = "\n".join(f"Generation: {g}\nAction: {a}" for g, a in zip(generations, actions))
candidate = CandidateExperience("label-0", args.checkpoint // 5, candidate_text, actions[-1],
                                observation, args.checkpoint, 50)
base_env.close()

def rollout(adapter_mode):
    _wrapper, env = make_game_env(config, gamefile)
    obs, inf = replay(env, actions)
    if adapter_mode == "learn":
        policy.learner.update(candidate, "attt")
    branch_history = list(history)
    total = 0.0
    done = False
    for _ in range(args.horizon):
        obs, inf, action, _generation, reward, done = step_env(
            env, obs, inf, policy, branch_history, args.max_new_tokens)
        branch_history.append((obs, action)); total += reward
        if done:
            break
    env.close()
    return total, done

keep_return, keep_done = rollout("keep")
policy.reset_episode()
learn_return, learn_done = rollout("learn")
row = {"episode_id": "label-0", "checkpoint": args.checkpoint, "gamefile": gamefile,
       "keep_return": keep_return, "learn_return": learn_return,
       "utility": learn_return - keep_return, "horizon": args.horizon,
       "keep_done": keep_done, "learn_done": learn_done}
with open(args.output, "w", encoding="utf-8") as stream:
    json.dump(row, stream, indent=2)
print(json.dumps(row, indent=2))
