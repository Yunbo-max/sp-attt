from __future__ import annotations

import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .hf_learner import HuggingFaceLoRALearner, attach_episode_lora
from .types import CandidateExperience


@dataclass
class EpisodeResult:
    episode: int
    method: str
    success: bool
    steps: int
    updates: int
    fallback_actions: int
    trajectory: list[dict[str, Any]]


def load_alfworld_config(config_path: str, *, data_dir: str, split: str, games: int) -> dict:
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config.setdefault("dataset", {})
    config.setdefault("logic", {})
    config.setdefault("env", {})
    config.setdefault("general", {})
    config["dataset"]["data_path"] = str(Path(data_dir) / "json_2.1.1" / split)
    config["dataset"]["eval_id_data_path"] = str(Path(data_dir) / "json_2.1.1" / "valid_seen")
    config["dataset"]["eval_ood_data_path"] = str(Path(data_dir) / "json_2.1.1" / "valid_unseen")
    config["dataset"]["num_train_games"] = games if split == "train" else -1
    config["dataset"]["num_eval_games"] = games if split != "train" else -1
    # Keep the requested split alongside the ALFWorld config.  The wrapper's
    # ``train_eval`` argument controls which of data_path/eval_id_data_path/
    # eval_ood_data_path is enumerated; losing this value silently turns a
    # meta-train request into valid_seen evaluation.
    config["_sp_split"] = split
    config["logic"]["domain"] = str(Path(data_dir) / "logic" / "alfred.pddl")
    config["logic"]["grammar"] = str(Path(data_dir) / "logic" / "alfred.twl2")
    config["general"]["use_cuda"] = torch.cuda.is_available()
    # The experiment config contains SP-aTTT settings; add the small subset of
    # ALFWorld's environment config required by AlfredTWEnv at runtime.
    config.setdefault("env", {}).setdefault("type", "AlfredTWEnv")
    config["env"].setdefault("task_types", [1, 2, 3, 4, 5, 6])
    config["env"].setdefault("domain_randomization", False)
    config["env"].setdefault("expert_type", "handcoded")
    config["env"].setdefault("goal_desc_human_anns_prob", 0.0)
    config.setdefault("general", {})["training_method"] = "dagger"
    config.setdefault("dagger", {}).setdefault("training", {})["max_nb_steps_per_episode"] = 50
    config.setdefault("rl", {}).setdefault("training", {})["max_nb_steps_per_episode"] = 50
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_alfworld_game_env(config: dict, gamefile: str | None = None,
                           train_eval: str | None = None):
    """Create a one-game eval environment, optionally forcing an exact gamefile."""
    from alfworld.agents.environment import get_environment
    if train_eval is None:
        split = config.get("_sp_split", "valid_seen")
        train_eval = "train" if split == "train" else (
            "eval_out_of_distribution" if split == "valid_unseen" else "eval_in_distribution"
        )
    env_class = get_environment(config["env"]["type"])
    if gamefile is None:
        wrapper = env_class(config, train_eval=train_eval)
    else:
        # AlfredTWEnv normally walks every task directory during __init__. A
        # counterfactual branch already names its exact gamefile, so bypass
        # that scan without changing any environment behavior after init.
        class SingleGameEnv(env_class):
            def collect_game_files(self, verbose=False):
                self.game_files = [gamefile]
                self.num_games = 1

        wrapper = SingleGameEnv(config, train_eval=train_eval)
    return wrapper, wrapper.init_env(batch_size=1)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^```[^\n]*\n|```$", "", text).strip()
    return re.sub(r"\s+", " ", text)


def choose_action(generation: str, admissible: list[str]) -> tuple[str, bool]:
    normalized = _normalize(generation)
    # Prefer a complete exact command, then a line containing one.
    for command in admissible:
        c = _normalize(command)
        if normalized == c or re.search(rf"(?m)^\s*{re.escape(c)}\s*[.!]?\s*$", normalized):
            return command, False
    for command in admissible:
        if _normalize(command) in normalized:
            return command, False
    # Qwen3.5-4B often omits a disambiguating numeric suffix while still
    # naming the intended action (e.g. ``go to sidetable`` for ``go to
    # sidetable 1``).  Resolve only a sufficiently specific prefix and only
    # when it maps to one command; otherwise retain the deterministic first
    # admissible fallback rather than guessing between objects.
    generation_tokens = normalized.split()
    if len(generation_tokens) >= 2:
        prefix = " ".join(generation_tokens)
        prefix_matches = [command for command in admissible
                          if _normalize(command).startswith(prefix)]
        if len(prefix_matches) == 1:
            return prefix_matches[0], False
    return admissible[0], True


class QwenTextPolicy:
    def __init__(self, model_name: str, *, use_lora: bool, rank: int = 8, alpha: int = 16,
                 lr: float = 5e-4, gradient_steps: int = 2):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None, low_cpu_mem_usage=True,
        )
        self.learner = None
        self.initial_adapter: dict[str, torch.Tensor] | None = None
        if use_lora:
            self.model = attach_episode_lora(self.model, rank=rank, alpha=alpha)
            self.learner = HuggingFaceLoRALearner(self.model, self.tokenizer, lr=lr,
                                                  gradient_steps=gradient_steps)
            self.initial_adapter = {name: p.detach().cpu().clone()
                                    for name, p in self.model.named_parameters()
                                    if p.requires_grad}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def reset_episode(self) -> None:
        if self.initial_adapter is None:
            return
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(self.initial_adapter[name].to(parameter.device))
        if self.learner is not None:
            self.learner.optimizer.state.clear()
            self.learner.token_history.clear()

    def snapshot_adapter(self) -> dict:
        if self.learner is None:
            return {"parameters": {}, "optimizer": None, "token_history": []}
        return {
            "parameters": {name: parameter.detach().cpu().clone()
                           for name, parameter in self.model.named_parameters()
                           if parameter.requires_grad},
            "optimizer": deepcopy(self.learner.optimizer.state_dict()),
            "token_history": deepcopy(self.learner.token_history),
        }

    def restore_adapter(self, snapshot: dict) -> None:
        if self.learner is None:
            return
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(snapshot["parameters"][name].to(parameter.device))
        self.learner.optimizer.load_state_dict(deepcopy(snapshot["optimizer"]))
        self.learner.token_history = deepcopy(snapshot["token_history"])

    def act(self, observation: str, history: list[tuple[str, str]], admissible: list[str],
            max_new_tokens: int = 96) -> tuple[str, str]:
        recent = "\n".join(f"Observation: {o}\nAction: {a}" for o, a in history[-6:])
        command_list = "\n".join(f"- {a}" for a in admissible)
        prompt = ("You are an ALFWorld household agent. Reason briefly, then output exactly one "
                  "valid action from the list.\n" + recent + f"\nObservation: {observation}\n"
                  f"Valid actions:\n{command_list}\nAction:")
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()
                  if torch.is_tensor(value)}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:],
                                           skip_special_tokens=True)
        return choose_action(generated, admissible)[0], generated


def run_alfworld(method: str, *, model_name: str, config_path: str,
                 data_dir: str = "/root/.cache/alfworld", split: str = "valid_seen",
                 episodes: int = 1, seed: int = 0, max_steps: int = 50,
                 candidate_every: int = 5, max_new_tokens: int = 24) -> list[EpisodeResult]:
    seed_everything(seed)
    config = load_alfworld_config(config_path, data_dir=data_dir, split=split, games=episodes)
    _wrapper, env = make_alfworld_game_env(config)
    policy = QwenTextPolicy(model_name, use_lora=method in {"ttt", "attt"})
    results = []
    for episode in range(episodes):
        policy.reset_episode()
        observation, info = env.reset()
        observation = observation[0]
        history: list[tuple[str, str]] = []
        trajectory: list[dict[str, Any]] = []
        updates = fallbacks = 0
        success = False
        for step in range(1, max_steps + 1):
            admissible = list(info["admissible_commands"][0])
            action, generation = policy.act(observation, history, admissible,
                                             max_new_tokens=max_new_tokens)
            if not any(_normalize(action) == _normalize(item) for item in admissible):
                action, fallback = admissible[0], True
            else:
                fallback = False
            fallback_count = int(fallback or not generation.strip())
            fallbacks += fallback_count
            next_obs, scores, dones, infos = env.step([action])
            next_observation = next_obs[0]
            info_won = infos.get("won", [False])[0] if isinstance(infos, dict) else infos[0].get("won", False)
            success = bool(info_won or scores[0] > 0)
            trajectory.append({"step": step, "observation": observation, "generation": generation,
                               "action": action, "reward": float(scores[0]), "done": bool(dones[0])})
            history.append((observation, action))
            if method in {"ttt", "attt"} and step % candidate_every == 0 and not dones[0]:
                # aTTT's primary Self signal trains only the latest reasoning/action,
                # not a replay of the whole prefix.
                text = f"Generation: {generation}\nAction: {action}"
                experience = CandidateExperience(str(episode), step // candidate_every, text, action,
                                                 observation, step, max_steps)
                policy.learner.update(experience, method)
                updates += 1
            if isinstance(infos, dict):
                info = {key: value for key, value in infos.items()}
            else:
                info = infos[0]
            observation = next_observation
            if dones[0]:
                break
        results.append(EpisodeResult(episode, method, success, step, updates, fallbacks, trajectory))
    env.close()
    return results
