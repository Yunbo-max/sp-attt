from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import urlopen

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


def _command_forms(text: str) -> set[str]:
    normalized = _normalize(text)
    forms = {normalized}
    if " in/on " in normalized:
        forms.add(normalized.replace(" in/on ", " in "))
        forms.add(normalized.replace(" in/on ", " on "))
    # ReAct demonstrations use ``put X in/on Y`` while ALFWorld's
    # admissible-command surface uses the equivalent ``move X to Y`` form.
    match = re.match(r"^put (.+) (?:in/on|in|on) (.+)$", normalized)
    if match:
        forms.add(f"move {match.group(1)} to {match.group(2)}")
    return forms


def choose_action(generation: str, admissible: list[str]) -> tuple[str, bool]:
    normalized = _normalize(generation)
    # Prefer a complete exact command, then a line containing one.
    for command in admissible:
        if _command_forms(generation) & _command_forms(command):
            return command, False
    for command in admissible:
        if any(form in normalized for form in _command_forms(command)):
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
                          if any(form.startswith(prefix) for form in _command_forms(command))]
        if len(prefix_matches) == 1:
            return prefix_matches[0], False
    return admissible[0], True


_REACT_TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def _alfworld_observation(observation: str) -> str:
    """Match the observation normalization used by the public ReAct runner."""
    parts = observation.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else observation


def _react_task_prefix(gamefile: str | None) -> str:
    if gamefile:
        for prefix, task in _REACT_TASK_PREFIXES.items():
            if prefix in gamefile:
                return task
    return "put"


@lru_cache(maxsize=1)
def _react_prompts() -> dict[str, str]:
    url = ("https://raw.githubusercontent.com/ysymyth/ReAct/"
           "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/prompts/alfworld.json")
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_react_prompt(task: str) -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "assets" / f"alfworld_react_{task}.json"
    if prompt_path.exists():
        with prompt_path.open(encoding="utf-8") as stream:
            prompts = json.load(stream)
    else:
        # Pin the public ReAct prompt revision so a fresh checkout reproduces
        # the same few-shot demonstrations without carrying a generated asset.
        prompts = _react_prompts()
    return ("Interact with a household to solve a task. Here are two examples.\n"
            + prompts[f"react_{task}_1"] + prompts[f"react_{task}_0"]
            + "\nHere is the task.\n")


def _is_think_action(action: str) -> bool:
    return _normalize(action).startswith("think:")


def _clean_generation_line(generation: str) -> str:
    line = next((line.strip() for line in generation.splitlines() if line.strip()), generation.strip())
    return re.sub(r"^>\s*", "", line).strip()


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
        self._task_prompt = _load_react_prompt("put")
        self.initial_adapter: dict[str, torch.Tensor] | None = None
        if use_lora:
            self.model = attach_episode_lora(self.model, rank=rank, alpha=alpha)
            self.learner = HuggingFaceLoRALearner(self.model, self.tokenizer, lr=lr,
                                                  gradient_steps=gradient_steps)
            self.initial_adapter = {name: p.detach().cpu().clone()
                                    for name, p in self.model.named_parameters()
                                    if p.requires_grad}

    def set_gamefile(self, gamefile: str | None) -> None:
        self._task_prompt = _load_react_prompt(_react_task_prefix(gamefile))

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
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtNewline(StoppingCriteria):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, input_ids, scores, **kwargs):
                tail = self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=False)
                return "\n" in tail

        initial = _alfworld_observation(history[0][0] if history else observation)
        trajectory = ""
        for index, (_before, action) in enumerate(history):
            response = history[index + 1][0] if index + 1 < len(history) else observation
            trajectory += f" {action}\n{('OK.' if _is_think_action(action) else response)}\n>"
        prompt = self._task_prompt + initial + "\n>" + trajectory
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()
                  if torch.is_tensor(value)}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         stopping_criteria=StoppingCriteriaList([StopAtNewline(self.tokenizer)]))
        generated = self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:],
                                           skip_special_tokens=True)
        first_line = _clean_generation_line(generated)
        if _is_think_action(first_line):
            return first_line, generated
        return choose_action(first_line, admissible)[0], generated


def run_alfworld(method: str, *, model_name: str, config_path: str,
                 data_dir: str = "/root/.cache/alfworld", split: str = "valid_seen",
                 episodes: int = 1, seed: int = 0, max_steps: int = 50,
                 candidate_every: int = 5, max_new_tokens: int = 24) -> list[EpisodeResult]:
    seed_everything(seed)
    config = load_alfworld_config(config_path, data_dir=data_dir, split=split, games=episodes)
    listing_wrapper, listing_env = make_alfworld_game_env(config)
    gamefiles = list(listing_wrapper.game_files)[:episodes]
    listing_env.close()
    policy = QwenTextPolicy(model_name, use_lora=method in {"ttt", "attt"})
    results = []
    for episode, gamefile in enumerate(gamefiles):
        policy.reset_episode()
        _wrapper, env = make_alfworld_game_env(config, gamefile)
        observation, info = env.reset()
        observation = observation[0]
        gamefile = info.get("extra.gamefile", [None])[0] if isinstance(info, dict) else None
        policy.set_gamefile(gamefile)
        history: list[tuple[str, str]] = []
        trajectory: list[dict[str, Any]] = []
        updates = fallbacks = 0
        success = False
        for step in range(1, max_steps + 1):
            admissible = list(info["admissible_commands"][0])
            action, generation = policy.act(observation, history, admissible,
                                             max_new_tokens=max_new_tokens)
            if not _is_think_action(action) and not any(_normalize(action) == _normalize(item) for item in admissible):
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
