from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Protocol, TypeVar

from .types import CandidateExperience, PlasticityLabel

StateT = TypeVar("StateT")
AdapterT = TypeVar("AdapterT")


class BranchRunner(Protocol[StateT, AdapterT]):
    def rollout(self, state: StateT, adapter: AdapterT, horizon: int, *, allow_updates: bool) -> float: ...


def counterfactual_label(
    *, episode_id: str, checkpoint: int, state: StateT, adapter: AdapterT,
    experience: CandidateExperience, horizon: int, runner: BranchRunner[StateT, AdapterT],
    update: Callable[[AdapterT, CandidateExperience], AdapterT],
) -> PlasticityLabel:
    """Paired KEEP/LEARN label; downstream updates are disabled in both branches."""
    keep_state, learn_state = copy.deepcopy(state), copy.deepcopy(state)
    keep_adapter = copy.deepcopy(adapter)
    learn_adapter = update(copy.deepcopy(adapter), experience)
    keep = runner.rollout(keep_state, keep_adapter, horizon, allow_updates=False)
    learn = runner.rollout(learn_state, learn_adapter, horizon, allow_updates=False)
    return PlasticityLabel(episode_id, checkpoint, keep, learn, horizon)


def alfworld_return(success: bool, finish_step: int, max_steps: int = 50) -> float:
    return float(success) + (0.11 * (1.0 - finish_step / max_steps) if success else 0.0)


def mineexplorer_return(milestones_before: int, milestones_after: int,
                        total_milestones: int, solved: bool) -> float:
    progress = (milestones_after - milestones_before) / max(1, total_milestones)
    return progress + float(solved)
