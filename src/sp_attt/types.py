from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CandidateExperience:
    episode_id: str
    checkpoint: int
    text: str
    action: str
    observation: str
    step: int
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateFeatures:
    hidden: torch.Tensor
    novelty: float
    normalized_nll: float
    repeat_last: float
    repeat_recent3: float
    observation_novelty: float
    relative_position: float

    def scalars(self, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [self.novelty, self.normalized_nll, self.repeat_last, self.repeat_recent3,
             self.observation_novelty, self.relative_position],
            dtype=torch.float32,
            device=device,
        )


@dataclass(frozen=True)
class PlasticityLabel:
    episode_id: str
    checkpoint: int
    keep_return: float
    learn_return: float
    horizon: int

    @property
    def utility(self) -> float:
        return self.learn_return - self.keep_return

