from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random


class Decision(str, Enum):
    LEARN = "learn"
    SKIP = "skip"


@dataclass
class GatePolicy:
    method: str
    threshold: float = 0.0
    update_rate: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def decide(self, *, predicted_value: float = 0.0, novelty: float = 0.0,
               uncertainty: float = 0.0, oracle_value: float | None = None) -> Decision:
        score = {
            "sp": predicted_value, "myopic": predicted_value, "novelty": novelty,
            "uncertainty": uncertainty, "oracle": oracle_value,
        }.get(self.method)
        if self.method in {"always", "ttt", "attt"}:
            return Decision.LEARN
        if self.method == "never":
            return Decision.SKIP
        if self.method == "random":
            return Decision.LEARN if self._rng.random() < (self.update_rate or 0.5) else Decision.SKIP
        if score is None:
            raise ValueError(f"Unsupported gate policy: {self.method}")
        return Decision.LEARN if score > self.threshold else Decision.SKIP

