from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .policy import Decision, GatePolicy
from .types import CandidateExperience, GateFeatures


class InnerLearner(Protocol):
    def update(self, experience: CandidateExperience, method: str) -> float: ...


class ValuePredictor(Protocol):
    def __call__(self, features: GateFeatures) -> float: ...


@dataclass
class OpportunityResult:
    decision: Decision
    predicted_value: float
    update_loss: float | None


class SelectivePlasticity:
    """The critical invariant: `learner.update` is never called on SKIP."""

    def __init__(self, learner: InnerLearner, predictor: ValuePredictor, policy: GatePolicy,
                 inner_method: str = "attt"):
        if inner_method not in {"ttt", "attt"}:
            raise ValueError("inner_method must be 'ttt' or 'attt'")
        self.learner, self.predictor, self.policy = learner, predictor, policy
        self.inner_method = inner_method

    def opportunity(self, experience: CandidateExperience,
                    features: GateFeatures) -> OpportunityResult:
        value = float(self.predictor(features))
        decision = self.policy.decide(predicted_value=value, novelty=features.novelty,
                                      uncertainty=features.normalized_nll)
        loss = self.learner.update(experience, self.inner_method) if decision is Decision.LEARN else None
        return OpportunityResult(decision, value, loss)

