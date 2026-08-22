import torch

from sp_attt.online import SelectivePlasticity
from sp_attt.policy import Decision, GatePolicy
from sp_attt.types import CandidateExperience, GateFeatures


class Learner:
    def __init__(self): self.calls = 0
    def update(self, experience, method): self.calls += 1; return 0.2


def data():
    exp = CandidateExperience("e", 1, "text", "act", "obs", 5, 50)
    feat = GateFeatures(torch.zeros(2), 1, 1, 0, 0, 1, .1)
    return exp, feat


def test_skip_means_no_backward_call():
    learner = Learner()
    sp = SelectivePlasticity(learner, lambda _: -0.1, GatePolicy("sp"))
    result = sp.opportunity(*data())
    assert result.decision is Decision.SKIP and learner.calls == 0 and result.update_loss is None


def test_learn_calls_inner_update():
    learner = Learner()
    sp = SelectivePlasticity(learner, lambda _: 0.1, GatePolicy("sp"))
    result = sp.opportunity(*data())
    assert result.decision is Decision.LEARN and learner.calls == 1

