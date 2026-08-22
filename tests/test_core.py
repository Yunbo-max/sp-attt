import torch

from sp_attt.alfworld_runner import choose_action
from sp_attt.counterfactual import alfworld_return, mineexplorer_return
from sp_attt.gate import PlasticityGate
from sp_attt.metrics import plasticity_metrics
from sp_attt.novelty import attt_token_weights, sequence_novelty
from sp_attt.sampling import stratified_checkpoints
from sp_attt.types import GateFeatures


def test_attt_repeated_ngram_is_downweighted():
    weights = attt_token_weights([1, 2, 3, 4], [[0, 1, 2, 3]], n=3)
    assert weights == [1.0, 1.0, 0.5, 1.0]


def test_novelty_extremes():
    assert sequence_novelty("a b c", []) == 1.0
    assert sequence_novelty("a b c", ["a b c"]) == 0.0


def test_gate_shape_and_decision():
    gate = PlasticityGate(16)
    features = GateFeatures(torch.zeros(16), 1, 2, 0, 0, 1, .5)
    decision, value = gate.decide(features)
    assert isinstance(decision, bool) and isinstance(value, float)


def test_returns_and_metrics():
    assert alfworld_return(True, 25) == 1.055
    assert mineexplorer_return(1, 3, 4, True) == 1.5
    result = plasticity_metrics([-1, 2, 3], [True, False, True])
    assert result["harmful_update_rate"] == 0.5
    assert result["missed_beneficial_rate"] == 1.0
    assert result["plasticity_regret"] == 3.0


def test_stratified_checkpoints():
    points = stratified_checkpoints(50, seed=7)
    assert len(points) == 3 and points == sorted(points)
    assert points[0] <= 17 < points[1] <= 34 < points[2]


def test_action_prefix_resolves_unique_object_suffix():
    action, fallback = choose_action("go to sidetable", ["go to bed 1", "go to sidetable 1"])
    assert action == "go to sidetable 1" and not fallback
