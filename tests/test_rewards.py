"""Scalar reward composition, raw gates, and configuration failure contracts."""
from copy import deepcopy
import math

import pytest

from rl_matdesign.rewards import RewardCombiner


def _gated():
    return {
        "type": "gated",
        "objective": {"type": "weighted_sum", "terms": {"performance": 2.0}},
        "gates": [
            {"name": "quality", "score": {"type": "weighted_sum", "terms": {"pure": 1, "nearly": .5}},
             "min": .8, "penalty_weight": 2.0},
            {"name": "cost", "score": {"terms": {"cost": 1}},
             "min": 0, "max": 5, "penalty_weight": 3.0},
        ],
        "failure": {"reward": -4.0},
    }


WEIGHTS = {"performance": 9.0, "pure": 7.0, "nearly": 3.0, "cost": 1.0}


def test_omitted_reward_preserves_signed_legacy_weighted_sum():
    for cfg in (None, {}, {"type": "weighted_sum"}):
        result = RewardCombiner(cfg, {"a": 2, "b": -3, "c": 0}).evaluate(
            {}, {"a": -5.0, "b": 4.0, "c": 999.0},
        )
        assert result == {"reward": -22.0, "objective": -22.0,
                          "type": "weighted_sum", "gate_passed": True, "gates": []}


def test_explicit_objective_replaces_not_multiplies_property_weights():
    combiner = RewardCombiner({"objective": {"terms": {"performance": 2}}}, WEIGHTS)
    assert combiner.evaluate({}, {"performance": 6})["reward"] == 12.0


def test_gates_use_raw_means_and_objective_uses_adjusted_values():
    result = RewardCombiner(_gated(), WEIGHTS).evaluate(
        {"pure": .7, "nearly": .4, "cost": 3, "performance": 1000},
        {"performance": 6, "pure": -100, "nearly": -100, "cost": -300},
    )
    assert result["gate_passed"]
    assert result["reward"] == 12.0
    assert result["gates"][0]["score"] == pytest.approx(.9)
    assert result["gates"][1]["score"] == 3.0


def test_all_gates_required_and_violations_add_in_raw_units():
    combiner = RewardCombiner(_gated(), WEIGHTS)
    result = combiner.evaluate({"pure": .4, "nearly": .4, "cost": 7}, {"performance": 1e6})
    assert result["objective"] == 2e6  # no hidden normalization or clipping
    assert result["gate_passed"] is False
    assert result["reward"] == pytest.approx(-4 - 2 * .2 - 3 * 2)
    assert [g["passed"] for g in result["gates"]] == [False, False]
    assert [g["violation"] for g in result["gates"]] == pytest.approx([.2, 2])


@pytest.mark.parametrize("cost", [0.0, 5.0])
def test_equality_at_either_bound_passes(cost):
    result = RewardCombiner(_gated(), WEIGHTS).evaluate(
        {"pure": .8, "nearly": 0, "cost": cost}, {"performance": 1000},
    )
    assert result["gate_passed"]
    assert result["reward"] == 2000


def test_zero_penalty_still_enforces_gate_and_preserves_explicit_failure_reward():
    cfg = _gated()
    cfg["gates"][0]["penalty_weight"] = 0
    result = RewardCombiner(cfg, WEIGHTS).evaluate(
        {"pure": .1, "nearly": 0, "cost": 1}, {"performance": 2},
    )
    assert result["gate_passed"] is False
    assert result["reward"] == -4


def test_gated_default_objective_keeps_existing_weights_and_does_not_force_positive_reward():
    cfg = _gated()
    del cfg["objective"]
    result = RewardCombiner(cfg, WEIGHTS).evaluate(
        {"pure": 1, "nearly": 0, "cost": 1},
        {"performance": -100, "pure": 0, "nearly": 0, "cost": 0},
    )
    assert result["gate_passed"]
    assert result["reward"] == -900  # gating alone is not lexicographic feasibility


def test_omitted_penalty_defaults_to_one_and_inputs_and_outputs_are_not_shared():
    cfg = _gated()
    del cfg["gates"][0]["penalty_weight"]
    original = deepcopy(cfg)
    combiner = RewardCombiner(cfg, WEIGHTS)
    cfg["gates"][0]["score"]["terms"]["pure"] = 900
    first = combiner.evaluate({"pure": .5, "nearly": 0, "cost": 1}, {"performance": 2})
    first["gates"][0]["score"] = -1
    second = combiner.evaluate({"pure": .5, "nearly": 0, "cost": 1}, {"performance": 2})
    assert second["reward"] == pytest.approx(-4.3)
    assert second["gates"][0]["score"] == .5
    assert original["gates"][0]["score"]["terms"]["pure"] == 1


@pytest.mark.parametrize("cfg,match", [
    ([], "reward must be a mapping"),
    ({"type": "product"}, "reward.type"),
    ({"type": "weighted_sum", "gates": []}, "unsupported key"),
    ({"typo": 1}, "unsupported key"),
    ({"objective": None}, "objective must be a mapping"),
    ({"objective": {"type": "product", "terms": {"a": 1}}}, "objective.type"),
    ({"objective": {}}, "terms must be a mapping"),
    ({"objective": {"terms": {}}}, "terms must be non-empty"),
    ({"objective": {"terms": []}}, "terms must be a mapping"),
    ({"objective": {"terms": {"absent": 1}}}, "unknown property"),
    ({"objective": {"terms": {"a": math.nan}}}, "finite"),
    ({"objective": {"terms": {"a": True}}}, "finite number"),
    ({"objective": {"terms": {"a": "1"}}}, "finite number"),
    ({"type": "gated"}, "non-empty list"),
    ({"type": "gated", "gates": []}, "non-empty list"),
])
def test_bad_reward_configuration_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        RewardCombiner(cfg, {"a": 1})


@pytest.mark.parametrize("change,match", [
    ({"name": ""}, "name must be"),
    ({"name": None}, "name must be"),
    ({"score": {"terms": {"missing": 1}}}, "unknown property"),
    ({"score": {"type": "product", "terms": {"pure": 1}}}, "type must be"),
    ({"min": math.nan}, "finite"),
    ({"min": None}, "finite number"),
    ({"min": 2, "max": 1}, "min must be <= max"),
    ({"max": math.inf}, "finite"),
    ({"penalty_weight": -1}, "penalty_weight must be >= 0"),
    ({"penalty_weight": math.inf}, "finite"),
    ({"threshold": 1}, "unsupported key"),
])
def test_bad_gate_fields_rejected(change, match):
    cfg = _gated()
    cfg["gates"][0].update(change)
    with pytest.raises(ValueError, match=match):
        RewardCombiner(cfg, WEIGHTS)


@pytest.mark.parametrize("missing,match", [("score", "score is required"), ("min", "at least one bound")])
def test_missing_gate_fields_rejected(missing, match):
    cfg = _gated()
    del cfg["gates"][0][missing]
    with pytest.raises(ValueError, match=match):
        RewardCombiner(cfg, WEIGHTS)


def test_duplicate_gate_names_rejected():
    cfg = _gated()
    cfg["gates"][1]["name"] = "quality"
    with pytest.raises(ValueError, match="duplicates gate"):
        RewardCombiner(cfg, WEIGHTS)


@pytest.mark.parametrize("failure,match", [(None, "failure must be a mapping"), ({}, "failure.reward is required"),
                                         ({"reward": math.inf}, "finite"), ({"reward": 0, "penalty": 1}, "unsupported")])
def test_explicit_finite_failure_reward_required(failure, match):
    cfg = _gated()
    cfg["failure"] = failure
    with pytest.raises(ValueError, match=match):
        RewardCombiner(cfg, WEIGHTS)


@pytest.mark.parametrize("weights,match", [({}, "at least one property"), ({"": 1}, "non-empty strings"),
                                         ({1: 1}, "non-empty strings"), ({"a": math.inf}, "finite")])
def test_bad_property_declarations_rejected(weights, match):
    with pytest.raises(ValueError, match=match):
        RewardCombiner(None, weights)


@pytest.mark.parametrize("raw,adjusted,match", [
    ({"pure": 1, "nearly": 0, "cost": 1}, {}, "missing property 'performance'"),
    ({"pure": 1, "cost": 1}, {"performance": 1}, "missing property 'nearly'"),
    ({"pure": math.nan, "nearly": 0, "cost": 1}, {"performance": 1}, "finite"),
    ({"pure": 1, "nearly": 0, "cost": 1}, {"performance": math.inf}, "finite"),
    ({"pure": 1, "nearly": 0, "cost": 1}, {"performance": 1e308}, "finite"),
])
def test_invalid_runtime_values_fail_with_useful_errors(raw, adjusted, match):
    with pytest.raises(ValueError, match=match):
        RewardCombiner(_gated(), WEIGHTS).evaluate(raw, adjusted)


def test_extra_runtime_metadata_is_ignored():
    result = RewardCombiner(None, {"a": 2}).evaluate({"temperature": 900}, {"a": 3, "temperature": 900})
    assert result["reward"] == 6
