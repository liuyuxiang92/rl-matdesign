from copy import deepcopy
import math

import pytest

from rl_matdesign.hpo.scoring import score_records, quality


def plan(direction="minimize", threshold=10, kind="positive_threshold_ratio"):
    return {"objective": {"field": "x", "direction": direction, "threshold": threshold,
                           "threshold_inclusive": False, "unit": "u",
                           "normalization": {"kind": kind, **({"scale": 2} if kind == "sigmoid" else {})}},
            "metrics": {"best_fraction": .1,
                        "training_weights": {"qualified_fraction": .45, "best_fraction_quality": .30, "best_value_quality": .10, "sampling_efficiency": .15},
                        "generation_weights": {"qualified_fraction": .50, "best_fraction_quality": .40, "best_value_quality": .10}},
            "efficiency": {"max_attempts_per_unique": 1.5, "comparison": "lt", "stop_when_completion_is_impossible": True, "train": {}, "generation": {}}}


def records(values, passed=None):
    return [{"identity": str(i), "properties": {"x": value}, "constraint_passed": True if passed is None else passed[i]}
            for i, value in enumerate(values)]


def ledger(n, attempts=None, status="COMPLETED", target=None):
    return {"target_unique": n if target is None else target, "unique_evaluated": n,
            "attempts_started": n if attempts is None else attempts, "status": status}


def test_direction_changes_threshold_best_tenth_and_best_value_together():
    data = records(list(range(1, 21)))
    lower = score_records(data, ledger(20), plan(), "train")
    higher = score_records(data, ledger(20), plan("maximize"), "train")
    assert lower["raw_metrics"]["qualified_fraction"] == 9 / 20
    assert higher["raw_metrics"]["qualified_fraction"] == 10 / 20
    assert lower["raw_metrics"]["best_fraction_mean"] == 1.5
    assert higher["raw_metrics"]["best_fraction_mean"] == 19.5
    assert lower["raw_metrics"]["best_value"] == 1
    assert higher["raw_metrics"]["best_value"] == 20
    assert lower["eligible_for_selection"] and higher["eligible_for_selection"]


def test_constraints_do_not_remove_denominator_and_missing_slots_penalized():
    data = records([2] + [1] * 19, passed=[True] + [False] * 19)
    result = score_records(data, ledger(20), plan(), "train")
    assert result["raw_metrics"]["qualified_fraction"] == .05
    assert result["raw_metrics"]["best_fraction_mean"] == 2
    assert result["normalized_metrics"]["best_fraction_quality"] == pytest.approx((10 / 12) * .5)


def test_transform_is_applied_after_mean_and_not_before():
    result = score_records(records([1, 9] + [100] * 18), ledger(20), plan(), "train")
    assert result["raw_metrics"]["best_fraction_mean"] == 5
    assert result["normalized_metrics"]["best_fraction_quality"] == pytest.approx(10 / 15)
    assert result["normalized_metrics"]["best_fraction_quality"] != pytest.approx((10 / 11 + 10 / 19) / 2)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_threshold_inclusion_is_explicit(direction):
    p = plan(direction)
    assert score_records(records([10]), ledger(1), p, "train")["raw_metrics"]["qualified_count"] == 0
    p["objective"]["threshold_inclusive"] = True
    assert score_records(records([10]), ledger(1), p, "train")["raw_metrics"]["qualified_count"] == 1


def test_strict_efficiency_gate_and_finite_penalty():
    data = records([1] * 10)
    assert score_records(data, ledger(10, 14), plan(), "train")["eligible_for_selection"]
    failed = score_records(data, ledger(10, 15), plan(), "train")
    assert failed["status"] == "INFEASIBLE_EFFICIENCY"
    assert failed["score"] == -1 and not failed["eligible_for_selection"]


def test_phase_override_is_independent_and_training_seen_candidate_counts_in_generation():
    p = plan()
    p["efficiency"]["generation"] = {"max_attempts_per_unique": 2.0}
    data = records([2] * 10)
    assert not score_records(data, ledger(10, 15), p, "train")["eligible_for_selection"]
    assert score_records(data, ledger(10, 15), p, "generation")["eligible_for_selection"]


def test_no_feasible_material_preserves_null_quality_and_real_denominator():
    result = score_records(records([1, 2], [False, False]), ledger(2), plan(), "generation")
    assert result["score"] == 0
    assert result["raw_metrics"]["best_fraction_mean"] is None
    assert result["raw_metrics"]["best_value"] is None
    assert result["raw_metrics"]["qualified_count"] == 0


@pytest.mark.parametrize("status", ["RUNNING", "SYSTEM_ERROR", "NUMERICAL_INSTABILITY", "INFEASIBLE_EFFICIENCY", "TIME_CENSORED"])
def test_partial_or_failed_jobs_never_enter_normal_ranking(status):
    result = score_records(records([1, 2]), ledger(2, 3, status, 10), plan(), "train")
    assert result["status"] == status
    assert result["score"] == -1 and not result["eligible_for_selection"]
    assert result["raw_metrics"]["target_unique"] == 10


def test_archive_is_deduplicated_without_changing_denominator():
    data = records([1, 2])
    result = score_records(data + [deepcopy(data[0])], ledger(2), plan(), "train")
    assert result["eligible_for_selection"]
    conflicting = deepcopy(data[0])
    conflicting["properties"]["x"] = 3
    assert score_records(data + [conflicting], ledger(2), plan(), "train")["status"] == "INVALID_RECORDS"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, True, "1"])
def test_invalid_properties_are_not_silently_filtered(bad):
    result = score_records(records([1, bad]), ledger(2), plan(), "train")
    assert result["status"] == "INVALID_RECORDS"
    assert not result["eligible_for_selection"]


def test_missing_property_and_ledger_disagreement_are_errors():
    data = records([1, 2])
    assert score_records(data, ledger(3), plan(), "train")["status"] == "INVALID_RECORDS"
    data[1]["properties"].clear()
    assert score_records(data, ledger(2), plan(), "train")["status"] == "INVALID_RECORDS"


def test_typed_identity_and_sequence_order_are_preserved():
    data = records([1] * 5)
    for record, identity in zip(data, [1, "1", True, ["A", "B"], ["B", "A"]]):
        record["identity"] = identity
    assert score_records(data, ledger(5), plan(), "generation")["eligible_for_selection"]


def test_sigmoid_handles_negative_and_zero_values_for_both_directions():
    data = records([-10000, 0, 10000])
    for direction in ("minimize", "maximize"):
        result = score_records(data, ledger(3), plan(direction, 0, "sigmoid"), "train")
        assert result["eligible_for_selection"]
        assert math.isfinite(result["score"])
        assert result["raw_metrics"]["qualified_count"] == 1
    assert quality(0, plan("minimize", 0, "sigmoid")["objective"]) == .5


def test_positive_ratio_rejects_wrong_domain_and_remains_finite_at_large_values():
    assert score_records(records([-1]), ledger(1), plan(), "train")["status"] == "INVALID_RECORDS"
    objective = plan(threshold=1e308)["objective"]
    assert quality(1e308, objective) == .5


def test_generation_score_does_not_reward_efficiency_twice():
    data = records([2] * 10)
    a = score_records(data, ledger(10, 10), plan(), "generation")
    b = score_records(data, ledger(10, 14), plan(), "generation")
    assert a["score"] == b["score"]
    assert score_records(data, ledger(10, 10), plan(), "train")["score"] > score_records(data, ledger(10, 14), plan(), "train")["score"]


def test_preparation_diagnostics_do_not_earn_unique_credit():
    data = records([2] * 10) + [{"evaluation_status": "preparation_only", "identity": "warmup"}]
    result = score_records(data, ledger(10, 12), plan(), "train")
    assert result["eligible_for_selection"]
    assert result["raw_metrics"]["attempts_per_unique"] == 1.2


def test_infrastructure_failure_is_not_relabelled_as_parameter_efficiency_failure():
    result = score_records(records([1] * 10), ledger(10, 15, "SYSTEM_ERROR"), plan(), "train")
    assert result["status"] == "SYSTEM_ERROR"
    assert not result["eligible_for_selection"]
