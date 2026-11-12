"""Scientific score contracts for the simplified HPO interface."""
from copy import deepcopy
import json

import pytest

from rl_matdesign.hpo.artifacts import atomic_json
from rl_matdesign.hpo.budget_search import budget_grid, budget_schedule, select_budget
from rl_matdesign.hpo.calibration import (QualityCalibrationError, fit_quality_scale,
                                         calibrate_quality_scale, restore_quality_scale)
from rl_matdesign.hpo.scoring import quality, score_records


def plan():
    return {"protocol": "hpo_v2", "scoring_scope": "score_only",
            "objective": {"field": "energy", "direction": "minimize", "threshold": 0,
                          "threshold_inclusive": True, "unit": "eV",
                          "normalization": {"kind": "sigmoid", "scale": None, "calibration": "reference"}},
            "metrics": {"best_fraction": .1,
                        "training_weights": {"qualified_fraction": .45, "best_fraction_quality": .3,
                                             "best_value_quality": .1, "sampling_efficiency": .15},
                        "generation_weights": {"qualified_fraction": .5, "best_fraction_quality": .4,
                                               "best_value_quality": .1}},
            "efficiency": {"max_attempts_per_unique": 1.5, "comparison": "lt",
                           "stop_when_completion_is_impossible": True, "train": {}, "generation": {}},
            "budget": {"mode": "per_phase", "tolerances": {"score": .01}, "upper_confirmation_levels": 2,
                       "train": {"mode": "fraction_of_space", "min_fraction": .001, "initial_fraction": .002,
                                 "max_fraction": .004, "factor": 2, "min_unique": 1},
                       "generation": {"mode": "absolute_unique", "min_unique": 10, "initial_unique": 20,
                                      "max_unique": 40, "factor": 2}},
            "_budget_grids": {"train": [10, 20, 40], "generation": [10, 20, 40]}}


def records(values):
    return [{"identity": str(i), "properties": {"energy": value}, "constraint_passed": True}
            for i, value in enumerate(values)]


def ledger(n, status="COMPLETED"):
    return {"target_unique": n, "unique_evaluated": n, "attempts_started": n, "status": status}


def row(budget, score=.7, **raw):
    return {"parameter_id": "a", "budget": budget, "score": score, "status": "COMPLETED",
            "eligible_for_selection": True, "raw_metrics": {"qualified_fraction": .5,
            "best_fraction_mean": -2, "best_value": -3, "attempts_per_unique": 1.1, **raw}}


def test_mixed_budget_modes_preserve_bidirectional_search_without_hidden_100_floor():
    p = plan()
    assert budget_grid(p, "train", 10_000) == [10, 20, 40]
    assert budget_schedule(p, "train", 10_000) == [20, 10, 40]
    assert budget_grid(p, "generation") == [10, 20, 40]


def test_only_total_score_controls_plateau_even_when_components_improve_or_cancel():
    p = plan()
    rows = [row(10), row(20, best_fraction_mean=-20, best_value=-30),
            row(40, qualified_fraction=.9, best_fraction_mean=-40, attempts_per_unique=1.4)]
    result = select_budget(rows, p, "train")
    assert result["selected_budget"] == 10
    assert result["convergence_status"] == "OBSERVED_PLATEAU"
    assert result["evidence"]["tolerances"] == {"score": .01}
    assert all(set(change["changes"]) == {"score"} for change in result["evidence"]["higher_budget_changes"])


def test_efficiency_and_missing_higher_jobs_still_block_score_plateau():
    p = plan()
    assert select_budget([row(10), row(20), row(40, attempts_per_unique=1.5)], p, "train")["convergence_status"] == "EVIDENCE_INCOMPLETE"
    assert select_budget([row(10), row(20)], p, "train")["convergence_status"] == "EVIDENCE_INCOMPLETE"
    assert select_budget([row(10), row(20), row(40, score=.8)], p, "train")["selected_budget"] == 40


def test_no_feasible_quality_is_reported_without_an_extra_component_plateau_gate():
    p = plan()
    result = select_budget([row(b, score=.1, qualified_fraction=0, best_fraction_mean=None,
                              best_value=None) for b in [10, 20, 40]], p, "train")
    assert result["selected_budget"] == 10
    assert result["convergence_status"] == "OBSERVED_PLATEAU"
    # This is a total-score plateau only, not a claim of scientifically good material.
    assert result["selected"]["raw_metrics"]["best_value"] is None


@pytest.mark.parametrize("values", [[-3, -1, 0, 2], [0, 1, 2, 3], [-4, -3, -2, -1]])
def test_fixed_sigmoid_handles_signs_and_includes_threshold(values):
    p = plan()
    data = records(values)
    p["objective"]["normalization"]["scale"] = fit_quality_scale(data, p["objective"])["scale"]
    scored = score_records(data, ledger(len(data)), p, "train")
    assert scored["eligible_for_selection"]
    assert scored["raw_metrics"]["qualified_count"] == sum(value <= 0 for value in values)
    assert quality(-2, p["objective"]) > quality(0, p["objective"]) > quality(2, p["objective"])
    assert quality(0, p["objective"]) == .5


def test_extreme_finite_values_keep_scaled_difference_correct():
    objective = plan()["objective"]
    objective["threshold"] = 1e308
    objective["normalization"]["scale"] = 1e308
    assert quality(-1e308, objective) == pytest.approx(.8807970779778823)


def test_automatic_scale_fallbacks_and_degenerate_reference():
    objective = plan()["objective"]
    assert fit_quality_scale(records([2, 2, 2]), objective) == {
        "scale": 2, "method": "reference_threshold_distance", "sample_count": 3, "minimum": 2, "maximum": 2}
    assert fit_quality_scale(records([0] * 9 + [10]), objective)["method"] == "reference_range"
    with pytest.raises(QualityCalibrationError, match="all equal"):
        fit_quality_scale(records([0, 0]), objective)


def test_reference_scale_is_frozen_restorable_and_archive_verified(tmp_path):
    p = plan()
    data = records([-3, -1, 0, 2])
    archive = tmp_path / "records.jsonl"
    archive.write_text("\n".join(json.dumps(r) for r in data))
    calibrated = calibrate_quality_scale(p, tmp_path, data, ledger(4), archive, "reference", "train")
    original_scale = calibrated["scale"]
    assert calibrate_quality_scale(p, tmp_path, records([100, 200]), ledger(2), archive, "other", "train") is None
    assert p["objective"]["normalization"]["scale"] == original_scale
    restored = plan()
    assert restore_quality_scale(restored, tmp_path)["scale"] == original_scale
    assert restored["objective"]["normalization"]["scale"] == original_scale
    calibrated["scale"] *= 2
    atomic_json(tmp_path / "quality_calibration.json", calibrated)
    with pytest.raises(QualityCalibrationError, match="changed"):
        restore_quality_scale(plan(), tmp_path)
    calibrated["scale"] = original_scale
    atomic_json(tmp_path / "quality_calibration.json", calibrated)
    archive.write_text(archive.read_text() + "\n")
    with pytest.raises(QualityCalibrationError, match="archive changed"):
        restore_quality_scale(plan(), tmp_path)


def test_failed_reference_does_not_change_normalization_or_failure_status(tmp_path):
    p = plan()
    data = records([-3, -1])
    accounting = ledger(2, "INFEASIBLE_EFFICIENCY")
    assert calibrate_quality_scale(p, tmp_path, data, accounting, tmp_path / "unused", "job", "train") is None
    assert score_records(data, accounting, p, "train")["status"] == "INFEASIBLE_EFFICIENCY"
    assert not (tmp_path / "quality_calibration.json").exists()
