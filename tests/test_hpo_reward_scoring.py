"""HPO reward scoring uses paired evaluation snapshots and one fixed scale."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rl_matdesign.hpo.calibration import (
    QualityCalibrationError, calibrate_quality_scale, fit_quality_scale,
    restore_quality_scale,
)
from rl_matdesign.hpo.scoring import score_records
from rl_matdesign.hpo.spec import _efficiency
from rl_matdesign.utils.candidate_records import episode_record


def plan(metric="reward", field="reward", threshold=0, direction="maximize", scale=1):
    objective = {
        "field": field, "direction": direction, "threshold": threshold,
        "threshold_inclusive": True, "unit": None,
        "normalization": {"kind": "sigmoid", "scale": scale, "calibration": "reference"},
    }
    if metric == "reward":
        objective["metric"] = metric
    return {
        "objective": objective,
        "metrics": {
            "best_fraction": .5,
            "training_weights": {"qualified_fraction": .45, "best_fraction_quality": .3,
                                 "best_value_quality": .1, "sampling_efficiency": .15},
            "generation_weights": {"qualified_fraction": .5, "best_fraction_quality": .4,
                                   "best_value_quality": .1},
        },
        "efficiency": _efficiency({}),
    }


def ledger(count, attempts=None):
    return {"status": "COMPLETED", "target_unique": count, "unique_evaluated": count,
            "attempts_started": count if attempts is None else attempts}


def records(values, passed=None):
    return [{"identity": index, "properties": {"reward": 999},
             "prediction_record": {"reward": value},
             "constraint_passed": True if passed is None else passed[index]}
            for index, value in enumerate(values)]


@pytest.mark.parametrize("phase", ["train", "generation"])
def test_combined_reward_uses_evaluated_scalar_instead_of_first_or_named_property(phase):
    data = records([-150, -200, -125, -250])
    for row, sinter, calcine in zip(data, [100, 50, 100, 200], [50, 150, 25, 50]):
        row["properties"].update(sinter=sinter, calcine=calcine)
    result = score_records(data, ledger(4), plan(threshold=-175), phase)
    assert result["eligible_for_selection"]
    assert result["raw_metrics"]["qualified_count"] == 2
    assert result["raw_metrics"]["best_fraction_mean"] == -137.5
    assert result["raw_metrics"]["best_value"] == -125


def test_reward_retains_uncertainty_and_calibration_from_the_same_evaluation():
    # Equal means but different uncertainty penalties must remain distinguishable.
    data = records([6.5, 9.0])
    for row, std in zip(data, [3.5, 1.0]):
        row["properties"] = {"pIC50": 10}
        row["prediction_record"].update(obj_pIC50_mean=10, obj_pIC50_std=std,
                                         reward_objective=100, reward_standardized_utility=200)
    result = score_records(data, ledger(2), plan(threshold=8), "train")
    assert result["raw_metrics"]["qualified_count"] == 1
    assert result["raw_metrics"]["best_value"] == 9
    # An actual calibrated reward is consumed verbatim, without choosing a
    # differently named objective or trying to reconstruct it from physical means.
    data[0]["prediction_record"]["reward"] = 12.25
    assert score_records(data, ledger(2), plan(), "train")["raw_metrics"]["best_value"] == 12.25


def test_candidate_archive_preserves_exact_reward_without_another_prediction():
    episode = SimpleNamespace(
        prediction_record={"reward": -350, "obj_sinter_mean": 200,
                           "obj_sinter_std": None, "obj_calcine_mean": 150,
                           "obj_calcine_std": None},
        terminal_key=("oxide", (1, 2)), formula="example",
    )
    row = episode_record(episode, "train")
    assert row["prediction_record"]["reward"] == -350
    result = score_records([row], ledger(1), plan(threshold=-400), "train")
    assert result["eligible_for_selection"]
    assert result["raw_metrics"]["best_value"] == -350


@pytest.mark.parametrize("bad", [None, True, False, "1", float("nan"), float("inf"), -float("inf")])
def test_invalid_reward_is_rejected_in_scoring_and_reference_calibration(bad):
    data = records([1, bad])
    p = plan()
    result = score_records(data, ledger(2), p, "train")
    assert result["status"] == "INVALID_RECORDS"
    assert "Recorded reward" in result["error"]
    assert result["score"] == -1
    with pytest.raises(QualityCalibrationError, match="Recorded reward"):
        fit_quality_scale(data, p["objective"])


@pytest.mark.parametrize("snapshot", [None, {}, {"reward_objective": 1}, [1]])
def test_missing_reward_does_not_fall_back_to_property_or_objective(snapshot):
    data = records([1, 2])
    data[0]["prediction_record"] = snapshot
    result = score_records(data, ledger(2), plan(), "train")
    assert result["status"] == "INVALID_RECORDS"
    assert "lacks recorded reward" in result["error"]
    with pytest.raises(QualityCalibrationError, match="lacks recorded reward"):
        fit_quality_scale(data, plan()["objective"])


def test_conflicting_rewards_for_one_identity_fail_even_if_properties_match():
    data = records([1, 2])
    duplicate = deepcopy(data[0])
    duplicate["prediction_record"]["reward"] = 3
    result = score_records(data + [duplicate], ledger(2), plan(), "train")
    assert result["status"] == "INVALID_RECORDS"
    assert "Conflicting evaluated snapshots" in result["error"]
    with pytest.raises(QualityCalibrationError, match="conflicting duplicate"):
        fit_quality_scale(data + [duplicate], plan()["objective"])


def test_gate_penalties_enter_reference_scale_but_do_not_earn_feasible_slots():
    data = records([-100, -100, 1, 3], passed=[False, False, True, True])
    p = plan(scale=None)
    fit = fit_quality_scale(data, p["objective"])
    assert fit["sample_count"] == 4
    assert fit["minimum"] == -100
    assert fit["scale"] != fit_quality_scale(data[2:], p["objective"])["scale"]
    p["objective"]["normalization"]["scale"] = fit["scale"]
    result = score_records(data, ledger(4), p, "train")
    assert result["raw_metrics"]["qualified_fraction"] == .5
    assert result["raw_metrics"]["best_fraction_mean"] == 2
    assert result["raw_metrics"]["best_value"] == 3


def test_reward_and_same_named_property_cannot_reuse_a_quality_calibration(tmp_path):
    data = records([1, 2, 3])
    archive = tmp_path / "records.jsonl"
    archive.write_text("\n".join(json.dumps(row) for row in data))
    reward_plan = plan(scale=None)
    calibrated = calibrate_quality_scale(reward_plan, tmp_path, data, ledger(3), archive, "reference", "train")
    assert calibrated["objective"]["metric"] == "reward"
    assert restore_quality_scale(plan(scale=None), tmp_path)["scale"] == calibrated["scale"]
    with pytest.raises(QualityCalibrationError, match="does not match"):
        restore_quality_scale(plan(metric="property", scale=None), tmp_path)


def test_existing_property_mode_ignores_reward_and_retains_artifact_contract(tmp_path):
    data = records([100, 200, 300])
    for row, value in zip(data, [1, 2, 3]):
        row["properties"]["reward"] = value
    archive = tmp_path / "records.jsonl"
    archive.write_text("\n".join(json.dumps(row) for row in data))
    p = plan(metric="property", scale=None)
    calibration = calibrate_quality_scale(p, tmp_path, data, ledger(3), archive, "reference", "train")
    assert calibration["objective"] == {"field": "reward", "direction": "maximize", "threshold": 0, "unit": None}
    result = score_records(data, ledger(3), p, "generation")
    assert result["raw_metrics"]["best_value"] == 3
    with pytest.raises(QualityCalibrationError, match="does not match"):
        restore_quality_scale(plan(scale=None), tmp_path)


@pytest.mark.parametrize("phase", ["train", "generation"])
def test_glassnet_affine_reward_preserves_scores_and_trial_ranking(phase):
    # Shared fixed scale comes from the same reference candidates, including
    # gate failures. Threshold equality is included in the reference data.
    cohorts = [
        [1300., 1390., 1400., 1410., 1420.15, 1450., 1600., 1800.],
        [1310., 1400., 1410., 1420., 1440., 1460., 1700., 1900.],
        [1290., 1380., 1390., 1400., 1420.15, 1440., 1500., 1700.],
    ]
    physical_plan = plan(metric="property", field="T3", threshold=1420.15,
                         direction="minimize", scale=None)
    reward_plan = plan(threshold=-1.42015, scale=None)
    physical_scores, reward_scores = [], []
    for index, temperatures in enumerate(cohorts):
        data = records([-t / 1000 for t in temperatures], passed=[False] + [True] * 6 + [False])
        for row, value in zip(data, temperatures):
            row["properties"] = {"T3": value}
        if index == 0:
            for p in (physical_plan, reward_plan):
                p["objective"]["normalization"]["scale"] = fit_quality_scale(data, p["objective"])["scale"]
            assert reward_plan["objective"]["normalization"]["scale"] == pytest.approx(
                physical_plan["objective"]["normalization"]["scale"] / 1000)
        physical = score_records(data, ledger(8, 10), physical_plan, phase)
        reward = score_records(data, ledger(8, 10), reward_plan, phase)
        assert physical["eligible_for_selection"] and reward["eligible_for_selection"]
        assert physical["normalized_metrics"] == pytest.approx(reward["normalized_metrics"], abs=1e-14)
        assert reward["raw_metrics"]["best_value"] == pytest.approx(-physical["raw_metrics"]["best_value"] / 1000)
        assert reward["raw_metrics"]["best_fraction_mean"] == pytest.approx(-physical["raw_metrics"]["best_fraction_mean"] / 1000)
        physical_scores.append(physical["score"])
        reward_scores.append(reward["score"])
    assert physical_scores == pytest.approx(reward_scores, abs=1e-14)
    assert sorted(range(3), key=physical_scores.__getitem__) == sorted(range(3), key=reward_scores.__getitem__)
