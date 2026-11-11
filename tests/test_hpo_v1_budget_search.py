from copy import deepcopy

import pytest

from rl_matdesign.hpo.budget_search import (
    budget_grid, budget_schedule, minimum_unique_for_preparation, select_budget,
)


def plan(direction="minimize"):
    return {
        "objective": {"direction": direction, "threshold": 100},
        "budget": {
            "mode": "fraction_of_space",
            "train": {"min_fraction": 0.000025, "initial_fraction": 0.0001, "max_fraction": 0.0004, "factor": 2, "min_unique": 1},
            "generation": {"min_fraction": 0.000025, "initial_fraction": 0.0001, "max_fraction": 0.0004, "factor": 2, "min_unique": 1},
            "tolerances": {"score": 0.01, "qualified_fraction": 0.02, "best_fraction_mean": 1, "best_value": 1, "attempts_per_unique": 0.1},
            "upper_confirmation_levels": 2,
        },
        "_budget_grids": {"train": [25, 50, 100, 200, 400], "generation": [25, 50, 100, 200, 400]},
    }


def result(budget, score=0.7, fraction=0.7, mean=50, best=40, ratio=1.1, eligible=True, parameter="reference"):
    return {"parameter_id": parameter, "budget": budget, "score": score,
            "eligible_for_selection": eligible, "status": "COMPLETED" if eligible else "INFEASIBLE_EFFICIENCY",
            "raw_metrics": {"qualified_fraction": fraction, "best_fraction_mean": mean, "best_value": best, "attempts_per_unique": ratio}}


def test_grid_contains_both_smaller_and_larger_endpoints():
    assert budget_grid(plan(), "train", 1_000_000) == [25, 50, 100, 200, 400]
    assert budget_schedule(plan(), "train", 1_000_000) == [100, 50, 200, 25, 400]


def test_non_geometric_endpoints_and_integer_duplicates():
    config = plan()
    config["budget"]["train"].update(min_fraction=0.00001, max_fraction=0.00013)
    assert budget_grid(config, "train", 10000) == [1, 2]
    config["budget"]["train"].update(min_fraction=0.00003, max_fraction=0.0003)
    assert budget_grid(config, "train", 1_000_000) == [30, 50, 100, 200, 300]


def test_no_rounding_error_at_decimal_boundary():
    config = plan()
    config["budget"]["train"].update(min_fraction=0.07, initial_fraction=0.07, max_fraction=0.07)
    assert budget_grid(config, "train", 100) == [7]


def test_preparation_floor_is_strict_and_moves_to_existing_tier():
    assert minimum_unique_for_preparation(200) == 401
    assert minimum_unique_for_preparation(0) == 1
    assert minimum_unique_for_preparation(2, 1.1) == 21
    config = plan()
    config["budget"]["train"]["max_fraction"] = 0.0008
    assert budget_grid(config, "train", 1_000_000, minimum_unique=401) == [800]
    assert budget_schedule(config, "train", 1_000_000, minimum_unique=401) == [800]


def test_absolute_grid_and_invalid_bounds():
    config = plan()
    config["budget"]["mode"] = "absolute_unique"
    config["budget"]["train"] = {"initial_unique": 100, "min_unique": 30, "max_unique": 350, "factor": 2}
    assert budget_grid(config, "train") == [30, 50, 100, 200, 350]
    with pytest.raises(ValueError, match="exceeds"):
        budget_grid(config, "train", 300)
    with pytest.raises(ValueError, match="no legal"):
        budget_grid(config, "train", minimum_unique=500)


@pytest.mark.parametrize("change", [{"factor": 1}, {"factor": float("nan")}, {"initial_fraction": 1}, {"min_fraction": 0}, {"max_fraction": 2}])
def test_invalid_fraction_grid(change):
    config = plan()
    config["budget"]["train"].update(change)
    with pytest.raises(ValueError):
        budget_grid(config, "train", 100000)


def test_percentage_requires_exact_size():
    with pytest.raises(ValueError, match="exact"):
        budget_grid(plan(), "train")


def test_selects_below_initial_with_two_higher_confirmations():
    rows = [result(25, 0.6, 0.6, 60, 50)] + [result(value) for value in [50, 100, 200, 400]]
    chosen = select_budget(rows, plan(), "train")
    assert chosen["selected_budget"] == 50
    assert chosen["convergence_status"] == "OBSERVED_PLATEAU"
    assert not chosen["lower_bound_reached"]


def test_later_improvement_prevents_short_false_plateau():
    rows = [result(value) for value in [25, 50, 100, 200]] + [result(400, 0.8, 0.8, 40, 30)]
    chosen = select_budget(rows, plan(), "train")
    assert chosen["selected_budget"] == 400
    assert chosen["convergence_status"] == "BUDGET_LIMIT_REACHED"
    assert chosen["upper_bound_reached"]


def test_maximize_direction_changes_raw_metric_acceptance():
    config = plan("maximize")
    rows = [result(25, mean=48, best=48)] + [result(value, mean=50, best=50) for value in [50, 100, 200, 400]]
    chosen = select_budget(rows, config, "train")
    # Reference tie selects the smaller budget; better larger values are accepted,
    # but cannot establish a plateau for that lower-quality reference.
    assert chosen["convergence_status"] == "BUDGET_LIMIT_REACHED"
    rows[0]["score"] = 0.68
    chosen = select_budget(rows, config, "train")
    assert chosen["selected_budget"] == 50
    assert chosen["convergence_status"] == "OBSERVED_PLATEAU"


def test_efficiency_failure_is_not_plateau():
    rows = [result(value) for value in [25, 50, 100, 200]] + [result(400, eligible=False)]
    chosen = select_budget(rows, plan(), "train")
    assert chosen["selected_budget"] == 25
    assert chosen["convergence_status"] == "EVIDENCE_INCOMPLETE"
    assert chosen["evidence"]["missing_or_ineligible_higher_budgets"] == [400]


def test_missing_entire_higher_level_is_incomplete():
    chosen = select_budget([result(value) for value in [25, 50, 100, 200]], plan(), "train")
    assert chosen["convergence_status"] == "EVIDENCE_INCOMPLETE"


def test_worse_large_budget_is_degradation_not_plateau():
    rows = [result(value) for value in [25, 50, 100, 200]] + [result(400, 0.6, 0.5, 60, 50)]
    assert select_budget(rows, plan(), "train")["convergence_status"] == "BUDGET_DEGRADATION"


def test_all_ineligible_and_all_null_are_distinct():
    rows = [result(value, eligible=False) for value in [25, 50, 100, 200, 400]]
    assert select_budget(rows, plan(), "train")["convergence_status"] == "NO_FEASIBLE_CONFIGURATION"
    rows = [result(value, mean=None, best=None) for value in [25, 50, 100, 200, 400]]
    chosen = select_budget(rows, plan(), "train")
    assert chosen["convergence_status"] == "NO_FEASIBLE_MATERIAL_EVIDENCE"
    assert chosen["selected_budget"] is None


def test_null_higher_scientific_metrics_are_missing_evidence():
    rows = [result(value) for value in [25, 50, 100, 200]] + [result(400, mean=None, best=None)]
    assert select_budget(rows, plan(), "train")["convergence_status"] == "EVIDENCE_INCOMPLETE"


def test_generation_does_not_use_ratio_as_quality_plateau_metric():
    rows = [result(value, ratio=(1.05 if value == 25 else 1.4)) for value in [25, 50, 100, 200, 400]]
    assert select_budget(rows, plan(), "train")["convergence_status"] == "BUDGET_DEGRADATION"
    assert select_budget(rows, plan(), "generation")["convergence_status"] == "OBSERVED_PLATEAU"


def test_ratio_gate_is_strict_and_accepts_per_phase_yaml_override():
    rows = [result(value, ratio=1.5) for value in [25, 50, 100, 200, 400]]
    assert select_budget(rows, plan(), "train")["convergence_status"] == "NO_FEASIBLE_CONFIGURATION"
    config = plan()
    config["efficiency"] = {"max_attempts_per_unique": 1.5, "generation": {"max_attempts_per_unique": 1.6}}
    assert select_budget(rows, config, "generation")["convergence_status"] == "OBSERVED_PLATEAU"


def test_unfinished_alternative_configuration_does_not_establish_complete_scan():
    rows = [result(value) for value in [25, 50, 100, 200, 400]]
    other = [result(value, eligible=False, parameter="other") for value in [25, 50, 100, 200, 400]]
    other[-1]["status"] = "RUNNING"
    assert select_budget(rows + other, plan(), "train")["convergence_status"] == "EVIDENCE_INCOMPLETE"


def test_score_tie_band_is_anchored_at_maximum():
    rows = [result(25, 0.7), result(50, 0.709), result(100, 0.718), result(200, 0.718), result(400, 0.718)]
    chosen = select_budget(rows, plan(), "train")
    assert chosen["reference"]["budget"] == 50
    assert chosen["selected_budget"] == 50


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_better_large_budget_other_configuration_prevents_false_global_plateau(direction):
    config = plan(direction)
    config["_budget_grids"]["train"] = [25, 50, 100]
    sign = 1 if direction == "minimize" else -1
    best = 100 - 10 * sign
    rows = [result(value, score=0.8, mean=100, best=best, parameter="A") for value in [25, 50, 100]]
    rows += [result(25, score=0.7, mean=100 + 10 * sign, best=best + 10 * sign, parameter="B"),
             result(50, score=0.75, mean=100 + 5 * sign, best=best + 5 * sign, parameter="B"),
             result(100, score=0.805, mean=100 - 20 * sign, best=best - 20 * sign, parameter="B")]
    chosen = select_budget(rows, config, "train")
    # Preserve the documented score-band / economical-reference selection.
    assert chosen["selected"]["parameter_id"] == "A"
    assert chosen["selected_budget"] == 25
    # Independent convergence evidence must include B's larger-budget quality.
    assert chosen["convergence_status"] == "BUDGET_LIMIT_REACHED"
    evidence = chosen["evidence"]["global_higher_budget_improvements"]
    assert evidence == [{"parameter_id": "B", "budget": 100,
                         "improvements": {"best_fraction_mean": 20.0, "best_value": 20.0},
                         "one_way_degradation": {"score": pytest.approx(-0.005), "qualified_fraction": 0.0,
                                                 "best_fraction_mean": -20.0, "best_value": -20.0,
                                                 "attempts_per_unique": 0.0}}]


def test_other_configuration_ratio_improvement_only_affects_training_plateau():
    config = plan()
    config["_budget_grids"] = {phase: [25, 50, 100] for phase in ["train", "generation"]}
    rows = [result(value, score=0.8, ratio=1.4, parameter="A") for value in [25, 50, 100]]
    rows += [result(25, score=0.7, ratio=1.0, parameter="B"),
             result(50, score=0.75, ratio=1.0, parameter="B"),
             result(100, score=0.805, ratio=1.0, parameter="B")]
    assert select_budget(rows, config, "train")["convergence_status"] == "BUDGET_LIMIT_REACHED"
    generated = select_budget(rows, config, "generation")
    assert generated["convergence_status"] == "OBSERVED_PLATEAU"
    assert generated["evidence"]["global_higher_budget_improvements"] == []


def test_inputs_not_mutated():
    config = plan()
    rows = [result(value) for value in [25, 50, 100, 200, 400]]
    original = deepcopy((config, rows))
    select_budget(rows, config, "train")
    assert (config, rows) == original
