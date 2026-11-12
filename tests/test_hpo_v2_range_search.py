"""Range diagnosis must honor the fixed public training repeat count."""
from copy import deepcopy
import statistics

import pytest

from rl_matdesign.hpo.range_search import run_range_search


def plan(seeds):
    return {
        "protocol": "hpo_v2",
        "search_space": {"x": {"type": "float", "low": 1., "high": 9.}},
        "range_search": {"seeds": list(seeds), "score_improvement": .01,
                         "max_extra_values_per_direction": 2},
    }


def evaluator(calls, observations):
    def evaluate(parameters, seeds):
        calls.append({"parameters": deepcopy(parameters), "seeds": list(seeds)})
        rows = []
        for seed in seeds:
            observed = observations(parameters["x"], seed)
            eligible = isinstance(observed, (int, float))
            rows.append({"seed": seed, "score": observed if eligible else -1,
                         "status": "COMPLETED" if eligible else observed,
                         "eligible_for_selection": eligible})
        eligible = all(row["eligible_for_selection"] for row in rows)
        return {"eligible_for_selection": eligible,
                "score": statistics.mean(row["score"] for row in rows) if eligible else -1,
                "status": "COMPLETED" if eligible else next(
                    row["status"] for row in rows if not row["eligible_for_selection"]),
                "seed_results": rows}
    return evaluate


@pytest.mark.parametrize("legacy_confirmation_field", [False, True])
def test_disagreeing_seed_trends_stay_unresolved_without_adding_repeats(legacy_confirmation_field):
    settings = plan([11, 29])
    if legacy_confirmation_field:
        # Even stale normalized metadata must not override v2's fixed count.
        settings["range_search"]["confirmation_seed"] = 47
    original = deepcopy(settings["search_space"])
    calls = []

    def observations(value, seed):
        if value == 9:
            return .8 if seed == 11 else .4
        return .5

    result = run_range_search(settings, evaluator(calls, observations))
    assert result["status"] == "COMPLETED"
    assert result["ranges"] == original
    assert result["boundary_status"]["x"]["high"] == "unresolved"
    assert result["range_status"] == "unresolved"
    assert len(calls) == 3  # Reference and two initial endpoints only.
    assert all(call["seeds"] == [11, 29] for call in calls)
    assert all(row["requested_seeds"] == [11, 29] for row in result["measurement_history"])
    assert not any(decision["action"] in {"paired_confirmation", "expand", "contract"}
                   for decision in result["decisions"])
    assert any(decision["action"] == "unresolved" and "training.repeats" in decision["reason"]
               for decision in result["decisions"])


def test_one_numerical_failure_in_fixed_cohort_does_not_buy_confirmation_or_contract():
    settings = plan([11, 29])
    calls = []

    def observations(value, seed):
        return "INFEASIBLE_NUMERICAL" if value == 9 and seed == 11 else .5

    result = run_range_search(settings, evaluator(calls, observations))
    assert result["status"] == "COMPLETED"
    assert result["ranges"] == settings["search_space"]
    assert result["boundary_status"]["x"]["high"] == "unresolved"
    assert all(call["seeds"] == [11, 29] for call in calls)
    assert len(calls) == 3
    assert not any(decision["action"] in {"paired_confirmation", "contract"}
                   for decision in result["decisions"])


@pytest.mark.parametrize("unstable_endpoints", [False, True])
def test_single_training_repeat_is_supported_without_claiming_repeated_instability(unstable_endpoints):
    settings = plan([11])
    calls = []

    def observations(value, seed):
        return "INFEASIBLE_NUMERICAL" if unstable_endpoints and value in {1, 9} else .5

    result = run_range_search(settings, evaluator(calls, observations))
    assert result["status"] == "COMPLETED"
    assert result["ranges"] == settings["search_space"]
    assert len(calls) == 3
    assert all(call["seeds"] == [11] for call in calls)
    assert all(len(row["seed_results"]) == 1 for row in result["measurement_history"])
    assert not any(decision["action"] in {"contract", "tested_discrete_only", "paired_confirmation"}
                   for decision in result["decisions"])
    if unstable_endpoints:
        assert result["boundary_status"]["x"] == {"low": "unresolved", "high": "unresolved"}
        assert result["range_status"] == "unresolved"


def test_two_observed_numerical_failures_can_still_support_conservative_contraction():
    settings = plan([11, 29])
    calls = []

    def observations(value, seed):
        return "INFEASIBLE_NUMERICAL" if value >= 9 else .5

    result = run_range_search(settings, evaluator(calls, observations))
    assert result["status"] == "COMPLETED"
    assert result["ranges"]["x"]["high"] < 9
    assert result["boundary_status"]["x"]["high"] == "conditional_stability_boundary"
    assert any(decision["action"] == "contract" for decision in result["decisions"])
    assert all(call["seeds"] == [11, 29] for call in calls)
