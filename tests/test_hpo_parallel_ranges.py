"""Independent range proposals can batch without changing scientific evidence."""
from copy import deepcopy

import pytest

from rl_matdesign.hpo.range_search import run_range_search


def plan(space=None, protocol="hpo_v2"):
    return {
        "protocol": protocol,
        "search_space": space or {"x": {"type": "float", "low": 0, "high": 10},
                                  "y": {"type": "float", "low": 0, "high": 10}},
        "range_search": {"seeds": [11, 29], "confirmation_seed": 47,
                         "max_extra_values_per_direction": 2, "score_improvement": .02},
    }


class Evaluator:
    def __init__(self, function):
        self.function = function
        self.scalar_calls = []
        self.batch_calls = []
        self.logical_calls = []

    def result(self, parameters, seeds):
        rows = []
        for seed in seeds:
            score, status = self.function(parameters, seed)
            rows.append({"seed": seed, "score": score, "status": status,
                         "eligible_for_selection": status == "COMPLETED"})
        eligible = all(row["eligible_for_selection"] for row in rows)
        return {"score": sum(row["score"] for row in rows) / len(rows),
                "status": "COMPLETED" if eligible else next(
                    row["status"] for row in rows if not row["eligible_for_selection"]),
                "eligible_for_selection": eligible, "seed_results": rows}

    def scalar(self, parameters, seeds):
        call = (deepcopy(parameters), tuple(seeds))
        self.scalar_calls.append(call)
        self.logical_calls.append(call)
        return self.result(parameters, seeds)

    def batch(self, parameters_list, seeds):
        self.batch_calls.append((deepcopy(parameters_list), tuple(seeds)))
        self.logical_calls.extend((deepcopy(parameters), tuple(seeds)) for parameters in parameters_list)
        return [self.result(parameters, seeds) for parameters in parameters_list]


def compare_serial_and_batch(settings, function):
    original = deepcopy(settings)
    serial, batched = Evaluator(function), Evaluator(function)
    serial_result = run_range_search(settings, serial.scalar)
    batch_result = run_range_search(settings, batched.scalar, evaluate_many=batched.batch)
    assert batch_result == serial_result
    assert batched.logical_calls == serial.logical_calls
    assert settings == original
    return batch_result, batched


def test_initial_reference_and_two_endpoints_per_parameter_are_one_batch():
    result, measured = compare_serial_and_batch(plan(), lambda parameters, seed: (.5, "COMPLETED"))
    assert measured.batch_calls == [([
        {"x": 5., "y": 5.}, {"x": 0, "y": 5.}, {"x": 10, "y": 5.},
        {"x": 5., "y": 0}, {"x": 5., "y": 10},
    ], (11, 29))]
    assert not measured.scalar_calls
    assert len(result["measurement_history"]) == 1 + 2 * 2
    assert [row["parameters"] for row in result["measurement_history"]] == measured.batch_calls[0][0]


def test_batch_deduplicates_fixed_and_reference_equal_endpoints():
    settings = plan({"x": {"type": "categorical", "choices": [3]},
                     "y": {"type": "categorical", "choices": [0, 10]}})
    result, measured = compare_serial_and_batch(settings, lambda parameters, seed: (.5, "COMPLETED"))
    assert measured.batch_calls == [([{"x": 3, "y": 0}, {"x": 3, "y": 10}], (11, 29))]
    assert len(result["measurement_history"]) == 2
    assert not measured.scalar_calls


def test_replacement_reference_batches_only_previously_unmeasured_endpoints():
    def response(parameters, seed):
        return (.5, "COMPLETED") if parameters["x"] == 0 else (-1, "INFEASIBLE_EFFICIENCY")

    result, measured = compare_serial_and_batch(plan(), response)
    assert result["reference"] == {"x": 0, "y": 5.}
    assert len(measured.batch_calls) == 2
    assert measured.batch_calls[1] == ([{"x": 0, "y": 0}, {"x": 0, "y": 10}], (11, 29))
    assert len(result["measurement_history"]) == 7
    assert not measured.scalar_calls
    assert any(decision["action"] == "replace_reference" for decision in result["decisions"])


def test_adaptive_probes_remain_ordered_after_initial_batch():
    def response(parameters, seed):
        return .4 + .01 * abs(parameters["x"] - 5) + .02 * abs(parameters["y"] - 5), "COMPLETED"

    result, measured = compare_serial_and_batch(plan(), response)
    assert len(measured.batch_calls) == 1
    assert len(measured.scalar_calls) == 8
    assert result["ranges"]["x"]["low"] < 0
    assert result["ranges"]["y"]["high"] > 10
    assert all(seeds == (11, 29) for _, seeds in measured.scalar_calls)


def test_random_replacement_fallback_stays_adaptive_then_batches_its_endpoints():
    def response(parameters, seed):
        valid = all(value not in {0, 5, 10} for value in parameters.values())
        return (.5, "COMPLETED") if valid else (-1, "INFEASIBLE_EFFICIENCY")

    result, measured = compare_serial_and_batch(plan(), response)
    assert result["status"] == "COMPLETED"
    assert len(measured.batch_calls) == 2
    assert len(measured.batch_calls[1][0]) == 4
    assert len(measured.scalar_calls) == 1
    assert measured.scalar_calls[0][0] == result["reference"]


def test_legacy_confirmation_uses_distinct_seed_cohort_cache_keys():
    settings = plan({"x": {"type": "float", "low": 0, "high": 10}}, protocol="hpo_v1")

    def response(parameters, seed):
        if parameters["x"] == 10:
            return (.7 if seed in {11, 47} else .4), "COMPLETED"
        return .5, "COMPLETED"

    result, measured = compare_serial_and_batch(settings, response)
    assert len(measured.batch_calls) == 1
    assert measured.scalar_calls == [({"x": 10}, (11, 29, 47)), ({"x": 5.}, (11, 29, 47))]
    assert len(result["measurement_history"]) == 5
    assert any(decision["action"] == "paired_confirmation" for decision in result["decisions"])


def test_serial_callback_retains_immediate_reference_error_short_circuit():
    measured = Evaluator(lambda parameters, seed: (-1, "SYSTEM_ERROR"))
    result = run_range_search(plan(), measured.scalar)
    assert result["status"] == "SYSTEM_ERROR"
    assert len(measured.scalar_calls) == 1
    assert len(result["measurement_history"]) == 1


def test_batch_failure_retains_all_already_executed_proposals_in_logical_order():
    measured = Evaluator(lambda parameters, seed: (-1, "SYSTEM_ERROR"))
    result = run_range_search(plan(), measured.scalar, evaluate_many=measured.batch)
    assert result["status"] == "SYSTEM_ERROR"
    assert len(measured.batch_calls) == 1
    assert not measured.scalar_calls
    assert len(result["measurement_history"]) == 5
    assert [row["parameters"] for row in result["measurement_history"]] == measured.batch_calls[0][0]


@pytest.mark.parametrize("response,exception,message", [
    ([], ValueError, "one aggregate"),
    ([None] * 5, TypeError, "mapping aggregates"),
])
def test_batch_callback_contract_errors_are_explicit(response, exception, message):
    measured = Evaluator(lambda parameters, seed: (.5, "COMPLETED"))
    with pytest.raises(exception, match=message):
        run_range_search(plan(), measured.scalar, evaluate_many=lambda parameters, seeds: response)
    assert not measured.scalar_calls
