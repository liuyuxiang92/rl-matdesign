from copy import deepcopy
import math

import pytest

from rl_matdesign.hpo.range_search import initial_reference, run_range_search


def plan(space):
    return {"search_space": space, "range_search": {"seeds": [11, 29], "confirmation_seed": 47, "max_extra_values_per_direction": 2, "score_improvement": 0.02}}


def evaluator(function, calls=None):
    def evaluate(parameters, seeds=None):
        if calls is not None:
            calls.append((dict(parameters), tuple(seeds)))
        records = []
        for seed in seeds:
            score, status = function(parameters, seed)
            records.append({"seed": seed, "score": score, "status": status, "eligible_for_selection": status == "COMPLETED"})
        eligible = all(record["eligible_for_selection"] for record in records)
        return {"score": sum(record["score"] for record in records) / len(records), "status": "COMPLETED" if eligible else next(record["status"] for record in records if not record["eligible_for_selection"]), "eligible_for_selection": eligible, "seed_results": records}
    return evaluate


def test_reference_uses_midpoints_and_lower_middle_category():
    reference = initial_reference({"linear": {"type": "float", "low": 0, "high": 10}, "log": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}, "category": {"type": "categorical", "choices": [64, 128, 256, 512]}})
    assert reference["linear"] == 5
    assert reference["log"] == pytest.approx(math.sqrt(1e-7))
    assert reference["category"] == 128


def test_first_pass_is_one_plus_two_d_and_no_generation():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}, "y": {"type": "float", "low": 0, "high": 10}})
    calls = []
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.5, "COMPLETED"), calls))
    assert len(calls) == 5
    assert outcome["ranges"] == config["search_space"]
    assert all(set(parameters) == {"x", "y"} for parameters, _ in calls)


def test_stable_low_score_keeps_low_boundary():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.1 if parameters["x"] == 0 else 0.5, "COMPLETED")))
    assert outcome["ranges"]["x"]["low"] == 0
    assert len(outcome["evaluations"]) == 3


def test_numerical_upper_boundary_contracts_geometrically():
    config = plan({"dqn_lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "NUMERICAL_INSTABILITY") if parameters["dqn_lr"] > 0.001 else (0.5, "COMPLETED")))
    assert outcome["ranges"]["dqn_lr"]["high"] == pytest.approx(0.0007498942093324558)
    assert outcome["ranges"]["dqn_lr"]["low"] == 1e-5
    assert len([decision for decision in outcome["decisions"] if decision["action"] == "test_inward"]) == 2


def test_runtime_infeasible_numerical_status_contracts_boundary():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "INFEASIBLE_NUMERICAL") if parameters["x"] == 10 else (0.5, "COMPLETED")))
    assert outcome["ranges"]["x"]["high"] == 7.5


def test_reproducible_endpoint_improvement_expands_only_tested_values():
    config = plan({"dqn_lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.7 + 0.05 * math.log10(parameters["dqn_lr"]), "COMPLETED")))
    assert outcome["ranges"]["dqn_lr"]["high"] == 1
    assert outcome["ranges"]["dqn_lr"]["low"] == 1e-5
    assert outcome["boundary_status"]["dqn_lr"]["high"] == "unknown"


def test_outward_stable_but_worse_keeps_existing_boundary():
    config = plan({"dqn_lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.7 if parameters["dqn_lr"] == 0.01 else 0.5, "COMPLETED")))
    assert outcome["ranges"]["dqn_lr"]["high"] == 0.01
    assert any(item["parameters"]["dqn_lr"] == 0.1 for item in outcome["evaluations"])


def test_both_endpoints_improve_without_single_peak_assumption():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.4 + 0.01 * abs(parameters["x"] - 5), "COMPLETED")))
    assert outcome["ranges"]["x"] == {"type": "float", "low": -10.0, "high": 20.0}


def test_efficiency_failure_does_not_contract_individual_range():
    config = plan({"dqn_eps_min": {"type": "float", "low": 0.02, "high": 0.4}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "INFEASIBLE_EFFICIENCY") if parameters["dqn_eps_min"] == 0.02 else (0.5, "COMPLETED")))
    assert outcome["ranges"] == config["search_space"]
    assert outcome["boundary_status"]["dqn_eps_min"]["low"] == "unresolved"


def test_discrete_extension_uses_declared_adjacent_values():
    config = plan({"dqn_batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.1 * math.log2(parameters["dqn_batch_size"]), "COMPLETED")))
    assert outcome["ranges"]["dqn_batch_size"]["choices"] == [64, 128, 256, 512, 1024, 2048]


def test_profile_extension_choices_override_builtin_values():
    config = plan({"dqn_batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]}})
    config["extension_choices"] = {"dqn_batch_size": [64, 128, 256, 512, 768]}
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (0.1 * math.log2(parameters["dqn_batch_size"]), "COMPLETED")))
    assert outcome["ranges"]["dqn_batch_size"]["choices"] == [64, 128, 256, 512, 768]


def test_warmup_expansion_does_not_round_to_full_training_budget():
    config = plan({"warmup_budget_fraction": {"type": "float", "low": 0.1, "high": 0.8}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (parameters["warmup_budget_fraction"], "COMPLETED")))
    assert outcome["ranges"]["warmup_budget_fraction"]["high"] == 0.8
    assert len(outcome["evaluations"]) == 3


def test_conflicting_seeds_trigger_independent_confirmation_without_expansion():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    calls = []
    def response(parameters, seed):
        if parameters["x"] == 10:
            return (0.7 if seed in [11, 47] else 0.4), "COMPLETED"
        return 0.5, "COMPLETED"
    outcome = run_range_search(config, evaluator(response, calls))
    assert outcome["ranges"] == config["search_space"]
    assert any(seeds == (11, 29, 47) for _, seeds in calls)


def test_single_numerical_failure_confirmed_twice_then_contracts():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    def response(parameters, seed):
        return (-1, "NUMERICAL_INSTABILITY") if parameters["x"] == 10 and seed in [29, 47] else (0.5, "COMPLETED")
    outcome = run_range_search(config, evaluator(response))
    assert outcome["ranges"]["x"]["high"] == 7.5


def test_reference_failure_replaced_and_comparisons_redone():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}, "y": {"type": "float", "low": 0, "high": 10}})
    def response(parameters, seed):
        return (0.5, "COMPLETED") if parameters["x"] == 0 else (-1, "INFEASIBLE_EFFICIENCY")
    outcome = run_range_search(config, evaluator(response))
    assert outcome["reference"]["x"] == 0
    assert any(item["action"] == "replace_reference" for item in outcome["decisions"])
    assert any(item["parameters"] == {"x": 0, "y": 10} for item in outcome["evaluations"])


def test_all_failed_references_have_bounded_joint_fallback():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "INFEASIBLE_EFFICIENCY")))
    assert outcome["status"] == "NO_FEASIBLE_CONFIGURATION"
    assert outcome["ranges"] is None
    assert len(outcome["evaluations"]) <= 5


def test_system_failure_does_not_trigger_parameter_random_search():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "SYSTEM_ERROR")))
    assert outcome["status"] == "SYSTEM_ERROR"
    assert len(outcome["evaluations"]) == 1


def test_unstable_interior_exports_tested_discrete_range():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, evaluator(lambda parameters, seed: (-1, "NUMERICAL_INSTABILITY") if parameters["x"] == 5 else (0.5, "COMPLETED")))
    assert outcome["ranges"]["x"] == {"type": "categorical", "choices": [0, 10]}
    assert outcome["boundary_status"]["x"]["low"] == "conditional_discrete"


def test_missing_seed_evidence_never_manufactures_paired_improvement():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    outcome = run_range_search(config, lambda parameters: {"score": parameters["x"] / 10, "eligible_for_selection": True, "status": "COMPLETED"})
    assert outcome["ranges"] == config["search_space"]
    assert outcome["range_status"] == "unresolved"


def test_initial_plan_not_mutated():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    original = deepcopy(config)
    run_range_search(config, evaluator(lambda parameters, seed: (0.4 + 0.01 * parameters["x"], "COMPLETED")))
    assert config == original


def test_failed_reference_confirmation_keeps_boundary_and_supersedes_stale_success():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    def response(parameters, seed):
        value = parameters["x"]
        failed = (value == 10 and seed in [29, 47]) or (value == 5 and seed == 47) or (5 < value < 10)
        return (-1, "INFEASIBLE_NUMERICAL") if failed else (0.5, "COMPLETED")
    outcome = run_range_search(config, evaluator(response))
    assert outcome["ranges"] == config["search_space"]
    assert outcome["boundary_status"]["x"]["high"] == "unresolved"
    assert outcome["range_status"] == "unresolved"
    assert not outcome["reference_result"]["eligible_for_selection"]
    assert outcome["reference_result"]["requested_seeds"] == [11, 29, 47]
    current = [item for item in outcome["evaluations"] if item["parameters"] == {"x": 5}]
    assert len(current) == 1 and not current[0]["eligible_for_selection"]
    history = [item for item in outcome["measurement_history"] if item["parameters"] == {"x": 5}]
    assert len(history) == 2 and history[0]["eligible_for_selection"]


def test_failed_reference_cannot_drive_later_parameter_expansion():
    config = plan({"x": {"type": "float", "low": 0, "high": 10},
                   "y": {"type": "float", "low": 0, "high": 10}})
    def response(parameters, seed):
        x, y = parameters["x"], parameters["y"]
        if (x == 10 and y == 5 and seed in [29, 47]) or (x == y == 5 and seed == 47):
            return -1, "INFEASIBLE_NUMERICAL"
        return (0.8 if y == 10 else 0.5), "COMPLETED"
    outcome = run_range_search(config, evaluator(response))
    assert outcome["ranges"] == config["search_space"]
    assert outcome["boundary_status"]["y"] == {"low": "unresolved", "high": "unresolved"}
    assert not any(item["parameters"]["y"] > 10 for item in outcome["evaluations"])


def test_outward_baseline_new_seed_failure_cannot_be_hidden_by_old_success():
    config = plan({"x": {"type": "float", "low": 0, "high": 10}})
    def response(parameters, seed):
        value = parameters["x"]
        if value == 10:
            return (-1, "INFEASIBLE_NUMERICAL") if seed == 47 else (0.7, "COMPLETED")
        if value == 15:
            return (0.6 if seed == 29 else 0.9), "COMPLETED"
        return 0.5, "COMPLETED"
    outcome = run_range_search(config, evaluator(response))
    assert outcome["ranges"] == config["search_space"]
    assert outcome["boundary_status"]["x"]["high"] == "unresolved"
    latest_high = [item for item in outcome["evaluations"] if item["parameters"] == {"x": 10}]
    assert len(latest_high) == 1 and not latest_high[0]["eligible_for_selection"]
