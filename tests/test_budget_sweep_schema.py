"""Training-budget sweeps cross budgets with the existing paired seed rows."""
from copy import deepcopy

import pytest

from rl_matdesign.config_schema import ConfigError, resolve_config, resolve_run_configs


def test_budget_major_cartesian_product_preserves_all_seed_pairings_and_provenance():
    raw = {
        "method": "a2c", "train_budget": [100, 200], "eval_seed": [1, 2, 3],
        "train_seed": [11, 12, 13], "gen_seed": [21, 22, 23], "parallel_jobs": 4,
    }
    batch = resolve_run_configs(raw, {"gamma": .8})
    assert batch.is_sweep and batch.parallel_jobs == 4
    assert batch.train_budget_values == (100, 200)
    assert batch.seed_list_keys == ("eval_seed", "train_seed", "gen_seed")
    assert [tuple(run.config[key] for key in ("train_budget", "eval_seed", "train_seed", "gen_seed"))
            for run in batch.runs] == [
                (budget, seed, seed + 10, seed + 20)
                for budget in (100, 200) for seed in (1, 2, 3)
            ]
    for run in batch.runs:
        scalar = {key: value for key, value in raw.items() if key != "parallel_jobs"}
        scalar.update({key: run.config[key] for key in ("train_budget", *batch.seed_list_keys)})
        assert run == resolve_config(scalar, {"gamma": .8})
        assert run.sources["train_budget"] == run.sources["train_seed"] == "yaml"
        assert run.sources["gamma"] == "override"
        for orchestration_key in ("parallel_jobs", "train_budget_values", "seed_list_keys", "is_sweep"):
            assert orchestration_key not in run.config and orchestration_key not in run.sources


def test_twenty_budgets_and_twenty_training_seeds_produce_four_hundred_runs():
    budgets = list(range(100, 2001, 100))
    seeds = list(range(20))
    batch = resolve_run_configs({"train_budget": budgets, "train_seed": seeds, "eval_seed": 42})
    assert len(batch.runs) == 400
    assert [(run.config["train_budget"], run.config["train_seed"]) for run in batch.runs] == [
        (budget, seed) for budget in budgets for seed in seeds
    ]
    assert all(run.config["eval_seed"] == run.config["gen_seed"] == 42 for run in batch.runs)


def test_budget_sweep_broadcasts_scalar_seeds_and_keeps_null_fallback():
    batch = resolve_run_configs({"train_budget": [100, 200], "eval_seed": [7, 8],
                                 "train_seed": 9, "gen_seed": None})
    for index, run in enumerate(batch.runs):
        assert run.config["eval_seed"] == run.config["gen_seed"] == 7 + index % 2
        assert run.config["train_seed"] == 9
        assert run.sources["gen_seed"] == "derived:eval_seed"


@pytest.mark.parametrize("authored", [[100, 200], [], [None]])
def test_scalar_cli_budget_supersedes_authored_list(authored):
    batch = resolve_run_configs({"train_budget": authored, "train_seed": [3, 4]},
                                {"train_budget": 300})
    assert batch.train_budget_values == ()
    assert batch.is_sweep and len(batch.runs) == 2
    assert all(run.config["train_budget"] == 300 and run.sources["train_budget"] == "override"
               for run in batch.runs)


def test_scalar_budget_override_removes_budget_only_sweep():
    raw = {"train_budget": [100, 200]}
    batch = resolve_run_configs(raw, {"train_budget": 300})
    assert not batch.is_sweep and batch.train_budget_values == ()
    assert batch.runs == (resolve_config({"train_budget": 100}, {"train_budget": 300}),)


def test_none_override_does_not_replace_authored_budget_list():
    batch = resolve_run_configs({"train_budget": [100]}, {"train_budget": None})
    assert batch.is_sweep and batch.train_budget_values == (100,)
    assert batch.runs[0].sources["train_budget"] == "yaml"


@pytest.mark.parametrize("raw", [{"train_budget": 100}, {"train_budget": None}, {}])
def test_scalar_requests_preserve_exact_config_and_sources(raw):
    batch = resolve_run_configs(raw)
    assert batch.runs == (resolve_config(raw),)
    assert not batch.is_sweep and batch.train_budget_values == ()


def test_seed_only_singleton_and_budget_only_singleton_are_explicit_sweeps():
    seed = resolve_run_configs({"train_seed": [9]})
    budget = resolve_run_configs({"train_budget": [100]})
    assert seed.is_sweep and seed.train_budget_values == () and len(seed.runs) == 1
    assert budget.is_sweep and budget.seed_list_keys == () and len(budget.runs) == 1
    assert budget.train_budget_values == (100,)
    assert budget.runs == (resolve_config({"train_budget": 100}),)


def test_repeated_budget_entries_preserve_authored_order():
    batch = resolve_run_configs({"train_budget": [200, 100, 200]})
    assert batch.train_budget_values == (200, 100, 200)
    assert batch.runs[0] == batch.runs[2]


def test_budget_list_cannot_be_empty():
    with pytest.raises(ConfigError, match="train_budget must be a nonempty budget list"):
        resolve_run_configs({"train_budget": []})


@pytest.mark.parametrize("value", [0, -1, None, True, False, 100.0, "100", [100], {"a2c": 100}])
def test_budget_list_elements_are_strict_positive_non_null_integers(value):
    with pytest.raises(ConfigError, match=r"train_budget\[1\]"):
        resolve_run_configs({"train_budget": [100, value]})


def test_budget_sweep_does_not_relax_paired_seed_length_validation():
    with pytest.raises(ConfigError, match="Seed lists must have the same length"):
        resolve_run_configs({"train_budget": [100, 200], "eval_seed": [1, 2], "train_seed": [3]})


@pytest.mark.parametrize("value", [[100, 200], {"a2c": 100}])
def test_cli_budget_overrides_remain_scalar(value):
    with pytest.raises(ConfigError, match="train_budget must be int"):
        resolve_run_configs({"train_budget": [100]}, {"train_budget": value})


@pytest.mark.parametrize("method,mode,budget", [("a2c", "bootstrap", 100),
                                               ("dqn", "bootstrap", 600), ("dqn", "mc", 700)])
def test_method_specific_scalar_budget_mapping_survives_seed_expansion(method, mode, budget):
    raw = {"method": method, "dqn_target_mode": mode, "train_seed": [1, 2],
           "train_budget": {"a2c": 100, "dqn-bootstrap": 600, "dqn-mc": 700}}
    batch = resolve_run_configs(raw)
    assert batch.train_budget_values == () and batch.is_sweep
    for index, run in enumerate(batch.runs):
        assert run == resolve_config({**raw, "train_seed": index + 1})
        assert run.config["train_budget"] == budget and run.sources["train_budget"] == "yaml"


def test_budget_lists_inside_method_mappings_are_not_supported_even_when_inactive():
    with pytest.raises(ConfigError, match=r"train_budget.dqn-mc must be int"):
        resolve_run_configs({"method": "a2c", "train_budget": {"a2c": 100, "dqn-mc": [600, 700]}})


def test_budget_expansion_and_nested_runs_do_not_mutate_or_share_inputs():
    raw = {"train_budget": [100, 200], "properties": [
        {"name": "score", "predictor": "dummy", "custom_options": ["original"]},
    ]}
    overrides = {"train_seed": [3, 4]}
    before_raw, before_overrides = deepcopy(raw), deepcopy(overrides)
    batch = resolve_run_configs(raw, overrides)
    batch.runs[0].config["properties"][0]["custom_options"].append("changed")
    assert raw == before_raw and overrides == before_overrides
    assert all(run.config["properties"] == raw["properties"] for run in batch.runs[1:])


@pytest.mark.parametrize("method", ["a2c", "reinforce"])
def test_small_policy_gradient_budget_ignores_dormant_dqn_warmup(method):
    raw = {"method": method, "train_budget": [100, 200], "pg_batch_eps": 16}
    batch = resolve_run_configs(raw)
    assert not any(key.startswith("dqn_") for key in raw)
    for run, iterations in zip(batch.runs, (7, 13)):
        assert run.config["pg_num_iters"] == iterations
        assert run.sources["pg_num_iters"] == "derived:ceil(train_budget/pg_batch_eps)"
        assert run.config["dqn_warmup_eps"] == 500 and run.sources["dqn_warmup_eps"] == "default"
        assert run.config["dqn_num_train_eps"] == 20000 and run.sources["dqn_num_train_eps"] == "default"
        assert resolve_config(run.config).config == run.config


def test_small_policy_gradient_budget_preserves_authored_dormant_dqn_settings():
    resolved = resolve_config({"method": "a2c", "train_budget": 100, "dqn_warmup_eps": 200,
                               "dqn_num_train_eps": 300})
    assert resolved.config["dqn_warmup_eps"] == 200
    assert resolved.config["dqn_num_train_eps"] == 300
    assert resolved.sources["dqn_num_train_eps"] == "yaml"


@pytest.mark.parametrize("method", ["a2c", "dqn", "reinforce"])
def test_previously_valid_budget_derivations_are_unchanged(method):
    batch = resolve_run_configs({"method": method, "train_budget": [600, 800], "pg_batch_eps": 16})
    for run, budget in zip(batch.runs, (600, 800)):
        assert run.config["dqn_num_train_eps"] == budget - 500
        assert run.sources["dqn_num_train_eps"] == "derived:train_budget-dqn_warmup_eps"
        assert run.config["pg_num_iters"] == (budget + 15) // 16
        assert resolve_config(run.config).config == run.config


@pytest.mark.parametrize("mode", ["bootstrap", "mc"])
def test_active_dqn_rejects_any_budget_below_warmup(mode):
    with pytest.raises(ConfigError, match="dqn_warmup_eps exceeds train_budget"):
        resolve_run_configs({"method": "dqn", "dqn_target_mode": mode, "train_budget": [600, 100]})


def test_active_dqn_still_rejects_conflicting_authored_episode_count():
    with pytest.raises(ConfigError, match="dqn_num_train_eps=100 conflicts with train_budget=600"):
        resolve_config({"method": "dqn", "train_budget": 600, "dqn_warmup_eps": 400,
                        "dqn_num_train_eps": 100})


def test_scalar_resolution_does_not_accept_budget_lists_directly():
    with pytest.raises(ConfigError, match="train_budget must be int"):
        resolve_config({"train_budget": [100, 200]})
