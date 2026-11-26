"""Direct-run seed batches resolve to the unchanged scalar scientific contract."""
from copy import deepcopy
import subprocess
import sys

import pytest

from rl_matdesign.config_schema import ConfigError, resolve_config, resolve_run_configs


@pytest.mark.parametrize("parallel_jobs", [1, 4])
def test_scalar_request_preserves_config_and_provenance(parallel_jobs):
    raw = {"method": "dqn", "eval_seed": 17, "train_seed": None, "gamma": .8}
    overrides = {"gen_seed": 19, "gamma": .9}
    expected = resolve_config(raw, overrides)
    batch = resolve_run_configs({**raw, "parallel_jobs": parallel_jobs}, overrides)
    assert batch.runs == (expected,)
    assert batch.parallel_jobs == parallel_jobs
    assert batch.seed_list_keys == ()
    assert "parallel_jobs" not in batch.runs[0].config
    assert "parallel_jobs" not in batch.runs[0].sources


def test_seed_lists_pair_by_index_instead_of_cartesian_product():
    batch = resolve_run_configs({"eval_seed": [1, 2], "train_seed": [11, 12], "gen_seed": [21, 22]})
    assert len(batch.runs) == 2
    assert batch.parallel_jobs == 1
    assert batch.seed_list_keys == ("eval_seed", "train_seed", "gen_seed")
    for index, run in enumerate(batch.runs):
        assert run == resolve_config({"eval_seed": index + 1, "train_seed": index + 11,
                                      "gen_seed": index + 21})


@pytest.mark.parametrize("explicit_nulls", [False, True])
def test_null_or_missing_training_generation_seeds_follow_each_evaluation_seed(explicit_nulls):
    raw = {"eval_seed": [0, 2**32 - 1]}
    if explicit_nulls:
        raw.update(train_seed=None, gen_seed=None)
    batch = resolve_run_configs(raw)
    for run, seed in zip(batch.runs, raw["eval_seed"]):
        assert [run.config[key] for key in ("eval_seed", "train_seed", "gen_seed")] == [seed] * 3
        assert run.sources["train_seed"] == run.sources["gen_seed"] == "derived:eval_seed"


def test_scalar_seed_broadcast_and_generation_fallback_are_independent():
    batch = resolve_run_configs({"eval_seed": 7, "train_seed": [30, 31], "gen_seed": None})
    assert [(run.config["eval_seed"], run.config["train_seed"], run.config["gen_seed"])
            for run in batch.runs] == [(7, 30, 7), (7, 31, 7)]


def test_length_one_list_remains_explicit_and_duplicate_seed_tuples_are_preserved():
    single = resolve_run_configs({"eval_seed": [9]})
    assert len(single.runs) == 1 and single.seed_list_keys == ("eval_seed",)
    repeated = resolve_run_configs({"eval_seed": [9, 9], "train_seed": [12, 12]})
    assert len(repeated.runs) == 2 and repeated.runs[0] == repeated.runs[1]


@pytest.mark.parametrize("raw", [
    {"eval_seed": [1], "train_seed": [2, 3]},
    {"train_seed": [1, 2], "gen_seed": [3]},
    {"eval_seed": [1, 2], "gen_seed": [3, 4, 5]},
])
def test_all_explicit_lists_require_equal_length_even_length_one(raw):
    with pytest.raises(ConfigError, match="same length.*scalar to broadcast"):
        resolve_run_configs(raw)


@pytest.mark.parametrize("key", ["eval_seed", "train_seed", "gen_seed"])
def test_seed_lists_cannot_be_empty(key):
    with pytest.raises(ConfigError, match=key + " must be a nonempty seed list"):
        resolve_run_configs({key: []})


@pytest.mark.parametrize("key", ["eval_seed", "train_seed", "gen_seed"])
@pytest.mark.parametrize("value", [True, False, -1, 2**32, 1.0, "1", None, [1], {"seed": 1}])
def test_seed_list_elements_are_strict_non_null_uint32(key, value):
    with pytest.raises(ConfigError, match=rf"{key}\[1\]"):
        resolve_run_configs({key: [0, value]})


@pytest.mark.parametrize("key", ["eval_seed", "train_seed", "gen_seed"])
@pytest.mark.parametrize("value", [True, -1, 2**32, 1.0, "1", (1, 2)])
def test_scalar_seeds_keep_strict_integer_validation(key, value):
    with pytest.raises(ConfigError, match=key):
        resolve_run_configs({key: value})


def test_null_evaluation_seed_is_rejected():
    with pytest.raises(ConfigError, match="eval_seed"):
        resolve_run_configs({"eval_seed": None})


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "2", [2]])
def test_parallel_jobs_is_a_positive_integer(value):
    with pytest.raises(ConfigError, match="parallel_jobs"):
        resolve_run_configs({"parallel_jobs": value})


def test_overrides_determine_effective_lists_and_keep_seed_provenance():
    batch = resolve_run_configs(
        {"eval_seed": [1], "train_seed": [], "gen_seed": [5, 6], "parallel_jobs": 2},
        {"eval_seed": [11, 12], "train_seed": 99, "gen_seed": None, "parallel_jobs": 4},
    )
    assert batch.parallel_jobs == 4
    assert batch.seed_list_keys == ("eval_seed", "gen_seed")
    for index, run in enumerate(batch.runs):
        assert run.config["eval_seed"] == 11 + index
        assert run.config["train_seed"] == 99
        assert run.config["gen_seed"] == 5 + index
        assert run.sources["eval_seed"] == run.sources["train_seed"] == "override"
        assert run.sources["gen_seed"] == "yaml"


def test_scalar_override_can_replace_list_and_none_override_is_absent():
    batch = resolve_run_configs({"eval_seed": [1, 2], "parallel_jobs": 3},
                                {"eval_seed": 0, "parallel_jobs": None})
    assert batch.seed_list_keys == () and len(batch.runs) == 1
    assert batch.runs[0].config["eval_seed"] == 0 and batch.parallel_jobs == 3


def test_method_values_and_legacy_generation_selection_survive_expansion():
    raw = {"method": "dqn", "dqn_target_mode": "mc", "exploration_adaptive": False,
           "eval_seed": [1, 2], "num_gen_eps": {"dqn-mc": 5, "a2c": 10},
           "gen_epsilon": .4, "gen_temperature": 0}
    overrides = {"method": "a2c", "gen_top_frac": .2}
    batch = resolve_run_configs(raw, overrides)
    for index, run in enumerate(batch.runs):
        assert run == resolve_config({**raw, "eval_seed": index + 1}, overrides)
        assert run.config["generation_selection"]["parameter"] == "gen_top_frac"


def test_expansion_never_mutates_inputs_or_shares_nested_run_config():
    raw = {"eval_seed": [1, 2], "properties": [{"name": "score", "predictor": "dummy",
                                                "custom_options": ["original"]}]}
    overrides = {"train_seed": [3, 4]}
    before_raw, before_overrides = deepcopy(raw), deepcopy(overrides)
    batch = resolve_run_configs(raw, overrides)
    batch.runs[0].config["properties"][0]["custom_options"].append("changed")
    assert raw == before_raw and overrides == before_overrides
    assert batch.runs[1].config["properties"] == raw["properties"]


def test_non_seed_validation_and_scalar_api_are_unchanged():
    with pytest.raises(ConfigError, match="Unknown experiment key 'parallel_jobs'"):
        resolve_config({"parallel_jobs": 1})
    with pytest.raises(ConfigError, match="eval_seed must be int"):
        resolve_config({"eval_seed": [1]})
    with pytest.raises(ConfigError, match="Did you mean 'dqn_batch_size'"):
        resolve_run_configs({"eval_seed": [1, 2], "dqn_batch_szie": 8})


def test_batch_resolution_does_not_import_ml_backends():
    result = subprocess.run([sys.executable, "-c", """
import sys
from rl_matdesign.config_schema import resolve_run_configs
resolve_run_configs({'eval_seed': [1, 2], 'parallel_jobs': 2})
assert not {'torch', 'ase', 'pymatgen', 'deepmd', 'matminer'} & set(sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
