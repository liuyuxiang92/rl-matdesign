"""Configuration behavior is testable without importing ML or oracle backends."""
from copy import deepcopy
from pathlib import Path

import pytest

from rl_matdesign.config_schema import (
    ConfigError,
    load_yaml_config,
    resolve_config,
    schema_markdown,
    validate_config,
)


@pytest.mark.parametrize("text", ["", "null\n", "- a\n- b\n", "42\n"])
def test_yaml_requires_a_mapping(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_config(path)


@pytest.mark.parametrize("text", [
    "method: dqn\nmethod: a2c\n",
    "properties:\n  - name: a\n    model: first\n    model: second\n",
    "groups:\n  - element_bounds: {Ni: [0, 1], Ni: [0.1, 0.9]}\n",
])
def test_yaml_duplicate_keys_are_never_silently_overwritten(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="Duplicate YAML key") as exc:
        load_yaml_config(path)
    assert str(path) in str(exc.value)
    assert "line" in str(exc.value)


def test_yaml_merge_can_override_inherited_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("defaults: &defaults {count: 2}\nmodel: {<<: *defaults, count: 4}\n")
    assert load_yaml_config(path)["model"] == {"count": 4}


@pytest.mark.parametrize("key", [
    "buffer_size", "num_train_eps", "grad_steps_per_ep", "target_update_freq",
    "eps_anneal_eps", "eps_min", "entropy_coef", "repeat_penalty_coef",
    "repeat_penalty_shape", "dp_seed", "dqn_gamma", "pg_gamma",
    "exploration_gen_eps", "predict_cache_size", "generated_columns",
])
@pytest.mark.parametrize("source", ["yaml", "override"])
def test_removed_core_parameters_are_rejected(key, source):
    raw, overrides = ({key: 1}, {}) if source == "yaml" else ({}, {key: 1})
    with pytest.raises(ConfigError, match=f"Unknown experiment key '{key}'"):
        resolve_config(raw, overrides)


def test_removed_alias_is_rejected_even_with_a_canonical_override():
    with pytest.raises(ConfigError, match="Unknown experiment key 'dp_seed'"):
        resolve_config({"dp_seed": 9}, {"eval_seed": 10})


def test_resolving_config_does_not_mutate_nested_authored_settings():
    raw = {"eval_seed": 12, "properties": [{"name": "score", "predictor": "dummy"}]}
    original = deepcopy(raw)
    result = resolve_config(raw)
    result.config["properties"][0]["name"] = "changed"
    assert raw == original


def test_defaults_preserve_distinct_dqn_and_pg_warmup():
    result = resolve_config({})
    assert result.config["dqn_warmup_eps"] == 500
    assert result.config["pg_warmup_eps"] == 200
    assert result.sources["dqn_warmup_eps"] == "default"
    assert result.config["train_budget"] is None
    assert result.config["train_count_duplicates"] is False
    assert result.sources["train_count_duplicates"] == "default"


def test_shared_budget_derives_both_algorithms_without_redundant_totals():
    result = resolve_config({"train_budget": 101, "dqn_warmup_eps": 11, "pg_batch_eps": 16,
                             "pg_warmup_eps": 999})
    assert result.config["dqn_num_train_eps"] == 90
    assert result.config["pg_num_iters"] == 7
    assert result.sources["dqn_num_train_eps"] == "derived:train_budget-dqn_warmup_eps"
    assert result.sources["pg_num_iters"] == "derived:ceil(train_budget/pg_batch_eps)"
    changed = resolve_config({"train_budget": 101, "dqn_warmup_eps": 11, "pg_batch_eps": 16},
                             {"pg_batch_eps": 20, "train_budget": 201})
    assert changed.config["dqn_num_train_eps"] == 190 and changed.config["pg_num_iters"] == 11


@pytest.mark.parametrize("extra", [{"dqn_num_train_eps": 100}, {"pg_num_iters": 1},
                                   {"dqn_warmup_eps": 102}])
def test_shared_budget_rejects_conflicting_legacy_limits_or_excess_warmup(extra):
    with pytest.raises(ConfigError, match="train_budget"):
        resolve_config({"train_budget": 101, "dqn_warmup_eps": 11, "pg_batch_eps": 16, **extra})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "100"])
def test_shared_budget_is_a_positive_integer_or_none(value):
    with pytest.raises(ConfigError, match="train_budget"):
        resolve_config({"train_budget": value})


def test_equal_legacy_limits_are_accepted_but_have_shared_budget_origin():
    resolved = resolve_config({"train_budget": 10, "dqn_warmup_eps": 10,
                               "dqn_num_train_eps": 0, "pg_batch_eps": 3, "pg_num_iters": 4})
    assert resolved.config["dqn_num_train_eps"] == 0
    assert resolved.sources["pg_num_iters"].startswith("derived:")


@pytest.mark.parametrize("raw,overrides,warmup,source", [
    ({"pg_warmup_eps": 40}, {}, 500, "default"),
    ({"pg_warmup_eps": 40}, {"pg_warmup_eps": 70}, 500, "default"),
    ({"pg_warmup_eps": 40, "dqn_warmup_eps": 80}, {}, 80, "yaml"),
    ({"pg_warmup_eps": 40}, {"dqn_warmup_eps": 0}, 0, "override"),
])
def test_dqn_warmup_is_independent_of_pg_warmup(raw, overrides, warmup, source):
    result = resolve_config(raw, overrides)
    assert result.config["dqn_warmup_eps"] == warmup
    assert result.sources["dqn_warmup_eps"] == source


def test_gamma_has_one_default_and_one_override():
    default = resolve_config({})
    assert default.config["gamma"] == 0.9
    assert default.sources["gamma"] == "default"
    result = resolve_config({"gamma": 0.8}, {"gamma": 0.95})
    assert result.config["gamma"] == 0.95
    assert result.sources["gamma"] == "override"
    assert "dqn_gamma" not in result.config
    assert "pg_gamma" not in result.config


def test_false_zero_and_none_cli_values_have_distinct_meanings():
    result = resolve_config(
        {"method": "dqn", "train_count_duplicates": True,
         "eval_seed": 9, "gen_epsilon": 0.2},
        {"method": None, "train_count_duplicates": False,
         "eval_seed": 0, "gen_epsilon": 0},
    )
    assert result.config["method"] == "dqn"
    assert result.sources["method"] == "yaml"
    assert result.config["train_count_duplicates"] is False
    assert result.config["eval_seed"] == 0
    assert result.config["gen_epsilon"] == 0
    assert result.sources["train_count_duplicates"] == "override"


def test_method_override_retains_inactive_algorithm_parameters():
    result = resolve_config(
        {"method": "dqn", "dqn_lr": 0.0002, "pg_lr_actor": 0.0004},
        {"method": "a2c"},
    )
    assert result.config["method"] == "a2c"
    assert result.config["dqn_lr"] == 0.0002
    assert result.config["pg_lr_actor"] == 0.0004


def test_seed_fallbacks_and_generation_cap_follow_effective_values():
    result = resolve_config({"eval_seed": 42, "train_seed": None, "num_gen_eps": 12},
                            {"gen_seed": 7, "num_gen_eps": 15})
    assert result.config["train_seed"] == 42
    assert result.sources["train_seed"] == "derived:eval_seed"
    assert result.config["gen_seed"] == 7
    assert result.sources["gen_seed"] == "override"
    assert result.config["max_gen_attempts"] == 150
    assert result.sources["max_gen_attempts"] == "derived:10*num_gen_eps"


@pytest.mark.parametrize("key,value", [
    ("eval_seed", -1), ("train_seed", 2**32), ("gen_seed", 1.5),
    ("train_count_duplicates", 1),
    ("dqn_batch_size", True), ("dqn_batch_size", 0),
    ("dqn_eps_anneal_eps", 0), ("dqn_lr", 0), ("pg_lr_actor", float("nan")),
    ("pg_lr_critic", float("inf")), ("gamma", 1.1), ("gen_epsilon", -0.1),
    ("pg_entropy_min", 1.1), ("dqn_target_mode", "bootstrapp"),
    ("method", "dqn-mc"), ("pg_repeat_penalty_shape", "typo"),
])
def test_invalid_core_values_fail_before_backend_construction(key, value):
    with pytest.raises(ConfigError, match=key):
        resolve_config({key: value})


def test_scientific_notation_from_pyyaml_is_normalized():
    result = resolve_config({"dqn_lr": "1e-3"})
    assert result.config["dqn_lr"] == 0.001
    assert isinstance(result.config["dqn_lr"], float)


def test_unknown_core_key_suggests_correct_spelling():
    with pytest.raises(ConfigError, match="Did you mean 'dqn_batch_size'"):
        resolve_config({"dqn_batch_szie": 32})


def test_nested_plugin_settings_are_preserved_without_claiming_validation():
    raw = {"properties": [{"name": "activity", "predictor": "custom:Predictor",
                            "custom_model_option": {"anything": [1, 2]}}]}
    result = resolve_config(raw)
    assert result.config["properties"] == raw["properties"]
    with pytest.raises(ConfigError, match="Unknown experiment key"):
        resolve_config({**raw, "custom_top_level_option": 3})


def test_structure_has_one_owner_without_materializing_defaults():
    raw = {"structure": {"builder": "external:Builder", "custom_option": [1, 2]},
           "properties": [{"name": "energy", "predictor": "external:Predictor"}]}
    before = deepcopy(raw)
    result = resolve_config(raw)
    assert result.config["structure"] == raw["structure"]
    assert result.sources["structure"] == "yaml"
    assert "n_random_configs" not in result.config["structure"]
    assert raw == before


@pytest.mark.parametrize("key,value", [("structure_sources", {}), ("sweep", {}),
    ("share_structure", True), ("builder", "custom:Builder"), ("filters", ["smact_charge"])])
def test_removed_parameters_fail_even_with_custom_predictor(key, value):
    with pytest.raises(ConfigError, match="Removed top-level"):
        resolve_config({key: value, "predictor": "external:Predictor"})


@pytest.mark.parametrize("source", [None, 1, [], "", "bulk"])
def test_removed_property_source_rejected(source):
    with pytest.raises(ConfigError, match="removed or structure-owned"):
        validate_config({"properties": [{"predictor": "external:Predictor", "source": source}]})


def test_structure_validation_does_not_import_ml_or_structure_backends():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", """
import sys
from rl_matdesign.config_schema import resolve_config
resolve_config({'structure': {'builder': 'external:Builder'},
                'properties': [{'predictor': 'external:Predictor'}]})
assert not {'torch', 'ase', 'pymatgen', 'deepmd', 'matminer'} & set(sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_inline_models_preserve_authored_config():
    raw = {"properties": [
        {"name": "cte", "predictor": "glassnet", "output": "CTEbelowTg", "unit": "K^-1", "representation": "physical", "objective": "mean"},
        {"name": "energy", "predictor": "dp_energy", "models": ["weights.pt"], "direction": "min"},
    ], "reward": {"objective": {"terms": {"energy": 1}}}}
    before = deepcopy(raw)
    result = resolve_config(raw)
    assert result.config["properties"] == before["properties"]
    assert raw == before


@pytest.mark.parametrize("field,value", [
    ("model_ref", None), ("output", 1), ("output", ""), ("unit", 0),
    ("representation", "log"), ("transform", "10^x"), ("uncertainty", "std"),
    ("uncertainty", 0), ("objective", "lower_bound"),
])
def test_property_output_contract_rejects_ambiguous_values(field, value):
    with pytest.raises(ConfigError, match=field):
        resolve_config({"properties": [{"predictor": "external:Predictor", field: value}]})


@pytest.mark.parametrize("key,value", [
    ("predictor", "ooh"), ("predictor", "external:Predictor"),
    ("models", {"g": {"predictor": "glassnet"}}), ("models", ["weights.pt"]),
    ("model", "weights.pt"), ("dp_models", ["weights.pt"]),
    ("objective", "mean"), ("uncertainty", "models"),
])
def test_flat_model_settings_are_rejected(key, value):
    with pytest.raises(ConfigError, match="Removed top-level"):
        resolve_config({key: value, "properties": [{"predictor": "dummy"}]})


@pytest.mark.parametrize("key,value", [
    ("model_ref", "glass"), ("model", "weights.pt"), ("dp_models", ["weights.pt"]),
])
def test_retired_property_model_settings_are_rejected(key, value):
    with pytest.raises(ConfigError, match=key):
        resolve_config({"properties": [{"predictor": "external:Predictor", key: value}]})


@pytest.mark.parametrize("value", [0, 1, None])
def test_removed_property_weight_is_rejected(value):
    with pytest.raises(ConfigError, match=r"weight was removed; put coefficients in reward terms"):
        resolve_config({"properties": [{"name": "p", "predictor": "external:Predictor", "weight": value}],
                        "reward": {"objective": {"terms": {"p": 1}}}})


@pytest.mark.parametrize("reward", [None, {}, {"type": "weighted_sum"}])
def test_multi_property_reward_must_select_its_terms(reward):
    cfg = {"properties": [{"name": "performance"}, {"name": "diagnostic"}]}
    if reward is not None:
        cfg["reward"] = reward
    with pytest.raises(ConfigError, match="Multiple properties require explicit reward.objective.terms"):
        resolve_config(cfg)
    cfg["reward"] = {"objective": {"terms": {"performance": 1}}}
    assert resolve_config(cfg).config["reward"] == cfg["reward"]


@pytest.mark.parametrize("models", [None, [], "weights.pt", {}, [None], [""], [1]])
def test_models_require_nonempty_path_lists(models):
    with pytest.raises(ConfigError, match="models"):
        resolve_config({"properties": [{"predictor": "dp_energy", "models": models}]})


def test_property_fqn_does_not_hide_a_top_level_typo():
    with pytest.raises(ConfigError, match="Did you mean 'method'"):
        resolve_config({"properties": [{"predictor": "custom:Predictor"}], "methodd": "dqn"})


def test_validate_does_not_materialize_defaults_before_hpo_overrides():
    cfg = {"gamma": 0.9, "pg_warmup_eps": 10}
    validate_config(cfg)
    assert cfg == {"gamma": 0.9, "pg_warmup_eps": 10}
    cfg.update(gamma=0.7, pg_warmup_eps=20)
    result = resolve_config(cfg)
    assert result.config["gamma"] == 0.7
    assert result.config["pg_warmup_eps"] == 20
    assert result.config["dqn_warmup_eps"] == 500


@pytest.mark.parametrize("raw", [
    {"groups": {}}, {"properties": ["wrong"]}, {"filters": [12]},
    {"constraint_filter": "smact_charge"}, {42: "not a string key"},
])
def test_invalid_boundaries_are_reported(raw):
    with pytest.raises(ConfigError):
        resolve_config(raw)


def test_tracked_experiment_config_inventory_resolves():
    configs = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(configs.glob("*.yaml"))
    assert paths
    for path in paths:
        raw = load_yaml_config(path)
        result = resolve_config(raw)
        assert set(result.config) == set(result.sources), path
        assert raw == load_yaml_config(path), path


def test_documented_parameter_table_is_generated_from_definitions():
    path = Path(__file__).resolve().parents[1] / "docs" / "configuration_schema.md"
    content = path.read_text()
    assert content.split("<!-- schema:start -->\n")[1].split("<!-- schema:end -->")[0] == schema_markdown()


def test_value_normalization_requires_a_value_network():
    assert resolve_config({}).config["value_normalization"] == "none"
    for method in ("dqn", "a2c"):
        cfg = resolve_config({"value_normalization": "popart"}, {"method": method}).config
        assert cfg["value_normalization_profile"] == "popart_v1"
    with pytest.raises(ConfigError, match="REINFORCE has no value network"):
        resolve_config({"value_normalization": "popart"}, {"method": "reinforce"})
    with pytest.raises(ConfigError, match="value_normalization_profile"):
        resolve_config({"value_normalization": "popart", "value_normalization_profile": "unknown"})


def test_calibrated_glassnet_examples_use_two_gates_and_maximize_t3():
    configs = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(configs.glob("glassnet_*.yaml"))
    assert paths
    for path in paths:
        cfg = resolve_config(load_yaml_config(path)).config
        assert cfg["value_normalization"] == "popart"
        t3 = next(prop for prop in cfg["properties"] if prop["output"] == "T3")
        assert t3["direction"] == "max"
        assert "scale" not in t3
        assert cfg["reward"]["objective"]["terms"] == {t3["name"]: 1.0}
        assert {gate["name"] for gate in cfg["reward"]["gates"]} == {
            "low_expansion", "sufficient_transition_temperature",
        }
        assert "failure" not in cfg["reward"]
        assert all("penalty_weight" not in gate for gate in cfg["reward"]["gates"])


def test_model_free_validation_rejects_conflicting_calibration_settings():
    path = Path(__file__).resolve().parents[1] / "configs" / "glassnet_coarse_2pct_a2c.yaml"
    cfg = load_yaml_config(path)
    cfg["properties"][0]["scale"] = 1000
    with pytest.raises(ConfigError, match="cannot declare both scale and normalization"):
        validate_config(cfg)
    del cfg["properties"][0]["scale"]
    cfg["reward"]["gates"][0]["penalty_weight"] = 1.0
    with pytest.raises(ConfigError, match="legacy penalty_weight"):
        validate_config(cfg)


def test_perovskite_example_calibrates_existing_combined_quality():
    path = Path(__file__).resolve().parents[1] / "configs" / "perovskite_calibrated.example.yaml"
    cfg = resolve_config(load_yaml_config(path)).config
    reward = cfg["reward"]
    assert reward["calibration"]["quality_mode"] == "aggregate"
    assert reward["objective"]["terms"] == dict.fromkeys(
        ("delta_T400", "delta_T500", "delta_T600"), 0.33333,
    )
    assert all(prop["objective"] == "mean_plus_kstd" for prop in cfg["properties"][:3])
    gate, = reward["gates"]
    assert gate["score"]["terms"] == {"pure_probability": 1.0}
    assert gate["min"] == 0.8
    assert "failure" not in reward and "penalty_weight" not in gate


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("source", ["yaml", "override"])
def test_removed_training_filter_toggle_is_rejected(value, source):
    config, overrides = ({"constrain_training": value}, {}) if source == "yaml" else ({}, {"constrain_training": value})
    with pytest.raises(ConfigError, match="constrain_training"):
        resolve_config(config, overrides)


@pytest.mark.parametrize("flag", ["--constrain-training", "--no-constrain-training"])
def test_removed_training_filter_cli_flags_are_rejected(flag, capsys):
    from rl_matdesign.experiment import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args(["--config", "unused.yaml", "--out", "unused-output", flag])
    assert exc.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--dp-seed", "1"], ["--no-train-count-duplicates"]])
def test_removed_cli_spellings_are_rejected(arguments, capsys):
    from rl_matdesign.experiment import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args(["--config", "unused.yaml", "--out", "unused-output", *arguments])
    assert exc.value.code == 2
    assert f"unrecognized arguments: {' '.join(arguments)}" in capsys.readouterr().err


@pytest.mark.parametrize("yaml_value", [None, False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_duplicate_cli_opt_in_preserves_yaml_when_absent(yaml_value, opt_in):
    from rl_matdesign.experiment import parse_args
    arguments = ["--config", "unused.yaml", "--out", "unused-output"]
    if opt_in:
        arguments.append("--train-count-duplicates")
    args = parse_args(arguments)
    assert args.train_count_duplicates is (True if opt_in else None)
    raw = {} if yaml_value is None else {"train_count_duplicates": yaml_value}
    resolved = resolve_config(raw, {"train_count_duplicates": args.train_count_duplicates})
    assert resolved.config["train_count_duplicates"] is (True if opt_in else bool(yaml_value))
    expected_source = "override" if opt_in else "default" if yaml_value is None else "yaml"
    assert resolved.sources["train_count_duplicates"] == expected_source
