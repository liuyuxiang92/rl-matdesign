"""Configuration behavior is testable without importing ML or oracle backends."""
from copy import deepcopy
from pathlib import Path

import pytest

from rl_matdesign.config_schema import (
    ConfigDeprecationWarning,
    ConfigError,
    load_yaml_config,
    normalize_aliases,
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


def test_aliases_normalize_without_mutation():
    raw = {"dp_seed": 12, "buffer_size": 300, "groups": [{"name": "x"}]}
    original = deepcopy(raw)
    with pytest.warns(ConfigDeprecationWarning):
        cfg = normalize_aliases(raw)
    assert cfg == {"eval_seed": 12, "dqn_buffer_size": 300, "groups": [{"name": "x"}]}
    cfg["groups"][0]["name"] = "changed"
    assert raw == original


def test_equal_alias_and_canonical_value_is_accepted_with_warning():
    with pytest.warns(ConfigDeprecationWarning):
        result = resolve_config({"buffer_size": 20, "dqn_buffer_size": 20})
    assert result.config["dqn_buffer_size"] == 20
    assert "buffer_size" not in result.config


def test_numeric_alias_values_compare_after_normalization():
    with pytest.warns(ConfigDeprecationWarning):
        result = resolve_config({"eps_min": 0, "dqn_eps_min": 0.0})
    assert result.config["dqn_eps_min"] == 0.0


def test_conflicting_alias_and_canonical_value_is_an_error():
    with pytest.raises(ConfigError, match="Conflicting config aliases"):
        resolve_config({"buffer_size": 20, "dqn_buffer_size": 30})


def test_cli_can_override_a_deprecated_yaml_spelling():
    with pytest.warns(ConfigDeprecationWarning):
        result = resolve_config({"dp_seed": 9}, {"eval_seed": 10})
    assert result.config["eval_seed"] == 10
    assert result.sources["eval_seed"] == "override"


def test_defaults_preserve_distinct_dqn_and_pg_warmup():
    result = resolve_config({})
    assert result.config["dqn_warmup_eps"] == 500
    assert result.config["pg_warmup_eps"] == 200
    assert result.sources["dqn_warmup_eps"] == "default"
    assert result.config["train_budget"] is None


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
    ({"pg_warmup_eps": 40}, {}, 40, "derived:pg_warmup_eps"),
    ({"pg_warmup_eps": 40}, {"pg_warmup_eps": 70}, 70, "derived:pg_warmup_eps"),
    ({"pg_warmup_eps": 40, "dqn_warmup_eps": 80}, {}, 80, "yaml"),
    ({"pg_warmup_eps": 40}, {"dqn_warmup_eps": 0}, 0, "override"),
])
def test_explicit_warmup_fallback_and_override(raw, overrides, warmup, source):
    result = resolve_config(raw, overrides)
    assert result.config["dqn_warmup_eps"] == warmup
    assert result.sources["dqn_warmup_eps"] == source


def test_gamma_fallbacks_apply_after_overrides():
    result = resolve_config({"gamma": 0.8, "dqn_gamma": 0.6}, {"gamma": 0.95})
    assert result.config["dqn_gamma"] == 0.6
    assert result.config["pg_gamma"] == 0.95
    assert result.sources["pg_gamma"] == "derived:gamma"


def test_false_zero_and_none_cli_values_have_distinct_meanings():
    result = resolve_config(
        {"method": "dqn", "constrain_training": True, "train_count_duplicates": True,
         "eval_seed": 9, "gen_epsilon": 0.2},
        {"method": None, "constrain_training": False, "train_count_duplicates": False,
         "eval_seed": 0, "gen_epsilon": 0},
    )
    assert result.config["method"] == "dqn"
    assert result.sources["method"] == "yaml"
    assert result.config["constrain_training"] is False
    assert result.config["train_count_duplicates"] is False
    assert result.config["eval_seed"] == 0
    assert result.config["gen_epsilon"] == 0
    assert result.sources["constrain_training"] == "override"


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
    ("constrain_training", "false"), ("train_count_duplicates", 1),
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
        resolve_config({"predictor": "dummy", "dqn_batch_szie": 32})


def test_nested_plugin_settings_are_preserved_without_claiming_validation():
    raw = {"properties": [{"name": "activity", "predictor": "custom:Predictor",
                            "custom_model_option": {"anything": [1, 2]}}]}
    result = resolve_config(raw)
    assert result.config["properties"] == raw["properties"]
    with pytest.raises(ConfigError, match="Unknown experiment key"):
        resolve_config({**raw, "custom_top_level_option": 3})


def test_named_structure_sources_have_an_owner_without_materializing_source_defaults():
    raw = {
        "structure_sources": {"bulk": {"builder": "external:Builder", "custom_option": [1, 2]}},
        "properties": [{"name": "energy", "predictor": "external:Predictor", "source": "bulk"}],
    }
    before = deepcopy(raw)
    result = resolve_config(raw)
    assert result.config["structure_sources"] == raw["structure_sources"]
    assert result.sources["structure_sources"] == "yaml"
    assert "n_random_configs" not in result.config["structure_sources"]["bulk"]
    assert raw == before


@pytest.mark.parametrize("sources", [
    [], {1: {}}, {"": {}}, {" bulk ": {}}, {"bulk": []},
    {"bulk": {"n_random_configs": True}}, {"bulk": {"n_random_configs": 0}},
    {"bulk": {"geo_opt": True}}, {"bulk": {"direction": "max"}},
])
def test_malformed_source_definitions_fail_in_lightweight_validation(sources):
    with pytest.raises(ConfigError, match="structure_sources"):
        resolve_config({"structure_sources": sources})


@pytest.mark.parametrize("source", [None, 1, [], "", " bulk "])
def test_property_source_requires_an_explicit_nonempty_name(source):
    with pytest.raises(ConfigError, match=r"properties\[0\].source"):
        validate_config({"properties": [{"predictor": "external:Predictor", "source": source}]})


def test_source_validation_does_not_import_ml_or_structure_backends():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", """
import sys
from rl_matdesign.config_schema import resolve_config
resolve_config({'structure_sources': {'bulk': {'builder': 'external:Builder'}},
                'properties': [{'predictor': 'external:Predictor', 'source': 'bulk'}]})
assert not {'torch', 'ase', 'pymatgen', 'deepmd', 'matminer'} & set(sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_named_models_resolve_through_one_owner_without_mutating_authored_config():
    raw = {"models": {"glass": {"predictor": "glassnet"}}, "properties": [
        {"name": "cte", "model_ref": "glass", "output": "CTEbelowTg", "unit": "K^-1", "representation": "physical", "objective": "mean"},
        {"name": "tg", "model_ref": "glass", "output": "Tg", "unit": "K"},
    ]}
    result = resolve_config(raw)
    assert result.config["models"] == raw["models"]
    assert result.config["properties"] == raw["properties"]
    assert result.sources["models"] == "yaml"


@pytest.mark.parametrize("field,value", [
    ("model_ref", None), ("output", 1), ("output", ""), ("unit", 0),
    ("representation", "log"), ("transform", "10^x"), ("uncertainty", "std"),
    ("uncertainty", 0), ("objective", "lower_bound"),
])
def test_property_output_contract_rejects_ambiguous_values(field, value):
    with pytest.raises(ConfigError, match=field):
        resolve_config({"properties": [{"predictor": "external:Predictor", field: value}]})


@pytest.mark.parametrize("definition,prop", [
    ({"predictor": "glassnet", "direction": "min"}, {"model_ref": "g"}),
    ({"predictor": "glassnet"}, {"model_ref": "unknown"}),
    ({"predictor": "glassnet"}, {"model_ref": "g", "predictor": "glassnet"}),
    ({"predictor": "glassnet", "batch_size": 32}, {"model_ref": "g", "batch_size": 32}),
])
def test_model_definition_conflicts_fail_before_backend_construction(definition, prop):
    with pytest.raises(ConfigError, match="Named model"):
        resolve_config({"models": {"g": definition}, "properties": [prop]})


def test_legacy_flat_models_list_is_not_a_named_resource_table():
    assert resolve_config({"predictor": "external:Predictor", "models": ["weights.pt"]}).config["models"] == ["weights.pt"]
    with pytest.raises(ConfigError, match="models"):
        resolve_config({"models": ["weights.pt"], "properties": [{"predictor": "dp_energy"}]})


@pytest.mark.parametrize("selector", [
    {"predictor": "custom:Predictor"},
    {"builder": "custom:Builder"},
    {"filters": [{"type": "custom:Filter"}]},
])
def test_flat_fqn_plugin_owns_its_custom_keys_but_not_core_typos(selector):
    result = resolve_config({**selector, "custom_option": {"foo": 1}})
    assert result.config["custom_option"] == {"foo": 1}
    with pytest.raises(ConfigError, match="dqn_batch_szie"):
        resolve_config({**selector, "dqn_batch_szie": 32})


def test_flat_extension_does_not_hide_a_nonprefixed_core_typo():
    with pytest.raises(ConfigError, match="Did you mean 'method'"):
        resolve_config({"predictor": "custom:Predictor", "methodd": "dqn"})


def test_validate_does_not_materialize_defaults_before_hpo_overrides():
    cfg = {"gamma": 0.9, "pg_warmup_eps": 10}
    validate_config(cfg)
    assert cfg == {"gamma": 0.9, "pg_warmup_eps": 10}
    cfg.update(gamma=0.7, pg_warmup_eps=20)
    result = resolve_config(cfg)
    assert result.config["dqn_gamma"] == 0.7
    assert result.config["dqn_warmup_eps"] == 20


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
