"""A scene's last authored generation policy survives overrides and exports."""
from copy import deepcopy
from itertools import permutations

import pytest
import yaml

from rl_matdesign.config_schema import (
    ConfigError,
    GENERATION_PARAMETERS,
    load_yaml_config,
    resolve_config,
    selected_generation_parameter,
    validate_config,
    with_generation_selection,
)


@pytest.mark.parametrize("order", list(permutations(GENERATION_PARAMETERS)))
def test_last_authored_generation_field_wins_exclusively(order):
    raw = dict.fromkeys(order, 0.25)
    resolved = resolve_config(raw)
    assert selected_generation_parameter(raw) == order[-1]
    assert selected_generation_parameter(resolved.config) == order[-1]
    assert resolved.config[order[-1]] == 0.25
    assert all(resolved.config[key] == 0 for key in order[:-1])
    assert raw == dict.fromkeys(order, 0.25)


@pytest.mark.parametrize("selected", GENERATION_PARAMETERS)
def test_explicit_zero_is_a_policy_choice_and_defaults_do_not_win(selected):
    cfg = resolve_config({selected: 0}).config
    assert selected_generation_parameter(cfg) == selected
    assert all(cfg[key] == 0 for key in GENERATION_PARAMETERS)


def test_no_explicit_policy_keeps_temperature_default():
    cfg = resolve_config({}).config
    assert selected_generation_parameter({}) == "gen_temperature"
    assert cfg["gen_temperature"] == 1
    assert selected_generation_parameter(cfg) == "gen_temperature"
    assert cfg["generation_selection"]["source"] == "default"


@pytest.mark.parametrize("selected", GENERATION_PARAMETERS)
@pytest.mark.parametrize("value", [0, 0.25])
@pytest.mark.parametrize("sort_keys", [False, True])
def test_resolved_export_reload_preserves_selected_policy(tmp_path, selected, value, sort_keys):
    original = resolve_config({selected: value}).config
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(original, sort_keys=sort_keys))
    reloaded = resolve_config(load_yaml_config(path)).config
    assert reloaded == original
    assert selected_generation_parameter(reloaded) == selected


def test_merged_defaults_do_not_override_later_explicit_policy(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("<<: {gen_temperature: 1, gen_epsilon: 0.2}\ngen_temperature: 0\n")
    raw = load_yaml_config(path)
    cfg = resolve_config(raw).config
    assert selected_generation_parameter(raw) == "gen_temperature"
    assert cfg["gen_epsilon"] == 0
    assert cfg["gen_temperature"] == 0


@pytest.mark.parametrize("new_parameter", ["gen_epsilon", "gen_top_frac"])
def test_cli_policy_overrides_saved_yaml_even_when_explicit_zero(new_parameter):
    saved = resolve_config({"gen_temperature": 1}).config
    cfg = resolve_config(saved, {new_parameter: 0}).config
    assert selected_generation_parameter(cfg) == new_parameter
    assert all(cfg[key] == 0 for key in GENERATION_PARAMETERS)
    assert cfg["generation_selection"]["source"] == "override"


def test_absent_cli_values_do_not_override_yaml_policy():
    saved = resolve_config({"gen_top_frac": 0.2}).config
    cfg = resolve_config(saved, {"gen_temperature": None, "gen_epsilon": None}).config
    assert selected_generation_parameter(cfg) == "gen_top_frac"
    assert cfg["gen_top_frac"] == 0.2


def test_cli_mapping_order_selects_last_explicit_policy():
    cfg = resolve_config({"gen_temperature": 1}, {"gen_epsilon": 0.2, "gen_top_frac": 0}).config
    assert selected_generation_parameter(cfg) == "gen_top_frac"
    assert cfg["gen_epsilon"] == cfg["gen_temperature"] == 0


def test_edits_to_resolved_config_override_saved_selection():
    saved = resolve_config({"gen_temperature": 1}).config
    saved["gen_epsilon"] = 0.3
    cfg = resolve_config(saved).config
    assert selected_generation_parameter(cfg) == "gen_epsilon"
    assert cfg["gen_temperature"] == 0
    assert cfg["gen_epsilon"] == 0.3


def test_last_changed_field_in_resolved_config_wins():
    saved = resolve_config({"gen_temperature": 1}).config
    saved["gen_epsilon"] = 0.3
    # Place the changed temperature after epsilon, including a zero edit.
    saved.pop("gen_temperature")
    saved["gen_temperature"] = 0
    cfg = resolve_config(saved).config
    assert selected_generation_parameter(cfg) == "gen_temperature"
    assert all(cfg[key] == 0 for key in GENERATION_PARAMETERS)


@pytest.mark.parametrize("change", [
    {"parameter": "invalid"},
    {"source": "unknown"},
    {"values": {"gen_temperature": 1}},
    {"values": {"gen_temperature": 1, "gen_epsilon": 0.2, "gen_top_frac": 0}},
    {"values": {"gen_temperature": float("nan"), "gen_epsilon": 0, "gen_top_frac": 0}},
    {"unexpected": 1},
])
def test_malformed_saved_selection_is_rejected(change):
    cfg = resolve_config({"gen_temperature": 1}).config
    cfg["generation_selection"].update(change)
    with pytest.raises(ConfigError, match="generation_selection|finite"):
        resolve_config(cfg)


def test_saved_metadata_requires_complete_effective_fields():
    cfg = resolve_config({"gen_temperature": 1}).config
    del cfg["gen_epsilon"]
    with pytest.raises(ConfigError, match="all three effective"):
        resolve_config(cfg)


@pytest.mark.parametrize("selected", GENERATION_PARAMETERS)
def test_job_builder_explicit_selection_is_immune_to_stale_metadata(selected):
    raw = resolve_config({"gen_temperature": 1}).config
    raw.update(dict.fromkeys(GENERATION_PARAMETERS, 0.25))
    raw[selected] = 0
    original = deepcopy(raw)
    selected_cfg = with_generation_selection(raw, selected)
    resolved = resolve_config(selected_cfg).config
    assert selected_generation_parameter(resolved) == selected
    assert all(resolved[key] == 0 for key in GENERATION_PARAMETERS)
    assert raw == original


def test_selection_does_not_change_generation_horizon_or_extra_exploration():
    raw = {"gen_top_frac": 0.3, "num_gen_eps": 17, "max_gen_attempts": 40,
           "exploration_gen_eps": 5}
    cfg = resolve_config(raw).config
    assert (cfg["num_gen_eps"], cfg["max_gen_attempts"], cfg["exploration_gen_eps"]) == (17, 40, 5)


def test_read_only_validation_accepts_internal_metadata_without_default_materialization():
    raw = {"gen_epsilon": 0.25}
    original = deepcopy(raw)
    validate_config(raw)
    assert raw == original
    validate_config(resolve_config(raw).config)
