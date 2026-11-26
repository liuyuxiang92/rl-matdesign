"""The public exploration switch selects complete, explicit sampling contracts."""
from copy import deepcopy

import pytest
import yaml

from rl_matdesign.config_schema import (
    ConfigError, LEGACY_EXPLORATION_KEYS, load_yaml_config, resolve_config,
    selected_generation_parameter, validate_config, with_generation_selection,
)


@pytest.mark.parametrize("value", [0, .25, 1])
@pytest.mark.parametrize("sort_keys", [False, True])
def test_target_export_reload_preserves_explicit_endpoints(tmp_path, value, sort_keys):
    original = resolve_config({"gen_exploration_target": value, "exploration_adaptive": True}).config
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(original, sort_keys=sort_keys))
    assert resolve_config(load_yaml_config(path)).config == original
    assert "generation_selection" not in original


def test_defaults_are_shared_targets_with_adaptation_enabled():
    cfg = resolve_config({}).config
    assert cfg["exploration_target"] == cfg["gen_exploration_target"] == .7
    assert cfg["exploration_adaptive"] is True
    assert not (set(cfg) & LEGACY_EXPLORATION_KEYS)


def test_yaml_merge_and_cli_zero_override_preserve_target(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("<<: {gen_exploration_target: 1, exploration_target: 0.2}\ngen_exploration_target: 0\n")
    cfg = resolve_config(load_yaml_config(path)).config
    assert cfg["gen_exploration_target"] == 0
    assert cfg["exploration_target"] == .2
    cfg = resolve_config(cfg, {"gen_exploration_target": None, "exploration_adaptive": True}).config
    assert cfg["gen_exploration_target"] == 0 and cfg["exploration_adaptive"] is True
    assert resolve_config(cfg, {"exploration_target": 0}).config["exploration_target"] == 0


@pytest.mark.parametrize("key", ["exploration_target", "gen_exploration_target"])
@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True])
def test_targets_reject_invalid_domain(key, value):
    with pytest.raises(ConfigError, match=key):
        resolve_config({key: value})


@pytest.mark.parametrize("key", ["gen_temperature", "gen_epsilon", "gen_top_frac", "generation_selection",
                                  "dqn_eps_min", "dqn_eps_anneal_eps", "pg_entropy_coef", "pg_entropy_min",
                                  "pg_repeat_penalty_coef", "pg_repeat_penalty_shape"])
def test_old_sampler_configuration_is_rejected_without_migration(key):
    with pytest.raises(ConfigError, match=key):
        resolve_config({key: 0})


@pytest.mark.parametrize("key", ["exploration_target", "gen_exploration_target"])
def test_targets_support_method_maps_and_validate_inactive_entries(key):
    raw = {"method": "dqn", "dqn_target_mode": "mc", key: {"a2c": .2, "dqn-mc": .8}}
    assert resolve_config(raw).config[key] == .8
    assert resolve_config(raw, {"method": "a2c"}).config[key] == .2
    raw[key]["a2c"] = 2
    with pytest.raises(ConfigError, match=key):
        resolve_config(raw)


def test_validation_is_read_only_and_targets_leave_horizon_unchanged():
    raw = {"gen_exploration_target": .3, "num_gen_eps": 17, "max_gen_attempts": 40}
    original = deepcopy(raw)
    validate_config(raw)
    assert raw == original
    cfg = resolve_config(raw).config
    assert (cfg["num_gen_eps"], cfg["max_gen_attempts"]) == (17, 40)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_adaptation_requires_an_explicit_boolean(value):
    with pytest.raises(ConfigError, match="exploration_adaptive"):
        resolve_config({"exploration_adaptive": value})


@pytest.mark.parametrize("key", ["exploration_target", "gen_exploration_target"])
def test_legacy_mode_rejects_targets_instead_of_selecting_fixed_entropy(key):
    with pytest.raises(ConfigError, match="exploration_adaptive=true"):
        resolve_config({"exploration_adaptive": False, key: .7})


def test_legacy_defaults_preserve_old_controls_without_entropy_targets():
    cfg = resolve_config({"exploration_adaptive": False}).config
    assert cfg["dqn_eps_min"] == .05 and cfg["dqn_eps_anneal_eps"] == 10000
    assert cfg["pg_entropy_coef"] == .01 and cfg["pg_entropy_min"] == .3
    assert cfg["pg_repeat_penalty_coef"] == 0 and cfg["pg_repeat_penalty_shape"] == "log"
    assert "exploration_target" not in cfg and "gen_exploration_target" not in cfg
    assert selected_generation_parameter(cfg) == "gen_temperature"
    assert resolve_config(cfg).config == cfg


@pytest.mark.parametrize("parameter,value", [("gen_temperature", .3), ("gen_top_frac", .2), ("gen_epsilon", 0)])
@pytest.mark.parametrize("sort_keys", [False, True])
def test_legacy_generation_selection_survives_resolution_and_sorted_export(tmp_path, parameter, value, sort_keys):
    original = resolve_config({"exploration_adaptive": False, parameter: value}).config
    assert selected_generation_parameter(original) == parameter
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(original, sort_keys=sort_keys))
    reloaded = resolve_config(load_yaml_config(path)).config
    assert reloaded == original
    for key in ("gen_temperature", "gen_top_frac", "gen_epsilon"):
        assert reloaded[key] == (value if key == parameter else 0)


def test_legacy_yaml_merge_authored_selection_and_hpo_override(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("exploration_adaptive: false\n<<: {gen_epsilon: 0.5, gen_temperature: 2}\ngen_epsilon: 0.1\n")
    cfg = resolve_config(load_yaml_config(path)).config
    assert selected_generation_parameter(cfg) == "gen_epsilon"
    changed = with_generation_selection({**cfg, "gen_top_frac": .3}, "gen_top_frac")
    changed = resolve_config(changed).config
    assert selected_generation_parameter(changed) == "gen_top_frac"
    assert changed["gen_top_frac"] == .3 and changed["gen_epsilon"] == 0


def test_legacy_training_values_survive_without_adjusting_budget():
    raw = {"exploration_adaptive": False, "dqn_eps_min": .231,
           "dqn_eps_anneal_eps": 1292, "pg_entropy_coef": .017,
           "pg_entropy_min": .487, "pg_repeat_penalty_coef": .2,
           "pg_repeat_penalty_shape": "sqrt", "train_budget": 916,
           "dqn_warmup_eps": 40, "pg_batch_eps": 6}
    cfg = resolve_config(raw).config
    assert all(cfg[key] == value for key, value in raw.items())
    assert cfg["dqn_num_train_eps"] == 876 and cfg["pg_num_iters"] == 153
