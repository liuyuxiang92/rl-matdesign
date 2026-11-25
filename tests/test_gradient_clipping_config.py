"""Adaptive clipping has one public switch and an exact learning identity."""
import copy
import json
from types import SimpleNamespace

import pytest

from rl_matdesign.config_schema import ConfigError, PARAMETERS, resolve_config
from rl_matdesign.utils.execution_contract import execution_contract, require_execution_contract


def test_clipping_is_one_optional_parameter_and_explicit_none_overrides_yaml():
    default = resolve_config({})
    assert default.config["gradient_clipping"] == "none"
    assert default.sources["gradient_clipping"] == "default"
    enabled = resolve_config({"gradient_clipping": "clipping"})
    assert enabled.sources["gradient_clipping"] == "yaml"
    disabled = resolve_config(enabled.config, {"gradient_clipping": "none"})
    assert disabled.config["gradient_clipping"] == "none"
    assert disabled.sources["gradient_clipping"] == "override"
    assert {key for key in PARAMETERS if "clipping" in key} == {"gradient_clipping"}


@pytest.mark.parametrize("value", ["fixed", "adaptive", "clip", 1, True, None, {"mode": "clipping"}])
def test_clipping_rejects_other_modes_and_nested_options(value):
    with pytest.raises(ConfigError, match="gradient_clipping"):
        resolve_config({"gradient_clipping": value})


@pytest.mark.parametrize("key", ["gradient_clip_norm", "gradient_clipping_threshold",
    "gradient_clipping_percentile", "gradient_clipping_window", "gradient_clipping_profile"])
def test_clipping_internal_settings_are_not_public_config_keys(key):
    with pytest.raises(ConfigError, match="Unknown experiment key"):
        resolve_config({"gradient_clipping": "clipping", key: 1})


@pytest.mark.parametrize("mode", ["none", "clipping"])
def test_clipping_cli_override(mode):
    from rl_matdesign.experiment import parse_args
    args = parse_args(["--config", "unused.yaml", "--out", "unused", "--gradient-clipping", mode])
    assert args.gradient_clipping == mode
    absent = parse_args(["--config", "unused.yaml", "--out", "unused"])
    assert absent.gradient_clipping is None


def test_clipping_cli_rejects_fixed_threshold_mode(capsys):
    from rl_matdesign.experiment import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args(["--config", "unused.yaml", "--out", "unused", "--gradient-clipping", "fixed"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def _manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"input_files": [], "runtime": {}, "runtime_args": {}}))
    return path


@pytest.mark.parametrize("method", ["dqn", "a2c", "reinforce"])
def test_clipping_changes_learning_identity_without_changing_reward_identity(tmp_path, method):
    path = _manifest(tmp_path)
    before = resolve_config({"method": method}).config
    after = resolve_config({"method": method, "gradient_clipping": "clipping"}).config
    first = execution_contract(before, SimpleNamespace(), path, "cpu")
    second = execution_contract(after, SimpleNamespace(), path, "cpu")
    assert first["evaluation"] == second["evaluation"]
    assert first["learning"]["gradient_clipping"] == "none"
    assert "gradient_clipping_profile" not in first["learning"]
    profile = second["learning"]["gradient_clipping_profile"]
    assert profile["calibration_candidates"] == 100
    assert profile["calibration_budget"] == "outside_training_budget"
    assert profile["initialization"] == "pilot_positive_gradient_p90"
    with pytest.raises(ValueError, match="training protocol"):
        require_execution_contract(second, first)


def test_missing_and_explicit_none_have_identical_learning_identity(tmp_path):
    path = _manifest(tmp_path)
    predictor = SimpleNamespace()
    assert execution_contract({"method": "dqn"}, predictor, path, "cpu") == execution_contract(
        {"method": "dqn", "gradient_clipping": "none"}, predictor, path, "cpu")


def test_changed_internal_profile_rejects_exact_resume(tmp_path, monkeypatch):
    from rl_matdesign import gradient_clipping
    path = _manifest(tmp_path)
    cfg = resolve_config({"gradient_clipping": "clipping"}).config
    first = execution_contract(cfg, SimpleNamespace(), path, "cpu")
    profile = copy.deepcopy(first["learning"]["gradient_clipping_profile"])
    profile["window"] += 1
    monkeypatch.setattr(gradient_clipping, "gradient_clipping_profile", lambda: copy.deepcopy(profile))
    second = execution_contract(cfg, SimpleNamespace(), path, "cpu")
    assert first["evaluation"] == second["evaluation"]
    with pytest.raises(ValueError, match="gradient_clipping_profile.window"):
        require_execution_contract(second, first)
