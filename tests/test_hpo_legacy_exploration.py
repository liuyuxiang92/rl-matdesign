"""Legacy HPO preserves the pre-Step-3 search and synthesized job contract.

Workers in this file are deterministic artifact fakes; no predictor or learner
is started. Historical numeric expectations are from revision 381a8c198.
"""
from copy import deepcopy

import pytest
import yaml

from rl_matdesign.config_schema import (
    ConfigError, load_yaml_config, resolve_config, selected_generation_parameter,
    with_generation_selection,
)
from rl_matdesign.hpo.algorithm_profiles import (
    get_profile, resolve_parameters, validate_parameter_value,
)
from rl_matdesign.hpo.controller import Campaign
from rl_matdesign.hpo.domain import training_budget_preflight
from rl_matdesign.hpo.job_runner import JobRunner
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_job_runner import FakeWorker


LEGACY_KEYS = {
    "a2c": {"pg_entropy_coef", "pg_entropy_min"},
    "dqn-bootstrap": {"dqn_eps_min", "epsilon_anneal_budget_ratio"},
    "dqn-mc": {"dqn_eps_min", "epsilon_anneal_budget_ratio"},
}
GENERATION_DEFAULTS = {
    "gen_temperature": [0, .001, .01, .1, 1, 10, 100, 1000],
    "gen_epsilon": [0, .01, .05, .1, .2, .4, .7, 1],
    "gen_top_frac": [.001, .01, .05, .1, .25, .5, 1],
}


def write_plan(tmp_path, *, protocol="hpo_v2", method="a2c", adaptive=False,
               generation_parameter="gen_temperature", base_extra=None,
               plan_extra=None):
    base = {
        "exploration_adaptive": adaptive,
        "method": "a2c" if method == "a2c" else "dqn",
        "dqn_target_mode": "mc" if method == "dqn-mc" else "bootstrap",
        "pg_warmup_eps": 0,
        "properties": [{"name": "energy", "predictor": "dummy",
                        "direction": "min", "unit": "eV"}],
    }
    if not adaptive:
        base[generation_parameter] = .5
    base.update(base_extra or {})
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    data = {"method": method, "base_config": "scenario.yaml"}
    if protocol == "hpo_v1":
        data["protocol"] = protocol
        data["objective"] = {
            "field": "energy", "direction": "minimize", "threshold": 1000,
            "unit": "eV", "normalization": {"kind": "sigmoid", "scale": 1},
        }
    else:
        data["evaluation"] = {"threshold": 0}
    data.update(plan_extra or {})
    path = tmp_path / "hpo.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.mark.parametrize("method", LEGACY_KEYS)
def test_legacy_profiles_restore_only_historically_active_search_dimensions(method):
    current = get_profile(method)
    legacy = get_profile(method, exploration_adaptive=False)
    assert set(legacy["search_space"]) - set(current["search_space"]) == LEGACY_KEYS[method]
    assert "exploration_target" not in legacy["search_space"]
    assert ("dqn_target_update_freq" in legacy["search_space"]) == (method == "dqn-bootstrap")
    expected = ({
        "pg_entropy_coef": {"type": "float", "low": 1e-4, "high": .1, "log": True},
        "pg_entropy_min": {"type": "float", "low": 0., "high": .8},
    } if method == "a2c" else {
        "dqn_eps_min": {"type": "float", "low": .02, "high": .4},
        "epsilon_anneal_budget_ratio": {"type": "float", "low": .25, "high": 2., "log": True},
    })
    assert {key: legacy["search_space"][key] for key in expected} == expected


@pytest.mark.parametrize("protocol", ["hpo_v1", "hpo_v2"])
@pytest.mark.parametrize("method", LEGACY_KEYS)
def test_search_mode_rejects_inactive_exploration_dimensions(tmp_path, protocol, method):
    for adaptive, name in [(True, name) for name in LEGACY_KEYS[method]] + [(False, "exploration_target")]:
        ranges = {name: {"type": "float", "low": .1, "high": .2}}
        extra = {"search_space": ranges} if protocol == "hpo_v1" else {"training": {"parameter_ranges": ranges}}
        with pytest.raises(ConfigError, match="not active"):
            load_plan(write_plan(tmp_path, protocol=protocol, method=method,
                                 adaptive=adaptive, plan_extra=extra))
        with pytest.raises(ConfigError, match="not active"):
            validate_parameter_value(method, name, .15, exploration_adaptive=adaptive)


@pytest.mark.parametrize("protocol", ["hpo_v1", "hpo_v2"])
def test_historical_bootstrap_epsilon_schedule_uses_online_budget(tmp_path, protocol):
    plan = load_plan(write_plan(tmp_path, protocol=protocol, method="dqn-bootstrap"))
    before = deepcopy(plan)
    params = {"warmup_budget_fraction": .15284408691191873,
              "epsilon_anneal_budget_ratio": 1.6899975432792926,
              "dqn_eps_min": .3391643803805721}
    overrides = resolve_parameters(plan, params, 458)
    assert overrides["dqn_warmup_eps"] == 70
    assert overrides["dqn_num_train_eps"] == 388
    assert overrides["dqn_eps_anneal_eps"] == 656
    assert overrides["dqn_eps_min"] == params["dqn_eps_min"]
    effective = resolve_config(plan["base_config_data"], overrides).config
    assert effective["exploration_adaptive"] is False
    assert "exploration_target" not in effective
    assert "gen_exploration_target" not in effective
    assert training_budget_preflight(plan, params, 458, {"episode_steps": 8})["eligible"]
    assert plan == before


def test_legacy_hpo_disables_base_repeat_penalty_as_historical_protocol_did(tmp_path):
    plan = load_plan(write_plan(tmp_path, base_extra={"pg_repeat_penalty_coef": .2}))
    effective = resolve_config(plan["base_config_data"], resolve_parameters(plan, {}, 128)).config
    assert effective["pg_repeat_penalty_coef"] == 0.


@pytest.mark.parametrize("protocol,parameter", [
    ("hpo_v1", "gen_temperature"),
    *( ("hpo_v2", parameter) for parameter in GENERATION_DEFAULTS ),
])
def test_legacy_generation_defaults_match_historical_protocol(tmp_path, protocol, parameter):
    plan = load_plan(write_plan(tmp_path, protocol=protocol, generation_parameter=parameter))
    settings = plan["generation_search"]
    assert settings["parameter"] == parameter
    assert settings["initial_values"] == GENERATION_DEFAULTS[parameter]
    assert "gen_exploration_target" not in settings
    if protocol == "hpo_v1":
        assert settings["fixed"] == {"gen_epsilon": 0, "gen_top_frac": 0}


@pytest.mark.parametrize("protocol,parameter", [
    ("hpo_v1", "gen_temperature"),
    *( ("hpo_v2", parameter) for parameter in GENERATION_DEFAULTS ),
])
@pytest.mark.parametrize("value", [0., .25])
def test_legacy_jobs_preserve_selected_generation_through_sorted_roundtrip(
        tmp_path, monkeypatch, protocol, parameter, value):
    plan = load_plan(write_plan(tmp_path, protocol=protocol, generation_parameter=parameter))
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "campaign")
    params = {"pg_batch_eps": 2, "pg_entropy_coef": .03430250729588482,
              "pg_entropy_min": .6185688993892955}
    trained = runner(params, 8, 11, "train")
    generated = runner({**params, parameter: value}, 4, 401, "generation", trained)
    assert trained["eligible_for_selection"] and generated["eligible_for_selection"]
    assert len(worker.calls) == 2
    for call in worker.calls:
        cfg = call["config"]
        assert cfg["exploration_adaptive"] is False
        assert not {"exploration_target", "gen_exploration_target", "exploration_mode"} & cfg.keys()
        assert cfg["pg_entropy_coef"] == params["pg_entropy_coef"]
        assert cfg["pg_entropy_min"] == params["pg_entropy_min"]
        assert selected_generation_parameter(cfg) == parameter
        assert all(cfg[name] == 0 for name in GENERATION_DEFAULTS if name != parameter)
        path = call["directory"] / "sorted.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        restored = resolve_config(load_yaml_config(path)).config
        assert restored == cfg
        assert selected_generation_parameter(restored) == parameter
    assert worker.calls[1]["config"][parameter] == value
    assert "--skip-generation" in worker.calls[0]["args"]
    assert "--only-generate" in worker.calls[1]["args"]


@pytest.mark.parametrize("parameter", GENERATION_DEFAULTS)
def test_generation_selection_cannot_cross_exploration_modes(parameter):
    with pytest.raises(ConfigError, match="exploration_adaptive=false"):
        with_generation_selection({"exploration_adaptive": True, parameter: .5}, parameter)
    with pytest.raises(ConfigError, match="exploration_adaptive=true"):
        with_generation_selection({"exploration_adaptive": False}, "gen_exploration_target")


def test_generation_endpoint_probes_respect_historical_parameter_domains():
    assert Campaign.generation_extensions("gen_temperature", [0, .001, 1, 1000]) == [
        ("low", .001, .0001), ("high", 1000, 10000)]
    for parameter in ("gen_epsilon", "gen_top_frac", "gen_exploration_target"):
        assert Campaign.generation_extensions(parameter, [.1, .5]) == [
            ("low", .1, 0.), ("high", .5, 1.)]
        assert Campaign.generation_extensions(parameter, [0., .5, 1.]) == [
            ("low", 0., 0.), ("high", 1., 1.)]
