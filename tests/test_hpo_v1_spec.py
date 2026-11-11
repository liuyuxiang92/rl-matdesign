from copy import deepcopy

import pytest
import yaml

from rl_matdesign.config_schema import ConfigError, resolve_config
from rl_matdesign.hpo.algorithm_profiles import get_profile, initial_parameters, resolve_parameters
from rl_matdesign.hpo.spec import load_plan, phase_efficiency


def write_plan(tmp_path, extra=None, base=None):
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(base or {"properties": [{"name": "temperature", "unit": "K"}]}))
    data = {"protocol": "hpo_v1", "method": "a2c", "base_config": "scenario.yaml",
            "objective": {"field": "temperature", "direction": "minimize", "threshold": 1000, "unit": "K"}}
    data.update(extra or {})
    path = tmp_path / "hpo.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_are_complete_and_paths_use_scenario_directory(tmp_path):
    path = write_plan(tmp_path, base={"properties": [{"name": "temperature", "unit": "K", "model": "assets/net.pt"}]})
    plan = load_plan(path)
    assert plan["base_config"] == str(tmp_path / "scenario.yaml")
    assert plan["base_config_data"]["properties"][0]["model"] == str(tmp_path / "assets/net.pt")
    assert len(plan["search_space"]) == 5
    assert plan["budget"]["train"]["min_fraction"] < plan["budget"]["train"]["initial_fraction"]
    assert phase_efficiency(plan, "generation")["max_attempts_per_unique"] == 1.5
    assert plan["generation_search"]["fixed"] == {"gen_epsilon": 0, "gen_top_frac": 0, "exploration_gen_eps": 0}
    assert plan["sources"]["efficiency"] == "default"


def test_method_cli_can_supply_but_cannot_silently_override(tmp_path):
    path = write_plan(tmp_path)
    with pytest.raises(ConfigError, match="conflicts"):
        load_plan(path, "dqn-mc")
    data = yaml.safe_load(path.read_text())
    data.pop("method")
    path.write_text(yaml.safe_dump(data))
    assert load_plan(path, "dqn-mc")["method"] == "dqn-mc"


@pytest.mark.parametrize("extra,match", [
    ({"unknown": 1}, "Unknown"),
    ({"budget": {"train": {"initial_frac": .1}}}, "Unknown"),
    ({"efficiency": {"max_attempts_per_unique": 1}}, "> 1"),
    ({"efficiency": {"max_attempts_per_unique": float("inf")}}, "finite"),
    ({"efficiency": {"comparison": "le"}}, "lt"),
    ({"range_search": {"confirmation_seed": 11}}, "fresh"),
    ({"generation_search": {"parameter": "gen_temp"}}, "gen_temperature"),
    ({"generation_search": {"fixed": {"gen_epsilon": .1}}}, "temperature-only"),
    ({"generation_search": {"initial_values": [0]}}, "positive"),
    ({"generation_search": {"initial_values": [0, 1, 1]}}, "distinct"),
    ({"search_space": {"dqn_grad_steps_per_ep": {"type": "int", "low": 1, "high": 10}}}, "not active"),
    ({"search_space": {"pg_lr_actor": {"type": "float", "low": 0, "high": 1}}}, "> 0"),
    ({"search_space": {"pg_batch_eps": {"type": "categorical", "choices": [1, 1]}}}, "duplicates"),
    ({"search_space": {"pg_entropy_min": {"type": "float", "low": 0, "high": 2}}}, "<= 1"),
    ({"metrics": {"generation_weights": {"qualified_fraction": 1}}}, "sum to 1"),
    ({"training_search": {"validation_seeds": [11, 211, 307]}}, "disjoint"),
    ({"training_search": {"trials": 2}}, "trial counts"),
])
def test_malformed_and_conflicting_inputs_are_rejected(tmp_path, extra, match):
    with pytest.raises(ConfigError, match=match):
        load_plan(write_plan(tmp_path, extra))


def test_duplicate_yaml_keys_rejected_recursively(tmp_path):
    path = write_plan(tmp_path)
    path.write_text(path.read_text() + "efficiency: {max_attempts_per_unique: 1.5, max_attempts_per_unique: 1.8}\n")
    with pytest.raises(ConfigError, match="Duplicate YAML"):
        load_plan(path)


@pytest.mark.parametrize("objective,match", [
    ({"direction": "max"}, "minimize or maximize"),
    ({"threshold": None}, "finite number"),
    ({"threshold": 0}, "positive threshold"),
    ({"field": "missing"}, "exactly one"),
    ({"unit": "C"}, "unit must match"),
    ({"normalization": {"kind": "sigmoid", "scale": 0}}, "> 0"),
])
def test_objective_contract(tmp_path, objective, match):
    defaults = {"field": "temperature", "direction": "minimize", "threshold": 1000, "unit": "K"}
    with pytest.raises(ConfigError, match=match):
        load_plan(write_plan(tmp_path, {"objective": {**defaults, **objective}}))


def test_other_application_negative_maximize_yaml_only(tmp_path):
    plan = load_plan(write_plan(tmp_path, {
        "objective": {"field": "energy", "direction": "maximize", "threshold": -2,
                      "unit": "eV", "normalization": {"kind": "sigmoid", "scale": .5}},
        "efficiency": {"max_attempts_per_unique": 1.7, "generation": {"max_attempts_per_unique": 2.2}},
        "budget": {"mode": "absolute_unique", "train": {"initial_unique": 400, "min_unique": 200, "max_unique": 800},
                   "generation": {"initial_unique": 20, "min_unique": 10, "max_unique": 80}},
    }, base={"properties": [{"name": "energy", "unit": "eV"}], "dqn_grad_steps_per_ep": 17}))
    assert plan["objective"]["direction"] == "maximize"
    assert phase_efficiency(plan, "train")["max_attempts_per_unique"] == 1.7
    assert phase_efficiency(plan, "generation")["max_attempts_per_unique"] == 2.2
    assert plan["base_config_data"]["dqn_grad_steps_per_ep"] == 17


@pytest.mark.parametrize("method,count", [("dqn-bootstrap", 6), ("dqn-mc", 5), ("a2c", 5)])
def test_profiles_active_parameters_only(method, count):
    profile = get_profile(method)
    assert len(profile["search_space"]) == count
    assert "dqn_grad_steps_per_ep" not in profile["search_space"]
    assert ("dqn_target_update_freq" in profile["search_space"]) == (method == "dqn-bootstrap")


def test_mc_inert_target_update_cannot_be_searched_but_can_remain_in_base(tmp_path):
    path = write_plan(tmp_path, {"method": "dqn-mc"}, base={"dqn_target_update_freq": 7})
    assert load_plan(path)["base_config_data"]["dqn_target_update_freq"] == 7
    with pytest.raises(ConfigError, match="not active"):
        load_plan(write_plan(tmp_path, {"method": "dqn-mc", "search_space": {"dqn_target_update_freq": {"type": "categorical", "choices": [10, 20]}}}))


def test_midpoint_and_budget_derivations_do_not_clip_anneal(tmp_path):
    plan = load_plan(write_plan(tmp_path, {"method": "dqn-bootstrap"}, base={"dqn_grad_steps_per_ep": 9}))
    reference = initial_parameters(plan)
    assert reference["dqn_lr"] == pytest.approx((1e-5 * 1e-2)**.5)
    assert reference["dqn_batch_size"] == 128
    before = deepcopy(plan)
    overrides = resolve_parameters(plan, {"warmup_budget_fraction": .1, "epsilon_anneal_budget_ratio": 2}, 100)
    assert overrides["dqn_warmup_eps"] == 10
    assert overrides["dqn_num_train_eps"] == 90
    assert overrides["dqn_eps_anneal_eps"] == 180
    assert overrides["dqn_grad_steps_per_ep"] == 9
    assert not overrides["train_count_duplicates"]
    assert resolve_config(plan["base_config_data"], overrides).config["train_budget"] == 100
    assert plan == before


def test_a2c_batch_does_not_change_unique_budget(tmp_path):
    plan = load_plan(write_plan(tmp_path))
    a = resolve_parameters(plan, {"pg_batch_eps": 8}, 101)
    b = resolve_parameters(plan, {"pg_batch_eps": 64}, 101)
    assert a["train_budget"] == b["train_budget"] == 101
    assert a["pg_num_iters"] == 13
    assert b["pg_num_iters"] == 2
    assert resolve_config(plan["base_config_data"], a).config["method"] == "a2c"


def test_custom_profile_and_partial_hpo_range_overrides(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"method": "a2c", "search_space": {"pg_batch_eps": {"type": "categorical", "choices": [2, 4]}}, "extension_choices": {"pg_batch_eps": [1, 2, 4, 8]}}))
    plan = load_plan(write_plan(tmp_path, {"parameter_profile": "profile.yaml", "search_space": {"pg_lr_actor": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True}}}))
    assert len(plan["search_space"]) == 5
    assert plan["search_space"]["pg_batch_eps"]["choices"] == [2, 4]
    assert plan["extension_choices"]["pg_batch_eps"] == [1, 2, 4, 8]


def test_fraction_and_absolute_budget_fields_cannot_mix(tmp_path):
    with pytest.raises(ConfigError, match="Unknown fields"):
        load_plan(write_plan(tmp_path, {"budget": {"train": {"initial_unique": 500}}}))


def test_bounds_allow_bidirectional_search_and_validate_order(tmp_path):
    with pytest.raises(ConfigError, match="min <= initial <= max"):
        load_plan(write_plan(tmp_path, {"budget": {"generation": {"min_fraction": .001}}}))
