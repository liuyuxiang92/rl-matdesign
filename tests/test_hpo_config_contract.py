"""HPO trial reconstruction uses the experiment's configuration contract."""
import optuna
import pytest
import yaml

import hpo


def test_hpo_overrides_keep_one_shared_discount():
    config = hpo._build_trial_yaml(
        base_cfg={"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}], "gamma": .9, "dqn_num_train_eps": 100},
        fixed_overrides={"gamma": .8}, sampled={"dqn_lr": .002},
        budget_scaling={"dqn_num_train_eps": .5}, method="dqn",
    )
    assert config["gamma"] == .8
    assert config["dqn_num_train_eps"] == 50
    assert "pg_gamma" not in config and "dqn_gamma" not in config
    assert hpo.resolve_config(config).config["gamma"] == .8


def test_hpo_rejects_unknown_core_parameters():
    with pytest.raises(ValueError, match="dqn_lrr"):
        hpo._build_trial_yaml(base_cfg={"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}]},
                              fixed_overrides={}, sampled={"dqn_lrr": .002},
                              budget_scaling={}, method="dqn")


def test_hpo_scales_omitted_defaults_without_freezing_other_defaults():
    config = hpo._build_trial_yaml(
        base_cfg={"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}]}, fixed_overrides={}, sampled={},
        budget_scaling={"dqn_num_train_eps": .25, "dqn_warmup_eps": .25}, method="dqn",
    )
    assert config["dqn_num_train_eps"] == 5000
    assert config["dqn_warmup_eps"] == 125
    assert "pg_warmup_eps" not in config


@pytest.mark.parametrize("scaling", [{"gamma": .25}, {"dqn_num_train_ep": .5},
                                     {"pg_num_iters": False}, {"pg_num_iters": float("nan")}])
def test_hpo_rejects_ambiguous_budget_scaling(scaling):
    with pytest.raises(ValueError, match="budget_scaling"):
        hpo._build_trial_yaml(base_cfg={}, fixed_overrides={}, sampled={},
                              budget_scaling=scaling, method="a2c")


def test_hpo_sampled_batch_size_preserves_shared_query_budget():
    base = {"train_budget": 1001, "dqn_warmup_eps": 101, "pg_batch_eps": 10}
    for batch in [7, 32, 100]:
        cfg = hpo._build_trial_yaml(base_cfg=base, fixed_overrides={}, sampled={"pg_batch_eps": batch},
                                    budget_scaling={}, method="a2c")
        assert cfg["train_budget"] == 1001
        assert "pg_num_iters" not in cfg and "dqn_num_train_eps" not in cfg
        effective = hpo.resolve_config(cfg).config
        assert effective["pg_num_iters"] == (1001 + batch - 1) // batch
        assert effective["dqn_num_train_eps"] == 900


def test_hpo_scales_shared_budget_and_warmup_before_deriving_totals():
    cfg = hpo._build_trial_yaml(base_cfg={"train_budget": 1000, "dqn_warmup_eps": 100},
                                fixed_overrides={}, sampled={"pg_batch_eps": 7},
                                budget_scaling={"train_budget": .25, "dqn_warmup_eps": .5}, method="dqn")
    assert cfg["train_budget"] == 250 and cfg["dqn_warmup_eps"] == 50
    assert "dqn_num_train_eps" not in cfg and "pg_num_iters" not in cfg
    assert hpo.resolve_config(cfg).config["dqn_num_train_eps"] == 200
    assert hpo.resolve_config(cfg).config["pg_num_iters"] == 36


@pytest.mark.parametrize("scaling", [{"dqn_num_train_eps": .5}, {"pg_num_iters": .5}])
def test_hpo_rejects_algorithm_budget_scaling_with_shared_budget(scaling):
    with pytest.raises(ValueError, match="scale train_budget"):
        hpo._build_trial_yaml(base_cfg={"train_budget": 1000}, fixed_overrides={}, sampled={},
                             budget_scaling=scaling, method="dqn")


def test_hpo_does_not_invent_a_shared_budget_or_allow_scaled_warmup_to_exceed_it():
    with pytest.raises(ValueError, match="explicitly configured"):
        hpo._build_trial_yaml(base_cfg={}, fixed_overrides={}, sampled={},
                             budget_scaling={"train_budget": .5}, method="a2c")
    with pytest.raises(ValueError, match="exceeds train_budget"):
        hpo._build_trial_yaml(base_cfg={"train_budget": 1000, "dqn_warmup_eps": 300},
                             fixed_overrides={}, sampled={}, budget_scaling={"train_budget": .1}, method="dqn")


def test_hpo_full_warmup_budget_leaves_no_online_episodes():
    cfg = hpo._build_trial_yaml(base_cfg={"train_budget": 10, "dqn_warmup_eps": 10},
                              fixed_overrides={}, sampled={}, budget_scaling={}, method="dqn")
    assert hpo.resolve_config(cfg).config["dqn_num_train_eps"] == 0


def test_stage_two_replays_fraction_before_integer_validation(tmp_path, monkeypatch):
    stage_one = tmp_path / "stage_one.yaml"
    stage_one.write_text("dqn_num_train_eps: 25\ndqn_warmup_eps: 12\n")
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        value=1.0, params={"dqn_warmup_eps": .5},
        distributions={"dqn_warmup_eps": optuna.distributions.FloatDistribution(.1, .9)},
        user_attrs={"config_path": str(stage_one)},
    ))
    seen = []

    def run(**kwargs):
        seen.append(yaml.safe_load(open(kwargs["trial_yaml"])))
        return 1.0, 1

    monkeypatch.setattr(hpo, "_run_one_seed", run)
    result = hpo._run_stage2(
        out_root=str(tmp_path), stage1_study=study,
        base_cfg={"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}], "dqn_num_train_eps": 100},
        hpo_cfg={"method": "dqn", "fixed_overrides": {}, "metric": {"k": 1},
                 "stage2": {"n_top_to_confirm": 1, "seeds_per_trial": 1},
                 "search_space": {"dqn_warmup_eps": {
                     "dist": "frac_of", "base": "dqn_num_train_eps", "low": .1, "high": .9}}},
        gpu_pool=None, timeout_s=None,
    )
    assert len(result) == 1
    assert seen[0]["dqn_num_train_eps"] == 100
    assert seen[0]["dqn_warmup_eps"] == 50


@pytest.mark.parametrize("key", [
    "dqn_gamma", "pg_gamma", "exploration_gen_eps", "generated_columns",
    "predict_cache_size", "buffer_size", "num_train_eps", "grad_steps_per_ep",
    "target_update_freq", "eps_anneal_eps", "eps_min", "entropy_coef",
    "repeat_penalty_coef", "repeat_penalty_shape", "dp_seed",
])
@pytest.mark.parametrize("layer", ["base_cfg", "fixed_overrides", "sampled"])
def test_hpo_trial_rejects_retired_parameters_in_every_layer(key, layer):
    args = dict(base_cfg={}, fixed_overrides={}, sampled={},
                budget_scaling={}, method="dqn")
    args[layer][key] = 1
    with pytest.raises(ValueError, match=key):
        hpo._build_trial_yaml(**args)


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
def test_hpo_sampled_gamma_stays_shared_across_methods(method):
    config = hpo._build_trial_yaml(
        base_cfg={"gamma": .9}, fixed_overrides={}, sampled={"gamma": .75},
        budget_scaling={}, method=method,
    )
    assert config == {"gamma": .75, "method": method}
    assert hpo.resolve_config(config).config["gamma"] == .75
