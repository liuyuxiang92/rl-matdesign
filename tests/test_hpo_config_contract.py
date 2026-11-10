"""HPO trial reconstruction uses the experiment's configuration contract."""
import optuna
import pytest
import yaml

import hpo


def test_hpo_overrides_keep_shared_defaults_derived_at_execution():
    config = hpo._build_trial_yaml(
        base_cfg={"predictor": "dummy", "gamma": .9, "dqn_num_train_eps": 100},
        fixed_overrides={"gamma": .8}, sampled={"dqn_lr": .002},
        budget_scaling={"dqn_num_train_eps": .5}, method="dqn",
    )
    assert config["gamma"] == .8
    assert config["dqn_num_train_eps"] == 50
    assert "pg_gamma" not in config and "dqn_gamma" not in config
    assert hpo.resolve_config(config).config["dqn_gamma"] == .8


def test_hpo_rejects_unknown_core_parameters():
    with pytest.raises(ValueError, match="dqn_lrr"):
        hpo._build_trial_yaml(base_cfg={"predictor": "dummy"},
                              fixed_overrides={}, sampled={"dqn_lrr": .002},
                              budget_scaling={}, method="dqn")


def test_hpo_scales_omitted_defaults_without_freezing_other_defaults():
    config = hpo._build_trial_yaml(
        base_cfg={"predictor": "dummy"}, fixed_overrides={}, sampled={},
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
                             fixed_overrides={}, sampled={}, budget_scaling={"train_budget": .1}, method="a2c")


def test_hpo_full_warmup_budget_does_not_make_anneal_zero_invalid():
    cfg = hpo._build_trial_yaml(base_cfg={"train_budget": 10, "dqn_warmup_eps": 10,
                                        "dqn_eps_anneal_eps": 50},
                              fixed_overrides={}, sampled={}, budget_scaling={}, method="dqn")
    assert cfg["dqn_eps_anneal_eps"] == 1
    assert hpo.resolve_config(cfg).config["dqn_num_train_eps"] == 0


def test_stage_two_replays_fraction_before_integer_validation(tmp_path, monkeypatch):
    stage_one = tmp_path / "stage_one.yaml"
    stage_one.write_text("dqn_num_train_eps: 25\ndqn_eps_anneal_eps: 12\n")
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        value=1.0, params={"dqn_eps_anneal_eps": .5},
        distributions={"dqn_eps_anneal_eps": optuna.distributions.FloatDistribution(.1, .9)},
        user_attrs={"config_path": str(stage_one)},
    ))
    seen = []

    def run(**kwargs):
        seen.append(yaml.safe_load(open(kwargs["trial_yaml"])))
        return 1.0, 1

    monkeypatch.setattr(hpo, "_run_one_seed", run)
    result = hpo._run_stage2(
        out_root=str(tmp_path), stage1_study=study,
        base_cfg={"predictor": "dummy", "dqn_num_train_eps": 100},
        hpo_cfg={"method": "dqn", "fixed_overrides": {}, "metric": {"k": 1},
                 "stage2": {"n_top_to_confirm": 1, "seeds_per_trial": 1},
                 "search_space": {"dqn_eps_anneal_eps": {
                     "dist": "frac_of", "base": "dqn_num_train_eps", "low": .1, "high": .9}}},
        gpu_pool=None, timeout_s=None,
    )
    assert len(result) == 1
    assert seen[0]["dqn_num_train_eps"] == 100
    assert seen[0]["dqn_eps_anneal_eps"] == 50
