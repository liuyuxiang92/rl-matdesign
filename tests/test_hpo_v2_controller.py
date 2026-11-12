"""Execution contracts for shared training repeats and model-wise generation."""
from copy import deepcopy

import pytest
import yaml

from rl_matdesign.hpo.artifacts import atomic_json, digest, read_json
from rl_matdesign.hpo.controller import Campaign, CampaignError, rank_results
from rl_matdesign.hpo.spec import load_plan


def internal_plan(tmp_path, parameter="gen_temperature"):
    """Keep orchestration tests independent of the public-schema migration."""
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump({
        "pg_warmup_eps": 0, "properties": [{"name": "x", "unit": "u"}],
    }))
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump({
        "protocol": "hpo_v1", "method": "a2c", "base_config": "scenario.yaml",
        "objective": {"field": "x", "direction": "minimize", "threshold": 10, "unit": "u"},
        "budget": {"mode": "absolute_unique",
                   "train": {"min_unique": 4, "initial_unique": 16, "max_unique": 32},
                   "generation": {"min_unique": 4, "initial_unique": 16, "max_unique": 32}},
        "search_space": {"pg_batch_eps": {"type": "categorical", "choices": [2, 4]}},
        "range_search": {"max_extra_values_per_direction": 0},
        "training_search": {"trials": 2, "startup_random_trials": 1, "finalists": 1,
                            "max_budget_revisions": 1, "additional_trials_per_revision": 2,
                            "additional_random_trials": 1},
    }))
    plan = load_plan(source)
    plan["protocol"] = "hpo_v2"
    plan["scoring_scope"] = "score_only"
    plan["range_search"]["seeds"] = [11, 29, 47]
    plan["range_search"]["confirmation_seed"] = 73
    plan["training_search"]["seeds"] = [11, 29, 47]
    plan["training_search"]["validation_seeds"] = [101, 211, 307]
    plan["generation_search"] = {
        "parameter": parameter, "initial_values": [0, .1, 1] if parameter == "gen_temperature" else [.2, .8],
        "max_extra_values_per_direction": 1, "repeats_per_model": 2, "master_seed": 42,
    }
    return plan


class ReusableRunner:
    def __init__(self, parameter="gen_temperature", prefer="high"):
        self.parameter, self.prefer = parameter, prefer
        self.calls, self.cache = [], {}

    def __call__(self, parameters, budget, seed, phase, model=None):
        # Deliberately omit the model from this fake key: distinct derived
        # sampling seeds must prevent accidental cross-model reuse as well.
        key = digest({"parameters": parameters, "budget": budget, "seed": seed, "phase": phase})
        if key in self.cache:
            return deepcopy(self.cache[key])
        self.calls.append({"parameters": parameters, "budget": budget, "seed": seed,
                           "phase": phase, "model": model})
        score = .8 - (.1 if budget < 8 else 0)
        if phase == "generation":
            value = parameters[self.parameter]
            if self.parameter == "gen_temperature":
                score -= 0 if value == 1 else .15
            else:
                score = .4 + .3 * (value if self.prefer == "high" else 1 - value)
        row = {"job_id": key[:24], "status": "COMPLETED", "eligible_for_selection": True,
               "score": score, "seed": seed, "budget": budget, "wall_seconds": .01,
               "raw_metrics": {"qualified_fraction": score, "best_fraction_mean": 2,
                               "best_value": 1, "attempts_per_unique": 1.1},
               "checkpoint_paths": {"policy.pt": f"/fake/{key}/policy.pt",
                                    "std_scaler.bin": f"/fake/{key}/std_scaler.bin"}}
        self.cache[key] = deepcopy(row)
        return row


@pytest.fixture
def campaign_environment(monkeypatch):
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "test-software")
    monkeypatch.setattr(Campaign, "export_reports", lambda self: None)


def campaign(plan, out, runner):
    return Campaign(plan, out, runner=runner,
                    domain={"status": "unknown", "size": None, "episode_steps": 1})


def test_shared_training_repeats_and_generation_cartesian_repeats_resume(tmp_path, campaign_environment):
    plan = internal_plan(tmp_path)
    runner = ReusableRunner()
    run = campaign(plan, tmp_path / "run", runner)
    assert run.run()["status"] == "COMPLETED"
    stages = {stage: read_json(run.stage_path(stage)) for stage in
              ("ranges", "train-budget", "train-search", "generation")}
    assert stages["ranges"]["evaluations"]
    assert all(row["seeds"] == [11, 29, 47] for row in stages["ranges"]["evaluations"])
    assert all(row["seeds"] == [11, 29, 47] for row in stages["train-budget"]["results"])
    assert all(row["seeds"] == [11, 29, 47] for study in stages["train-search"]["studies"]
               for row in study["results"])
    assert all(row["seeds"] == [101, 211, 307] for row in stages["train-search"]["results"])
    generation = stages["generation"]
    pairs = generation["generation_pairings"]
    assert len(pairs) == 6
    assert len({pair["seed"] for pair in pairs}) == 6
    assert {pair["training_seed"] for pair in pairs} == {101, 211, 307}
    assert {pair["generation_repeat_index"] for pair in pairs} == {0, 1}
    generation_calls = [call for call in runner.calls if call["phase"] == "generation"]
    assert len(generation_calls) == len(generation["parameter_values"]) * 4 * 6
    for row in generation["results"]:
        assert row["generation_pairings"] == pairs
        assert len(row["seed_results"]) == 6
        assert len(row["model_results"]) == 3
        assert all(len(model["seed_results"]) == 2 for model in row["model_results"])
        assert {seed["model_id"] for seed in row["seed_results"]} == {
            model["job_id"] for model in stages["train-search"]["models"]}
    count = len(runner.calls)
    assert campaign(plan, run.out, runner).run()["status"] == "COMPLETED"
    assert len(runner.calls) == count
    assert read_json(run.out / "manifest.json")["protocol"] == "hpo_v2"


@pytest.mark.parametrize("parameter", ["gen_epsilon", "gen_top_frac"])
@pytest.mark.parametrize("prefer,expected", [("high", 1), ("low", 0)])
def test_bounded_generation_search_selects_and_exports_only_active_method(
        tmp_path, campaign_environment, parameter, prefer, expected):
    plan = internal_plan(tmp_path, parameter)
    runner = ReusableRunner(parameter, prefer)
    run = campaign(plan, tmp_path / "run", runner)
    models = [runner({"pg_batch_eps": 2}, 16, seed, "train") for seed in [101, 211, 307]]
    atomic_json(run.stage_path("train-search"), {
        "status": "COMPLETED", "models": models,
        "selection": {"selected": {"parameters": {"pg_batch_eps": 2}}}})
    initial_calls = len(runner.calls)
    result = run.run_generation()
    assert result["status"] == "COMPLETED"
    assert result["parameter"] == parameter
    assert expected in result["parameter_values"]
    assert all(0 <= value <= 1 for value in result["parameter_values"])
    assert result["selection"]["selected"]["parameters"][parameter] == expected
    assert all(call["phase"] == "generation" for call in runner.calls[initial_calls:])
    exported = yaml.safe_load((run.out / "best_generation.yaml").read_text())
    assert exported[parameter] == expected
    assert not (set(exported) & ({"gen_temperature", "gen_top_frac", "gen_epsilon"} - {parameter}))
    assert "exploration_gen_eps" not in exported


def test_near_tie_ranking_does_not_use_component_metrics(tmp_path):
    plan = internal_plan(tmp_path)
    rows = [
        {"parameter_id": "a", "eligible_for_selection": True, "score": .8,
         "raw_metrics": {"attempts_per_unique": 1.49, "best_fraction_mean": 100}},
        {"parameter_id": "b", "eligible_for_selection": True, "score": .805,
         "raw_metrics": {"attempts_per_unique": 1.0, "best_fraction_mean": 1}},
        {"parameter_id": "c", "eligible_for_selection": False, "score": 1,
         "raw_metrics": {"attempts_per_unique": 1.0, "best_fraction_mean": 0}},
    ]
    assert [row["parameter_id"] for row in rank_results(rows, plan)] == ["a", "b"]
    plan["budget"]["tolerances"]["score"] = .001
    assert [row["parameter_id"] for row in rank_results(rows, plan)] == ["b", "a"]


def test_duplicate_search_trials_do_not_consume_validation_config_slots(
        tmp_path, campaign_environment, monkeypatch):
    plan = public_plan(tmp_path)
    runner = ReusableRunner()
    run = campaign(plan, tmp_path / "run", runner)
    trials = []
    for index, rate in enumerate([.0001, .0002, .0003, .0004]):
        trial = run.evaluate({"pg_batch_eps": 2, "pg_lr_actor": rate}, 8,
                             plan["training_search"]["seeds"])
        trial["score"] = .95 - index * .05
        trials.append(trial)
    # Repeated proposals are legitimate (and cached) in categorical TPE spaces.
    # Three distinct finalists must reach validation; a duplicate of the winner
    # must not displace third place, and fourth place must not enter the panel.
    monkeypatch.setattr(run, "train_study", lambda *args: [trials[0], deepcopy(trials[0]), *trials[1:]])
    atomic_json(run.stage_path("train-budget"), {
        "selection": {"selected_budget": 8}, "panel": []})
    result = run.run_train_search()
    assert result["status"] == "COMPLETED"
    validated = result["results"]
    assert {row["parameters"]["pg_lr_actor"] for row in validated} == {.0001, .0002, .0003}
    for budget in {row["budget"] for row in validated}:
        assert len([row for row in validated if row["budget"] == budget]) == 3
    assert all(row["seeds"] == plan["training_search"]["validation_seeds"] for row in validated)


def test_generation_measurement_identity_includes_frozen_model(tmp_path, campaign_environment):
    plan = internal_plan(tmp_path)
    runner = ReusableRunner()
    run = campaign(plan, tmp_path / "run", runner)
    source_a = runner({"pg_batch_eps": 2}, 16, 101, "train")
    source_b = runner({"pg_batch_eps": 2}, 32, 101, "train")
    seeds_a, models_a, pairings_a = run.generation_pairs([source_a])
    seeds_b, models_b, pairings_b = run.generation_pairs([source_b])
    assert seeds_a == seeds_b  # Same reproducible RNG pairing, different model.
    assert pairings_a[0]["model_id"] != pairings_b[0]["model_id"]
    first = run.evaluate({"gen_temperature": 1}, 8, seeds_a, "generation", models_a, pairings_a)
    second = run.evaluate({"gen_temperature": 1}, 8, seeds_b, "generation", models_b, pairings_b)
    assert first["seed_results"][0]["model_id"] != second["seed_results"][0]["model_id"]
    assert len(list((run.out / "measurements").glob("*.json"))) == 2


def test_one_failed_generation_repeat_invalidates_aggregate(tmp_path, campaign_environment):
    plan = internal_plan(tmp_path)
    runner = ReusableRunner()
    run = campaign(plan, tmp_path / "run", runner)
    source = runner({"pg_batch_eps": 2}, 16, 101, "train")
    seeds, models, pairings = run.generation_pairs([source])
    original = runner.__call__

    def one_failure(parameters, budget, seed, phase, model=None):
        row = original(parameters, budget, seed, phase, model)
        if seed == seeds[1]:
            row.update(eligible_for_selection=False, status="INFEASIBLE_EFFICIENCY", score=-1)
        return row

    run.runner = one_failure
    result = run.evaluate({"gen_temperature": 1}, 8, seeds, "generation", models, pairings)
    assert not result["eligible_for_selection"]
    assert not result["model_results"][0]["eligible_for_selection"]
    assert result["score"] == -1


def public_plan(tmp_path):
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump({
        "pg_warmup_eps": 0, "gen_temperature": 1,
        "properties": [{"name": "working_temperature", "output": "T3",
                        "predictor": "dummy", "direction": "min", "unit": "K"}],
        "reward": {"objective": {"terms": {"working_temperature": 1}}},
    }, sort_keys=False))
    budget = {"mode": "absolute_unique", "min_unique": 4, "initial_unique": 16, "max_unique": 32}
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump({
        "base_config": "scenario.yaml", "method": "a2c", "random_seed": 42,
        "evaluation": {"threshold": 10, "score_tolerance": .01},
        "training": {"repeats": 3,
                     "parameter_ranges": {"pg_batch_eps": {"type": "categorical", "choices": [2, 4]}},
                     "budget": budget,
                     "search": {"max_rounds": 2, "trials_per_round": 2, "startup_random_trials": 1}},
        "generation": {"repeats": 2, "initial_values": [0, .1, 1],
                       "max_extra_values_per_direction": 0, "budget": budget},
    }, sort_keys=False))
    return load_plan(source)


def test_public_v2_yaml_runs_all_stages_with_two_repeat_controls(tmp_path, campaign_environment):
    plan = public_plan(tmp_path)
    assert plan["protocol"] == "hpo_v2"
    assert plan["objective"]["field"] == "working_temperature"
    assert plan["objective"]["threshold_inclusive"] is True
    exploration = plan["seed_schedule"]["exploration"]
    validation = plan["seed_schedule"]["validation"]
    assert len(exploration) == len(validation) == 3
    assert set(exploration).isdisjoint(validation)
    runner = ReusableRunner()
    run = campaign(plan, tmp_path / "run", runner)
    assert run.run()["status"] == "COMPLETED"
    trained = read_json(run.stage_path("train-search"))
    assert {model["seed"] for model in trained["models"]} == set(validation)
    assert len(trained["models"]) == 3
    generation = read_json(run.stage_path("generation"))
    assert len(generation["generation_pairings"]) == 6
    assert all(len(row["seed_results"]) == 6 for row in generation["results"])
    assert set(plan["authored_config"]) == {"base_config", "method", "random_seed", "evaluation", "training", "generation"}


def test_failed_quality_calibration_stops_before_buying_more_probes(tmp_path, campaign_environment):
    plan = public_plan(tmp_path)
    runner = ReusableRunner()
    original = runner.__call__

    def cannot_calibrate(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update(status="QUALITY_CALIBRATION_REQUIRED", eligible_for_selection=False,
                   reason="The reference archive has no variation", score=-1)
        return row

    run = campaign(plan, tmp_path / "run", cannot_calibrate)
    with pytest.raises(CampaignError, match="QUALITY_CALIBRATION_REQUIRED.*no variation"):
        run.run("ranges")
    assert len(runner.calls) == 1
    state = read_json(run.out / "state.json")
    assert state["status"] == "SYSTEM_ERROR"
    assert "no variation" in state["reason"]
