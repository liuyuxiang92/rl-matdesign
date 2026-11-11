"""Fast protocol integration: fake observations and subprocesses, no ML models."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from rl_matdesign.hpo.artifacts import atomic_json, digest, read_json
from rl_matdesign.hpo.algorithm_profiles import get_profile, initial_parameters
from rl_matdesign.hpo.controller import Campaign, CampaignError, aggregate_results
from rl_matdesign.hpo.domain import design_space_info, minimum_training_budget, training_budget_preflight
from rl_matdesign.hpo.job_runner import JobError, JobRunner
from rl_matdesign.hpo.spec import load_plan


def small_plan(tmp_path):
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump({
        "pg_warmup_eps": 0, "properties": [{"name": "x", "unit": "u"}],
    }))
    data = {
        "method": "a2c", "base_config": "scenario.yaml",
        "objective": {"field": "x", "direction": "minimize", "threshold": 10, "unit": "u"},
        "budget": {"mode": "absolute_unique",
                   "train": {"min_unique": 4, "initial_unique": 16, "max_unique": 32},
                   "generation": {"min_unique": 4, "initial_unique": 16, "max_unique": 32}},
        "search_space": {"pg_batch_eps": {"type": "categorical", "choices": [2, 4]}},
        "range_search": {"max_extra_values_per_direction": 0},
        "training_search": {"trials": 3, "startup_random_trials": 1, "finalists": 1,
                            "max_budget_revisions": 1, "additional_trials_per_revision": 2, "additional_random_trials": 1},
        "generation_search": {"initial_values": [0, .1, 1], "max_extra_values_per_direction": 1},
    }
    path = tmp_path / "hpo.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_plan(path)


class FakeRunner:
    def __init__(self, fail_seed=None):
        self.calls = []
        self.cache = {}
        self.fail_seed = fail_seed

    def __call__(self, parameters, budget, seed, phase, model=None):
        key = digest({"parameters": parameters, "budget": budget, "seed": seed, "phase": phase})
        if key in self.cache:
            return deepcopy(self.cache[key])
        self.calls.append({"parameters": deepcopy(parameters), "budget": budget, "seed": seed, "phase": phase, "model": model})
        temperature = parameters.get("gen_temperature", 1)
        loss = (.1 if budget < 8 else 0) + (0 if phase == "train" or temperature == 1 else .15)
        eligible = seed != self.fail_seed
        result = {"status": "COMPLETED" if eligible else "INFEASIBLE_EFFICIENCY", "eligible_for_selection": eligible,
                  "score": .8 - loss if eligible else -1, "seed": seed, "budget": budget, "wall_seconds": .01,
                  "raw_metrics": {"qualified_fraction": .8 - loss, "best_fraction_mean": 2 + 10 * loss,
                                  "best_value": 1 + 10 * loss, "attempts_per_unique": 1.1,
                                  "qualified_count": int((.8 - loss) * budget)},
                  "checkpoint_paths": {"policy.pt": f"/fake/{key}/policy.pt", "std_scaler.bin": f"/fake/{key}/std_scaler.bin"}}
        self.cache[key] = deepcopy(result)
        return result


@pytest.fixture
def campaign_environment(monkeypatch):
    # Parallel implementation edits are irrelevant to these isolated fixtures.
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "test-software")
    monkeypatch.setattr(Campaign, "export_reports", lambda self: None)


def campaign(plan, out, runner):
    return Campaign(plan, out, runner=runner, domain={"status": "unknown", "size": None, "episode_steps": 1})


def test_full_flow_selects_smaller_budget_then_temperature_only(tmp_path, campaign_environment):
    plan = small_plan(tmp_path)
    runner = FakeRunner()
    run = campaign(plan, tmp_path / "run", runner)
    assert run.run()["status"] == "COMPLETED"
    trained = read_json(run.stage_path("train-search"))
    generated = read_json(run.stage_path("generation"))
    assert trained["selection"]["selected_budget"] == 8 < 16
    assert generated["selection"]["selected_budget"] == 8
    assert generated["selection"]["selected"]["parameters"]["gen_temperature"] == 1
    assert generated["selection"]["convergence_status"] == "OBSERVED_PLATEAU"
    assert 10 in generated["temperatures"]  # best positive boundary is probed outward
    generation_calls = [call for call in runner.calls if call["phase"] == "generation"]
    chosen = trained["selection"]["selected"]["parameters"]
    for call in generation_calls:
        assert {k: v for k, v in call["parameters"].items() if k != "gen_temperature"} == chosen
        assert call["model"] in trained["models"]
    for temperature in generated["temperatures"]:
        assert {c["budget"] for c in generation_calls if c["parameters"]["gen_temperature"] == temperature} == {4, 8, 16, 32}
    assert {c["budget"] for c in runner.calls if c["phase"] == "train"} == {4, 8, 16, 32}
    assert (run.out / "best_training.yaml").is_file()
    assert (run.out / "best_generation.yaml").is_file()


def test_ranges_and_search_can_run_separately_and_completed_stages_reuse(tmp_path, campaign_environment):
    plan = small_plan(tmp_path)
    runner = FakeRunner()
    first = campaign(plan, tmp_path / "run", runner)
    assert first.run("ranges")["completed_stages"] == ["ranges"]
    assert all(call["phase"] == "train" for call in runner.calls)
    resumed = campaign(plan, tmp_path / "run", runner)
    assert resumed.run("search")["status"] == "COMPLETED"
    count = len(runner.calls)
    assert campaign(plan, tmp_path / "run", runner).run()["status"] == "COMPLETED"
    assert len(runner.calls) == count


def test_missing_stage_dependency_does_not_start_jobs(tmp_path, campaign_environment):
    runner = FakeRunner()
    run = campaign(small_plan(tmp_path), tmp_path / "run", runner)
    with pytest.raises(CampaignError, match="requires completed train-search"):
        run.run("generation")
    assert not runner.calls


def test_manifest_rejects_changed_source_even_when_parsed_content_is_identical(tmp_path, campaign_environment):
    plan = small_plan(tmp_path)
    out = tmp_path / "run"
    campaign(plan, out, FakeRunner())
    source = Path(plan["source_path"])
    source.write_text(source.read_text() + "\n# Edited experiment specification\n")
    with pytest.raises(CampaignError, match="changed"):
        campaign(plan, out, FakeRunner())


def test_required_seed_failure_is_not_averaged_away(tmp_path, campaign_environment):
    runner = FakeRunner(fail_seed=29)
    run = campaign(small_plan(tmp_path), tmp_path / "run", runner)
    status = run.run("ranges")
    assert status["status"] == "NO_FEASIBLE_CONFIGURATION"
    assert all(not r["eligible_for_selection"] for r in read_json(run.stage_path("ranges"))["evaluations"])
    assert not (run.out / "best_training.yaml").exists()


def test_aggregate_requires_every_seed_and_keeps_missing_scientific_metrics():
    rows = [{"seed": 1, "eligible_for_selection": True, "score": .8, "status": "COMPLETED", "raw_metrics": {"best_value": 2}},
            {"seed": 2, "eligible_for_selection": False, "score": -1, "status": "INFEASIBLE_EFFICIENCY", "raw_metrics": {"best_value": None}}]
    result = aggregate_results(rows, {"a": 1}, 10)
    assert not result["eligible_for_selection"] and result["score"] == -1
    assert result["raw_metrics"]["best_value"] is None


def test_training_minimum_supports_range_updated_categorical_and_integer_specs(tmp_path):
    plan = small_plan(tmp_path)
    plan["search_space"]["pg_batch_eps"] = {"type": "int", "low": 2, "high": 8}
    assert minimum_training_budget(plan, {"episode_steps": 1}) == 8
    plan["method"] = "dqn-bootstrap"
    plan["search_space"] = {"dqn_batch_size": {"type": "int", "low": 4, "high": 8},
                            "dqn_target_update_freq": {"type": "int", "low": 2, "high": 4},
                            "warmup_budget_fraction": {"type": "categorical", "choices": [.2]}}
    assert minimum_training_budget(plan, {"episode_steps": 2}) == 7


def fake_worker(monkeypatch):
    calls = []
    class Process:
        returncode = 0
        def __init__(self, args, **kwargs):
            cfg = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
            out = Path(args[args.index("--out") + 1])
            calls.append({"config": cfg, "args": args, "out": out, **kwargs})
            budget = cfg["hpo_target_unique"]
            atomic_json(out / "candidate_budget.json", {"target_unique": budget, "unique_evaluated": budget,
                                                       "attempts_started": budget, "status": "COMPLETED"})
            rows = [{"identity": str(i), "properties": {"x": 2}, "constraint_passed": True} for i in range(budget)]
            (out / "candidate_records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            if cfg["hpo_phase"] == "train":
                for name in ("policy.pt", "std_scaler.bin"):
                    (out / name).write_text("frozen-test-artifact")
        def wait(self, timeout=None):
            return self.returncode
    monkeypatch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", Process)
    return calls


def test_job_runner_resolves_generation_temperature_and_reuses_completed_jobs(tmp_path, monkeypatch):
    plan = small_plan(tmp_path)
    calls = fake_worker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "jobs_test")
    trained = runner({}, 16, 101, "train")
    generated = runner({"gen_temperature": .1}, 8, 401, "generation", trained)
    assert generated["eligible_for_selection"]
    cfg = calls[-1]["config"]
    assert cfg["train_budget"] == 16 and cfg["num_gen_eps"] == 8
    assert cfg["gen_temperature"] == .1
    assert cfg["gen_epsilon"] == cfg["gen_top_frac"] == cfg["exploration_gen_eps"] == 0
    assert "--only-generate" in calls[-1]["args"]
    assert calls[-1]["env"]["PYTHONPATH"].split(":")[0].endswith("/src")
    count = len(calls)
    assert runner({}, 16, 101, "train") == trained
    assert runner({"gen_temperature": .1}, 8, 401, "generation", trained) == generated
    assert len(calls) == count
    Path(trained["checkpoint_paths"]["policy.pt"]).write_text("mutated")
    with pytest.raises(JobError, match="Frozen model artifact changed"):
        runner({"gen_temperature": 1}, 8, 401, "generation", trained)


def test_zero_attempt_initialization_artifacts_are_not_silently_deleted(tmp_path, monkeypatch):
    plan = small_plan(tmp_path)
    calls = fake_worker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "runner")
    key = {"parameters": {}, "budget": 16, "seed": 101, "phase": "train", "input_models": {}}
    directory = runner.out / "jobs" / digest(key)[:24]
    atomic_json(directory / "candidate_budget.json", {"attempts_started": 0, "unique_evaluated": 0, "target_unique": 16, "status": "SYSTEM_ERROR"})
    with pytest.raises(JobError, match="never silently erased"):
        runner({}, 16, 101, "train")
    assert not calls
    assert read_json(directory / "candidate_budget.json")["attempts_started"] == 0


def test_generation_resume_requires_checkpoint_and_sets_explicit_mode(tmp_path, monkeypatch):
    plan = small_plan(tmp_path)
    calls = fake_worker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "runner")
    trained = runner({}, 16, 101, "train")
    completed = runner({"gen_temperature": 1}, 8, 401, "generation", trained)
    directory = Path(completed["directory"])
    (directory / "result.json").unlink()
    atomic_json(directory / "candidate_budget.json", {"attempts_started": 1, "unique_evaluated": 1, "target_unique": 8, "status": "INTERRUPTED"})
    with pytest.raises(JobError, match="without a resumable checkpoint"):
        runner({"gen_temperature": 1}, 8, 401, "generation", trained)
    (directory / "generation_checkpoint.pt").write_text("runtime verifies the real synchronized checkpoint")
    resumed = runner({"gen_temperature": 1}, 8, 401, "generation", trained)
    assert resumed["eligible_for_selection"]
    assert calls[-1]["config"]["hpo_resume_generation"] is True
    assert "--only-generate" in calls[-1]["args"]
    assert "--resume-training" not in calls[-1]["args"]
    assert resumed["wall_seconds"] is None
    assert resumed["current_execution_seconds"] >= 0
    assert resumed["timing_status"] == "partial_after_resume"


def test_terminal_artifacts_recover_result_without_running_or_spending_again(tmp_path, monkeypatch):
    runner = JobRunner(small_plan(tmp_path), tmp_path / "runner")
    calls = fake_worker(monkeypatch)
    original = runner({}, 16, 101, "train")
    result_path = Path(original["directory"]) / "result.json"
    result_path.unlink()  # Parent interrupted between worker completion and result write.
    count = len(calls)
    restored = runner({}, 16, 101, "train")
    assert len(calls) == count
    assert restored["eligible_for_selection"]
    assert restored["score"] == original["score"]
    assert restored["ledger"] == original["ledger"]
    assert restored["wall_seconds"] is None
    assert restored["recovered_from_terminal_artifacts"]
    assert result_path.exists()


def test_missing_scientific_evidence_remains_a_distinct_terminal_result(tmp_path, campaign_environment):
    class NoMaterialRunner(FakeRunner):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result["raw_metrics"].update(qualified_fraction=0, best_fraction_mean=None, best_value=None)
            return result
    runner = NoMaterialRunner()
    plan = small_plan(tmp_path)
    run = campaign(plan, tmp_path / "run", runner)
    assert run.run()["status"] == "NO_FEASIBLE_MATERIAL_EVIDENCE"
    assert not (run.out / "best_training.yaml").exists()
    count = len(runner.calls)
    assert campaign(plan, tmp_path / "run", runner).run()["status"] == "NO_FEASIBLE_MATERIAL_EVIDENCE"
    assert len(runner.calls) == count


def test_diagnostic_reports_are_exported_when_stage_dependency_fails(tmp_path, campaign_environment, monkeypatch):
    reported = []
    monkeypatch.setattr(Campaign, "export_reports", lambda self: reported.append(read_json(self.out / "state.json")))
    run = campaign(small_plan(tmp_path), tmp_path / "run", FakeRunner())
    with pytest.raises(CampaignError, match="requires completed"):
        run.run("generation")
    assert reported[-1]["status"] == "SYSTEM_ERROR"


def test_reporting_exception_does_not_hide_original_failure(tmp_path, campaign_environment, monkeypatch):
    def broken_report(self):
        raise ValueError("report unavailable")
    monkeypatch.setattr(Campaign, "export_reports", broken_report)
    run = campaign(small_plan(tmp_path), tmp_path / "run", FakeRunner())
    with pytest.raises(CampaignError, match="requires completed"):
        run.run("generation")
    assert read_json(run.out / "state.json")["report_error"] == "report unavailable"


@pytest.mark.parametrize("filters_location", [None, "top", "group", "env"])
def test_exact_domain_uses_default_composition_kind_and_rejects_every_filter(tmp_path, monkeypatch, filters_location):
    # Exercise only the adapter contract; do not import a materials backend.
    configuration = ModuleType("rl_matdesign.configuration")
    env = SimpleNamespace(max_steps=3, phase_filter=object() if filters_location == "env" else None)
    configuration.build_env = lambda cfg, predictor: (env, "fraction", cfg)
    counter = ModuleType("rl_matdesign.utils.design_count")
    counter.sampler_from_env = lambda _: SimpleNamespace(size=120)
    monkeypatch.setitem(sys.modules, "rl_matdesign.configuration", configuration)
    monkeypatch.setitem(sys.modules, "rl_matdesign.utils.design_count", counter)
    plan = small_plan(tmp_path)
    plan["base_config_data"]["groups"] = [{"name": "composition"}]
    if filters_location == "top":
        plan["base_config_data"]["filters"] = [{"type": "test"}]
    elif filters_location == "group":
        plan["base_config_data"]["groups"][0]["filters"] = [{"type": "test"}]
    info = design_space_info(plan)
    assert info["episode_steps"] == 3
    assert info["status"] == ("exact" if filters_location is None else "unknown")
    if filters_location is None:
        assert info["size"] == 120


def small_dqn_plan(tmp_path, method="dqn-mc"):
    plan = small_plan(tmp_path)
    plan["method"] = method
    plan["parameter_profile"] = "builtin:" + method
    plan["extension_choices"] = get_profile(method)["extension_choices"]
    plan["extension_choices"]["dqn_batch_size"] = [4, 8, 16, 32]
    plan["search_space"] = {
        "dqn_lr": {"type": "categorical", "choices": [.001]},
        "dqn_batch_size": {"type": "categorical", "choices": [4, 8]},
        "warmup_budget_fraction": {"type": "categorical", "choices": [.1]},
        "dqn_eps_min": {"type": "categorical", "choices": [.2]},
        "epsilon_anneal_budget_ratio": {"type": "categorical", "choices": [1]},
    }
    if method == "dqn-bootstrap":
        plan["search_space"]["dqn_target_update_freq"] = {"type": "categorical", "choices": [10]}
    plan["range_search"]["max_extra_values_per_direction"] = 1
    return plan


@pytest.mark.parametrize("method,parameters,required", [
    ("a2c", {"pg_batch_eps": 64}, 64),
    ("dqn-mc", {"dqn_batch_size": 32}, 33),
    ("dqn-bootstrap", {"dqn_target_update_freq": 100}, 113),
])
def test_specific_learning_budget_preflight_rejects_all_seeds_without_sampling(tmp_path, campaign_environment, method, parameters, required):
    plan = small_plan(tmp_path) if method == "a2c" else small_dqn_plan(tmp_path, method)
    runner = FakeRunner()
    run = campaign(plan, tmp_path / "run", runner)
    result = run.evaluate(parameters, 16, [11, 29])
    assert result["status"] == "INFEASIBLE_LEARNING_BUDGET"
    assert result["required_budget"] == required
    assert result["score"] == -1 and not result["eligible_for_selection"]
    assert str(required) in result["reason"]
    assert not runner.calls and run.completed_jobs == 0 and not run.known_seconds
    assert len(result["seed_results"]) == 2
    assert all(row["ledger"]["attempts_started"] == 0 and not row["executed"] for row in result["seed_results"])
    persisted = list((run.out / "measurements").glob("*.json"))
    assert len(persisted) == 1 and read_json(persisted[0])["budget"] == 16


def test_replay_capacity_preflight_rejects_even_an_arbitrarily_large_budget(tmp_path, campaign_environment):
    plan = small_dqn_plan(tmp_path)
    plan["base_config_data"]["dqn_buffer_size"] = 8
    runner = FakeRunner()
    run = campaign(plan, tmp_path / "run", runner)
    result = run.evaluate({"dqn_batch_size": 16}, 1000000, [11, 29])
    assert result["status"] == "INFEASIBLE_LEARNING_BUDGET"
    assert result["required_budget"] is None
    assert result["preflight"]["failure_kind"] == "replay_capacity"
    assert "no finite sampling budget" in result["reason"]
    assert not runner.calls


def test_range_outward_batch_probe_is_not_mistaken_for_improved_learning(tmp_path, campaign_environment):
    class BatchRunner(FakeRunner):
        def __call__(self, parameters, *args, **kwargs):
            result = super().__call__(parameters, *args, **kwargs)
            result["score"] = parameters["dqn_batch_size"] / 10
            return result
    plan = small_dqn_plan(tmp_path)
    runner = BatchRunner()
    run = campaign(plan, tmp_path / "run", runner)
    result = run.run_ranges()
    assert result["status"] == "COMPLETED"
    assert result["diagnostic_budget"] == 16
    assert result["ranges"]["dqn_batch_size"]["choices"] == [4, 8]
    rejected = [r for r in result["evaluations"] if r["parameters"]["dqn_batch_size"] == 16]
    assert rejected and all(r["status"] == "INFEASIBLE_LEARNING_BUDGET" for r in rejected)
    assert all(call["parameters"]["dqn_batch_size"] <= 8 for call in runner.calls)
    assert all(call["budget"] == 16 for call in runner.calls)
    assert not any(d["action"] == "contract" for d in result["decisions"])


def test_generation_budget_does_not_inherit_training_batch_minimum(tmp_path, campaign_environment):
    runner = FakeRunner()
    run = campaign(small_plan(tmp_path), tmp_path / "run", runner)
    result = run.evaluate({"pg_batch_eps": 64, "gen_temperature": 1}, 1, [401, 503],
                          phase="generation", models=[{"budget": 128}, {"budget": 128}])
    assert result["eligible_for_selection"]
    assert len(runner.calls) == 2


def test_preflight_does_not_mutate_plan_or_expand_the_comparison_budget(tmp_path):
    plan = small_dqn_plan(tmp_path)
    before = deepcopy(plan)
    result = training_budget_preflight(plan, {"dqn_batch_size": 1024}, 196, {"episode_steps": 5})
    assert not result["eligible"] and result["required_budget"] == 206
    assert plan == before


def test_budget_panel_does_not_revive_reference_invalidated_by_confirmation(tmp_path, campaign_environment):
    plan = small_plan(tmp_path)
    runner = FakeRunner()
    run = campaign(plan, tmp_path / "run", runner)
    reference = initial_parameters(plan)
    replacement = {**reference, "pg_lr_actor": .002}
    valid = aggregate_results([runner(replacement, 16, seed, "train") for seed in [11, 29]], replacement, 16)
    invalid = {**valid, "parameters": reference, "eligible_for_selection": False,
               "status": "INFEASIBLE_EFFICIENCY", "score": -1}
    atomic_json(run.stage_path("ranges"), {"status": "COMPLETED", "reference": reference,
                "reference_result": invalid, "evaluations": [invalid, valid]})
    assert run.panel()[0] == replacement
    assert reference not in run.panel()
    atomic_json(run.stage_path("ranges"), {"status": "COMPLETED", "reference": reference,
                "reference_result": invalid, "evaluations": [invalid]})
    count = len(runner.calls)
    assert run.run_train_budget()["status"] == "NO_FEASIBLE_CONFIGURATION"
    assert len(runner.calls) == count
