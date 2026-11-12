"""Parallel scheduling preserves HPO evidence and experimental pairing."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import threading

import pytest

from rl_matdesign.hpo.artifacts import digest, read_json
from rl_matdesign.hpo.controller import Campaign
from tests.test_hpo_v2_controller import ReusableRunner, campaign, internal_plan


@pytest.fixture(autouse=True)
def stable_campaign_environment(monkeypatch):
    # Concurrent implementation edits are irrelevant to these isolated fixtures.
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "test-software")
    monkeypatch.setattr(Campaign, "export_reports", lambda self: None)


def plan_for_parallel(tmp_path, parallel_jobs=4):
    plan = internal_plan(tmp_path)
    plan["parallel_jobs"] = parallel_jobs
    plan["search_space"] = {
        "pg_batch_eps": {"type": "categorical", "choices": [2, 4]},
        "pg_lr_actor": {"type": "categorical", "choices": [.0001, .001]},
    }
    return plan


class ObservedRunner(ReusableRunner):
    """Hold the first worker wave until its declared peers really execute."""

    def __init__(self, first_wave=0, fail_seed=None):
        super().__init__()
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(first_wave, timeout=5) if first_wave else None
        self.first_wave = first_wave
        self.fail_seed = fail_seed
        self.active = self.maximum_active = 0
        self.active_keys = set()
        self.duplicate_inflight = []
        self.invocations = []

    def __call__(self, parameters, budget, seed, phase, model=None):
        job = dict(parameters=parameters, budget=budget, seed=seed, phase=phase, model=model)
        key = digest(job)
        with self.lock:
            index = len(self.invocations)
            self.invocations.append(deepcopy(job))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if key in self.active_keys:
                self.duplicate_inflight.append(key)
            self.active_keys.add(key)
        try:
            if self.barrier is not None and index < self.first_wave:
                self.barrier.wait()
            # Only the tiny in-memory cache/fixture update is protected; the
            # barrier above proves workers overlap without timing assumptions.
            with self.lock:
                row = super().__call__(parameters, budget, seed, phase, model)
            if seed == self.fail_seed:
                row.update(status="INFEASIBLE_EFFICIENCY", eligible_for_selection=False, score=-1.)
            return row
        finally:
            with self.lock:
                self.active -= 1
                self.active_keys.discard(key)


def requests():
    return [dict(parameters={"pg_batch_eps": batch}, budget=16, seeds=[11, 29, 47])
            for batch in (2, 4)]


@pytest.mark.parametrize("entry", ["evaluate_many", "ranges", "scan"])
def test_independent_configurations_and_seeds_share_one_global_limit(tmp_path, entry):
    runner = ObservedRunner(first_wave=4)
    run = campaign(plan_for_parallel(tmp_path), tmp_path / "run", runner)
    if entry == "evaluate_many":
        results = run.evaluate_many(requests())
    elif entry == "ranges":
        result = run.run_ranges()
        assert result["status"] == "COMPLETED"
        results = result["evaluations"]
    else:
        results = run.scan([request["parameters"] for request in requests()], "train", [11, 29, 47])
    assert runner.maximum_active == 4
    assert runner.active == 0
    assert not runner.duplicate_inflight
    first_wave = runner.invocations[:4]
    assert len({digest(job["parameters"]) for job in first_wave}) == 2
    assert {job["seed"] for job in first_wave} == {11, 29, 47}
    assert all(row["seeds"] == [11, 29, 47] for row in results)
    assert all(row["eligible_for_selection"] for row in results)


def test_duplicate_requests_share_workers_without_double_execution(tmp_path):
    runner = ObservedRunner(first_wave=4)
    run = campaign(plan_for_parallel(tmp_path), tmp_path / "run", runner)
    first, second = requests()
    results = run.evaluate_many([
        first, deepcopy(first), {**first, "seeds": [29, 47]}, second,
    ])
    counts = Counter(digest(job) for job in runner.invocations)
    assert len(counts) == 6
    assert set(counts.values()) == {1}
    assert not runner.duplicate_inflight
    assert results[0] == results[1]
    assert results[2]["seeds"] == [29, 47]
    assert results[2]["seed_results"] == results[0]["seed_results"][1:]
    assert run.completed_jobs == 6
    assert len(list((run.out / "measurements").glob("*.json"))) == 3


def test_failed_parallel_training_seed_disqualifies_whole_group(tmp_path):
    runner = ObservedRunner(first_wave=4, fail_seed=29)
    run = campaign(plan_for_parallel(tmp_path), tmp_path / "run", runner)
    results = run.evaluate_many(requests())
    assert runner.maximum_active == 4
    assert len(runner.invocations) == 6
    for result in results:
        assert result["seeds"] == [11, 29, 47]
        assert not result["eligible_for_selection"]
        assert result["status"] == "INFEASIBLE_EFFICIENCY"
        assert result["score"] == -1
        assert [row["eligible_for_selection"] for row in result["seed_results"]] == [True, False, True]


def test_serial_and_parallel_full_flow_preserve_all_scientific_outputs(tmp_path):
    plan = plan_for_parallel(tmp_path)
    plan["training_search"].update(trials=5, startup_random_trials=3,
                                   additional_trials_per_revision=5, additional_random_trials=3)
    plan["generation_search"]["repeats_per_model"] = 3
    runs, runners = [], []
    for parallel_jobs in (1, 4):
        runner = ObservedRunner()
        run = campaign({**plan, "parallel_jobs": parallel_jobs}, tmp_path / f"jobs-{parallel_jobs}", runner)
        assert run.run()["status"] == "COMPLETED"
        runs.append(run)
        runners.append(runner)
    for stage in ("ranges", "train-budget", "train-search", "generation"):
        assert read_json(runs[0].stage_path(stage)) == read_json(runs[1].stage_path(stage))
    for name in ("best_training.yaml", "best_generation.yaml", "generation_pairings.json", "frozen_models.json"):
        assert (runs[0].out / name).read_text() == (runs[1].out / name).read_text()
    for phase in ("train", "generation"):
        assert Counter(digest(call) for call in runners[0].calls if call["phase"] == phase) == Counter(
            digest(call) for call in runners[1].calls if call["phase"] == phase)
    assert runners[0].maximum_active == 1
    assert runners[1].maximum_active <= 4
    before = len(runners[1].invocations)
    assert campaign(plan, runs[1].out, runners[1]).run()["status"] == "COMPLETED"
    assert len(runners[1].invocations) == before


@pytest.mark.parametrize("failed_repeat", [None, 1])
def test_generation_keeps_nine_model_repeat_pairs_and_frozen_artifacts(tmp_path, failed_repeat):
    plan = plan_for_parallel(tmp_path)
    plan["generation_search"]["repeats_per_model"] = 3
    sources = ReusableRunner()
    models = [sources({"pg_batch_eps": 2}, 16, seed, "train") for seed in (101, 211, 307)]
    original_files = {}
    for index, model in enumerate(models):
        for name in model["checkpoint_paths"]:
            path = tmp_path / f"model-{index}-{name}"
            path.write_bytes(f"frozen model {index}: {name}".encode())
            path.chmod(0o444)
            model["checkpoint_paths"][name] = str(path)
            original_files[path] = path.read_bytes()
    original_models = deepcopy(models)
    runner = ObservedRunner(first_wave=4)
    run = campaign(plan, tmp_path / "run", runner)
    seeds, paired_models, pairings = run.generation_pairs(models)
    if failed_repeat is not None:
        runner.fail_seed = seeds[failed_repeat]
    results = run.evaluate_many([
        dict(parameters={"pg_batch_eps": 2, "gen_temperature": temperature}, budget=8,
             seeds=seeds, phase="generation", models=paired_models, pairings=pairings)
        for temperature in (.1, 1)
    ])
    assert len(seeds) == len(set(seeds)) == len(pairings) == 9
    assert Counter(pair["training_seed"] for pair in pairings) == {101: 3, 211: 3, 307: 3}
    assert {pair["generation_repeat_index"] for pair in pairings} == {0, 1, 2}
    assert runner.maximum_active == 4
    assert len(runner.invocations) == 18
    assert all(call["phase"] == "generation" for call in runner.invocations)
    for result in results:
        assert result["generation_pairings"] == pairings
        assert result["seeds"] == seeds
        assert len(result["seed_results"]) == 9
        assert [len(model["seed_results"]) for model in result["model_results"]] == [3, 3, 3]
        assert result["eligible_for_selection"] is (failed_repeat is None)
        if failed_repeat is not None:
            assert result["score"] == -1
            assert [model["eligible_for_selection"] for model in result["model_results"]] == [False, True, True]
    assert models == original_models
    assert all(path.read_bytes() == contents for path, contents in original_files.items())
    assert all(Path(call["model"]["checkpoint_paths"]["policy.pt"]).read_bytes()
               == original_files[Path(call["model"]["checkpoint_paths"]["policy.pt"])]
               for call in runner.invocations)


def test_random_proposals_batch_but_adaptive_proposals_wait_for_all_prior_scores(tmp_path, monkeypatch):
    import optuna
    from optuna.trial import TrialState

    plan = plan_for_parallel(tmp_path)
    # A continuous parameter gives the random startup distinct configurations,
    # so the first worker wave crosses proposals as well as their seed repeats.
    plan["search_space"] = {
        "pg_batch_eps": {"type": "categorical", "choices": [2]},
        "pg_lr_actor": {"type": "float", "low": .0001, "high": .001, "log": True},
    }
    events = []
    original_ask, original_tell = optuna.study.Study.ask, optuna.study.Study.tell

    def checked_ask(study, *args, **kwargs):
        frozen = study.get_trials(deepcopy=False)
        number = len(frozen)
        if number >= 3:
            assert all(trial.state == TrialState.COMPLETE for trial in frozen)
        events.append(("ask", number))
        return original_ask(study, *args, **kwargs)

    def checked_tell(study, trial, *args, **kwargs):
        result = original_tell(study, trial, *args, **kwargs)
        events.append(("tell", result.number))
        return result

    monkeypatch.setattr(optuna.study.Study, "ask", checked_ask)
    monkeypatch.setattr(optuna.study.Study, "tell", checked_tell)
    runner = ObservedRunner(first_wave=4)
    run = campaign(plan, tmp_path / "run", runner)
    results = run.train_study(16, "ordering", trials=5, startup=3)
    assert events == [
        ("ask", 0), ("ask", 1), ("ask", 2),
        ("tell", 0), ("tell", 1), ("tell", 2),
        ("ask", 3), ("tell", 3), ("ask", 4), ("tell", 4),
    ]
    assert runner.maximum_active == 4
    assert len({digest(job["parameters"]) for job in runner.invocations[:4]}) == 2
    assert len(results) == 5
    assert all(result["seeds"] == [11, 29, 47] for result in results)
    assert all(len(result["seed_results"]) == 3 for result in results)
