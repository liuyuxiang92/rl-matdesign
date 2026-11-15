"""A settled irrecoverable wave keeps its paid data and skips wasted learning."""
from __future__ import annotations

import csv
import json

import pytest
import torch

from rl_matdesign import training
from rl_matdesign.hpo.restart import _inspect_job
from rl_matdesign.utils.candidate_budget import CandidateBudget, EfficiencyLimitReached
from rl_matdesign.utils.candidate_records import CandidateRecorder
from rl_matdesign.utils.checkpoint_store import load_checkpoint
from rl_matdesign.utils.training_samples import TrainingSampleWriter
from tests.test_cross_formula_training import Predictor, ScriptedEnv, pg_setup, run_dqn


class Timer:
    def __init__(self):
        self.rows = []

    def record_episode(self, row):
        self.rows.append(dict(row))

    def snapshot(self):
        return {}

    def mark(self, _name):
        pass


def _rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_settled_attainability_keeps_recoverable_prefix():
    budget = CandidateBudget(1000)
    for index in range(300):
        budget.start()
        budget.settle(str(index % 150))
    assert budget.snapshot()["ratio"] == 2
    budget.check_efficiency()  # Best possible final ratio is 1.15, not 2.
    assert budget.status == "RUNNING"
    for index in range(450):
        budget.start()
        budget.settle(str(index + 150))
    for _ in range(350):
        budget.start()
        budget.settle("0")
    assert (budget.attempts_started, budget.unique_evaluated) == (1100, 600)
    with pytest.raises(EfficiencyLimitReached):
        budget.check_efficiency()  # Even 400 all-new attempts reaches 1.5.
    assert budget.status == "INFEASIBLE_EFFICIENCY"


def test_preparation_only_stops_before_scaler_fit(tmp_path, monkeypatch):
    env = ScriptedEnv(["A", "B", "C"], Predictor(1))
    recorder = CandidateRecorder(tmp_path, 4)
    env.hpo_recorder = recorder
    monkeypatch.setattr(training.StandardScaler, "fit",
                        lambda *a, **k: pytest.fail("An infeasible preparation must not fit"))
    with pytest.raises(EfficiencyLimitReached):
        training._fit_scaler_from_warmup(env, 2)
    recorder.close()
    assert env.starts == ["A", "B"]
    assert recorder.budget.preparation_only == 2
    assert recorder.budget.status == "INFEASIBLE_EFFICIENCY"


@pytest.mark.parametrize("mode", ["mc", "bootstrap"])
def test_dqn_skips_last_gradient_checkpoint_and_preserves_sample(tmp_path, monkeypatch, mode):
    env = ScriptedEnv(["A", "A", "A", "B", "C"], Predictor(1))
    recorder = CandidateRecorder(tmp_path, 4)
    env.hpo_recorder = recorder
    samples = TrainingSampleWriter(tmp_path / "training_samples.csv")
    timer = Timer()
    updates = []
    monkeypatch.setattr(training, "_dqn_gradient_step", lambda *a, **k: updates.append(1) or 0.)
    with pytest.raises(EfficiencyLimitReached):
        run_dqn(env, tmp_path, n_warmup_eps=1, dqn_num_train_eps=3,
                dqn_target_mode=mode, train_count_duplicates=False,
                timer=timer, training_samples=samples,
                checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 1})
    recorder.close()
    assert env.starts == ["A", "A", "A"]
    assert len(updates) == 1
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")
    assert checkpoint["training_episodes_completed"] == 1
    assert len(_rows(samples.path)) == recorder.budget.unique_evaluated == 1
    assert len(timer.rows) == 3
    assert timer.rows[-1]["training_attempts"] == 3
    assert timer.rows[-1]["rl_update_s"] == 0
    assert timer.rows[-1]["optimizer_steps"] == 0
    assert timer.rows[-1]["update_skipped_reason"] == "INFEASIBLE_EFFICIENCY"
    assert recorder.budget.snapshot()["attempts_started"] == 3
    assert not (tmp_path / "partial_checkpoint.pt").exists()
    # A stale training checkpoint is irrelevant once the paid ledger is terminal.
    assert _inspect_job(tmp_path, "dqn-" + mode)["classification"] == "RECOVERABLE_TERMINAL"


def test_dqn_settled_wave_keeps_successes_failures_and_no_update(tmp_path, monkeypatch):
    env = ScriptedEnv(["A", "X", "B", "A", "C"], Predictor(4))
    recorder = CandidateRecorder(tmp_path, 4)
    env.hpo_recorder = recorder
    timer = Timer()
    samples = TrainingSampleWriter(tmp_path / "training_samples.csv")
    monkeypatch.setattr(training, "_dqn_gradient_step",
                        lambda *a, **k: pytest.fail("A rejected wave must not learn"))
    with pytest.raises(EfficiencyLimitReached):
        run_dqn(env, tmp_path, n_warmup_eps=1, dqn_num_train_eps=3,
                train_count_duplicates=False, timer=timer, training_samples=samples,
                checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 100})
    recorder.close()
    assert env.starts == ["A", "X", "B", "A"]
    assert recorder.budget.pending == 0
    assert recorder.budget.failures == 1
    assert len(_rows(samples.path)) == recorder.budget.unique_evaluated == 2
    assert len(timer.rows) == 4
    assert sum(row["status"] == "failed" for row in timer.rows) == 1
    assert all(row["rl_update_s"] == 0 for row in timer.rows)
    assert not (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "partial_checkpoint.pt").exists()


@pytest.mark.parametrize("size,attempts", [(1, 3), (4, 4)])
@pytest.mark.parametrize("method", ["a2c", "reinforce"])
def test_pg_stops_partial_batch_and_keeps_every_observation(tmp_path, monkeypatch, size, attempts, method):
    _, sampled, kwargs = pg_setup(monkeypatch, [0, 0, 0, 0, 1], size=size, method=method)
    recorder = CandidateRecorder(tmp_path, 4)
    kwargs["env"].hpo_recorder = recorder
    initial = kwargs["policy"].logits.detach().clone()
    samples = TrainingSampleWriter(tmp_path / "training_samples.csv")
    timer = Timer()
    monkeypatch.setattr(training, "_episode_pg_terms",
                        lambda *a, **k: pytest.fail("A rejected PG batch must not learn"))
    with pytest.raises(EfficiencyLimitReached):
        training.train_pg(**kwargs, train_count_duplicates=False,
                          timer=timer, training_samples=samples,
                          checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 1})
    recorder.close()
    assert len(sampled) == attempts
    assert len(_rows(samples.path)) == recorder.budget.unique_evaluated == 1
    assert len(timer.rows) == attempts
    assert all(row["rl_update_s"] == 0 for row in timer.rows)
    assert timer.rows[-1]["training_attempts"] == attempts
    assert torch.equal(initial, kwargs["policy"].logits)
    assert recorder.budget.status == "INFEASIBLE_EFFICIENCY"
    assert recorder.budget.pending == 0
    assert not (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "partial_checkpoint.pt").exists()
    assert _inspect_job(tmp_path, method)["classification"] == "RECOVERABLE_TERMINAL"


@pytest.mark.parametrize("method", ["a2c", "dqn"])
def test_native_cli_efficiency_rejection_is_terminal_with_paid_logs(tmp_path, method):
    from tests.test_duplicate_budget_cli import _config, _run
    config = _config(method=method)
    config.pop("dqn_num_train_eps")
    config.pop("pg_num_iters")
    config.update(hpo_recording=True, hpo_phase="train", hpo_target_unique=8,
                  hpo_ratio_limit=1.5, train_budget=8)
    result, out = _run(tmp_path, config)
    # The public CLI handles scientific rejection, rather than reporting a
    # system/worker crash. The coordinator reads eligibility from the ledger.
    assert result.returncode == 0, result.stderr
    assert "Strict attempts/unique limit is no longer attainable" in result.stdout
    ledger = json.loads((out / "candidate_budget.json").read_text())
    assert ledger["status"] == "INFEASIBLE_EFFICIENCY"
    assert ledger["pending"] == 0
    assert ledger["attempts_started"] < ledger["max_attempts"]
    from rl_matdesign.hpo.scoring import score_records
    from tests.test_hpo_v1_scoring import plan
    scoring_plan = plan()
    scoring_plan["objective"].update(field="dp_mean", unit=None)
    records = [json.loads(line) for line in (out / "candidate_records.jsonl").read_text().splitlines()]
    result = score_records(records, ledger, scoring_plan, "train")
    assert result["status"] == "INFEASIBLE_EFFICIENCY"
    assert result["eligible_for_selection"] is False
    assert result["score"] == -1
    samples = _rows(out / "training_samples.csv")
    assert len(samples) == ledger["unique_evaluated"]
    timings = _rows(out / "episode_timing.csv")
    assert len(timings) == ledger["attempts_started"] - ledger["preparation_only_attempts"]
    assert not (out / "partial_checkpoint.pt").exists()
    assert _inspect_job(out, method)["classification"] == "RECOVERABLE_TERMINAL"


@pytest.mark.parametrize("size,attempts", [(1, 3), (4, 4)])
def test_generation_rejected_wave_keeps_timing_without_final_checkpoint(tmp_path, monkeypatch, size, attempts):
    import numpy as np
    env = ScriptedEnv(["A", "A", "A", "A", "B"], Predictor(size))
    recorder = CandidateRecorder(tmp_path, 4, phase="generation")
    env.hpo_recorder = recorder
    timer = Timer()
    monkeypatch.setattr(training, "_rollout_policy_episode",
                        lambda **kwargs: training._rollout_random_episode(kwargs["env"]))
    with pytest.raises(EfficiencyLimitReached):
        training.generate_candidates(
            env=env, predictor=timer, scaler=None, device=torch.device("cpu"),
            elem_feats_scaled=np.zeros((1, 1)), fraction_set=["1"],
            qnet=torch.nn.Linear(1, 1), n_exploit=4,
        )
    recorder.close()
    assert len(env.starts) == attempts
    assert len(timer.rows) == attempts
    assert recorder.budget.attempts_started == attempts
    assert recorder.budget.status == "INFEASIBLE_EFFICIENCY"
    assert recorder.budget.pending == 0
    assert not (tmp_path / "generation_checkpoint.pt").exists()
    assert _inspect_job(tmp_path, "dqn-bootstrap")["classification"] == "RECOVERABLE_TERMINAL"
