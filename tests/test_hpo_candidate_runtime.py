"""Strict HPO caps count the work actually done without changing RL clocks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_matdesign import training
from rl_matdesign.utils.candidate_budget import CandidateBudget, EfficiencyLimitReached
from rl_matdesign.utils.candidate_records import CandidateRecorder, candidate_identity
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator
from tests.test_cross_formula_training import Predictor, ScriptedEnv, run_dqn


def test_strict_decimal_boundary_and_transient_prefix():
    b = CandidateBudget(1000, 1.5)
    assert b.max_attempts == 1499
    for i in range(20):
        b.start(); b.settle(str(i % 10))
    assert b.snapshot()["ratio"] == 2
    b.check_can_start()
    for i in range(10, 1000):
        b.start(); b.settle(str(i))
    assert b.finish()["status"] == "COMPLETED"
    assert b.attempts_started == 1010
    assert CandidateBudget(100, "1.01").max_attempts == 100


def test_irreversible_limit_and_pending_wave():
    b = CandidateBudget(4, 1.5)
    assert b.max_attempts == 5
    # Two starts before settling; do not reject on a pessimistic interim prefix.
    b.start(); b.start(); b.settle("A"); b.settle("A")
    assert b.wave_limit(10) == 3
    b.start(); b.settle("A")
    with pytest.raises(EfficiencyLimitReached):
        b.start()
    assert b.attempts_started == 3
    assert b.status == "INFEASIBLE_EFFICIENCY"


def test_identity_keeps_types_and_float_precision():
    assert candidate_identity((1,)) != candidate_identity((True,))
    assert candidate_identity((1,)) != candidate_identity([1])
    assert candidate_identity((.1,)) != candidate_identity((.100000000001,))


class SnapshotPredictor(Predictor):
    def prediction_record(self, candidate, prediction, *, k=1):
        value = prediction[0]
        return {"obj_T3_mean": 1200 + value, "obj_Tg_mean": 700 + value,
                "gate_passed": next(iter(candidate)) != "B", "dp_mean": value}


@pytest.mark.parametrize("size", [1, 20])
def test_actual_retries_duplicates_raw_gates_and_no_reprediction(tmp_path, size):
    pred = SnapshotPredictor(size)
    env = ScriptedEnv(["A", "X", "A", "B", "C"], pred)
    env.hpo_recorder = CandidateRecorder(tmp_path / "records", 3, 2.0)
    _, _, rows = run_dqn(env, tmp_path, n_warmup_eps=1, dqn_num_train_eps=2,
                         train_count_duplicates=False)
    env.hpo_recorder.close()
    budget = json.loads((tmp_path / "records/candidate_budget.json").read_text())
    records = [json.loads(line) for line in (tmp_path / "records/candidate_records.jsonl").read_text().splitlines()]
    assert budget["status"] == "COMPLETED"
    assert (budget["attempts_started"], budget["unique_evaluated"], budget["failed_attempts"]) == (5, 3, 1)
    assert [r["formula"] for r in records] == ["A1", "B1", "C1"]
    assert [r["constraint_passed"] for r in records] == [True, False, True]
    assert records[1]["properties"] == {"T3": 1202., "Tg": 702., "dp_mean": 2.}
    assert not pred.raw
    # Failed evaluation never advances epsilon's successful-episode clock.
    assert [row["episode"] for row in rows if row["phase"] == "dqn_train"] == [1, 2, 3]


def test_batch_does_not_launch_or_discard_extra_candidates(tmp_path):
    pred = SnapshotPredictor(50)
    env = ScriptedEnv(["A", "A", "A", "B", "C"], pred)
    env.hpo_recorder = CandidateRecorder(tmp_path / "records", 3, 1.5)
    with pytest.raises(EfficiencyLimitReached):
        run_dqn(env, tmp_path, n_warmup_eps=1, dqn_num_train_eps=2,
                train_count_duplicates=False)
    env.hpo_recorder.close()
    assert env.starts == ["A", "A", "A"]
    assert env.hpo_recorder.budget.status == "INFEASIBLE_EFFICIENCY"
    assert env.hpo_recorder.budget.pending == 0
    partial = torch.load(tmp_path / "records/partial_checkpoint.pt", weights_only=False)
    assert partial["resume_safe"] is False
    assert "qnet_state" in partial


def test_preparation_sampling_counts_without_unique_credit(tmp_path, monkeypatch):
    pred = SnapshotPredictor(1)
    env = ScriptedEnv(["A", "B", "A"], pred)
    env.hpo_recorder = CandidateRecorder(tmp_path / "records", 10, 1.5)
    monkeypatch.setattr(training, "bind_scaler", lambda *_: None)
    training._fit_scaler_from_warmup(env, 2)
    assert env.hpo_recorder.budget.attempts_started == 2
    assert env.hpo_recorder.budget.unique_evaluated == 0
    assert env.hpo_recorder.budget.preparation_only == 2
    assert not pred.scalar
    evaluator = EpisodeBatchEvaluator(env)
    evaluator.next(lambda: training._rollout_random_episode(env), limit=1)
    assert env.hpo_recorder.budget.unique_evaluated == 1


def test_generation_starts_fresh_and_keeps_failed_scientific_gates(tmp_path, monkeypatch):
    pred = SnapshotPredictor(20)
    env = ScriptedEnv(["A", "B", "C"], pred)
    env.hpo_recorder = CandidateRecorder(tmp_path, 3, 1.5, "generation")
    monkeypatch.setattr(training, "_rollout_policy_episode", lambda **kw: training._rollout_random_episode(kw["env"]))
    qnet = torch.nn.Linear(1, 1)
    rows = training.generate_candidates(env=env, predictor=pred, scaler=None,
        device=torch.device("cpu"), elem_feats_scaled=np.zeros((1, 1)), fraction_set=["1"],
        qnet=qnet, n_exploit=3, gen_temperature=.1)
    env.hpo_recorder.close()
    assert len(rows) == 3
    assert rows[1]["gate_passed"] is False
    assert env.hpo_recorder.budget.attempts_started == 3
    assert not pred.raw
    with pytest.raises(ValueError, match="already exist"):
        CandidateRecorder(tmp_path, 3)


def test_exact_resume_keeps_spent_attempts_and_refuses_rollback(tmp_path):
    from rl_matdesign.utils.episode_batch import EpisodeSnapshot
    rec = CandidateRecorder(tmp_path, 10)
    rec.start(); rec.failure(preparation=True)
    rec.start(); rec.success(EpisodeSnapshot([], (("A", 1),), {}, "A1", prediction_record={"dp_mean": 2.}))
    state = rec.checkpoint_state()
    rec.close()
    resumed = CandidateRecorder(tmp_path, 10, resume_state=state)
    assert resumed.budget.attempts_started == 2
    assert resumed.budget.unique_evaluated == 1
    resumed.start(); resumed.failure()
    resumed.close()
    before = (tmp_path / "candidate_budget.json").read_bytes()
    with pytest.raises(ValueError, match="advanced beyond checkpoint"):
        CandidateRecorder(tmp_path, 10, resume_state=state)
    assert (tmp_path / "candidate_budget.json").read_bytes() == before


def test_changed_candidate_archive_refuses_resume(tmp_path):
    rec = CandidateRecorder(tmp_path, 10)
    state = rec.checkpoint_state()
    rec.close()
    (tmp_path / "candidate_records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="archive differs"):
        CandidateRecorder(tmp_path, 10, resume_state=state)


@pytest.mark.parametrize("method", ["dqn", "a2c"])
def test_real_cli_writes_opt_in_hpo_artifacts(tmp_path, method):
    from tests.test_duplicate_budget_cli import _config, _run
    config = _config(method=method)
    config.update(hpo_recording=True, hpo_phase="train", hpo_target_unique=3,
                  hpo_ratio_limit=20., train_budget=3)
    result, out = _run(tmp_path, config)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-4000:]
    budget = json.loads((out / "candidate_budget.json").read_text())
    assert budget["status"] == "COMPLETED"
    assert budget["unique_evaluated"] == 3
    assert budget["preparation_only_attempts"] == (2 if method == "a2c" else 0)
    records = [json.loads(line) for line in (out / "candidate_records.jsonl").read_text().splitlines()]
    assert len(records) == 3 and all("dp_mean" in row["properties"] for row in records)
    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["hpo_state"]["budget"]["attempts_started"] == budget["attempts_started"]


def test_numerical_failure_has_distinct_status(tmp_path):
    from contextlib import ExitStack
    from rl_matdesign.utils.candidate_budget import NumericalTrainingError
    with pytest.raises(NumericalTrainingError):
        with ExitStack() as stack:
            recorder = CandidateRecorder(tmp_path, 1)
            stack.callback(recorder.close)
            from rl_matdesign.utils.episode_batch import EpisodeSnapshot
            recorder.start()
            recorder.success(EpisodeSnapshot([], (("A", 1),), {}, "A1", prediction_record={"dp_mean": 2.}))
            assert recorder.budget.unique_evaluated == recorder.budget.target_unique
            assert recorder.budget.status == "RUNNING"
            env = SimpleNamespace(hpo_recorder=recorder)
            training._check_hpo_numerical(env, float("nan"))
    assert json.loads((tmp_path / "candidate_budget.json").read_text())["status"] == "INFEASIBLE_NUMERICAL"


@pytest.mark.parametrize("size", [1, 8])
def test_all_failed_candidates_exhaust_strict_cap_not_legacy_retry_count(tmp_path, size):
    pred = SnapshotPredictor(size)
    env = ScriptedEnv(["X"] * 100, pred)
    env.hpo_recorder = CandidateRecorder(tmp_path, 30, 1.5)
    evaluator = EpisodeBatchEvaluator(env, max_retries=2)
    with pytest.raises(EfficiencyLimitReached):
        evaluator.next(lambda: training._rollout_random_episode(env), limit=30,
                       retry_scalar=training._rollout_with_retry)
    # B=30 permits fourteen nonproductive attempts; a batched wave can
    # discover infeasibility when its in-flight outcomes all settle.
    assert 15 <= env.hpo_recorder.budget.attempts_started <= 22
    assert env.hpo_recorder.budget.pending == 0
    assert env.hpo_recorder.budget.status == "INFEASIBLE_EFFICIENCY"


def test_scenario_post_constraint_keeps_paid_candidate_in_archive(tmp_path):
    pred = SnapshotPredictor(1)
    env = ScriptedEnv(["A"], pred)
    recorder = CandidateRecorder(tmp_path, 1, 1.5)
    recorder.constraint_check = lambda episode: False
    env.hpo_recorder = recorder
    EpisodeBatchEvaluator(env).next(lambda: training._rollout_random_episode(env), limit=1)
    recorder.close()
    row = json.loads((tmp_path / "candidate_records.jsonl").read_text())
    assert row["constraint_passed"] is False
    assert row["post_constraint_passed"] is False
    assert recorder.budget.unique_evaluated == 1
    assert recorder.budget.status == "COMPLETED"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generation_resume_preserves_candidates_cost_rng_and_predictor(tmp_path, monkeypatch, batch_size):
    from contextlib import ExitStack
    from rl_matdesign.utils.timing import PredictorTimer
    from rl_matdesign.utils.seeding import capture_rng_state, restore_rng_state
    contract = {"weights": "immutable-hash", "temperature": .1}
    inner = SnapshotPredictor(batch_size)
    predictor = PredictorTimer(inner)
    predictor.set_phase("generate")
    env = ScriptedEnv(["A", "B", "C"], predictor)
    env.hpo_generation_contract = contract
    original_save = training._save_checkpoint
    draws = []
    def rollout(**kw):
        draws.append(float(np.random.random()))
        training._rollout_random_episode(kw["env"])
    monkeypatch.setattr(training, "_rollout_policy_episode", rollout)
    def interrupted_save(path, data, **kwargs):
        original_save(path, data, **kwargs)
        if data.get("type") == "hpo_generation" and data["hpo_state"]["budget"]["unique_evaluated"] == batch_size:
            raise KeyboardInterrupt
    monkeypatch.setattr(training, "_save_checkpoint", interrupted_save)
    qnet = torch.nn.Linear(1, 1)
    with pytest.raises(KeyboardInterrupt):
        with ExitStack() as stack:
            env.hpo_recorder = CandidateRecorder(tmp_path, 3, 1.5, "generation")
            stack.callback(env.hpo_recorder.close)
            training.generate_candidates(env=env, predictor=predictor, scaler=None,
                device=torch.device("cpu"), elem_feats_scaled=np.zeros((1, 1)),
                fraction_set=["1"], qnet=qnet, n_exploit=3, gen_temperature=.1)
    state = torch.load(tmp_path / "generation_checkpoint.pt", weights_only=False)
    assert state["hpo_state"]["budget"]["attempts_started"] == batch_size
    restore_rng_state(state["rng_state"])
    expected_next_draw = float(np.random.random())
    np.random.seed(9921)  # Simulate unrelated setup draws before restoration.
    resumed_inner = SnapshotPredictor(batch_size)
    resumed_predictor = PredictorTimer(resumed_inner)
    resumed_env = ScriptedEnv(["A", "B", "C"][batch_size:], resumed_predictor)
    resumed_env.hpo_generation_contract = contract
    resumed_env.hpo_generation_resume = state
    resumed_env.hpo_recorder = CandidateRecorder(tmp_path, 3, 1.5, "generation", resume_state=state["hpo_state"])
    monkeypatch.setattr(training, "_save_checkpoint", original_save)
    rows = training.generate_candidates(env=resumed_env, predictor=resumed_predictor, scaler=None,
        device=torch.device("cpu"), elem_feats_scaled=np.zeros((1, 1)),
        fraction_set=["1"], qnet=qnet, n_exploit=3, gen_temperature=.1)
    resumed_env.hpo_recorder.close()
    assert draws[batch_size] == expected_next_draw
    assert [row["formula"] for row in rows] == ["A1", "B1", "C1"]
    assert resumed_env.hpo_recorder.budget.attempts_started == 3
    assert resumed_predictor.n_calls == 3
    assert resumed_env.hpo_recorder.budget.status == "COMPLETED"
    assert "A" not in resumed_inner.scalar


def test_generation_resume_rejects_changed_frozen_contract():
    from rl_matdesign.utils.generation_resume import restore_generation
    with pytest.raises(ValueError, match="changed its frozen weights"):
        restore_generation(None, {"generation_contract": {"weights": "old"}}, {"weights": "new"})


def test_generation_cli_checkpoint_can_finish_reporting_after_interruption(tmp_path):
    import subprocess
    import sys
    import yaml
    from tests.test_duplicate_budget_cli import _config, _run, _REPO, _RUNNER
    config = _config(method="dqn")
    trained, training_out = _run(tmp_path, config)
    assert trained.returncode == 0, trained.stderr
    weight_before = (training_out / "qnet.pt").read_bytes()
    config.update(hpo_recording=True, hpo_phase="generation", hpo_target_unique=2,
                  hpo_ratio_limit=20., num_gen_eps=2, gen_temperature=1.)
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(yaml.safe_dump(config))
    generated_out = tmp_path / "generation"
    command = [sys.executable, str(_RUNNER), "--config", str(config_path),
               "--out", str(generated_out), "--only-generate", "--device", "cpu",
               "--load-qnet", str(training_out / "qnet.pt"),
               "--load-scaler", str(training_out / "std_scaler.bin")]
    fresh = subprocess.run(command, cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert fresh.returncode == 0, fresh.stdout[-3000:] + fresh.stderr[-4000:]
    budget_path = generated_out / "candidate_budget.json"
    previous = json.loads(budget_path.read_text())
    assert previous["status"] == "COMPLETED"
    before = (generated_out / "candidate_records.jsonl").read_bytes()
    # Models a process interruption after its final settled wave, before report
    # completion; the fully synchronized checkpoint is already durable.
    budget_path.write_text(json.dumps({**previous, "status": "INTERRUPTED"}))
    config["hpo_resume_generation"] = True
    config_path.write_text(yaml.safe_dump(config))
    resumed = subprocess.run(command, cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout[-3000:] + resumed.stderr[-4000:]
    after = json.loads(budget_path.read_text())
    assert after["status"] == "COMPLETED"
    assert after["attempts_started"] == previous["attempts_started"]
    assert (generated_out / "candidate_records.jsonl").read_bytes() == before
    assert (training_out / "qnet.pt").read_bytes() == weight_before
