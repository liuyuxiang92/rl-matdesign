"""Restart audit never repairs archives, spends attempts, or masks worker errors."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_matdesign.hpo.artifacts import file_digest
from rl_matdesign.hpo.restart import inspect_restart
from rl_matdesign.utils.candidate_records import CandidateRecorder
from rl_matdesign.utils.seeding import capture_rng_state
from rl_matdesign.utils.training_budget import TrainingBudget


def _json(path, value):
    path.write_text(json.dumps(value))


def _job(tmp_path, kind="a2c", phase="train", *, terminal=False):
    tmp_path.mkdir(exist_ok=True)
    _json(tmp_path / "manifest.json", {"plan": {"method": kind}})
    directory = tmp_path / "jobs" / "one"
    directory.mkdir(parents=True)
    recorder = CandidateRecorder(directory, 1 if terminal else 4, phase=phase)
    recorder.start()
    recorder.success(SimpleNamespace(terminal_key="A", formula="A", prediction_record={"dp_mean": 1.0}))
    state = recorder.checkpoint_state()
    recorder.close()
    training_budget = TrainingBudget(count_duplicates=False)
    training_budget.record("A")
    optimizer = {"state": {}, "param_groups": []}
    checkpoint = {
        "hpo_state": state, "rng_state": capture_rng_state(),
        "predictor_ledger": {"schema_version": 1, "n_calls": 1, "seen": ["A"],
                             "t_predict_s": 0.1, "phase": phase, "phase_costs": {}},
        "execution_contract": {"test": True}, "environment_contract": {"test": True},
        "scaler_fingerprint": "test", "episodes_completed": 1,
        "training_budget_state": training_budget.state_dict(),
    }
    if phase == "generation":
        checkpoint.update(type="hpo_generation", generation_contract={"execution": {}, "sampling": {}, "frozen_artifacts": {}})
        name = "generation_checkpoint.pt"
    elif kind == "a2c":
        checkpoint.update(type="pg", policy_state={"w": torch.tensor([1.])},
                          value_net_state={"w": torch.tensor([2.])}, opt_actor_state=optimizer,
                          opt_critic_state=optimizer, iteration=1, visit_counts={"A": 1},
                          entropy_coef_eff=0.01, rl_method="a2c")
        name = "checkpoint.pt"
    else:
        checkpoint.update(type="dqn", qnet_state={"w": torch.tensor([1.])},
                          target_net_state={"w": torch.tensor([2.])}, opt_state=optimizer,
                          buffer=[], training_episodes_completed=1, eps=0.9)
        name = "checkpoint.pt"
    torch.save(checkpoint, directory / name)
    (directory / "std_scaler.bin").write_bytes(b"original scaler")
    if terminal and phase == "train":
        (directory / ("policy.pt" if kind == "a2c" else "qnet.pt")).write_bytes(b"final model")
    return directory, checkpoint


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind,phase", [("a2c", "train"), ("dqn_mc", "train"),
                                          ("dqn_bootstrap", "train"), ("a2c", "generation")])
def test_synchronized_checkpoint_audit_is_read_only_and_cpu_only(tmp_path, monkeypatch, kind, phase):
    _job(tmp_path, kind, phase)
    before = _snapshot(tmp_path)
    actual_load, calls = torch.load, []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    report = inspect_restart(tmp_path)
    assert report["ready"]
    assert report["job_counts"]["SYNCHRONIZED"] == 1
    assert calls == [{"map_location": "cpu", "weights_only": False}]
    assert _snapshot(tmp_path) == before
    json.dumps(report)


def test_ledger_ahead_preserves_paid_attempts_and_refuses_restart(tmp_path):
    directory, _ = _job(tmp_path)
    ledger = json.loads((directory / "candidate_budget.json").read_text())
    ledger.update(attempts_started=2, failed_attempts=1)
    _json(directory / "candidate_budget.json", ledger)
    before = _snapshot(tmp_path)
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "advanced beyond checkpoint" in report["jobs"][0]["reason"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("change,expected", [
    ("checkpoint.pt", "UnpicklingError"),
    ("candidate_budget.json", "JSONDecodeError"),
    ("candidate_records.jsonl", "JSONDecodeError"),
])
def test_corrupt_job_does_not_stop_audit_of_other_jobs(tmp_path, change, expected):
    directory, _ = _job(tmp_path)
    (directory / change).write_bytes(b"broken")
    empty = tmp_path / "jobs" / "two"
    empty.mkdir()
    before = _snapshot(tmp_path)
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert report["job_counts"]["BLOCKED"] == 1
    assert report["job_counts"]["SYNCHRONIZED"] == 1
    assert expected in report["jobs"][0]["reason"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("missing", ["std_scaler.bin", "checkpoint.pt"])
def test_missing_restart_artifact_is_blocked(tmp_path, missing):
    directory, _ = _job(tmp_path)
    (directory / missing).unlink()
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert missing in report["jobs"][0]["reason"]


@pytest.mark.parametrize("kind,missing", [("a2c", "opt_critic_state"), ("a2c", "value_net_state"),
                                         ("dqn_mc", "buffer"), ("dqn_bootstrap", "opt_state"),
                                         ("a2c", "rng_state"), ("a2c", "predictor_ledger")])
def test_incomplete_model_or_optimizer_state_is_not_restartable(tmp_path, kind, missing):
    directory, checkpoint = _job(tmp_path, kind)
    checkpoint.pop(missing)
    torch.save(checkpoint, directory / "checkpoint.pt")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert missing in report["jobs"][0]["reason"]


def test_changed_archive_digest_is_rejected(tmp_path):
    directory, _ = _job(tmp_path)
    path = directory / "candidate_records.jsonl"
    row = json.loads(path.read_text())
    row["properties"]["dp_mean"] = 999.
    path.write_text(json.dumps(row) + "\n")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "archive differs" in report["jobs"][0]["reason"]


@pytest.mark.parametrize("phase", ["train", "generation"])
def test_terminal_job_without_result_needs_no_checkpoint_load(tmp_path, monkeypatch, phase):
    _job(tmp_path, phase=phase, terminal=True)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("terminal audit must not load a checkpoint"))
    report = inspect_restart(tmp_path)
    assert report["ready"]
    assert report["job_counts"]["RECOVERABLE_TERMINAL"] == 1


def test_terminal_train_missing_final_export_is_blocked(tmp_path):
    directory, _ = _job(tmp_path, terminal=True)
    (directory / "policy.pt").unlink()
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "missing its final model" in report["jobs"][0]["reason"]


@pytest.mark.parametrize("failure", [{"returncode": 1}, {"execution_failure": {"message": "disk full"}}])
def test_terminal_ledger_cannot_erase_recorded_worker_failure(tmp_path, failure):
    directory, _ = _job(tmp_path, terminal=True)
    _json(directory / "result.json", {"status": "SYSTEM_ERROR", **failure})
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "recorded worker failure" in report["jobs"][0]["reason"]


def test_cached_job_checks_every_fingerprint_without_loading_checkpoint(tmp_path, monkeypatch):
    directory, _ = _job(tmp_path, terminal=True)
    hashes = {str(path): file_digest(path) for path in directory.iterdir() if path.is_file()}
    _json(directory / "result.json", {"status": "COMPLETED", "phase": "train",
                                      "eligible_for_selection": True, "artifact_hashes": hashes})
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("cached audit must not load a checkpoint"))
    assert inspect_restart(tmp_path)["job_counts"]["CACHED"] == 1
    (directory / "checkpoint.pt").write_bytes(b"changed")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "Recorded artifact" in report["jobs"][0]["reason"]


def test_initialized_zero_attempt_job_keeps_current_fail_closed_policy(tmp_path):
    directory = tmp_path / "jobs" / "one"
    recorder = CandidateRecorder(directory, 4)
    recorder.close()
    before = _snapshot(tmp_path)
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "zero consumed attempts" in report["jobs"][0]["reason"]
    assert _snapshot(tmp_path) == before


def test_missing_out_and_corrupt_manifest_are_reported(tmp_path):
    assert not inspect_restart(tmp_path / "absent")["ready"]
    (tmp_path / "manifest.json").write_bytes(b"broken")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "manifest" in report["errors"][0]


@pytest.mark.parametrize("failure", [{}, {"execution_failure": {"message": "failure"}}])
def test_cached_result_needs_fingerprints_and_no_execution_failure(tmp_path, failure):
    directory, _ = _job(tmp_path, terminal=True)
    result = {"status": "COMPLETED", "phase": "train", "eligible_for_selection": True}
    if failure:
        result["artifact_hashes"] = {str(path): file_digest(path) for path in directory.iterdir() if path.is_file()}
    _json(directory / "result.json", {**result, **failure})
    assert not inspect_restart(tmp_path)["ready"]


def test_training_clock_cannot_disagree_with_hpo_seen_candidates(tmp_path):
    directory, checkpoint = _job(tmp_path)
    checkpoint["training_budget_state"]["seen_keys"] = ["different"]
    torch.save(checkpoint, directory / "checkpoint.pt")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "Training budget identities" in report["jobs"][0]["reason"]


def test_generation_contract_missing_fields_is_not_restartable(tmp_path):
    directory, checkpoint = _job(tmp_path, phase="generation")
    checkpoint["generation_contract"].pop("frozen_artifacts")
    torch.save(checkpoint, directory / "generation_checkpoint.pt")
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "frozen_artifacts" in report["jobs"][0]["reason"]


def test_checkpoint_without_original_candidate_accounting_is_blocked(tmp_path):
    directory, _ = _job(tmp_path)
    (directory / "candidate_budget.json").unlink()
    (directory / "candidate_records.jsonl").unlink()
    assert not inspect_restart(tmp_path)["ready"]


def test_terminal_execution_failure_survives_missing_result_json(tmp_path):
    directory, _ = _job(tmp_path, terminal=True)
    _json(directory / "execution.json", {"status": "SYSTEM_ERROR", "returncode": 1})
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert "execution.json" in report["jobs"][0]["reason"]


@pytest.mark.parametrize("status", ["INVALID_RECORDS", "QUALITY_CALIBRATION_REQUIRED"])
def test_cached_campaign_fatal_status_blocks_restart(tmp_path, status):
    directory, _ = _job(tmp_path, terminal=True)
    hashes = {str(path): file_digest(path) for path in directory.iterdir() if path.is_file()}
    _json(directory / "result.json", {"status": status, "phase": "train", "error": "preserved error",
                                      "eligible_for_selection": False, "artifact_hashes": hashes})
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert status in report["jobs"][0]["reason"]
