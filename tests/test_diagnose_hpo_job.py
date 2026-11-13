"""Single-job diagnostics retain damaged evidence and do not authorize recovery."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

_script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_hpo_job.py"
_spec = importlib.util.spec_from_file_location("diagnose_hpo_job", _script)
diagnose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnose)


def _write(path, value):
    path.write_text(json.dumps(value))


def _snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def job(tmp_path):
    out = tmp_path / "campaign"
    directory = out / "jobs" / "dd2b3a16877e54f40cdc52f3"
    directory.mkdir(parents=True)
    budget = {"phase": "generation", "attempts_started": 2, "unique_evaluated": 2,
              "target_unique": 10, "max_attempts": 14, "pending": 0,
              "failed_attempts": 0, "preparation_only_attempts": 0,
              "ratio_limit": 1.5, "ratio": 1.0, "status": "RUNNING"}
    records = b'{"identity":"A","attempts_started":1,"unique_evaluated":1}\n' + b'{"identity":"B","attempts_started":2,"unique_evaluated":2}\n'
    (directory / "candidate_records.jsonl").write_bytes(records)
    model = tmp_path / "frozen-policy.pt"
    model.write_bytes(b"original frozen policy")
    state = {"type": "hpo_generation", "hpo_state": {"phase": "generation", "budget": dict(budget),
             "seen": ["A", "B"], "records_sha256": hashlib.sha256(records).hexdigest()},
             "rng_state": {"schema_version": 1, "python": (1,), "numpy": (2,),
                           "torch_cpu": torch.tensor([1], dtype=torch.uint8), "torch_cuda": []},
             "predictor_ledger": {"schema_version": 1, "n_calls": 2, "seen": ["A", "B"],
                 "t_predict_s": .01, "phase": "generate", "phase_costs": {"generate": {"n_calls": 2, "seen": ["A", "B"]}}},
             "generation_contract": {"execution": {}, "sampling": {"temperature": 1},
                 "frozen_artifacts": {"policy.pt": hashlib.sha256(model.read_bytes()).hexdigest()}}}
    _write(directory / "candidate_budget.json", budget)
    _write(directory / "job.json", {"phase": "generation", "budget": 10, "seed": 42})
    _write(directory / "result.json", {"status": "INTERRUPTED", "reason": "disk full"})
    _write(directory / "execution.json", {"status": "SYSTEM_ERROR", "returncode": 1,
                                          "argv": ["python", "--load-policy", str(model)]})
    (directory / "stderr.log").write_text("head\n" * 10000 + "OSError: No space left on device\n")
    torch.save(state, directory / "generation_checkpoint.pt")
    return out, directory, budget, state


def _inspect(job):
    out, directory, _, _ = job
    return diagnose.inspect_job(out, directory.name)


def test_complete_read_only_matching_inspection_uses_cpu_and_no_state_restore(job, monkeypatch):
    out, directory, _, _ = job
    original_load, calls = torch.load, []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(torch, "set_rng_state", lambda *a: pytest.fail("Must not restore RNG"))
    before = _snapshot(out.parent)
    report = _inspect(job)
    assert report["inspection_complete"] and report["errors"] == []
    assert report["read_only"] and "no recovery attempted" in report["scope"]
    assert calls == [{"map_location": "cpu", "weights_only": False}]
    primary = report["checkpoints"]["generation_checkpoint.pt"]
    assert primary["budget_comparison"]["differing_keys"] == []
    assert primary["generation_contract"]["frozen_artifacts"]["policy.pt"]["sha256_matches"]
    assert all(primary["rng"]["fields_present"].values())
    assert "seen" not in primary["predictor_ledger"]["phase_costs"]["generate"]
    archive = report["archive"]
    assert archive["valid_records"] == archive["unique_identities"] == 2
    assert archive["checkpoint_comparisons"]["generation_checkpoint.pt"]["whole_archive_seen_matches"]
    assert report["stderr_tail"]["truncated"]
    assert len(report["stderr_tail"]["lines"]) == 30
    assert "No space left" in report["stderr_tail"]["lines"][-1]
    assert _snapshot(out.parent) == before


@pytest.mark.parametrize("pending", [0, 1])
def test_stale_checkpoint_reports_exact_prefix_without_reset_or_recovery_claim(job, pending):
    out, directory, budget, state = job
    budget.update(attempts_started=3 + pending, unique_evaluated=3, pending=pending)
    _write(directory / "candidate_budget.json", budget)
    with (directory / "candidate_records.jsonl").open("ab") as stream:
        stream.write(b'{"identity":"C"}\n')
    before = _snapshot(out.parent)
    report = _inspect(job)
    comparison = report["checkpoints"]["generation_checkpoint.pt"]["budget_comparison"]
    assert comparison["ledger_minus_checkpoint"]["attempts_started"] == 1 + pending
    assert comparison["ledger_minus_checkpoint"]["pending"] == pending
    evidence = report["archive"]["checkpoint_comparisons"]["generation_checkpoint.pt"]
    assert evidence["prefix_sha256_matches"] and evidence["prefix_seen_matches"]
    assert not evidence["whole_archive_sha256_matches"]
    assert evidence["lines_after_checkpoint_prefix"] == 1
    assert report["inspection_complete"]  # Evidence is complete; this is NOT readiness.
    assert "ready" not in report and "recoverable" not in report
    assert _snapshot(out.parent) == before


def test_checkpoint_ahead_is_distinguished_and_pending_is_not_hidden(job):
    _, directory, budget, _ = job
    budget.update(attempts_started=1, unique_evaluated=1, pending=1)
    _write(directory / "candidate_budget.json", budget)
    report = _inspect(job)
    delta = report["checkpoints"]["generation_checkpoint.pt"]["budget_comparison"]["ledger_minus_checkpoint"]
    assert delta["attempts_started"] == -1
    assert delta["unique_evaluated"] == -1
    assert delta["pending"] == 1
    assert not report["archive"]["record_count_matches_ledger"]


def test_corrupt_primary_does_not_prevent_independent_tmp_inspection(job, capsys):
    out, directory, _, state = job
    (directory / "generation_checkpoint.pt").write_bytes(b"partial torch archive")
    torch.save(state, directory / "generation_checkpoint.pt.tmp")
    before = _snapshot(out.parent)
    assert diagnose.main(["--out", str(out), "--job-id", directory.name]) == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["inspection_complete"]
    assert not report["checkpoints"]["generation_checkpoint.pt"]["loaded"]
    assert report["checkpoints"]["generation_checkpoint.pt.tmp"]["loaded"]
    assert report["archive"]["checkpoint_comparisons"]["generation_checkpoint.pt.tmp"]["whole_archive_sha256_matches"]
    assert _snapshot(out.parent) == before


@pytest.mark.parametrize("tail,partial", [(b'{"identity":', True), (b'{bad}\n', False)])
def test_archive_corruption_preserves_prefix_evidence_and_partial_tail(job, tail, partial):
    out, directory, _, _ = job
    with (directory / "candidate_records.jsonl").open("ab") as stream:
        stream.write(tail)
    before = _snapshot(out.parent)
    report = _inspect(job)
    assert not report["inspection_complete"]
    assert report["archive"]["parse_error_count"] == 1
    assert report["archive"]["truncated_tail_suspected"] is partial
    assert report["archive"]["checkpoint_comparisons"]["generation_checkpoint.pt"]["prefix_sha256_matches"]
    assert _snapshot(out.parent) == before


def test_reordered_or_tampered_prefix_and_duplicate_identities_do_not_match(job):
    _, directory, _, _ = job
    (directory / "candidate_records.jsonl").write_bytes(b'{"identity":"A"}\n{"identity":"A"}\n')
    report = _inspect(job)
    assert report["archive"]["duplicate_identities"] == 1
    evidence = report["archive"]["checkpoint_comparisons"]["generation_checkpoint.pt"]
    assert not evidence["prefix_sha256_matches"]
    assert not evidence["prefix_seen_matches"]


@pytest.mark.parametrize("name,value", [("result.json", []), ("execution.json", None),
                                        ("job.json", "bad object")])
def test_malformed_metadata_does_not_hide_checkpoint_or_archive(job, name, value):
    _, directory, _, _ = job
    _write(directory / name, value)
    report = _inspect(job)
    assert not report["inspection_complete"]
    assert any(error["location"] == name for error in report["errors"])
    assert report["checkpoints"]["generation_checkpoint.pt"]["loaded"]
    assert report["archive"]["valid_records"] == 2


def test_missing_ledger_is_reported_while_job_phase_allows_other_evidence(job):
    _, directory, _, _ = job
    (directory / "candidate_budget.json").unlink()
    report = _inspect(job)
    assert not report["inspection_complete"]
    assert report["checkpoints"]["generation_checkpoint.pt"]["loaded"]
    assert report["archive"]["valid_records"] == 2


@pytest.mark.parametrize("job_id", ["../escape", "a/b", "..", "/tmp/evil", "a\\b", ""])
def test_rejects_traversal_without_loading_checkpoint(job, job_id, monkeypatch):
    out, _, _, _ = job
    before = _snapshot(out.parent)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Must not load"))
    report = diagnose.inspect_job(out, job_id)
    assert not report["inspection_complete"]
    assert report["errors"][0]["location"] == "job_id"
    assert _snapshot(out.parent) == before


def test_absent_or_symlink_job_is_rejected_without_new_directories(job):
    out, directory, _, _ = job
    alias = out / "jobs" / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    before = _snapshot(out.parent)
    for name in ("absent", "alias"):
        report = diagnose.inspect_job(out, name)
        assert not report["inspection_complete"]
        assert report["errors"][0]["location"] == "jobdir"
    assert not (out / "jobs" / "absent").exists()
    assert _snapshot(out.parent) == before


def test_checkpoint_symlink_cannot_load_outside_selected_job(job, monkeypatch):
    out, directory, _, state = job
    other = out / "outside.pt"
    torch.save(state, other)
    primary = directory / "generation_checkpoint.pt"
    primary.unlink()
    primary.symlink_to(other)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Must not load external checkpoint"))
    report = _inspect(job)
    assert not report["inspection_complete"]
    assert not report["checkpoints"]["generation_checkpoint.pt"]["loaded"]
    assert any("escapes" in error["message"] for error in report["errors"])


def test_evaluation_log_is_bounded_and_omits_candidates(job):
    _, directory, _, _ = job
    rows = [{"phase": "generate", "episode": i, "formula": "SECRET", "candidate_key": ["SECRET"]}
            for i in range(10)]
    (directory / "generation_evaluations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = _inspect(job)
    assert report["generation_evaluations"]["rows"] == 10
    assert len(report["generation_evaluations"]["last_rows"]) == 3
    assert "SECRET" not in json.dumps(report)


def test_training_selects_only_training_checkpoint_and_tmp(job):
    _, directory, budget, state = job
    budget["phase"] = "train"
    state["type"] = "pg"
    state["hpo_state"]["phase"] = "train"
    state["hpo_state"]["budget"]["phase"] = "train"
    _write(directory / "candidate_budget.json", budget)
    torch.save(state, directory / "checkpoint.pt")
    (directory / "generation_checkpoint.pt").write_bytes(b"unrelated corrupt file must not load")
    report = _inspect(job)
    assert report["inspection_complete"]
    assert set(report["checkpoints"]) == {"checkpoint.pt", "checkpoint.pt.tmp"}


def test_main_mismatch_exit_zero_means_evidence_only(job, capsys):
    out, directory, budget, _ = job
    budget["pending"] = 1
    _write(directory / "candidate_budget.json", budget)
    assert diagnose.main(["--out", str(out), "--job-id", directory.name]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "pending" in report["checkpoints"]["generation_checkpoint.pt"]["budget_comparison"]["differing_keys"]
    assert "recoverability" in report["scope"]


def test_actual_observation_and_timing_fields_are_bounded_without_candidate_contents(job):
    _, directory, _, _ = job
    (directory / "generation_evaluations.jsonl").write_text(json.dumps({
        "phase": "generate", "request": 7, "invocation": "session", "status": "ok",
        "unique": 7, "new_successful_candidate": True, "prediction_elapsed_s": 1.5,
        "candidate": "SECRET", "candidate_key": "SECRET", "outputs": {"secret": 9}}) + "\n")
    (directory / "generation_episode_timing.csv").write_text(
        "timing_session,episode,evaluation_wave_id,evaluation_wave_size,formula,status\n"
        "session,7,wave1,4,SECRET,ok\n")
    (directory / "stdout.log").write_text("worker stopped\n")
    report = _inspect(job)
    observed = report["generation_evaluations"]["last_rows"][0]
    assert observed["request"] == 7 and observed["prediction_elapsed_s"] == 1.5
    assert report["generation_episode_timing"]["last_rows"][0]["evaluation_wave_size"] == "4"
    assert report["stdout_tail"]["lines"] == ["worker stopped"]
    assert report["archive"]["last_record_counters"][-1] == {"attempts_started": 2, "unique_evaluated": 2}
    assert "SECRET" not in json.dumps(report)


def test_missing_execution_does_not_invent_frozen_default_path(job):
    _, directory, _, _ = job
    (directory / "execution.json").unlink()
    report = _inspect(job)
    frozen = report["checkpoints"]["generation_checkpoint.pt"]["generation_contract"]["frozen_artifacts"]["policy.pt"]
    assert frozen["path"] is None and frozen["recorded_path"] is None
    assert "path_source" not in frozen
    assert "unavailable" in frozen["inspection_note"]
