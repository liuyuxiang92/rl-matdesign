"""One-observation repair preserves paid artifacts and rejects ambiguous evidence."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch

from rl_matdesign.utils.candidate_records import candidate_identity
from rl_matdesign.utils.composition import composition_key

_path = Path(__file__).resolve().parents[1] / "scripts" / "hpo_generation_repair_io.py"
_spec = importlib.util.spec_from_file_location("hpo_generation_repair_io", _path)
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)


def _write(path, value):
    path.write_text(json.dumps(value))


def _snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file() and repair.REPAIR_DIR not in path.parts}


@pytest.fixture
def job(tmp_path):
    job = {"phase": "generation", "budget": 6, "seed": 42, "parameters": {"gen_temperature": 1}, "input_models": {}}
    job_id = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    directory = tmp_path / "jobs" / job_id
    directory.mkdir(parents=True)
    ledger = {"phase": "generation", "status": "SYSTEM_ERROR", "attempts_started": 3,
              "unique_evaluated": 3, "target_unique": 6, "max_attempts": 8, "ratio_limit": 1.5,
              "pending": 0, "failed_attempts": 0, "preparation_only_attempts": 0}
    saved = {**ledger, "attempts_started": 2, "unique_evaluated": 2, "status": "RUNNING"}
    records, observations, timing, keys = [], [], [], []
    for i in range(1, 4):
        candidate = {"Al": i / 10, "Si": 1 - i / 10}
        key = composition_key(candidate)
        keys.append(key)
        records.append({"identity": candidate_identity(("candidate", i)), "phase": "generation",
                        "attempts_started": i, "unique_evaluated": i, "formula": f"row{i}",
                        "prediction_record": {"candidate_json": json.dumps(candidate)}})
        observations.append({"phase": "generate", "request": i, "unique": i,
             "new_successful_candidate": True, "status": "ok", "error": None, "invocation": "single-session",
             "candidate": candidate, "candidate_key": repr(key)})
        timing.append({"phase": "generate", "status": "ok", "episode": str(i),
                       "evaluation_wave_id": str(i), "evaluation_wave_size": "1", "formula": f"row{i}",
                       "timing_session": "single-session"})
    record_lines = [(json.dumps(row) + "\n").encode() for row in records]
    (directory / "candidate_records.jsonl").write_bytes(b"".join(record_lines))
    (directory / "generation_evaluations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in observations))
    with (directory / "generation_episode_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    state = {"type": "hpo_generation", "hpo_state": {"phase": "generation", "budget": saved,
        "seen": [row["identity"] for row in records[:2]], "records_sha256": hashlib.sha256(b"".join(record_lines[:2])).hexdigest()},
        "rng_state": {"torch_cpu": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        "generation_contract": {"execution": {"old": "preserved"}, "sampling": {"temperature": 1}, "frozen_artifacts": {}},
        "predictor_ledger": {"schema_version": 1, "phase": "generate", "n_calls": 2, "seen": keys[:2],
                             "phase_costs": {"generate": {"requests": 2, "successes": 2, "failures": 0, "seen": keys[:2]}}},
        "predictor_caches": {"example": torch.tensor([4., 5.])}}
    torch.save(state, directory / repair.PRIMARY)
    (directory / (repair.PRIMARY + ".tmp")).write_bytes(b"x" * 128)
    _write(directory / "candidate_budget.json", ledger)
    _write(directory / "job.json", job)
    _write(directory / "execution.json", {"status": "SYSTEM_ERROR", "returncode": 1, "argv": ["worker"]})
    _write(directory / "result.json", {"job_id": job_id, "phase": "generation", "status": "SYSTEM_ERROR",
                                       "returncode": 1, "eligible_for_selection": False})
    _write(directory / "run_config.json", {"hpo_phase": "generation"})
    provenance = directory / "provenance" / "invocation"
    provenance.mkdir(parents=True)
    _write(provenance / "manifest.json", {"phase": "generate", "config": {}})
    _write(provenance / "execution_contract.json", {"old": "preserved"})
    (directory / "stderr.log").write_text("RuntimeError: disk full\n")
    return directory


def _repaired(evidence):
    state = copy.deepcopy(evidence["checkpoint"])
    state["hpo_state"] = {"phase": "generation", "budget": {**evidence["ledger"], "status": "RUNNING"},
                          "seen": [row["identity"] for row in evidence["records"]],
                          "records_sha256": evidence["file_hashes"]["candidate_records.jsonl"]}
    state["rng_state"]["torch_cpu"] = torch.tensor([7, 8, 9], dtype=torch.uint8)
    return state


def test_guard_is_read_only_and_exposes_exact_paid_prefix(job):
    before = _snapshot(job)
    evidence = repair.validate_guard(job)
    assert evidence["checkpoint"]["hpo_state"]["budget"]["unique_evaluated"] == 2
    assert evidence["ledger"]["unique_evaluated"] == 3
    assert len(evidence["records"]) == len(evidence["timing_rows"]) == 3
    assert evidence["provenance"][0]["execution_contract"] == {"old": "preserved"}
    assert evidence["file_hashes"][repair.PRIMARY + ".tmp"]
    assert _snapshot(job) == before


@pytest.mark.parametrize("change", ["pending", "failed", "duplicate", "target", "terminal", "two_ahead"])
def test_guard_rejects_ambiguous_or_terminal_accounting(job, change):
    path = job / "candidate_budget.json"
    ledger = json.loads(path.read_text())
    if change == "pending": ledger["pending"] = 1
    if change == "failed": ledger["failed_attempts"] = 1
    if change == "duplicate": ledger["attempts_started"] = 4
    if change == "target": ledger["target_unique"] = 7
    if change == "terminal": ledger.update(status="COMPLETED", target_unique=3)
    if change == "two_ahead": ledger.update(attempts_started=4, unique_evaluated=4)
    _write(path, ledger)
    with pytest.raises(ValueError):
        repair.validate_guard(job)


@pytest.mark.parametrize("file,change", [("candidate_records.jsonl", "partial"), ("candidate_records.jsonl", "duplicate"),
    ("candidate_records.jsonl", "counter"), ("generation_evaluations.jsonl", "candidate"),
    ("generation_evaluations.jsonl", "invocation"), ("generation_evaluations.jsonl", "request")])
def test_guard_rejects_corrupt_archive_and_observation_evidence(job, file, change):
    path = job / file
    if change == "partial":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "duplicate": rows[-1]["identity"] = rows[0]["identity"]
        if change == "counter": rows[-1]["attempts_started"] = 99
        if change == "candidate": rows[-1]["candidate"]["Al"] = 0.999
        if change == "invocation": rows[-1]["invocation"] = "different-session"
        if change == "request": rows[-1]["request"] = 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = _snapshot(job)
    with pytest.raises(ValueError):
        repair.validate_guard(job)
    assert _snapshot(job) == before


@pytest.mark.parametrize("change", ["wave", "missing", "formula", "extra"])
def test_guard_requires_complete_single_candidate_timing(job, change):
    path = job / "generation_episode_timing.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    if change == "missing": rows.pop()
    if change == "wave": rows[-1]["evaluation_wave_size"] = "2"
    if change == "formula": rows[-1]["formula"] = "changed"
    if change == "extra": rows.append(rows[-1])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        repair.validate_guard(job)


def test_guard_rejects_changed_result_fingerprint_and_job_key(job):
    result = json.loads((job / "result.json").read_text())
    result["artifact_hashes"] = {str(job / "candidate_records.jsonl"): "0" * 64}
    _write(job / "result.json", result)
    with pytest.raises(ValueError, match="artifact changed"):
        repair.validate_guard(job)
    result.pop("artifact_hashes")
    _write(job / "result.json", result)
    data = json.loads((job / "job.json").read_text())
    data["seed"] = 99
    _write(job / "job.json", data)
    with pytest.raises(ValueError, match="identity mismatch"):
        repair.validate_guard(job)


def test_apply_backs_up_all_bytes_and_is_idempotent_without_resetting_cost(job):
    evidence = repair.validate_guard(job)
    before = _snapshot(job)
    receipt = repair.apply_repair(job, _repaired(evidence), evidence, {"replay": "verified elsewhere"})
    assert receipt["status"] == "APPLIED" and not receipt["already_applied"]
    after = _snapshot(job)
    assert {k: v for k, v in after.items() if k != repair.PRIMARY} == {k: v for k, v in before.items() if k != repair.PRIMARY}
    backup = Path(receipt["backup_directory"])
    for name, (content, _) in before.items():
        assert (backup / name).read_bytes() == content
    assert repair.applied_receipt(job)["already_applied"]
    again = repair.apply_repair(job, _repaired(evidence), evidence, {"ignored": True})
    assert again["already_applied"] and again["status"] == "APPLIED"
    assert _snapshot(job) == after
    saved = torch.load(job / repair.PRIMARY, map_location="cpu", weights_only=False)
    assert saved["hpo_state"]["budget"]["unique_evaluated"] == 3
    assert json.loads((job / "candidate_budget.json").read_text())["status"] == "SYSTEM_ERROR"


@pytest.mark.parametrize("change", ["mutate", "new", "remove"])
def test_apply_rejects_stale_input_evidence(job, change):
    evidence = repair.validate_guard(job)
    primary = (job / repair.PRIMARY).read_bytes()
    if change == "mutate": (job / "stderr.log").write_text("worker wrote new output")
    if change == "new": (job / "new-worker-output").write_text("late")
    if change == "remove": (job / "stderr.log").unlink()
    with pytest.raises(ValueError, match="changed since validation"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert (job / repair.PRIMARY).read_bytes() == primary


def test_failed_serialization_preserves_original_checkpoint_and_damaged_tmp(job, monkeypatch):
    evidence = repair.validate_guard(job)
    before = _snapshot(job)
    def fail(state, stream):
        stream.write(b"partial")
        raise OSError("No space left on device")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="No space"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert _snapshot(job) == before


def test_serialization_roundtrip_must_preserve_rng_and_caches(job, monkeypatch):
    evidence = repair.validate_guard(job)
    before = _snapshot(job)
    original = torch.save
    def damaged(state, stream):
        altered = copy.deepcopy(state)
        altered["rng_state"]["torch_cpu"] = torch.tensor([99], dtype=torch.uint8)
        original(altered, stream)
    monkeypatch.setattr(torch, "save", damaged)
    with pytest.raises(ValueError, match="verification"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert _snapshot(job) == before


def test_final_recheck_catches_worker_write_during_serialization(job, monkeypatch):
    evidence = repair.validate_guard(job)
    primary = (job / repair.PRIMARY).read_bytes()
    original = torch.save
    def raced(state, stream):
        original(state, stream)
        (job / "stderr.log").write_text("concurrent worker wrote here")
    monkeypatch.setattr(torch, "save", raced)
    with pytest.raises(ValueError, match="changed since validation"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert (job / repair.PRIMARY).read_bytes() == primary


def test_crash_after_atomic_replace_is_detected_without_reapplying(job, monkeypatch):
    evidence = repair.validate_guard(job)
    original = repair.os.replace
    def crash(source, target):
        original(source, target)
        raise OSError("simulated crash after replace")
    monkeypatch.setattr(repair.os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    receipt = repair.applied_receipt(job)
    assert receipt["already_applied"] and receipt["status"] == "APPLIED"
    again = repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert again["already_applied"]


def test_missing_log_or_provenance_is_not_inferred(job):
    (job / "generation_evaluations.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing required evidence"):
        repair.validate_guard(job)


def test_guard_rejects_missing_provenance(job):
    for path in (job / "provenance").rglob("*.json"):
        path.unlink()
    with pytest.raises(ValueError, match="No invocation provenance"):
        repair.validate_guard(job)


def test_applied_receipt_rejects_changed_preserved_backup(job):
    evidence = repair.validate_guard(job)
    receipt = repair.apply_repair(job, _repaired(evidence), evidence, {})
    (Path(receipt["backup_directory"]) / "stderr.log").write_text("altered backup")
    with pytest.raises(ValueError, match="Preserved repair evidence changed"):
        repair.applied_receipt(job)


def test_repair_cannot_change_generation_contract(job):
    evidence = repair.validate_guard(job)
    state = _repaired(evidence)
    state["generation_contract"]["sampling"]["temperature"] = 5
    before = _snapshot(job)
    with pytest.raises(ValueError, match="frozen generation contract"):
        repair.apply_repair(job, state, evidence, {})
    assert _snapshot(job) == before


def test_state_comparison_covers_statistics_outputs_and_mapping_order():
    from rl_matdesign.predictors.outputs import OutputSchema, ScientificOutput
    from rl_matdesign.predictors.structure_score import _Statistics
    left = _Statistics({"T3": (1000., 2.)}, outputs={"T3": ScientificOutput(OutputSchema("T3", "K"), 1000.)})
    right = copy.deepcopy(left)
    assert repair._same_state(left, right)
    right.outputs["T3"] = ScientificOutput(OutputSchema("T3", "K"), 999.)
    assert dict(left) == dict(right)
    assert not repair._same_state(left, right)
    assert not repair._same_state({"first": 1, "second": 2}, {"second": 2, "first": 1})


def test_serialization_verification_checks_statistics_attributes(job, monkeypatch):
    from rl_matdesign.predictors.outputs import OutputSchema, ScientificOutput
    from rl_matdesign.predictors.structure_score import _Statistics
    evidence = repair.validate_guard(job)
    state = _repaired(evidence)
    state["predictor_caches"]["stats"] = _Statistics(
        {"T3": (1000., 2.)}, outputs={"T3": ScientificOutput(OutputSchema("T3", "K"), 1000.)})
    original = torch.save
    def damaged(value, stream):
        altered = copy.deepcopy(value)
        altered["predictor_caches"]["stats"].outputs.clear()
        original(altered, stream)
    monkeypatch.setattr(torch, "save", damaged)
    before = _snapshot(job)
    with pytest.raises(ValueError, match="Serialized repair failed verification"):
        repair.apply_repair(job, state, evidence, {})
    assert _snapshot(job) == before


@pytest.mark.parametrize("stage", ["backup", "serialization"])
def test_external_model_change_before_final_replace_is_rejected(job, monkeypatch, stage):
    model = job.parent.parent / "external-model.pt"
    model.write_bytes(b"original frozen model")
    evidence = repair.validate_guard(job)
    evidence["external_hashes"][str(model)] = repair._sha(model)
    before = _snapshot(job)
    if stage == "backup":
        original = repair.shutil.copy2
        def raced(source, target):
            result = original(source, target)
            model.write_bytes(b"changed model during backup")
            return result
        monkeypatch.setattr(repair.shutil, "copy2", raced)
    else:
        original = torch.save
        def raced(state, stream):
            original(state, stream)
            model.write_bytes(b"changed model during serialization")
        monkeypatch.setattr(torch, "save", raced)
    with pytest.raises(ValueError, match="External evidence changed"):
        repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert _snapshot(job) == before


def test_external_fingerprints_are_preserved_and_checked_on_repeat(job):
    model = job.parent.parent / "external-model.pt"
    model.write_bytes(b"original frozen model")
    evidence = repair.validate_guard(job)
    evidence["external_hashes"][str(model)] = repair._sha(model)
    receipt = repair.apply_repair(job, _repaired(evidence), evidence, {})
    assert receipt["external_evidence_sha256"] == evidence["external_hashes"]
    assert repair.applied_receipt(job)["already_applied"]
    model.write_bytes(b"changed after repair")
    with pytest.raises(ValueError, match="External evidence changed"):
        repair.applied_receipt(job)
