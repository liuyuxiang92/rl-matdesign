"""Narrow evidence guard and atomic storage for one-observation generation repair.

This helper never evaluates a property, generates a candidate, restores RNGs or
claims that replay is valid. Its caller must prove replay equivalence first.
Only trusted local checkpoints may be loaded. Stop all campaign workers first.
"""
from __future__ import annotations

import ast
import csv
import fcntl
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


REPAIR_DIR = ".hpo-generation-repair"
PRIMARY = "generation_checkpoint.pt"
COUNTERS = ("attempts_started", "unique_evaluated", "target_unique", "max_attempts",
            "pending", "failed_attempts", "preparation_only_attempts")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, label):
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    try:
        return _object(json.loads(Path(path).read_text()), str(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read required metadata {path}: {exc}") from exc


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if relative.parts[0] == REPAIR_DIR:
            continue
        _require(not path.is_symlink(), f"Symlink evidence is unsupported: {relative}")
        if path.is_file():
            result[relative.as_posix()] = _sha(path)
        else:
            _require(path.is_dir(), f"Special evidence file is unsupported: {relative}")
    return result


def _lines(path):
    try:
        lines = Path(path).read_bytes().splitlines(keepends=True)
        _require(bool(lines) and all(line.endswith(b"\n") for line in lines),
                 f"Missing records or incomplete final line: {path}")
        rows = [_object(json.loads(line), f"{path} row {i}") for i, line in enumerate(lines, 1)]
        return lines, rows
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid complete JSONL evidence {path}: {exc}") from exc


def _load_checkpoint(path):
    import torch
    try:
        return _object(torch.load(path, map_location="cpu", weights_only=False), "checkpoint")
    except Exception as exc:
        raise ValueError(f"Cannot load trusted checkpoint {path}: {exc}") from exc


def _budget(value, label):
    value = _object(value, label)
    for key in COUNTERS:
        _require(type(value.get(key)) is int and value[key] >= 0, f"{label}.{key} is invalid")
    _require(value["target_unique"] > 0, f"{label} has no positive target")
    try:
        ratio = Decimal(str(value["ratio_limit"]))
        _require(ratio.is_finite() and ratio > 1, f"{label} ratio is invalid")
    except (KeyError, ArithmeticError) as exc:
        raise ValueError(f"{label} ratio is invalid") from exc
    _require(value["attempts_started"] == value["unique_evaluated"], f"{label} includes duplicate attempts")
    _require(all(value[key] == 0 for key in ("pending", "failed_attempts", "preparation_only_attempts")),
             f"{label} has pending, failed or preparation attempts")
    return value


def validate_guard(directory):
    """Read immutable evidence for the supported single missing generation save.

    Property/runtime equivalence and actual action/RNG replay belong to caller.
    Returned checkpoint is private mutable caller state; input files stay intact.
    """
    directory = Path(directory).expanduser().absolute()
    _require(directory.is_dir() and not directory.is_symlink(), "Job directory must be a real existing directory")
    directory = directory.resolve()
    before = _inventory(directory)
    for name in (PRIMARY, "candidate_budget.json", "candidate_records.jsonl", "generation_evaluations.jsonl",
                 "generation_episode_timing.csv", "job.json", "execution.json", "result.json", "run_config.json"):
        _require(name in before, f"Missing required evidence: {name}")
    checkpoint = _load_checkpoint(directory / PRIMARY)
    _require(checkpoint.get("type") == "hpo_generation" and checkpoint.get("resume_safe") is not False,
             "Only complete hpo_generation checkpoints are supported")
    hpo = _object(checkpoint.get("hpo_state"), "checkpoint.hpo_state")
    saved = _budget(hpo.get("budget"), "checkpoint budget")
    ledger = _budget(_json(directory / "candidate_budget.json"), "candidate ledger")
    _require(hpo.get("phase") == ledger.get("phase") == "generation", "Generation phase mismatch")
    _require(saved.get("status") == "RUNNING", "Checkpoint budget must be RUNNING")
    _require(ledger.get("status") in {"SYSTEM_ERROR", "INCOMPLETE"}, "Ledger must record a nonterminal failed generation")
    n = saved["unique_evaluated"]
    _require(ledger["unique_evaluated"] == n + 1 and ledger["unique_evaluated"] < ledger["target_unique"],
             "Only exactly one paid observation beyond checkpoint, below target, is supported")
    _require(all(saved[key] == ledger[key] for key in ("target_unique", "max_attempts"))
             and Decimal(str(saved["ratio_limit"])) == Decimal(str(ledger["ratio_limit"])),
             "Budget target, attempt cap or ratio changed")
    _require(ledger["attempts_started"] <= ledger["max_attempts"], "Consumed attempts exceed cap")
    metadata = {name: _json(directory / name) for name in ("job.json", "execution.json", "result.json", "run_config.json")}
    job, execution, result = (metadata[name] for name in ("job.json", "execution.json", "result.json"))
    job_id = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]
    _require(directory.name == job_id and result.get("job_id") == job_id, "Job key/directory/result identity mismatch")
    _require(job.get("phase") == result.get("phase") == "generation", "Job/result phase is not generation")
    _require(job.get("budget") == ledger["target_unique"], "Job budget differs from ledger")
    for name, data in (("result", result), ("execution", execution)):
        _require(data.get("status") == "SYSTEM_ERROR" and type(data.get("returncode")) is int
                 and data["returncode"] == 1, f"{name} must record SYSTEM_ERROR with returncode 1")
    _require(not result.get("finalization_pending") and not result.get("eligible_for_selection"),
             "Pending finalization or eligible results are unsupported")
    external_hashes = {}
    for name, expected in _object(result.get("artifact_hashes", {}), "result.artifact_hashes").items():
        path = Path(name)
        _require(path.is_absolute() and path.is_file(), f"Recorded artifact is unavailable: {name}")
        _require(_sha(path) == expected, f"Recorded artifact changed: {name}")
        if not path.is_relative_to(directory):
            external_hashes[str(path)] = expected
    record_lines, records = _lines(directory / "candidate_records.jsonl")
    _, observations = _lines(directory / "generation_evaluations.jsonl")
    _require(len(records) == len(observations) == n + 1, "Archive/observation counts do not equal paid unique count")
    _require(hashlib.sha256(b"".join(record_lines[:n])).hexdigest() == hpo.get("records_sha256"),
             "Checkpoint is not the exact archive byte prefix")
    identities = [record.get("identity") for record in records]
    _require(all(isinstance(value, str) and value for value in identities) and len(set(identities)) == n + 1,
             "Archive canonical identities are missing or duplicated")
    seen = hpo.get("seen")
    _require(isinstance(seen, (tuple, list)) and len(seen) == n and set(seen) == set(identities[:n]),
             "Checkpoint identities differ from archive prefix")
    from rl_matdesign.utils.composition import composition_key
    keys = []
    invocation = observations[0].get("invocation")
    _require(isinstance(invocation, str) and invocation, "Missing observation invocation")
    for i, (record, observed) in enumerate(zip(records, observations), 1):
        _require(record.get("phase") == "generation" and type(record.get("attempts_started")) is int and record["attempts_started"] == i
                 and type(record.get("unique_evaluated")) is int and record["unique_evaluated"] == i, f"Archive row {i} has inconsistent counters")
        _require(observed.get("phase") == "generate" and type(observed.get("request")) is int and observed["request"] == i
                 and type(observed.get("unique")) is int and observed["unique"] == i and observed.get("new_successful_candidate") is True
                 and observed.get("status") == "ok" and observed.get("error") is None
                 and observed.get("invocation") == invocation, f"Observation row {i} is not one successful unique request")
        try:
            candidate = json.loads(_object(record.get("prediction_record"), "prediction_record")["candidate_json"])
            key = composition_key(observed["candidate"])
            _require(key == composition_key(candidate), f"Observation/archive composition differs at row {i}")
            _require(ast.literal_eval(observed["candidate_key"]) == key, f"Observation key differs at row {i}")
        except (KeyError, TypeError, SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid candidate identity evidence at row {i}: {exc}") from exc
        keys.append(key)
    _require(len(set(keys)) == n + 1, "Observed compositions contain duplicates")
    try:
        with (directory / "generation_episode_timing.csv").open(newline="") as stream:
            timing_rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise ValueError(f"Cannot parse generation timing: {exc}") from exc
    _require(len(timing_rows) == n + 1, "Timing row count differs from paid observations")
    for i, (row, record) in enumerate(zip(timing_rows, records), 1):
        _require(None not in row and all(value is not None for value in row.values()), f"Malformed timing row {i}")
        _require(row.get("phase") == "generate" and row.get("status") == "ok"
                 and row.get("episode") == row.get("evaluation_wave_id") == str(i)
                 and row.get("evaluation_wave_size") == "1" and row.get("timing_session") == invocation,
                 f"Timing row {i} is not the corresponding single-candidate wave")
        _require(row.get("formula") == record.get("formula"), f"Timing/archive formula differs at row {i}")
    predictor = _object(checkpoint.get("predictor_ledger"), "predictor_ledger")
    _require(isinstance(predictor.get("seen"), (list, tuple)), "Predictor seen identities must be a sequence")
    _require(predictor.get("schema_version") == 1 and predictor.get("phase") == "generate"
             and predictor.get("n_calls") == n and len(predictor.get("seen", [])) == n
             and set(predictor.get("seen", [])) == set(keys[:n]), "Predictor checkpoint counts/identities differ from prefix")
    _require(predictor.get("diagnostic_requests", 0) == predictor.get("diagnostic_failures", 0) == 0,
             "Diagnostic predictor requests are unsupported")
    costs = _object(predictor.get("phase_costs"), "predictor phase costs")
    _require(set(costs) == {"generate"}, "Additional predictor phases are unsupported")
    cost = _object(costs["generate"], "generate phase costs")
    _require(isinstance(cost.get("seen"), (list, tuple)), "Predictor phase seen identities must be a sequence")
    _require(cost.get("requests") == cost.get("successes") == n and cost.get("failures") == 0
             and len(cost.get("seen", [])) == n and set(cost.get("seen", [])) == set(keys[:n]),
             "Predictor phase accounting differs from prefix")
    provenance = []
    for path in sorted((directory / "provenance").glob("*/manifest.json")):
        entry = {"path": str(path), "manifest": _json(path)}
        for name in ("execution_contract.json", "source_identity.json"):
            if (path.parent / name).exists():
                entry[name.removesuffix(".json")] = _json(path.parent / name)
        provenance.append(entry)
    _require(bool(provenance), "No invocation provenance is available")
    _require(_inventory(directory) == before, "Job evidence changed during inspection; stop workers and inspect again")
    return {"directory": str(directory), "checkpoint": checkpoint, "ledger": ledger,
            "records": records, "record_lines": record_lines, "observations": observations,
            "timing_rows": timing_rows, "observation_keys": keys, "invocation": invocation,
            "job": job, "execution": execution, "result": result,
            "run_config": metadata["run_config.json"], "provenance": provenance,
            "file_hashes": before, "external_hashes": external_hashes, "checkpoint_hash": before[PRIMARY]}


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _recheck(directory, evidence):
    _require(_inventory(directory) == evidence["file_hashes"],
             "Job evidence changed since validation; refusing to apply repair")
    for name, expected in evidence.get("external_hashes", {}).items():
        _require(Path(name).is_file() and _sha(name) == expected, f"External evidence changed: {name}")


def applied_receipt(directory):
    """Return an intact already-applied transaction, including after a crash."""
    directory = Path(directory).expanduser().resolve()
    primary_hash = _sha(directory / PRIMARY)
    repair_root = directory / REPAIR_DIR
    _require(not repair_root.is_symlink(), "Repair directory cannot be a symlink")
    for path in sorted(repair_root.glob("*/receipt.json")):
        _require(not path.is_symlink(), "Repair receipt cannot be a symlink")
        receipt = _json(path)
        if receipt.get("new_checkpoint_sha256") != primary_hash:
            continue
        _require(receipt.get("directory") == str(directory), "Receipt belongs to another job")
        original = _object(receipt.get("evidence_sha256"), "receipt evidence")
        expected = {**original, PRIMARY: primary_hash}
        _recheck(directory, {"file_hashes": expected,
                            "external_hashes": receipt.get("external_evidence_sha256", {})})
        backup = path.parent / "original"
        for name, old_hash in original.items():
            target = backup / name
            _require(not target.is_symlink() and target.is_file() and _sha(target) == old_hash,
                     f"Preserved repair evidence changed: {name}")
        return {**receipt, "already_applied": True, "status": "APPLIED"}
    return None


def _same_state(left, right):
    """Verify every deserialized state value, including cached dataclasses/RNGs."""
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, dict):
        return (list(left) == list(right)
                and all(_same_state(left[key], right[key]) for key in left)
                and (not hasattr(left, "__dict__")
                     or _same_state(vars(left), vars(right))))
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    if hasattr(left, "__dict__"):
        return _same_state(vars(left), vars(right))
    return bool(left == right)


def apply_repair(directory, repaired_state, evidence, details):
    """Serialize writers; a crashed process automatically releases its OS lock."""
    directory = Path(directory).expanduser().resolve()
    _require(str(directory) == evidence.get("directory"), "Evidence belongs to another job")
    repair_root = directory / REPAIR_DIR
    _require(not repair_root.is_symlink(), "Repair directory cannot be a symlink")
    repair_root.mkdir(exist_ok=True)
    lock = repair_root / "apply.lock"
    _require(not lock.is_symlink(), "Repair lock cannot be a symlink")
    with lock.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another repair process owns this job") from exc
        return _apply_repair_locked(directory, repaired_state, evidence, details)


def _apply_repair_locked(directory, repaired_state, evidence, details):
    """Apply a caller-proven state with preserved evidence and durable intent.

    A receipt is durable before replacement. After interruption, matching its
    new hash means applied; matching old hash means original remains untouched.
    The original damaged .tmp and every original metadata/log byte are retained.
    """
    directory = Path(directory).expanduser().resolve()
    _require(str(directory) == evidence.get("directory"), "Evidence belongs to another job")
    existing = applied_receipt(directory)
    if existing:
        _require(existing["old_checkpoint_sha256"] == evidence["checkpoint_hash"], "Different repair evidence supplied")
        return existing
    _recheck(directory, evidence)
    hpo = _object(_object(repaired_state, "repaired state").get("hpo_state"), "repaired hpo_state")
    ledger = evidence["ledger"]
    _require(repaired_state.get("type") == "hpo_generation" and hpo.get("phase") == "generation",
             "Repaired state must remain a generation checkpoint")
    budget = _object(hpo.get("budget"), "repaired budget")
    _require(_same_state(repaired_state.get("generation_contract"), evidence["checkpoint"].get("generation_contract")),
             "Repair cannot change the frozen generation contract")
    _require(all(budget.get(key) == ledger[key] for key in COUNTERS)
             and Decimal(str(budget.get("ratio_limit"))) == Decimal(str(ledger["ratio_limit"])),
             "Repaired accounting does not preserve the paid ledger")
    _require(hpo.get("records_sha256") == evidence["file_hashes"]["candidate_records.jsonl"]
             and sorted(hpo.get("seen", [])) == sorted(record["identity"] for record in evidence["records"]),
             "Repaired checkpoint does not describe the complete preserved archive")
    repair_root = directory / REPAIR_DIR
    _require(not repair_root.is_symlink(), "Repair directory cannot be a symlink")
    repair_root.mkdir(exist_ok=True)
    transaction = repair_root / uuid.uuid4().hex
    transaction.mkdir()
    _fsync_directory(repair_root)
    backup = transaction / "original"
    backup.mkdir()
    for name, expected in evidence["file_hashes"].items():
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / name, target)
        _require(_sha(target) == expected, f"Evidence changed while backing up: {name}")
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
    for folder in sorted((path for path in backup.rglob("*") if path.is_dir()), reverse=True):
        _fsync_directory(folder)
    _fsync_directory(backup)
    import torch
    prepared = transaction / "repaired-checkpoint.pt"
    with prepared.open("xb") as stream:
        torch.save(repaired_state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = _load_checkpoint(prepared)
    _require(_same_state(restored, repaired_state), "Serialized repair failed verification")
    del restored
    new_hash = _sha(prepared)
    receipt = {"schema_version": 1, "operation": "single_observation_generation_checkpoint_repair",
               "directory": str(directory), "created_at": datetime.now(timezone.utc).isoformat(),
               "backup_directory": str(backup), "old_checkpoint_sha256": evidence["checkpoint_hash"],
               "new_checkpoint_sha256": new_hash, "evidence_sha256": evidence["file_hashes"],
               "external_evidence_sha256": evidence.get("external_hashes", {}), "details": details,
               "status": "PREPARED", "status_note": "Current checkpoint hash determines whether replacement completed"}
    _write_json(transaction / "receipt.json", receipt)
    _fsync_directory(transaction)
    _recheck(directory, evidence)
    os.replace(prepared, directory / PRIMARY)
    _fsync_directory(directory)
    _require(_sha(directory / PRIMARY) == new_hash, "Replacement checkpoint hash changed unexpectedly")
    _write_json(transaction / "applied.json", {"checkpoint_sha256": new_hash,
                "applied_at": datetime.now(timezone.utc).isoformat()})
    _fsync_directory(transaction)
    return {**receipt, "already_applied": False, "status": "APPLIED"}
