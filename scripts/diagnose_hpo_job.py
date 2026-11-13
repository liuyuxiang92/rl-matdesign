#!/usr/bin/env python3
"""Read-only artifact diagnosis for one trusted local HPO job.

Loads only this job's phase checkpoint and its .tmp sibling with PyTorch on CPU.
PyTorch checkpoints may contain pickle: use only your own trusted local outputs.
Stop the campaign and workers first. No campaign or predictor is constructed,
no RNG is restored, and no file is repaired, promoted, removed or rewritten.
Unpickling cached values may import their classes from the installed RL package;
use the same Python environment as the CLI. This script does not alter sys.path.
Exit 0 means the inspection completed, NOT that restart or recovery is safe;
exit 2 means required evidence was missing, malformed or unreadable.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

COUNTERS = ("attempts_started", "unique_evaluated", "target_unique", "max_attempts",
            "pending", "failed_attempts", "preparation_only_attempts")
BUDGET_FIELDS = (*COUNTERS, "ratio_limit", "status", "ratio")
LOAD_FLAGS = {"policy.pt": "--load-policy", "qnet.pt": "--load-qnet",
              "value_net.pt": "--load-value-net", "std_scaler.bin": "--load-scaler"}


def _error(report, location, exc):
    report["errors"].append({"location": location, "type": type(exc).__name__,
                             "message": str(exc)[:800]})


def _small(value, depth=0):
    """Bound error/metadata output; never stringify tensors or raw state objects."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:800] + ("..." if len(value) > 800 else "")
    if depth > 3:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): _small(v, depth + 1) for k, v in list(value.items())[:25]}
    if isinstance(value, (list, tuple, set)):
        return [_small(v, depth + 1) for v in list(value)[:25]]
    return {"type": type(value).__name__}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path, report, *, required=False):
    row = {"path": str(path), "exists": path.exists(), "symlink": path.is_symlink()}
    try:
        if not row["exists"]:
            if required:
                raise FileNotFoundError(str(path))
            return row
        stat = path.stat()
        row.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=_sha(path))
        if row["symlink"]:
            row["resolved_path"] = str(path.resolve())
    except Exception as exc:
        _error(report, path.name, exc)
        row["inspection_error"] = str(exc)[:800]
    return row


def _json(path, report, *, required=False):
    if not path.exists():
        if required:
            _error(report, path.name, FileNotFoundError(str(path)))
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except Exception as exc:
        _error(report, path.name, exc)
        return None


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _state(path, directory, report):
    if not path.exists():
        return None
    try:
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Checkpoint symlink escapes the selected job; not loaded")
        import torch
        return _mapping(torch.load(path, map_location="cpu", weights_only=False), "Checkpoint")
    except Exception as exc:
        _error(report, path.name, exc)
        return None


def _compare(ledger, budget):
    result = {"ledger": {k: _small(ledger.get(k)) for k in BUDGET_FIELDS},
              "checkpoint": {k: _small(budget.get(k)) for k in BUDGET_FIELDS}}
    result["differing_keys"] = [k for k in BUDGET_FIELDS if ledger.get(k) != budget.get(k)]
    result["ledger_minus_checkpoint"] = {
        k: ledger[k] - budget[k] for k in COUNTERS
        if type(ledger.get(k)) is int and type(budget.get(k)) is int}
    return result


def _frozen(contract, execution, directory, report):
    artifacts = _mapping(contract.get("frozen_artifacts"), "generation_contract.frozen_artifacts")
    argv = execution.get("argv", [])
    argv_available = bool(argv)
    if not isinstance(argv, list) or any(not isinstance(x, str) for x in argv):
        raise ValueError("execution.json argv must be a string list")
    result = {}
    for name, expected in artifacts.items():
        row = {"expected_sha256": _small(expected), "path": None}
        result[str(name)] = row
        flag = LOAD_FLAGS.get(name)
        path = None
        if flag and flag in argv:
            index = len(argv) - 1 - argv[::-1].index(flag)
            if index + 1 < len(argv):
                path = Path(argv[index + 1]).expanduser()
                row["path_source"] = "execution.json " + flag
        elif isinstance(name, str) and Path(name).is_absolute():
            path = Path(name)
            row["path_source"] = "generation_contract"
        elif argv_available and isinstance(name, str) and Path(name).name == name:
            path = directory / name
            row["path_source"] = "worker default (no load flag recorded)"
        if path is None or not path.is_absolute():
            row["recorded_path"] = str(path) if path is not None else None
            row["inspection_note"] = "Path unavailable or relative; worker working directory is not inferred"
            continue
        row.update(_file(path, report))
        row["sha256_matches"] = row.get("sha256") == expected if row["exists"] else False
    return result


def _checkpoint_summary(state, ledger, execution, directory, report, name):
    row = {"loaded": state is not None}
    if state is None:
        return row
    row.update(type=_small(state.get("type")), resume_safe=_small(state.get("resume_safe")))
    try:
        hpo = _mapping(state.get("hpo_state"), "hpo_state")
        budget = _mapping(hpo.get("budget"), "hpo_state.budget")
        row.update(phase=_small(hpo.get("phase")), phase_matches_ledger=hpo.get("phase") == ledger.get("phase"),
                   budget_comparison=_compare(ledger, budget),
                   records_sha256=_small(hpo.get("records_sha256")))
        row["seen_count"] = len(hpo["seen"]) if isinstance(hpo.get("seen"), (list, tuple)) else None
    except Exception as exc:
        _error(report, name + ":hpo_state", exc)
    rng = state.get("rng_state")
    row["rng"] = {"is_mapping": isinstance(rng, dict),
                  "schema_version": _small(rng.get("schema_version")) if isinstance(rng, dict) else None,
                  "fields_present": {key: isinstance(rng, dict) and rng.get(key) is not None
                                     for key in ("python", "numpy", "torch_cpu", "torch_cuda")}}
    try:
        predictor = _mapping(state.get("predictor_ledger"), "predictor_ledger")
        row["predictor_ledger"] = {k: _small(predictor.get(k)) for k in
                                   ("schema_version", "n_calls", "t_predict_s", "phase")}
        row["predictor_ledger"]["seen_count"] = len(predictor["seen"]) if isinstance(predictor.get("seen"), (list, tuple, set)) else None
        costs = _mapping(predictor.get("phase_costs"), "predictor_ledger.phase_costs")
        row["predictor_ledger"]["phase_costs"] = {
            str(phase): {str(k): _small(v) for k, v in _mapping(cost, "phase cost").items() if k != "seen"}
            for phase, cost in costs.items()}
    except Exception as exc:
        _error(report, name + ":predictor_ledger", exc)
    if state.get("type") == "hpo_generation" or ledger.get("phase") == "generation":
        try:
            contract = _mapping(state.get("generation_contract"), "generation_contract")
            row["generation_contract"] = {
                "execution_present": isinstance(contract.get("execution"), dict),
                "sampling": _small(contract.get("sampling")),
                "frozen_artifacts": _frozen(contract, execution, directory, report)}
        except Exception as exc:
            _error(report, name + ":generation_contract", exc)
    return row


def _archive(path, states, ledger, report):
    """Stream exact raw bytes once; prefixes are whole archive lines, not reserialized JSON."""
    row = {"physical_lines": 0, "valid_records": 0, "unique_identities": 0,
           "duplicate_identities": 0, "parse_error_count": 0, "parse_errors": [],
           "ends_with_newline": None, "truncated_tail_suspected": False, "checkpoint_comparisons": {}}
    goals = {}
    for name, state in states.items():
        hpo = state.get("hpo_state") if isinstance(state, dict) else None
        budget = hpo.get("budget") if isinstance(hpo, dict) else None
        n = budget.get("unique_evaluated") if isinstance(budget, dict) else None
        if type(n) is int and n >= 0:
            goals[name] = (n, hpo)
    digest, identities = hashlib.sha256(), set()
    last_counters = deque(maxlen=3)
    prefixes = {0: (digest.hexdigest(), set(), 0)}
    targets = {n for n, _ in goals.values()}
    last_invalid = False
    try:
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row["physical_lines"] += 1
                row["ends_with_newline"] = line.endswith(b"\n")
                last_invalid = False
                try:
                    record = _mapping(json.loads(line), "Candidate record")
                    identity = record.get("identity")
                    if not isinstance(identity, str):
                        raise ValueError("Candidate record has no string identity")
                    row["valid_records"] += 1
                    last_counters.append({k: _small(record.get(k)) for k in
                                          ("attempts_started", "unique_evaluated")})
                    row["duplicate_identities"] += identity in identities
                    identities.add(identity)
                except Exception as exc:
                    last_invalid = True
                    row["parse_error_count"] += 1
                    if len(row["parse_errors"]) < 5:
                        row["parse_errors"].append({"line": row["physical_lines"], "type": type(exc).__name__, "message": str(exc)[:240]})
                if row["physical_lines"] in targets:
                    prefixes[row["physical_lines"]] = (digest.hexdigest(), identities.copy(), row["parse_error_count"])
        row.update(sha256=digest.hexdigest(), unique_identities=len(identities),
                   last_record_counters=list(last_counters),
                   truncated_tail_suspected=last_invalid and row["ends_with_newline"] is False,
                   record_count_matches_ledger=row["valid_records"] == ledger.get("unique_evaluated"),
                   unique_count_matches_ledger=len(identities) == ledger.get("unique_evaluated"))
        if row["parse_error_count"]:
            _error(report, path.name, ValueError(f'{row["parse_error_count"]} malformed candidate record(s); bytes preserved'))
        for name, (n, hpo) in goals.items():
            prefix = prefixes.get(n)
            seen = hpo.get("seen")
            valid_seen = isinstance(seen, (list, tuple)) and all(isinstance(x, str) for x in seen)
            exact_seen = valid_seen and len(set(seen)) == len(seen) == n
            row["checkpoint_comparisons"][name] = {
                "checkpoint_unique_evaluated": n,
                "whole_archive_sha256_matches": row["sha256"] == hpo.get("records_sha256"),
                "whole_archive_seen_matches": bool(exact_seen and set(seen) == identities and not row["parse_error_count"] and row["physical_lines"] == n),
                "prefix_lines_available": prefix is not None,
                "prefix_sha256": prefix[0] if prefix else None,
                "prefix_sha256_matches": bool(prefix and prefix[0] == hpo.get("records_sha256")),
                "prefix_seen_matches": bool(prefix and exact_seen and set(seen) == prefix[1] and not prefix[2]),
                "lines_after_checkpoint_prefix": max(0, row["physical_lines"] - n)}
    except Exception as exc:
        _error(report, path.name, exc)
    return row


def _tail(path, report, *, max_bytes=8192, lines=30):
    if not path.exists():
        return {"exists": False}
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            tail = stream.read(max_bytes).decode("utf-8", errors="replace").splitlines()
        return {"exists": True, "truncated": size > max_bytes or len(tail) > lines,
                "lines": tail[-lines:]}
    except Exception as exc:
        _error(report, path.name, exc)
        return {"inspection_error": str(exc)[:800]}


def _evaluation_log(path, report):
    if not path.exists():
        return {"exists": False}
    row, last = {"exists": True, "rows": 0, "parse_errors": 0}, deque(maxlen=3)
    try:
        with path.open("rb") as stream:
            for line in stream:
                row["rows"] += 1
                try:
                    item = _mapping(json.loads(line), "Evaluation row")
                    last.append({k: _small(v) for k, v in item.items()
                                 if k in {"phase", "request", "invocation", "episode", "status", "unique",
                                          "new_successful_candidate", "new_successful_unique", "cache_hit",
                                          "attempts_started", "unique_evaluated", "n_calls", "wall_seconds",
                                          "elapsed_s", "t_predict_s", "wall_elapsed_s", "prediction_elapsed_s"}})
                except Exception:
                    row["parse_errors"] += 1
        row["last_rows"] = list(last)
        row["order_note"] = "Observation/timing logs may lag durable candidate records; row order does not prove sampling order"
        if row["parse_errors"]:
            _error(report, path.name, ValueError(f'{row["parse_errors"]} malformed evaluation row(s)'))
    except Exception as exc:
        _error(report, path.name, exc)
    return row



def _episode_timing(path, report):
    if not path.exists():
        return {"exists": False}
    row, last = {"exists": True, "rows": 0}, deque(maxlen=3)
    fields = {"timing_session", "episode", "phase", "status", "evaluation_wave_id",
              "evaluation_wave_size", "rollout_s", "eval_wall_share_s", "rl_update_s",
              "episode_wall_share_s", "md_task_s", "opt_task_s", "structure_other_task_s",
              "property_s", "config_attempted", "config_succeeded", "timing_incomplete",
              "wall_elapsed_s", "cache_hit"}
    try:
        with path.open(newline="") as stream:
            for item in csv.DictReader(stream):
                row["rows"] += 1
                last.append({k: _small(v) for k, v in item.items() if k in fields})
        row["last_rows"] = list(last)
        row["order_note"] = "Batch/wave timing can lag observations; episode is not a per-attempt ordering proof"
    except Exception as exc:
        _error(report, path.name, exc)
    return row


def inspect_job(out, job_id):
    report = {"schema_version": 1, "read_only": True, "inspection_complete": False,
              "out": str(Path(out).expanduser().resolve()), "job_id": job_id, "python": sys.executable, "errors": [],
              "scope": "Evidence only: a matching prefix or .tmp file does not establish recoverability; no recovery attempted"}
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id):
        _error(report, "job_id", ValueError("job-id must be one safe directory name, without path separators"))
        return report
    directory = Path(report["out"]) / "jobs" / job_id
    report["jobdir"] = str(directory)
    if not directory.is_dir() or directory.is_symlink() or not directory.resolve().is_relative_to(Path(report["out"])):
        _error(report, "jobdir", ValueError("Selected job is absent, a symlink, or outside the campaign"))
        return report
    metadata = {name: _json(directory / name, report, required=name == "candidate_budget.json")
                for name in ("job.json", "result.json", "execution.json", "candidate_budget.json")}
    report["metadata"] = {name: {k: _small(v) for k, v in (data or {}).items() if k in {
        "phase", "method", "seed", "budget", "status", "returncode", "error", "errors", "reason",
        "execution_failure", "finalization_pending", "resume", "started_at", "finished_at", "wall_seconds",
        "eligible_for_selection", "input_models"}} for name, data in metadata.items()}
    ledger = metadata["candidate_budget.json"] or {}
    report["ledger"] = {k: _small(ledger.get(k)) for k in (*BUDGET_FIELDS, "phase")}
    phase = ledger.get("phase") or (metadata["job.json"] or {}).get("phase")
    if phase not in ("train", "generation"):
        _error(report, "phase", ValueError("No valid train/generation phase; checkpoint not loaded"))
        return report
    checkpoint = "generation_checkpoint.pt" if phase == "generation" else "checkpoint.pt"
    names = (checkpoint, checkpoint + ".tmp")
    report["files"] = {name: _file(directory / name, report, required=name == checkpoint)
                       for name in (*metadata, "candidate_records.jsonl", *names, "stderr.log", "stdout.log",
                                    "generation_evaluations.jsonl", "generation_episode_timing.csv")}
    states = {}
    execution = metadata["execution.json"] or {}
    report["checkpoints"] = {}
    for name in names:
        state = _state(directory / name, directory, report)
        report["checkpoints"][name] = _checkpoint_summary(state, ledger, execution, directory, report, name)
        # Replay buffers/caches can be large. Keep only archive comparison state
        # before loading another checkpoint; never retain two complete payloads.
        states[name] = {"hpo_state": state.get("hpo_state")} if state is not None else None
        del state
    report["archive"] = _archive(directory / "candidate_records.jsonl", states, ledger, report)
    report["stderr_tail"] = _tail(directory / "stderr.log", report)
    report["stdout_tail"] = _tail(directory / "stdout.log", report)
    report["generation_episode_timing"] = _episode_timing(directory / "generation_episode_timing.csv", report)
    report["generation_evaluations"] = _evaluation_log(directory / "generation_evaluations.jsonl", report)
    package = sys.modules.get("rl_matdesign")
    report["loaded_rl_package"] = getattr(package, "__file__", None)
    report["inspection_complete"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing HPO output directory")
    parser.add_argument("--job-id", required=True, help="One job directory name under OUT/jobs")
    args = parser.parse_args(argv)
    with redirect_stdout(sys.stderr):
        report = inspect_job(args.out, args.job_id)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["inspection_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
