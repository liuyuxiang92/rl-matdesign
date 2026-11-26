"""Opt-in HPO candidate archive built exclusively from evaluation snapshots."""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

from .candidate_budget import CandidateBudget, NumericalTrainingError


def candidate_identity(value):
    """Preserve canonical key types; never round or identify by display formula."""
    def typed(item):
        if item is None or isinstance(item, (str, bool, int)):
            return [type(item).__name__, item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Candidate identity contains nonfinite number")
            return ["float", item.hex()]
        if isinstance(item, (tuple, list)):
            return [type(item).__name__, [typed(v) for v in item]]
        if isinstance(item, dict):
            pairs = [[typed(k), typed(v)] for k, v in item.items()]
            return ["dict", sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
        raise TypeError(f"Unsupported canonical candidate key type: {type(item).__name__}")
    return json.dumps(typed(value), ensure_ascii=False, separators=(",", ":"))


def episode_record(episode, phase):
    raw = dict(episode.prediction_record)
    properties = {}
    for key, value in raw.items():
        if key.startswith("obj_") and key.endswith("_mean"):
            properties[key[4:-5]] = value
    # Single-property predictors use dp_mean; it remains explicitly selectable.
    if "dp_mean" in raw:
        properties["dp_mean"] = raw["dp_mean"]
    for name, value in properties.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Nonfinite or nonnumeric property {name}: {value!r}")
    constraints = [bool(value) for name, value in raw.items()
                   if name in {"gate_passed", "primary_ok", "constraint_passed"}
                   or (name.startswith("gate_") and name.endswith("_passed"))]
    sampling = {name: deepcopy(value) for name, value in getattr(episode, "metadata", {}).items()
                if name.startswith("exploration_")}
    return {"identity": candidate_identity(episode.terminal_key), "phase": phase,
            "formula": episode.formula, "properties": properties,
            "constraint_passed": all(constraints), "prediction_record": raw,
            **sampling}


class CandidateRecorder:
    """One phase per fresh job; no unsafe reset or weights-only resume."""
    def __init__(self, out, target_unique, ratio_limit=1.5, phase="train", resume_state=None):
        self.path = Path(out)
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "candidate_records.jsonl"
        self.budget_path = self.path / "candidate_budget.json"
        self.budget = CandidateBudget(target_unique, ratio_limit)
        self.phase = phase
        self.partial_checkpoint_factory = None
        self.constraint_check = None
        self._records_hash = hashlib.sha256()
        self._records_signature = None
        if resume_state is not None:
            self._restore(resume_state)
        elif self.records_path.exists() or self.budget_path.exists():
            raise ValueError("HPO candidate artifacts already exist; reuse the completed job or preserve this interrupted job and start a distinct job. Unsafe in-place restart is refused.")
        else:
            self.records_path.touch()
            self._records_signature = self._archive_signature()
        self.flush()

    def _archive_signature(self, stat=None):
        stat = self.records_path.stat() if stat is None else stat
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _check_archive_unchanged(self, stat=None):
        # The recorder is the archive's sole writer. Detect replacement,
        # truncation and external edits without rereading its entire history.
        try:
            signature = self._archive_signature(stat)
        except FileNotFoundError as exc:
            raise ValueError("HPO candidate archive changed outside the recorder") from exc
        if signature != self._records_signature:
            raise ValueError("HPO candidate archive changed outside the recorder")

    def checkpoint_state(self):
        if self.budget.pending:
            raise RuntimeError("Cannot checkpoint HPO accounting with pending evaluations")
        self._check_archive_unchanged()
        return {"budget": self.budget.snapshot(), "phase": self.phase,
                "seen": sorted(self.budget.seen),
                "records_sha256": self._records_hash.hexdigest()}

    def _restore(self, state):
        if not self.records_path.exists() or not self.budget_path.exists():
            raise ValueError("Exact HPO resume requires the original budget and candidate archive")
        disk = json.loads(self.budget_path.read_text())
        saved = state.get("budget", {})
        if state.get("phase") != self.phase or disk.get("phase") != self.phase:
            raise ValueError("HPO checkpoint phase mismatch")
        if (saved.get("target_unique") != self.budget.target_unique
                or str(saved.get("ratio_limit")) != str(float(self.budget.ratio_limit))):
            raise ValueError("HPO resume cannot change budget or efficiency limit")
        if disk.get("status") in {"COMPLETED", "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}:
            raise ValueError("Terminal HPO jobs cannot be resumed; reuse their result")
        compare = ("attempts_started", "unique_evaluated", "target_unique", "max_attempts",
                   "pending", "failed_attempts", "preparation_only_attempts")
        if any(disk.get(k) != saved.get(k) for k in compare) or saved.get("pending") != 0:
            raise ValueError("HPO accounting advanced beyond checkpoint; refusing to erase already spent attempts")
        # Verify legacy checkpoints against the original bytes once. Continue
        # the same digest as new records are durably appended after restart.
        signature = self._archive_signature()
        archived = self.records_path.read_bytes()
        records_hash = hashlib.sha256(archived)
        if (self._archive_signature() != signature
                or records_hash.hexdigest() != state.get("records_sha256")):
            raise ValueError("HPO candidate archive differs from synchronized checkpoint")
        rows = [json.loads(line) for line in archived.decode("utf-8").splitlines()]
        seen = set(state.get("seen", ()))
        if (len(rows) != saved["unique_evaluated"] or len(seen) != len(rows)
                or seen != {row["identity"] for row in rows}):
            raise ValueError("HPO checkpoint identities differ from candidate archive")
        self.budget.seen = seen
        self.budget.attempts_started = saved["attempts_started"]
        self.budget.failures = saved["failed_attempts"]
        self.budget.preparation_only = saved["preparation_only_attempts"]
        self._records_hash = records_hash
        self._records_signature = signature

    def flush(self):
        temporary = self.budget_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump({**self.budget.snapshot(), "phase": self.phase}, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.budget_path)

    def wave_limit(self, requested):
        try:
            return self.budget.wave_limit(requested)
        finally:
            self.flush()

    def start(self):
        try:
            self.budget.start()
        finally:
            self.flush()

    def failure(self, *, preparation=False):
        self.budget.settle(preparation=preparation)
        self.flush()

    def success(self, episode):
        record = episode_record(episode, self.phase)
        if self.constraint_check is not None:
            passed = bool(self.constraint_check(episode))
            record["post_constraint_passed"] = passed
            record["constraint_passed"] = record["constraint_passed"] and passed
            # The learner consumes this same completed snapshot after the
            # recorder. Carry the already-computed postfilter result forward;
            # exploration feedback must not re-run a scientific check/oracle.
            episode.metadata["post_constraint_passed"] = passed
        fresh = self.budget.settle(record["identity"])
        if fresh:
            record.update(attempts_started=self.budget.attempts_started,
                          unique_evaluated=self.budget.unique_evaluated)
            # Match the UTF-8 text archive's existing native newline bytes.
            encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + os.linesep).encode("utf-8")
            with self.records_path.open("ab") as handle:
                self._check_archive_unchanged(os.fstat(handle.fileno()))
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
                signature = self._archive_signature(os.fstat(handle.fileno()))
            if signature[2] != self._records_signature[2] + len(encoded):
                raise ValueError("HPO candidate archive changed outside the recorder")
            self._records_hash.update(encoded)
            self._records_signature = signature
        self.flush()
        return fresh

    def close(self):
        error_type, error, _ = sys.exc_info()
        if error is not None and self.budget.status == "RUNNING":
            if isinstance(error, NumericalTrainingError):
                self.budget.status = "INFEASIBLE_NUMERICAL"
            else:
                self.budget.status = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "SYSTEM_ERROR"
        self.budget.finish()
        self.flush()
        if self.budget.status != "COMPLETED" and self.partial_checkpoint_factory is not None:
            # Diagnostic snapshot only: a failed wave may contain evaluated
            # trajectories which have not yet received their learning update.
            # Never present this file as a safe training resume checkpoint.
            try:
                import torch
                temporary = self.path / "partial_checkpoint.tmp"
                torch.save({**self.partial_checkpoint_factory(),
                            "resume_safe": False, "hpo_budget": self.budget.snapshot()}, temporary)
                os.replace(temporary, self.path / "partial_checkpoint.pt")
            except Exception as checkpoint_error:
                print(f"[WARN] Could not preserve diagnostic partial checkpoint: {checkpoint_error}")
