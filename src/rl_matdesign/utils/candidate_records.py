"""Opt-in HPO candidate archive built exclusively from evaluation snapshots."""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
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
    return {"identity": candidate_identity(episode.terminal_key), "phase": phase,
            "formula": episode.formula, "properties": properties,
            "constraint_passed": all(constraints), "prediction_record": raw}


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
        if resume_state is not None:
            self._restore(resume_state)
        elif self.records_path.exists() or self.budget_path.exists():
            raise ValueError("HPO candidate artifacts already exist; reuse the completed job or preserve this interrupted job and start a distinct job. Unsafe in-place restart is refused.")
        else:
            self.records_path.touch()
        self.flush()

    def checkpoint_state(self):
        if self.budget.pending:
            raise RuntimeError("Cannot checkpoint HPO accounting with pending evaluations")
        return {"budget": self.budget.snapshot(), "phase": self.phase,
                "seen": sorted(self.budget.seen),
                "records_sha256": hashlib.sha256(self.records_path.read_bytes()).hexdigest()}

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
        if hashlib.sha256(self.records_path.read_bytes()).hexdigest() != state.get("records_sha256"):
            raise ValueError("HPO candidate archive differs from synchronized checkpoint")
        rows = [json.loads(line) for line in self.records_path.read_text().splitlines()]
        seen = set(state.get("seen", ()))
        if (len(rows) != saved["unique_evaluated"] or len(seen) != len(rows)
                or seen != {row["identity"] for row in rows}):
            raise ValueError("HPO checkpoint identities differ from candidate archive")
        self.budget.seen = seen
        self.budget.attempts_started = saved["attempts_started"]
        self.budget.failures = saved["failed_attempts"]
        self.budget.preparation_only = saved["preparation_only_attempts"]

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
        fresh = self.budget.settle(record["identity"])
        if fresh:
            record.update(attempts_started=self.budget.attempts_started,
                          unique_evaluated=self.budget.unique_evaluated)
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
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
