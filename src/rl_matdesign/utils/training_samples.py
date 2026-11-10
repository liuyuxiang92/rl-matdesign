"""Durable, generated-CSV-shaped records of budget-consuming training samples.

The caller supplies an immutable prediction snapshot taken during evaluation.
Training metrics only supply provenance: losses, discounted returns and other
learning statistics must never replace the material's predicted properties.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path


_LEADING_FIELDS = ("formula", "reward", "dp_mean")
_IDENTITY_FIELDS = (
    "phase", "episode", "iteration", "training_budget_used", "training_attempts",
    "terminal_comp_key", "budget_counted", "training_unique_candidates",
    "training_duplicate_episodes", "training_budget_history_complete",
    "training_samples_history_complete", "training_session_id",
)


def _counter(value, name: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"{name} must be a nonnegative integer")


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


class TrainingSampleWriter:
    """Write every counted episode, including duplicates, in collection order.

    ``start`` must be called after restoring the trainer's budget. Full resumes
    retain rows up to that checkpoint; any later rows are preserved in a sibling
    ``*.superseded-<unique>.csv`` before the active file is rewritten. A weights-
    only resume starts a new ``training_session_id`` and marks history incomplete
    instead of pretending its reset counters continue the previous session.

    Each record is flushed and fsynced before returning. A new prediction field
    expands the shared CSV schema with an atomic rewrite, preserving older rows.
    The writer is intended for one training process per output path.
    """

    def __init__(self, path, append: bool = False):
        self.path = Path(path)
        self.append = append
        self.session_id = uuid.uuid4().hex
        self.history_complete = True
        self._started = False
        self._last_budget = 0
        self._fields = [*_LEADING_FIELDS, *_IDENTITY_FIELDS]

    def _warn(self, message: str):
        warnings.warn(f"Training samples: {message}", RuntimeWarning, stacklevel=3)

    def _read(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return [], []
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
        if len(set(fields)) != len(fields) or any(None in row for row in rows):
            raise ValueError(f"Invalid training samples CSV schema: {self.path}")
        return fields, rows

    def _field_union(self, fields):
        properties = dict.fromkeys(
            field for field in (*self._fields, *fields)
            if field not in _LEADING_FIELDS and field not in _IDENTITY_FIELDS
        )
        return [*_LEADING_FIELDS, *properties, *_IDENTITY_FIELDS]

    def _rewrite(self, rows, fields):
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._fields = list(fields)

    def _archive(self):
        # Preserve the whole original file, including its original schema. This
        # is also safe when a fresh run replaces an older writer's CSV format.
        archive = self.path.with_name(
            f"{self.path.stem}.superseded-{uuid.uuid4().hex}{self.path.suffix}"
        )
        shutil.copyfile(self.path, archive)
        with archive.open("r+b") as handle:
            os.fsync(handle.fileno())
        self._warn(f"preserved previous CSV at {archive}")

    def start(self, budget_used: int, history_complete: bool = True,
              session_id: str | None = None):
        """Align history to the checkpoint's budget and, when known, session.

        Save ``session_id`` in training checkpoints and restore it here. Older
        checkpoints can infer identity only from a CSV containing one session;
        multiple sessions require a fresh, explicitly incomplete continuation.
        """
        if self._started:
            raise RuntimeError("TrainingSampleWriter.start() may only be called once")
        budget_used = _counter(budget_used, "budget_used")
        if type(history_complete) is not bool:
            raise ValueError("history_complete must be a boolean")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("session_id must be a nonempty string or None")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists and not self.append:
            self._archive()
            fields, rows = [], []
        else:
            fields, rows = self._read()
        self.history_complete = history_complete
        # CSVs written before session IDs are unambiguously one legacy session.
        # Preserve that provenance even when this start opens a different one.
        for row in rows:
            if not row.get("training_session_id"):
                row["training_session_id"] = "legacy"
        active = None
        if session_id is not None:
            self.session_id = session_id
            active = [row for row in rows if row["training_session_id"] == session_id]
            if not active:
                self.history_complete = False
                self._warn(
                    f"checkpoint session {session_id!r} has no prior sample rows; "
                    "new rows will retain that session with training_samples_history_complete=False"
                )
        elif not history_complete and budget_used == 0:
            self._warn(
                "budget history is incomplete; starting a separate training_session_id "
                "so reset counters cannot be confused with previous samples"
            )
        elif rows:
            sessions = {row["training_session_id"] for row in rows}
            if len(sessions) != 1:
                self.history_complete = False
                self._warn(
                    "CSV contains multiple sessions but the checkpoint has no session_id; "
                    "starting a separate training_session_id with "
                    "training_samples_history_complete=False instead of guessing its history"
                )
            else:
                self.session_id = next(iter(sessions))
                active = rows
        elif budget_used:
            self.history_complete = False
            self._warn(
                f"no prior sample rows for restored budget {budget_used}; "
                "new rows will have training_samples_history_complete=False"
            )

        if active:
            counters = [_counter(row.get("training_budget_used"), "CSV training_budget_used")
                        for row in active]
            if any(current <= previous for previous, current in zip([0, *counters], counters)):
                raise ValueError("Training samples budget counters must increase within a session")
            if counters[-1] > budget_used:
                self._archive()
                rows = [row for row in rows if row["training_session_id"] != self.session_id
                        or _counter(row["training_budget_used"], "CSV training_budget_used") <= budget_used]
                active = [row for row in rows if row["training_session_id"] == self.session_id]
                counters = [_counter(row["training_budget_used"], "CSV training_budget_used")
                            for row in active]
            self.history_complete = (
                history_complete and counters == list(range(1, budget_used + 1))
                and all(str(row.get("training_samples_history_complete", "True")).lower()
                        not in {"false", "0"} for row in active)
            )
            if not self.history_complete:
                self._warn(
                    f"CSV does not contain complete history through restored budget {budget_used}; "
                    "new rows will have training_samples_history_complete=False"
                )
        self._last_budget = budget_used
        self._rewrite(rows, self._field_union(fields))
        self._started = True

    def record(self, episode, metrics: dict):
        """Persist one successfully evaluated episode iff it consumed budget."""
        if not self._started:
            raise RuntimeError("Call TrainingSampleWriter.start() before record()")
        if metrics.get("budget_counted") is not True:
            return
        budget_used = _counter(metrics.get("training_budget_used"), "training_budget_used")
        if budget_used != self._last_budget + 1:
            raise ValueError(
                f"Training samples expected budget {self._last_budget + 1}, got {budget_used}; "
                "record every counted episode exactly once"
            )
        prediction = dict(getattr(episode, "prediction", {}) or {})
        metadata = getattr(episode, "metadata", {}) or {}
        row = {key: _csv_value(value) for key, value in prediction.items()
               if key not in _IDENTITY_FIELDS}
        row["formula"] = prediction.get("formula", getattr(episode, "formula", ""))
        if "reward" not in row:
            path = getattr(episode, "path", ())
            row["reward"] = float(path[-1].reward) if path else ""
        row.setdefault("dp_mean", "")
        for key in _IDENTITY_FIELDS:
            if key in metrics:
                row[key] = _csv_value(metrics[key])
            elif key in metadata:
                row[key] = _csv_value(metadata[key])
        row.setdefault("terminal_comp_key", str(getattr(episode, "terminal_key", "")))
        row.update(
            training_budget_used=budget_used,
            budget_counted=True,
            training_samples_history_complete=self.history_complete,
            training_session_id=self.session_id,
        )
        fields = self._field_union(row)
        if fields != self._fields:
            _, previous = self._read()
            self._rewrite([*previous, row], fields)
        else:
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
        self._last_budget = budget_used
