"""Episode-budget bookkeeping shared by training algorithms.

Only successful, completed rollouts should be recorded. Duplicate identity is the
environment's canonical terminal composition key, independent of action order,
reward caching, or the learning updates performed for that rollout.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class TrainingBudget:
    """Track successful rollouts and optionally count only new compositions.

    ``record`` never enforces the duplicate limit, so callers can learn from
    every successful rollout, including the last allowed duplicate. Call
    ``check_can_sample`` before requesting another rollout if the phase still
    has budget remaining. In unique-only mode, exactly ``max_duplicate_attempts``
    consecutive duplicate rollouts are allowed; attempting the next rollout
    raises. A new composition resets the streak. The default mode counts every
    rollout and does not impose this guard.

    A restored state must use the same counting mode once any rollouts have been
    recorded. The duplicate limit can be changed on resume, for example to allow
    more attempts after the guard stopped a run. Keys are retained as hashable
    Python objects, including nested tuples, for lossless torch checkpoints.
    """

    def __init__(
        self,
        count_duplicates: bool = True,
        max_duplicate_attempts: int = 1000,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        if type(count_duplicates) is not bool:
            raise ValueError("train_count_duplicates must be a YAML boolean (true or false)")
        _nonnegative_int(max_duplicate_attempts, "train_max_duplicate_attempts")
        if max_duplicate_attempts == 0:
            raise ValueError("train_max_duplicate_attempts must be a positive integer")
        self.count_duplicates = count_duplicates
        self.max_duplicate_attempts = max_duplicate_attempts
        self.attempted = 0
        self.counted = 0
        self.duplicates = 0
        self.consecutive_duplicates = 0
        self._seen_keys: dict[Hashable, None] = {}
        if state is not None:
            self._restore(state)

    @property
    def unique(self) -> int:
        return len(self._seen_keys)

    def record(self, key: Hashable) -> bool:
        """Record one successful rollout and return whether it consumes budget."""
        try:
            duplicate = key in self._seen_keys
        except TypeError as exc:
            raise ValueError("training composition keys must be hashable") from exc
        self.attempted += 1
        if duplicate:
            self.duplicates += 1
            self.consecutive_duplicates += 1
        else:
            self._seen_keys[key] = None
            self.consecutive_duplicates = 0
        counted = self.count_duplicates or not duplicate
        self.counted += int(counted)
        return counted

    def check_can_sample(self) -> None:
        """Stop an unfinished unique-only run after its duplicate retry limit."""
        if (
            not self.count_duplicates
            and self.consecutive_duplicates >= self.max_duplicate_attempts
        ):
            raise RuntimeError(
                f"Training produced {self.consecutive_duplicates} consecutive duplicate "
                f"episodes without a new composition ({self.unique} unique candidates "
                f"from {self.attempted} successful rollouts). These duplicate episodes "
                "were available for training but did not consume episode budget. "
                "Increase exploration, reduce the requested unique-episode budget, "
                "increase train_max_duplicate_attempts, or start a new run with "
                "train_count_duplicates=true. "
                "The policy and budget mode have not been changed automatically."
            )

    def snapshot(self) -> dict[str, int]:
        """Return cumulative counters suitable for training logs."""
        return {
            "training_attempts": self.attempted,
            "training_budget_used": self.counted,
            "training_unique_candidates": self.unique,
            "training_duplicate_episodes": self.duplicates,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint payload without sharing the mutable key container."""
        return {
            "version": 1,
            "count_duplicates": self.count_duplicates,
            "max_duplicate_attempts": self.max_duplicate_attempts,
            "seen_keys": list(self._seen_keys),
            "attempted": self.attempted,
            "counted": self.counted,
            "unique": self.unique,
            "duplicates": self.duplicates,
            "consecutive_duplicates": self.consecutive_duplicates,
        }

    def _restore(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("training budget state must be a mapping")
        required = set(self.state_dict())
        missing = required.difference(state)
        if missing:
            raise ValueError(f"training budget state is missing: {', '.join(sorted(missing))}")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported training budget state version")
        previous_mode = state["count_duplicates"]
        if type(previous_mode) is not bool:
            raise ValueError("training budget state's count_duplicates must be a boolean")
        previous_limit = state["max_duplicate_attempts"]
        _nonnegative_int(previous_limit, "training budget state's max_duplicate_attempts")
        if previous_limit == 0:
            raise ValueError("training budget state's max_duplicate_attempts must be positive")
        counters = {
            name: _nonnegative_int(state[name], f"training budget state's {name}")
            for name in ("attempted", "counted", "unique", "duplicates", "consecutive_duplicates")
        }
        keys = state["seen_keys"]
        if not isinstance(keys, (list, tuple)):
            raise ValueError("training budget state's seen_keys must be a list or tuple")
        try:
            seen_keys = dict.fromkeys(keys)
        except TypeError as exc:
            raise ValueError("training budget state's composition keys must be hashable") from exc
        if len(seen_keys) != len(keys):
            raise ValueError("training budget state's seen_keys contains duplicate keys")
        attempted, counted, unique, duplicates, streak = (
            counters[name]
            for name in ("attempted", "counted", "unique", "duplicates", "consecutive_duplicates")
        )
        if (
            unique != len(seen_keys)
            or attempted != unique + duplicates
            or counted != (attempted if previous_mode else unique)
            or streak > duplicates
            or (attempted > 0 and unique == 0)
        ):
            raise ValueError("training budget state has inconsistent counters or seen keys")
        if attempted and previous_mode != self.count_duplicates:
            raise ValueError(
                "Cannot change train_count_duplicates when resuming a nonempty training budget; "
                "use the checkpoint's counting mode or start a new run."
            )
        self._seen_keys = seen_keys
        self.attempted = attempted
        self.counted = counted
        self.duplicates = duplicates
        self.consecutive_duplicates = streak
