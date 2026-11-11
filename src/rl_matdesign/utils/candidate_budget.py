"""Strict HPO sampling accounting, separate from the learning episode clock."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING


class EfficiencyLimitReached(RuntimeError):
    """Even perfectly productive remaining attempts cannot meet the strict cap."""


class NumericalTrainingError(RuntimeError):
    """A trial produced nonfinite training losses or model parameters."""


class CandidateBudget:
    def __init__(self, target_unique: int, ratio_limit=1.5):
        if type(target_unique) is not int or target_unique < 1:
            raise ValueError("target_unique must be a positive integer")
        ratio = Decimal(str(ratio_limit))
        if not ratio.is_finite() or ratio <= 1:
            raise ValueError("ratio_limit must be finite and greater than 1")
        self.target_unique = target_unique
        self.ratio_limit = str(ratio)
        self.max_attempts = int((ratio * target_unique).to_integral_value(rounding=ROUND_CEILING)) - 1
        self.attempts_started = 0
        self.pending = 0
        self.seen = set()
        self.status = "RUNNING"
        self.failures = 0
        self.preparation_only = 0

    @property
    def unique_evaluated(self):
        return len(self.seen)

    def check_can_start(self):
        if self.status != "RUNNING":
            raise RuntimeError(f"Cannot sample from budget in state {self.status}")
        if self.pending:
            return  # Check impossibility only after the evaluation wave settles.
        if self.attempts_started + self.target_unique - self.unique_evaluated > self.max_attempts:
            self.status = "INFEASIBLE_EFFICIENCY"
            raise EfficiencyLimitReached("Strict attempts/unique limit is no longer attainable")
        if self.unique_evaluated >= self.target_unique:
            raise RuntimeError("Candidate target already reached")

    def wave_limit(self, requested: int):
        self.check_can_start()
        return min(requested, self.target_unique - self.unique_evaluated,
                   self.max_attempts - self.attempts_started)

    def start(self):
        self.check_can_start()
        if self.attempts_started >= self.max_attempts:
            raise RuntimeError("Attempt reservation exceeded strict cap")
        self.attempts_started += 1
        self.pending += 1

    def settle(self, identity=None, *, preparation=False):
        if self.pending < 1:
            raise RuntimeError("No started attempt to settle")
        self.pending -= 1
        if preparation:
            self.preparation_only += 1
            return False
        if identity is None:
            self.failures += 1
            return False
        fresh = identity not in self.seen
        if fresh:
            if self.unique_evaluated >= self.target_unique:
                raise RuntimeError("Evaluation wave exceeded unique budget")
            self.seen.add(identity)
        return fresh

    def snapshot(self):
        return {"version": 1, "status": self.status,
                "attempts_started": self.attempts_started,
                "unique_evaluated": self.unique_evaluated,
                "target_unique": self.target_unique,
                "ratio_limit": float(self.ratio_limit), "max_attempts": self.max_attempts,
                "pending": self.pending, "failed_attempts": self.failures,
                "preparation_only_attempts": self.preparation_only,
                "ratio": self.attempts_started / self.unique_evaluated if self.unique_evaluated else None}

    def finish(self):
        if self.status == "RUNNING":
            if self.pending:
                self.status = "SYSTEM_ERROR"
            elif self.unique_evaluated == self.target_unique:
                self.status = "COMPLETED"
            elif self.attempts_started + self.target_unique - self.unique_evaluated > self.max_attempts:
                self.status = "INFEASIBLE_EFFICIENCY"
            else:
                self.status = "INCOMPLETE"
        return self.snapshot()
