"""Masked entropy sampling and checkpointable exploration feedback.

The outer controller uses sequential observations, not causal policy probes. It
never changes scientific rewards or calls an evaluator. Its shared constants are
an experimental recipe, not a claim of optimal exploration or convergence.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from numbers import Real
from typing import Any

import numpy as np


EXPLORATION_PROFILE = "adaptive_entropy_v1"


def exploration_profile() -> dict[str, Any]:
    """Return every numerical and semantic choice affecting the controller."""
    return {
        "name": EXPLORATION_PROFILE,
        "version": 1,
        "entropy": "mean_per_unforced_decision_H_over_log_legal_count",
        "temperature_iterations": 64,
        "temperature_max_logit_gap": 1024.0,
        "score_resolution_relative": 32.0 * np.finfo(np.float64).eps,
        "score_span_over_scale_floor": 1e-8,
        "target_min": 0.0,
        "target_max": 1.0,
        "target_step": 0.05,
        "coefficient_initial": 0.01,
        "coefficient_min": 1e-6,
        "coefficient_max": 10.0,
        "coefficient_log_rate": 0.05,
        "window_attempts": 64,
        "window_closure": "at_collection_block_end_after_minimum_attempts",
        "cooldown_windows": 2,
        "sustained_windows": 2,
        "low_novelty": 0.2,
        "productive_novelty": 0.5,
        "min_success_rate": 0.5,
        "min_qualified_quality_samples": 8,
        "trial_settle_windows": 1,
        "trial_measure_windows": 2,
        "quality_direction": "higher_is_better_feasible_only",
        "archive": "canonical_successful_evaluations_including_infeasible",
        "trial_acceptance": "pareto_improvement_in_unique_feasible_yield_and_mean_quality",
        "feedback_evidence": "sequential_observational_not_causal",
        "failure_policy": "count_attempt_without_novelty_credit",
    }


def _number(value: Any, name: str, *, low: float | None = None,
            high: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (low is not None and result < low) or (
        high is not None and result > high
    ):
        raise ValueError(f"{name} is outside its finite allowed range")
    return result


def _integer(value: Any, name: str, *, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (
        high is not None and value > high
    ):
        raise ValueError(f"{name} must be a nonnegative integer in range")
    return value


def normalized_entropy(probs: Iterable[float]) -> float | None:
    """Return H/log(N), or None for a forced decision; require probabilities."""
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a nonempty finite vector")
    total = float(values.sum())
    if (values < 0).any() or not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    if values.size == 1:
        return None
    # float32 softmax outputs can sum a few ulps away from one. Normalize this
    # diagnostic copy in float64 without changing the actual sampling tensor.
    positive = values[values > 0] / total
    result = -float(np.dot(positive, np.log(positive))) / math.log(values.size)
    return float(np.clip(result, 0.0, 1.0))


def entropy_probabilities(scores: Iterable[float], target: float,
                          scale: float | None = None) -> np.ndarray:
    """Sample legal scores at a requested relative entropy, without RNG use.

    Callers supply only legal actions. Target zero is uniform over exact maxima;
    target one is uniform over all actions. Tied maxima bound achievable entropy.
    Numerically unresolved scores stay uniform. An optional trusted return scale
    additionally prevents amplifying score spans below 1e-8 of that scale. A
    common positive affine score transformation otherwise leaves probabilities
    unchanged (up to floating-point resolution); transform ``scale`` as well.
    """
    target = _number(target, "Exploration target", low=0.0, high=1.0)
    if scale is not None:
        scale = _number(scale, "Exploration score scale", low=0.0)
        if scale == 0:
            raise ValueError("Exploration score scale must be positive")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty finite vector")
    count = values.size
    uniform = np.full(count, 1.0 / count)
    if count == 1 or target == 1.0:
        return uniform
    maxima = values == values.max()
    greedy = maxima.astype(np.float64) / int(maxima.sum())
    if target == 0.0:
        return greedy
    # Scale before subtracting: longdouble is only float64 on some platforms.
    # This keeps opposite finite float64 extremes from overflowing their span.
    extended = values.astype(np.longdouble)
    magnitude = np.abs(extended).max()
    if magnitude == 0:
        return uniform
    scaled = extended / magnitude
    span = scaled.max() - scaled.min()
    profile = exploration_profile()
    if span == 0 or span <= profile["score_resolution_relative"]:
        return uniform
    if scale is not None and (
        math.log(float(span)) + math.log(float(magnitude)) - math.log(scale)
        <= math.log(profile["score_span_over_scale_floor"])
    ):
        return uniform
    # Tiny gaps between the best actions should not be amplified merely because
    # another action is clearly worse. Treat maxima indistinguishable at machine
    # resolution as tied for positive targets; explicit target zero remains exact.
    resolved_maxima = scaled.max() - scaled <= profile["score_resolution_relative"]
    resolved_greedy = resolved_maxima.astype(np.float64) / int(resolved_maxima.sum())
    minimum = math.log(int(resolved_maxima.sum())) / math.log(count)
    if target <= minimum:
        return resolved_greedy
    gaps = np.asarray((scaled - scaled.max()) / span, dtype=np.float64)
    gaps[resolved_maxima] = 0.0

    def probabilities(inverse_temperature: float) -> np.ndarray:
        with np.errstate(under="ignore"):
            weights = np.exp(gaps * inverse_temperature)
        return weights / weights.sum()

    lower, upper = 0.0, 1.0
    # The normalized gaps lie in [-1, 0]. This bracket remains finite even for
    # subnormal score differences and arbitrarily close but distinct maxima.
    max_inverse = profile["temperature_max_logit_gap"] / float(-gaps[gaps < 0].max())
    while normalized_entropy(probabilities(upper)) > target and upper < max_inverse:
        upper = min(upper * 2.0, max_inverse)
    for _ in range(profile["temperature_iterations"]):
        middle = (lower + upper) / 2.0
        if normalized_entropy(probabilities(middle)) > target:
            lower = middle
        else:
            upper = middle
    return probabilities((lower + upper) / 2.0)


def _empty_window() -> dict[str, Any]:
    return {"attempts": 0, "successful": 0, "unique": 0, "feasible": 0,
            "improvements": 0, "qualities": []}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    magnitude = max(abs(value) for value in values)
    return 0.0 if magnitude == 0 else magnitude * math.fsum(
        value / magnitude / len(values) for value in values
    )


def _summary(window: Mapping[str, Any]) -> dict[str, Any]:
    count = window["attempts"]
    return {**deepcopy(dict(window)),
            "novelty_rate": window["unique"] / count,
            "success_rate": window["successful"] / count,
            "useful_yield": window["feasible"] / count,
            "mean_quality": _mean(window["qualities"])}


def _combined(windows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = _empty_window()
    for window in windows:
        for name in ("attempts", "successful", "unique", "feasible", "improvements"):
            result[name] += window[name]
        result["qualities"].extend(window["qualities"])
    return _summary(result)


class ExplorationController:
    """Shared inner entropy feedback and conservative terminal-outcome feedback.

    ``adaptive=False`` freezes the target, while A2C's coefficient still tracks
    that target. ``observe`` accepts higher-is-better scientific quality; a
    caller must not feed an evolving novelty bonus as quality. Every completed
    proposal, including duplicates and failed evaluations, counts as an attempt.
    """

    def __init__(self, initial_target: float = 0.7, adaptive: bool = True) -> None:
        self.initial_target = _number(initial_target, "Initial exploration target", low=0.0, high=1.0)
        if not isinstance(adaptive, bool):
            raise ValueError("Exploration adaptive flag must be boolean")
        self.adaptive = adaptive
        self.target = self.initial_target
        self.coefficient = exploration_profile()["coefficient_initial"]
        self.entropy_updates = 0
        self.attempts = 0
        self.completed_attempts = 0
        self.windows = 0
        self.cooldown_windows = 0
        self.stagnant_windows = 0
        self.productive_windows = 0
        self.best_quality: float | None = None
        self._seen: set[str] = set()
        self._window = _empty_window()
        self._previous_window: dict[str, Any] | None = None
        self._trial: dict[str, Any] | None = None
        self.last_action = "initialized"

    @property
    def seen_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._seen))

    def seed_seen(self, keys: Iterable[str]) -> None:
        """Import known canonical evaluations without consuming proposals."""
        keys = list(keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Canonical candidate keys must be nonempty strings")
        self._seen.update(keys)

    def update_entropy(self, h: float | None) -> float:
        """Adjust A2C's coefficient for the next update; ignore forced decisions."""
        if h is None:
            return self.coefficient
        h = _number(h, "Observed normalized entropy", low=0.0, high=1.0)
        profile = exploration_profile()
        self.coefficient = math.exp(float(np.clip(
            math.log(self.coefficient) + profile["coefficient_log_rate"] * (self.target - h),
            math.log(profile["coefficient_min"]), math.log(profile["coefficient_max"]),
        )))
        self.entropy_updates += 1
        return self.coefficient

    def diagnostics(self) -> dict[str, Any]:
        previous = self._previous_window or {}
        return {"profile": EXPLORATION_PROFILE, "target": self.target,
                "coefficient": self.coefficient, "attempts": self.attempts,
                "unique": len(self._seen), "windows": self.windows,
                "cooldown_windows": self.cooldown_windows,
                "trial_active": self._trial is not None,
                "entropy_updates": self.entropy_updates, "last_action": self.last_action,
                "last_window_attempts": previous.get("attempts"),
                "novelty_rate": previous.get("novelty_rate"),
                "success_rate": previous.get("success_rate"),
                "useful_yield": previous.get("useful_yield"),
                "mean_quality": previous.get("mean_quality"),
                "best_quality": self.best_quality,
                "trial_settle_remaining": (self._trial["settle_remaining"] if self._trial else None),
                "trial_measured_windows": (len(self._trial["measurements"]) if self._trial else None)}

    def observe(self, key: str | None, quality: float | None,
                feasible: bool | None = True, successful: bool = True) -> dict[str, Any]:
        """Record a single-proposal collection block."""
        return self.observe_block([{"key": key, "quality": quality,
                                   "feasible": feasible, "successful": successful}])

    @staticmethod
    def _validate_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping) or set(observation) - {
            "key", "quality", "feasible", "successful"
        } or not {"key", "quality"}.issubset(observation):
            raise ValueError("Exploration observation fields are invalid")
        key, quality = observation["key"], observation["quality"]
        feasible = observation.get("feasible", True)
        successful = observation.get("successful", True)
        if key is not None and (not isinstance(key, str) or not key):
            raise ValueError("Canonical candidate key must be a nonempty string or None")
        if not isinstance(successful, bool) or (feasible is not None and not isinstance(feasible, bool)):
            raise ValueError("Successful and feasible flags must be boolean (feasible may be None)")
        if quality is not None:
            quality = _number(quality, "Scientific quality")
        return {"key": key, "quality": quality, "feasible": feasible, "successful": successful}

    def observe_block(self, observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Observe one completed collection block without attributing old samples
        to a new target. A window closes only at block end, with at least 64
        attempted proposals. Large blocks therefore produce one larger window,
        at most one target change, and rates with the actual proposal denominator.

        Validate the entire block before mutation. Callers must retain proposal
        order and include evaluation failures, and checkpoint between blocks.
        """
        observations = [self._validate_observation(item) for item in observations]
        is_new = False
        feasible = successful = None
        new_count = 0
        for observation in observations:
            key, quality = observation["key"], observation["quality"]
            feasible, successful = observation["feasible"], observation["successful"]
            is_new = successful and key is not None and key not in self._seen
            self.attempts += 1
            self._window["attempts"] += 1
            self._window["successful"] += int(successful)
            if is_new:
                new_count += 1
                self._seen.add(key)
                self._window["unique"] += 1
                if feasible is True:
                    self._window["feasible"] += 1
                    if quality is not None:
                        self._window["qualities"].append(quality)
                        if self.best_quality is None or quality > self.best_quality:
                            self.best_quality = quality
                            self._window["improvements"] += 1
        completed = self._window["attempts"] >= exploration_profile()["window_attempts"]
        summary = None
        action = "collecting"
        if completed:
            summary = _summary(self._window)
            self.windows += 1
            self.completed_attempts += self._window["attempts"]
            action = self._finish_window(summary)
            self.last_action = action
            self._previous_window = summary
            self._window = _empty_window()
        return {**self.diagnostics(), "new": bool(is_new), "feasible": feasible,
                "successful": successful, "window_completed": completed,
                "action": action, "window": summary, "new_count": new_count,
                "attempts_in_block": len(observations)}

    def _finish_window(self, summary: dict[str, Any]) -> str:
        profile = exploration_profile()
        if not self.adaptive:
            return "fixed_target"
        if self._trial is not None:
            if self._trial["settle_remaining"]:
                self._trial["settle_remaining"] -= 1
                return "trial_settling"
            self._trial["measurements"].append(summary)
            if len(self._trial["measurements"]) < profile["trial_measure_windows"]:
                return "trial_measuring"
            trial = _combined(self._trial["measurements"])
            baseline = self._trial["baseline"]
            quality_ok = trial["mean_quality"] is not None and baseline["mean_quality"] is not None
            accept = bool(quality_ok and
                trial["useful_yield"] >= baseline["useful_yield"] and
                trial["mean_quality"] >= baseline["mean_quality"] and
                (trial["useful_yield"] > baseline["useful_yield"] or
                 trial["mean_quality"] > baseline["mean_quality"]) and
                trial["success_rate"] >= baseline["success_rate"])
            if not accept:
                self.target = self._trial["prior_target"]
            self._trial = None
            self.cooldown_windows = profile["cooldown_windows"]
            return "trial_kept_observationally" if accept else "trial_reverted"
        if self.cooldown_windows:
            self.cooldown_windows -= 1
            return "cooldown"
        if summary["success_rate"] < profile["min_success_rate"]:
            self.stagnant_windows = self.productive_windows = 0
            return "evaluation_failure_hold"
        stagnant = summary["novelty_rate"] < profile["low_novelty"] and summary["feasible"] == 0
        productive = (summary["novelty_rate"] >= profile["productive_novelty"] and
                      len(summary["qualities"]) >= profile["min_qualified_quality_samples"])
        self.stagnant_windows = self.stagnant_windows + 1 if stagnant else 0
        self.productive_windows = self.productive_windows + 1 if productive else 0
        if self.stagnant_windows >= profile["sustained_windows"]:
            old = self.target
            self.target = min(profile["target_max"], old + profile["target_step"])
            self.stagnant_windows = self.productive_windows = 0
            self.cooldown_windows = profile["cooldown_windows"]
            return "raise_for_stagnation" if self.target > old else "upper_bound_hold"
        if self.productive_windows >= profile["sustained_windows"] and self.target > profile["target_min"]:
            self._trial = {"prior_target": self.target,
                           "baseline": _combined([self._previous_window, summary]),
                           "settle_remaining": profile["trial_settle_windows"],
                           "measurements": []}
            self.target = max(profile["target_min"], self.target - profile["target_step"])
            self.stagnant_windows = self.productive_windows = 0
            return "lower_target_trial"
        return "hold"

    def state_dict(self) -> dict[str, Any]:
        """JSON-compatible full continuation state, including in-flight trials."""
        return deepcopy({"profile": exploration_profile(),
                         "config": {"initial_target": self.initial_target, "adaptive": self.adaptive},
                         "target": self.target, "coefficient": self.coefficient,
                         "entropy_updates": self.entropy_updates, "attempts": self.attempts,
                         "completed_attempts": self.completed_attempts,
                         "windows": self.windows, "cooldown_windows": self.cooldown_windows,
                         "stagnant_windows": self.stagnant_windows,
                         "productive_windows": self.productive_windows,
                         "best_quality": self.best_quality, "seen_keys": list(self.seen_keys),
                         "window": self._window, "previous_window": self._previous_window,
                         "trial": self._trial, "last_action": self.last_action})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate all fields before replacing state; reject changed recipes."""
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("Exploration state fields do not match the current schema")
        candidate = deepcopy(dict(state))
        if candidate["profile"] != exploration_profile():
            raise ValueError("Exploration profile does not match this implementation")
        if candidate["config"] != {"initial_target": self.initial_target, "adaptive": self.adaptive}:
            raise ValueError("Exploration configuration changed on resume")
        profile = exploration_profile()
        candidate["target"] = _number(candidate["target"], "Saved target", low=0.0, high=1.0)
        if not self.adaptive and candidate["target"] != self.initial_target:
            raise ValueError("Fixed exploration target changed in saved state")
        candidate["coefficient"] = _number(candidate["coefficient"], "Saved coefficient",
                                             low=profile["coefficient_min"] * (1 - 1e-14),
                                             high=profile["coefficient_max"] * (1 + 1e-14))
        for name in ("entropy_updates", "attempts", "completed_attempts", "windows", "cooldown_windows",
                     "stagnant_windows", "productive_windows"):
            candidate[name] = _integer(candidate[name], f"Saved {name}")
        if candidate["cooldown_windows"] > profile["cooldown_windows"]:
            raise ValueError("Saved exploration cooldown exceeds profile")
        if candidate["best_quality"] is not None:
            candidate["best_quality"] = _number(candidate["best_quality"], "Saved best quality")
        keys = candidate["seen_keys"]
        if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Saved canonical candidate ledger is invalid")

        def window(value: Any, *, complete: bool) -> dict[str, Any]:
            expected = set(_empty_window())
            if complete:
                expected |= {"novelty_rate", "success_rate", "useful_yield", "mean_quality"}
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError("Saved exploration window fields are invalid")
            for name in ("attempts", "successful", "unique", "feasible", "improvements"):
                _integer(value[name], f"Window {name}")
            if not 0 <= value["improvements"] <= value["feasible"] <= value["unique"] <= value["successful"] <= value["attempts"]:
                raise ValueError("Saved exploration window counters are inconsistent")
            qualities = value["qualities"]
            if not isinstance(qualities, list) or len(qualities) > value["feasible"]:
                raise ValueError("Saved exploration quality samples are inconsistent")
            for quality in qualities:
                _number(quality, "Saved quality sample")
            if complete:
                if value["attempts"] == 0 or value != _summary({key: value[key] for key in _empty_window()}):
                    raise ValueError("Saved exploration window summary is inconsistent")
            return value

        partial = window(candidate["window"], complete=False)
        if (partial["attempts"] >= profile["window_attempts"] or
            candidate["attempts"] != candidate["completed_attempts"] + partial["attempts"] or
            candidate["completed_attempts"] < candidate["windows"] * profile["window_attempts"] or
            (candidate["windows"] == 0 and candidate["completed_attempts"] != 0)):
            raise ValueError("Saved exploration proposal clock is inconsistent")
        previous = candidate["previous_window"]
        if previous is not None:
            window(previous, complete=True)
            if previous["attempts"] < profile["window_attempts"]:
                raise ValueError("Saved previous window has wrong length")
        if (previous is None) != (candidate["windows"] == 0):
            raise ValueError("Saved exploration previous window is missing or premature")
        trial = candidate["trial"]
        if trial is not None:
            if not self.adaptive or not isinstance(trial, dict) or set(trial) != {
                "prior_target", "baseline", "settle_remaining", "measurements"
            }:
                raise ValueError("Saved exploration trial fields are invalid")
            prior = _number(trial["prior_target"], "Saved trial target", low=0.0, high=1.0)
            if candidate["target"] != max(0.0, prior - profile["target_step"]):
                raise ValueError("Saved exploration trial target is inconsistent")
            window(trial["baseline"], complete=True)
            if trial["baseline"]["attempts"] < 2 * profile["window_attempts"]:
                raise ValueError("Saved exploration trial baseline has wrong length")
            _integer(trial["settle_remaining"], "Saved trial settling", high=profile["trial_settle_windows"])
            measurements = trial["measurements"]
            if not isinstance(measurements, list) or len(measurements) >= profile["trial_measure_windows"]:
                raise ValueError("Saved exploration trial measurements are invalid")
            for measurement in measurements:
                window(measurement, complete=True)
                if measurement["attempts"] < profile["window_attempts"]:
                    raise ValueError("Saved exploration trial measurement has wrong length")
            if trial["settle_remaining"] and measurements:
                raise ValueError("Saved exploration trial measured before settling")
        if not isinstance(candidate["last_action"], str):
            raise ValueError("Saved exploration action must be a string")
        for name in ("target", "coefficient", "entropy_updates", "attempts", "completed_attempts", "windows",
                     "cooldown_windows", "stagnant_windows", "productive_windows", "best_quality", "last_action"):
            setattr(self, name, candidate[name])
        self._seen = set(keys)
        self._window = partial
        self._previous_window = previous
        self._trial = trial
