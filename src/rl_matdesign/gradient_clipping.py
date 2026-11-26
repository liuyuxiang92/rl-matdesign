"""Adaptive global gradient clipping with a separately calibrated startup cap.

The pilot's gradient norms set the startup threshold; they never enter the
rolling history of successful training updates. ``prepare`` applies clipping
using only prior history. ``commit`` advances the controller after its optimizer
step succeeds. Model/optimizer transactions belong to the caller.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import torch


GRADIENT_CLIPPING_PROFILE = "adaptive_grad_clip_v1"
_VERSION = 1
_PERCENTILE = 90.0
_WINDOW = 200
_MIN_POSITIVE_HISTORY = 20


class NonfiniteGradientError(ValueError):
    """A live gradient cannot be represented by a finite global norm."""


def gradient_clipping_profile() -> dict[str, Any]:
    """Return the complete, immutable-by-convention execution profile."""
    return {
        "name": GRADIENT_CLIPPING_PROFILE,
        "version": _VERSION,
        "percentile": _PERCENTILE,
        "percentile_interpolation": "linear",
        "norm": "global_l2_float64",
        "window": _WINDOW,
        "min_positive_history": _MIN_POSITIVE_HISTORY,
        "calibration_candidates": 100,
        "calibration_budget": "outside_training_budget",
        "initialization": "pilot_positive_gradient_p90",
        "history": "prior_successful_update_raw_norms",
        "zero_history": "positive_only_percentile",
        "all_zero_pilot": "first_positive_gradient_provisional",
        "probe_unit": "mean_trajectory_loss",
        "probe_weights": "fixed_disposable_copies",
        "probe_value_coordinates": "single_pooled_target_update_when_enabled",
        "probe_actor_scaling": "pilot_wide_advantage_standardization",
        "probe_actor_entropy": "initial_controller_coefficient",
    }


def _norm_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_percentile(values: list[float]) -> float | None:
    ordered = sorted(value for value in values if value > 0)
    if not ordered:
        return None
    position = (_PERCENTILE / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    # Subtract before interpolation to avoid adding two huge weighted values.
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _gradients(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    gradients = []
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            raise ValueError("Gradient norm parameters must not contain duplicates")
        seen.add(id(parameter))
        if parameter.grad is not None:
            if parameter.grad.layout != torch.strided:
                raise ValueError("Adaptive gradient clipping requires dense gradients")
            gradients.append(parameter.grad)
    return gradients


def _global_norm(gradients: list[torch.Tensor]) -> float:
    """Scaled sum of squares avoids overflow for finite float32/64 gradients."""
    magnitudes = []
    largest = 0.0
    for gradient in gradients:
        magnitude = gradient.detach().abs().to(dtype=torch.float64)
        if not torch.isfinite(magnitude).all().item():
            raise NonfiniteGradientError("Nonfinite gradient components cannot be clipped")
        if magnitude.numel():
            largest = max(largest, float(magnitude.max().item()))
            magnitudes.append(magnitude)
    if largest == 0:
        return 0.0
    squared_sum = math.fsum(
        float((magnitude / largest).square().sum().item())
        for magnitude in magnitudes
    )
    result = largest * math.sqrt(squared_sum)
    if not math.isfinite(result):
        raise NonfiniteGradientError("Global gradient norm exceeds finite float64 range")
    return result


def global_gradient_norm(parameters: Iterable[torch.Tensor]) -> float:
    """Return a finite global L2 norm without changing gradients or parameters."""
    return _global_norm(_gradients(parameters))


class AdaptiveGradientClipper:
    """One independent controller for one optimizer/network.

    A zero-only (or empty) pilot leaves startup unresolved. Zero gradients pass
    through; the first subsequent positive norm provides a provisional startup
    cap, without clipping that observation. It is saved only on ``commit``.
    """

    def __init__(self, calibration_norms: Iterable[float], *, network: str) -> None:
        if not isinstance(network, str) or not network.strip():
            raise ValueError("Gradient clipping network must be a nonempty string")
        self.network = network
        self.calibration_norms = [
            _norm_number(value, "Calibration gradient norm")
            for value in calibration_norms
        ]
        self.startup_cap = _positive_percentile(self.calibration_norms)
        self.startup_source = "pilot" if self.startup_cap is not None else "unresolved"
        self.startup_established_at_update: int | None = (
            0 if self.startup_cap is not None else None
        )
        self.history: list[float] = []
        self.successful_updates = 0
        self._pending: dict[str, Any] | None = None

    @torch.no_grad()
    def prepare(self, parameters: Iterable[torch.Tensor]) -> dict[str, Any]:
        """Clip finite gradients using prior state; do not advance that state."""
        if self._pending is not None:
            raise RuntimeError("Commit or discard the prepared clipping update first")
        gradients = _gradients(parameters)
        raw_norm = _global_norm(gradients)
        positive_count = sum(value > 0 for value in self.history)
        provisional = False
        if positive_count >= _MIN_POSITIVE_HISTORY:
            cap = _positive_percentile(self.history)
            source = "adaptive_history"
        elif self.startup_cap is not None:
            cap = self.startup_cap
            source = self.startup_source
        elif raw_norm > 0:
            cap = raw_norm
            source = "first_positive_gradient"
            provisional = True
        else:
            cap = None
            source = "unresolved_zero"
        coefficient = cap / raw_norm if cap is not None and raw_norm > cap else 1.0
        if coefficient < 1.0:
            for gradient in gradients:
                gradient.mul_(coefficient)
        diagnostics = {
            "network": self.network,
            "profile": GRADIENT_CLIPPING_PROFILE,
            "update": self.successful_updates + 1,
            "raw_norm": raw_norm,
            "cap": cap,
            "coefficient": coefficient,
            "clipped": coefficient < 1.0,
            "history_count": len(self.history),
            "positive_history_count": positive_count,
            "threshold_source": source,
            "provisional_startup": provisional,
        }
        self._pending = deepcopy(diagnostics)
        return diagnostics

    def commit(self, diagnostics: Mapping[str, Any]) -> None:
        """Record the raw norm only after the associated optimizer step succeeds."""
        if self._pending is None or dict(diagnostics) != self._pending:
            raise ValueError("Clipping commit does not match the prepared update")
        if self._pending["provisional_startup"]:
            self.startup_cap = self._pending["raw_norm"]
            self.startup_source = "first_positive_gradient"
            self.startup_established_at_update = self._pending["update"]
        self.history.append(self._pending["raw_norm"])
        self.history = self.history[-_WINDOW:]
        self.successful_updates += 1
        self._pending = None

    def discard_prepared(self) -> None:
        """Discard an uncommitted decision; this does not restore model gradients."""
        self._pending = None

    def state_dict(self) -> dict[str, Any]:
        """Serialize committed state; callers checkpoint only complete updates."""
        return {
            "version": _VERSION,
            "profile": gradient_clipping_profile(),
            "network": self.network,
            "calibration_norms": self.calibration_norms.copy(),
            "calibration_count": len(self.calibration_norms),
            "calibration_positive_count": sum(value > 0 for value in self.calibration_norms),
            "startup_cap": self.startup_cap,
            "startup_source": self.startup_source,
            "startup_established_at_update": self.startup_established_at_update,
            "history": self.history.copy(),
            "successful_updates": self.successful_updates,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, network: str | None = None,
    ) -> "AdaptiveGradientClipper":
        """Restore strictly validated controller state without recalibration."""
        expected_keys = {
            "version", "profile", "network", "calibration_norms", "calibration_count",
            "calibration_positive_count", "startup_cap", "startup_source",
            "startup_established_at_update", "history", "successful_updates",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("Gradient clipping state has missing or unexpected fields")
        if type(state["version"]) is not int or state["version"] != _VERSION:
            raise ValueError("Unsupported gradient clipping state version")
        expected_profile = gradient_clipping_profile()
        profile = state["profile"]
        if (not isinstance(profile, Mapping) or set(profile) != set(expected_profile)
                or any(type(profile[key]) is not type(value) or profile[key] != value
                       for key, value in expected_profile.items())):
            raise ValueError("Gradient clipping profile mismatch")
        if network is not None and state["network"] != network:
            raise ValueError("Gradient clipping network mismatch")
        if not isinstance(state["calibration_norms"], list) or not isinstance(state["history"], list):
            raise ValueError("Gradient clipping norms and history must be ordered lists")
        result = cls(state["calibration_norms"], network=state["network"])
        count = _integer(state["calibration_count"], "Calibration count")
        positive_count = _integer(state["calibration_positive_count"], "Calibration positive count")
        if count != len(result.calibration_norms) or positive_count != sum(
            value > 0 for value in result.calibration_norms
        ):
            raise ValueError("Gradient clipping calibration counts are inconsistent")
        result.successful_updates = _integer(state["successful_updates"], "Successful updates")
        result.history = [_norm_number(value, "History gradient norm") for value in state["history"]]
        if len(result.history) != min(result.successful_updates, _WINDOW):
            raise ValueError("Gradient clipping history length does not match successful updates")
        cap = state["startup_cap"]
        cap = None if cap is None else _norm_number(cap, "Startup cap")
        if cap == 0:
            raise ValueError("Gradient clipping startup cap must be positive or unresolved")
        established = state["startup_established_at_update"]
        established = None if established is None else _integer(established, "Startup update")
        source = state["startup_source"]
        if result.startup_cap is not None:
            if cap != result.startup_cap or source != "pilot" or established != 0:
                raise ValueError("Gradient clipping pilot startup state is inconsistent")
        elif source == "unresolved":
            if cap is not None or established is not None or any(result.history):
                raise ValueError("Unresolved clipping startup has positive gradient history")
        elif source == "first_positive_gradient":
            if cap is None or established is None or not 1 <= established <= result.successful_updates:
                raise ValueError("Provisional clipping startup state is inconsistent")
            history_start = result.successful_updates - len(result.history) + 1
            if established >= history_start:
                prefix = result.history[:established - history_start]
                if any(prefix) or result.history[established - history_start] != cap:
                    raise ValueError("Provisional clipping startup disagrees with gradient history")
        else:
            raise ValueError("Unknown gradient clipping startup source")
        result.startup_cap = cap
        result.startup_source = source
        result.startup_established_at_update = established
        return result
