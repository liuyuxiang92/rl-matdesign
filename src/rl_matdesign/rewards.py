"""Generic scalar reward composition and raw-property feasibility gates.

The caller supplies raw property means for gates and separately supplies
property values with direction, uncertainty objective, and scale already
applied for reward optimization. This module never normalizes either input.
An omitted reward configuration preserves the existing property-weighted sum.

A gated reward is a piecewise scalar objective, not a guarantee that every
feasible candidate outranks every infeasible one: that guarantee also needs a
known lower bound on the passing objective and a lower failure reward.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


def _mapping(value: Any, context: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _keys(value: Mapping, allowed: set[str], context: str) -> None:
    unknown = [key for key in value if key not in allowed]
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"{context} has unsupported key(s): {names}.")


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite number (not a boolean).")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{context} must be a finite number.") from None
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite (got {value!r}).")
    return result


@dataclass(frozen=True)
class _Gate:
    name: str
    terms: dict[str, float]
    lower: float | None
    upper: float | None
    penalty_weight: float


class RewardCombiner:
    """Validate and evaluate ``weighted_sum`` or ``gated`` reward configuration.

    ``cfg`` is the optional ``reward:`` mapping, not the whole experiment
    configuration. ``property_weights`` maps every available property name to
    its legacy weight. An explicit ``objective.terms`` replaces those weights;
    it never multiplies them a second time. Every gate scores raw property
    means, regardless of the property's direction, scale, or uncertainty mode.

    Gate bounds are inclusive. All gates must pass; otherwise the reward is
    ``failure.reward - sum(penalty_weight * bound_violation)``. A gate's omitted
    ``penalty_weight`` defaults to 1.0. The failure reward is required explicitly.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None,
        property_weights: Mapping[str, float],
    ) -> None:
        weights = _mapping(property_weights, "property_weights")
        if not weights:
            raise ValueError("property_weights must contain at least one property.")
        self._property_weights: dict[str, float] = {}
        for name, weight in weights.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Property names must be non-empty strings.")
            self._property_weights[name] = _number(weight, f"property_weights[{name!r}]")

        config = {} if cfg is None else _mapping(cfg, "reward")
        self.type = config.get("type", "weighted_sum")
        if self.type not in ("weighted_sum", "gated"):
            raise ValueError("reward.type must be 'weighted_sum' or 'gated'.")
        allowed = {"type", "objective"}
        if self.type == "gated":
            allowed |= {"gates", "failure"}
        _keys(config, allowed, "reward")

        self.objective_terms = (
            self._terms(config["objective"], "reward.objective")
            if "objective" in config else dict(self._property_weights)
        )
        self._gates: list[_Gate] = []
        self._failure_reward = 0.0
        if self.type == "weighted_sum":
            return

        gates = config.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError("reward.gates must be a non-empty list for a gated reward.")
        names: set[str] = set()
        for index, item in enumerate(gates):
            context = f"reward.gates[{index}]"
            gate = _mapping(item, context)
            _keys(gate, {"name", "score", "min", "max", "penalty_weight"}, context)
            name = gate.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{context}.name must be a non-empty string.")
            if name in names:
                raise ValueError(f"{context}.name duplicates gate {name!r}.")
            names.add(name)
            if "score" not in gate:
                raise ValueError(f"{context}.score is required.")
            terms = self._terms(gate["score"], f"{context}.score")
            if "min" not in gate and "max" not in gate:
                raise ValueError(f"{context} needs at least one bound: min or max.")
            lower = _number(gate["min"], f"{context}.min") if "min" in gate else None
            upper = _number(gate["max"], f"{context}.max") if "max" in gate else None
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{context}.min must be <= max.")
            penalty_weight = _number(gate.get("penalty_weight", 1.0), f"{context}.penalty_weight")
            if penalty_weight < 0:
                raise ValueError(f"{context}.penalty_weight must be >= 0.")
            self._gates.append(_Gate(name, terms, lower, upper, penalty_weight))

        failure = _mapping(config.get("failure"), "reward.failure")
        _keys(failure, {"reward"}, "reward.failure")
        if "reward" not in failure:
            raise ValueError("reward.failure.reward is required for a gated reward.")
        self._failure_reward = _number(failure["reward"], "reward.failure.reward")

    def _terms(self, value: Any, context: str) -> dict[str, float]:
        score = _mapping(value, context)
        _keys(score, {"type", "terms"}, context)
        if score.get("type", "weighted_sum") != "weighted_sum":
            raise ValueError(f"{context}.type must be 'weighted_sum'.")
        terms = _mapping(score.get("terms"), f"{context}.terms")
        if not terms:
            raise ValueError(f"{context}.terms must be non-empty.")
        result: dict[str, float] = {}
        for name, weight in terms.items():
            if name not in self._property_weights:
                raise ValueError(f"{context}.terms references unknown property {name!r}.")
            result[name] = _number(weight, f"{context}.terms[{name!r}]")
        return result

    @staticmethod
    def _sum(terms: Mapping[str, float], values: Mapping, context: str) -> float:
        result = 0.0
        for name, weight in terms.items():
            if name not in values:
                raise ValueError(f"{context} is missing property {name!r}.")
            result += weight * _number(values[name], f"{context}[{name!r}]")
        return _number(result, f"{context} weighted sum")

    def evaluate(
        self,
        raw_values: Mapping[str, float],
        objective_values: Mapping[str, float],
    ) -> dict[str, Any]:
        """Return the reward, objective, and per-gate diagnostics.

        Extra unreferenced input keys are harmless (for example, sweep metadata).
        Every referenced value must be present and finite. The returned mapping
        is fresh on each call and contains only ordinary JSON-compatible values.
        """
        raw = _mapping(raw_values, "raw_values")
        adjusted = _mapping(objective_values, "objective_values")
        objective = self._sum(self.objective_terms, adjusted, "objective_values")
        diagnostics = []
        total_penalty = 0.0
        for gate in self._gates:
            score = self._sum(gate.terms, raw, f"raw_values for gate {gate.name!r}")
            violation = 0.0
            if gate.lower is not None:
                violation += max(0.0, gate.lower - score)
            if gate.upper is not None:
                violation += max(0.0, score - gate.upper)
            violation = _number(violation, f"Gate {gate.name!r} bound violation")
            penalty = _number(gate.penalty_weight * violation, f"Gate {gate.name!r} penalty")
            total_penalty += penalty
            diagnostics.append({
                "name": gate.name,
                "score": score,
                "min": gate.lower,
                "max": gate.upper,
                "passed": violation == 0.0,
                "violation": violation,
                "penalty_weight": gate.penalty_weight,
                "penalty": penalty,
            })
        passed = all(gate["passed"] for gate in diagnostics)
        reward = objective if passed else self._failure_reward - total_penalty
        return {
            "reward": _number(reward, "Combined reward"),
            "type": self.type,
            "objective": objective,
            "gate_passed": passed,
            "gates": diagnostics,
        }
