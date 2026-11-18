"""Generic scalar reward composition and property feasibility gates.

The caller supplies gate scores in native property units and separately supplies
property values with direction, uncertainty objective, and scale already
applied for reward optimization. This module never normalizes either input.
Structure scoring uses mean minus std for classification gates, raw means for
other gates; those statistics are selected by the caller.
Reward terms own all coefficients. A sole property has implicit coefficient 1;
multiple properties require an explicit reward objective.

A gated reward is a piecewise scalar objective, not a guarantee that every
feasible candidate outranks every infeasible one: that guarantee also needs a
known lower bound on the passing objective and a lower failure reward.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    """Validate and evaluate ``weighted_sum``, ``gated``, or ``max`` rewards.

    ``cfg`` is the optional ``reward:`` mapping, not the whole experiment
    configuration. ``property_names`` identifies the available properties.
    All coefficients belong in reward terms; only a sole property's objective
    can be omitted, in which case its coefficient is 1. Every gate scores the
    supplied ``raw_values`` in native units; the caller selects their statistics without
    applying objective direction, scale, or uncertainty mode.

    Gate bounds are inclusive. All gates must pass; otherwise the reward is
    ``failure.reward - sum(penalty_weight * bound_violation)``. A gate's omitted
    ``penalty_weight`` defaults to 1.0. The failure reward is required explicitly.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None,
        property_names: Sequence[str],
    ) -> None:
        if isinstance(property_names, (str, bytes)) or not isinstance(property_names, Sequence):
            raise ValueError("property_names must be a sequence of names; coefficients belong in reward terms.")
        if not property_names:
            raise ValueError("property_names must contain at least one property.")
        self._property_names: set[str] = set()
        for name in property_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Property names must be non-empty strings.")
            if name in self._property_names:
                raise ValueError(f"Property name {name!r} duplicates an earlier property.")
            self._property_names.add(name)

        config = {} if cfg is None else _mapping(cfg, "reward")
        self.type = config.get("type", "weighted_sum")
        if self.type not in ("weighted_sum", "gated", "max"):
            raise ValueError("reward.type must be 'weighted_sum', 'gated', or 'max'.")
        allowed = {"type", "objective"}
        if self.type == "max":
            allowed = {"type", "alternatives"}
        if self.type == "gated":
            allowed |= {"gates", "failure"}
        _keys(config, allowed, "reward")

        self.objective_terms: dict[str, float] = {}
        self._gates: list[_Gate] = []
        self._failure_reward = 0.0
        self.alternatives: list[tuple[str, dict[str, float]]] = []
        if self.type == "max":
            alternatives = config.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError("reward.alternatives must be a non-empty list for a max reward.")
            names = set()
            for index, item in enumerate(alternatives):
                context = f"reward.alternatives[{index}]"
                item = _mapping(item, context)
                _keys(item, {"name", "terms"}, context)
                name = item.get("name")
                if not isinstance(name, str) or not name.strip() or name in names:
                    raise ValueError(f"{context}.name must be a unique non-empty string.")
                names.add(name)
                self.alternatives.append((name, self._terms({"terms": item.get("terms")}, context)))
            return
        if "objective" in config:
            self.objective_terms = self._terms(config["objective"], "reward.objective")
        elif len(self._property_names) == 1:
            self.objective_terms = {property_names[0]: 1.0}
        else:
            raise ValueError("Multiple properties require explicit reward.objective.terms.")
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
            if name not in self._property_names:
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

        Extra unreferenced input keys are harmless (for example, diagnostic metadata).
        Every referenced value must be present and finite. The returned mapping
        is fresh on each call and contains only ordinary JSON-compatible values.
        """
        raw = _mapping(raw_values, "raw_values")
        adjusted = _mapping(objective_values, "objective_values")
        if self.type == "max":
            alternatives = [{"name": name, "score": self._sum(terms, adjusted, "objective_values")}
                            for name, terms in self.alternatives]
            # Python's stable max preserves the first declared alternative on ties.
            best = max(alternatives, key=lambda alternative: alternative["score"])
            return {"reward": best["score"], "type": self.type, "objective": best["score"],
                    "selected_alternative": best["name"], "alternatives": alternatives,
                    "gate_passed": True, "gates": []}
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
