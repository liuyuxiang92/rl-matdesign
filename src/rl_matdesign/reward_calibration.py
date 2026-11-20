"""Fixed scientific reward calibration, independent of adaptive value normalization.

Observations contain native gate inputs and complete, signed scientific utilities.
No evaluator is called here. A fitted artifact is immutable and content-addressed;
collecting observations and charging evaluations belong to the experiment runner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

from .predictors.outputs import _mean, _population_std
from .rewards import RewardCombiner, _keys, _mapping, _number


class CalibrationPendingError(ValueError):
    """A scientific reward cannot be emitted before its calibration freezes."""


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _positive(value: Any, context: str, *, allow_zero: bool = False) -> float:
    value = _number(value, context)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{context} must be {'nonnegative' if allow_zero else 'strictly positive'}.")
    return value


def _normalization(value: Any, context: str, *, reference: bool = False) -> dict:
    if isinstance(value, str):
        value = {"mode": value}
    value = dict(_mapping(value, context))
    mode = value.get("mode", "auto_fixed")
    if mode not in ("auto_fixed", "fixed"):
        raise ValueError(f"{context}.mode must be auto_fixed or fixed.")
    allowed = {"mode", "scale_floor"} if mode == "auto_fixed" else {"mode", "scale"}
    if reference and mode == "fixed":
        allowed.add("reference")
    _keys(value, allowed, context)
    result = {"mode": mode}
    if mode == "fixed":
        result["scale"] = _positive(value.get("scale"), f"{context}.scale")
        if reference:
            result["reference"] = _number(value.get("reference", 0.0), f"{context}.reference")
    else:
        result["scale_floor"] = _positive(value.get("scale_floor", 0.0),
                                         f"{context}.scale_floor", allow_zero=True)
    return result


def _moments(values: Sequence[float]) -> tuple[float, float]:
    """Use the stable population statistics shared by scientific output handling."""
    values = [_number(v, "Calibration observation") for v in values]
    if not len(values):
        raise ValueError("Calibration requires valid observations.")
    return _mean(values), _population_std(values) or 0.0


def _rms(values: Sequence[float]) -> float:
    anchor = max((abs(value) for value in values), default=0.0)
    if anchor == 0:
        return 0.0
    scaled_mean_square = math.fsum((value / anchor) ** 2 for value in values) / len(values)
    return _number(math.sqrt(scaled_mean_square) * anchor, "Calibration RMS")


class RewardCalibration:
    """Fit and evaluate an explicitly selected immutable reward convention.

    ``objective_values`` in observations are already signed and uncertainty
    adjusted. Applying direction or uncertainty again would change the task.
    Fixed property references are declared in physical coordinates and oriented
    once by ``directions``. Automatically fitted references use signed utility
    coordinates, including any declared uncertainty adjustment.
    """

    SCHEMA_VERSION = 1

    def __init__(self, combiner: RewardCombiner, properties: Sequence[Mapping[str, Any]], *,
                 directions: Mapping[str, float | None] | None = None,
                 gate_domain_widths: Mapping[str, float] | None = None,
                 scientific_identity: Mapping[str, Any] | None = None) -> None:
        self.combiner = combiner
        specs = {p["name"]: dict(p) for p in properties}
        self._state = None
        if combiner.type not in ("weighted_sum", "gated"):
            raise ValueError("Automatic calibration supports weighted_sum and gated rewards, not max alternatives.")
        cfg = dict(combiner.calibration_config)
        _keys(cfg, {"profile", "num_candidates", "quality_mode", "quality_scale_floor"}, "reward.calibration")
        expected = "auto_gated_softplus_v1" if combiner.type == "gated" else "auto_weighted_sum_v1"
        self.profile = cfg.get("profile", expected)
        if self.profile != expected:
            raise ValueError(f"reward.calibration.profile must be {expected!r} for {combiner.type}.")
        if combiner.type == "gated" and not combiner.calibrated_gated:
            raise ValueError("Gated property calibration requires an explicit complete reward.calibration profile.")
        self.quality_mode = cfg.get("quality_mode", "per_property")
        if self.quality_mode not in ("per_property", "aggregate"):
            raise ValueError("reward.calibration.quality_mode must be per_property or aggregate.")
        if self.quality_mode == "aggregate" and combiner.type != "gated":
            raise ValueError("Aggregate quality calibration requires the gated profile.")
        self.quality_specs = {}
        if directions is None:
            aliases = {"min": -1.0, "minimize": -1.0, "max": 1.0, "maximize": 1.0,
                       "target": None}
            directions = {}
            for name, prop in specs.items():
                direction = str(prop.get("direction", "max")).lower()
                if direction not in aliases:
                    raise ValueError(f"Property {name!r} has an unsupported direction {direction!r}.")
                directions[name] = aliases[direction]
        for name, prop in specs.items():
            if "normalization" in prop and "scale" in prop:
                raise ValueError(f"Property {name!r} cannot declare both scale and normalization.")
            if combiner.type == "gated" and "scale" in prop:
                raise ValueError(f"Calibrated property {name!r} cannot also declare legacy scale.")
            if name not in combiner.objective_terms:
                if "normalization" in prop:
                    raise ValueError(f"Gate-only property {name!r} must use gate violation_normalization.")
                continue
            if self.quality_mode == "aggregate":
                if "normalization" in prop:
                    raise ValueError("Aggregate quality mode cannot also normalize individual properties.")
                continue
            normalization = prop.get("normalization")
            if combiner.objective_terms[name] == 0:
                if normalization is not None:
                    raise ValueError(f"Zero-weight objective {name!r} does not use normalization.")
                continue
            if normalization is None and combiner.type == "gated":
                normalization = {"mode": "auto_fixed"}
            if normalization is not None:
                spec = _normalization(normalization, f"Property {name!r} normalization", reference=True)
                if spec["mode"] == "fixed":
                    direction = directions.get(name, 1.0)
                    if direction is None:
                        raise ValueError("Fixed physical references for target-distance objectives are unsupported.")
                    spec["reference"] *= float(direction)
                self.quality_specs[name] = spec
        if combiner.type == "gated":
            weights = list(combiner.objective_terms.values())
            if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
                raise ValueError("Calibrated gated objective weights must be nonnegative with positive total.")
        self.aggregate_floor = _positive(cfg.get("quality_scale_floor", 0.0),
                                         "quality_scale_floor", allow_zero=True)
        if "quality_scale_floor" in cfg and self.quality_mode != "aggregate":
            raise ValueError("quality_scale_floor is for aggregate mode; use per-property scale_floor otherwise.")
        if combiner.type == "weighted_sum" and not self.quality_specs:
            raise ValueError("Weighted-sum reward calibration requires at least one normalized objective.")
        self.gate_specs = {}
        widths = dict(gate_domain_widths or {})
        for gate in combiner._gates:
            spec = _normalization(combiner.gate_normalizations[gate.name], f"Gate {gate.name!r} normalization")
            physical_floor = _positive(widths.get(gate.name, 0.0), "Gate domain width", allow_zero=True)
            if gate.lower is not None and gate.upper is not None:
                physical_floor = max(physical_floor, _number(gate.upper - gate.lower, "Gate width"))
            spec["physical_floor"] = physical_floor
            self.gate_specs[gate.name] = spec
        needs_pilot = (self.quality_mode == "aggregate"
                       or any(v["mode"] == "auto_fixed" for v in self.quality_specs.values())
                       or any(v["mode"] == "auto_fixed" for v in self.gate_specs.values()))
        count = cfg.get("num_candidates", 100)
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError("reward.calibration.num_candidates must be a positive integer.")
        if not needs_pilot and "num_candidates" in cfg:
            raise ValueError("reward.calibration.num_candidates is unused when all scales are fixed.")
        self.num_candidates = int(count) if needs_pilot else 0
        self.spec = {"schema_version": self.SCHEMA_VERSION, "profile": self.profile,
                     "quality_mode": self.quality_mode, "quality_specs": self.quality_specs,
                     "gate_specs": self.gate_specs, "aggregate_floor": self.aggregate_floor,
                     "objective_terms": combiner.objective_terms,
                     "gates": [{"name": g.name, "terms": g.terms, "min": g.lower, "max": g.upper,
                                "priority": g.penalty_weight} for g in combiner._gates],
                     "num_candidates": self.num_candidates,
                     "scientific_identity": deepcopy(dict(scientific_identity or {}))}
        # Work with the same JSON types that a later artifact load returns.
        self.spec = json.loads(json.dumps(self.spec, allow_nan=False))
        self.spec_hash = _hash(self.spec)
        if not needs_pilot:
            self.fit([])

    @property
    def frozen(self) -> bool:
        return self._state is not None

    def require_frozen(self) -> None:
        if not self.frozen:
            raise CalibrationPendingError("Scientific reward calibration is pending; collect raw observations and freeze it first.")

    def _sum(self, values: Mapping[str, float], *, relative: bool = False) -> float:
        terms = self.combiner.objective_terms
        if not relative:
            return self.combiner._sum(terms, values, "objective_values")
        largest = max(terms.values())
        denominator = math.fsum(w / largest for w in terms.values())
        return _number(math.fsum((w / largest / denominator) * _number(values[name], name)
                                for name, w in terms.items()), "Combined standardized utility")

    def _validate_observations(self, observations: Sequence[Mapping[str, Any]]) -> list[dict]:
        rows, identities = [], set()
        for raw in observations:
            row = deepcopy(dict(_mapping(raw, "Calibration observation")))
            if row.get("spec_hash") != self.spec_hash:
                raise ValueError("Calibration observation belongs to a different scientific contract.")
            key = row.get("candidate_key")
            if not isinstance(key, str) or not key or key in identities:
                raise ValueError("Calibration observations require distinct non-empty candidate identities.")
            identities.add(key)
            values = _mapping(row.get("objective_values"), "objective_values")
            for name in self.combiner.objective_terms:
                _number(values.get(name), f"Calibration objective {name!r}")
            self.combiner.gate_details(row.get("gate_values"))
            rows.append(row)
        if len(rows) < self.num_candidates:
            raise ValueError(f"Calibration requires {self.num_candidates} distinct valid candidates; got {len(rows)}.")
        return rows

    @staticmethod
    def _fit_quality(values: Sequence[float], spec: Mapping[str, Any]) -> dict:
        if spec["mode"] == "fixed":
            return {"reference": spec["reference"], "scale": spec["scale"], "source": "fixed"}
        mean, std = _moments(values)
        scale = max(std, spec.get("scale_floor", 0.0))
        if scale <= 0:
            raise ValueError("Quality calibration is unresolved: zero spread requires a meaningful declared scale_floor.")
        return {"reference": mean, "scale": scale, "population_std": std,
                "scale_floor": spec.get("scale_floor", 0.0), "source": "pilot"}

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> dict:
        if self._state is not None:
            raise ValueError("Reward calibration is frozen and cannot be refitted.")
        rows = self._validate_observations(observations)
        qualities = {}
        if self.quality_mode == "aggregate":
            qualities["__aggregate__"] = self._fit_quality(
                [self._sum(row["objective_values"]) for row in rows],
                {"mode": "auto_fixed", "scale_floor": self.aggregate_floor})
        else:
            for name, spec in self.quality_specs.items():
                qualities[name] = self._fit_quality([row["objective_values"][name] for row in rows], spec)
        gates = {}
        for gate in self.combiner._gates:
            spec = self.gate_specs[gate.name]
            if spec["mode"] == "fixed":
                gates[gate.name] = {"scale": spec["scale"], "source": "fixed"}
                continue
            scores = [self.combiner._sum(gate.terms, row["gate_values"], "gate_values") for row in rows]
            _, std = _moments(scores)
            distances = [_number(min(abs(score - bound) for bound in (gate.lower, gate.upper)
                                     if bound is not None), "Gate boundary distance") for score in scores]
            rms = _rms(distances)
            floor = max(spec["physical_floor"], spec.get("scale_floor", 0.0))
            scale = max(std, rms, floor)
            if scale <= 0:
                raise ValueError(f"Gate {gate.name!r} calibration unresolved: zero scale needs physical metadata.")
            gates[gate.name] = {"scale": scale, "population_std": std, "boundary_rms": rms,
                                "physical_floor": floor, "min": min(scores), "max": max(scores), "source": "pilot"}
        state = {"schema_version": self.SCHEMA_VERSION, "spec_hash": self.spec_hash,
                 "spec": deepcopy(self.spec), "quality": qualities, "gates": gates,
                 "candidate_keys": [r["candidate_key"] for r in rows], "observation_count": len(rows)}
        state["content_hash"] = _hash(state)
        self._state = state
        return self.state_dict()

    def state_dict(self) -> dict | None:
        return deepcopy(self._state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state = deepcopy(dict(_mapping(state, "Reward calibration artifact")))
        content_hash = state.pop("content_hash", None)
        if not content_hash or _hash(state) != content_hash:
            raise ValueError("Reward calibration artifact content hash mismatch.")
        state["content_hash"] = content_hash
        if state.get("schema_version") != self.SCHEMA_VERSION or state.get("spec_hash") != self.spec_hash:
            raise ValueError("Reward calibration artifact has a different scientific contract/profile.")
        if state.get("spec") != self.spec:
            raise ValueError("Reward calibration artifact specification mismatch.")
        quality_keys = {"__aggregate__"} if self.quality_mode == "aggregate" else set(self.quality_specs)
        quality = _mapping(state.get("quality"), "Stored quality calibration")
        gates = _mapping(state.get("gates"), "Stored gate calibration")
        if set(quality) != quality_keys or set(gates) != set(self.gate_specs):
            raise ValueError("Reward calibration artifact is missing quality/gate scales.")
        for item in quality.values():
            item = _mapping(item, "Stored quality scale")
            _number(item.get("reference"), "Stored quality reference")
            _positive(item.get("scale"), "Stored quality scale")
        for item in gates.values():
            item = _mapping(item, "Stored gate scale")
            _positive(item.get("scale"), "Stored gate scale")
        count = state.get("observation_count")
        keys = state.get("candidate_keys")
        if (not isinstance(keys, list) or isinstance(count, bool) or not isinstance(count, int)
                or count < self.num_candidates or count != len(keys)):
            raise ValueError("Reward calibration artifact has an invalid observation count.")
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Reward calibration artifact must identify distinct candidates.")
        if self._state is not None and self._state != state:
            raise ValueError("Cannot replace an already frozen reward calibration.")
        self._state = state

    def evaluate(self, gate_values: Mapping[str, float], objective_values: Mapping[str, float]) -> dict:
        self.require_frozen()
        values = {name: _number(objective_values.get(name), f"Objective {name!r}")
                  for name in self.combiner.objective_terms}
        quality = self._state["quality"]
        if self.quality_mode == "aggregate":
            q = quality["__aggregate__"]
            utility = _number((self._sum(values) - q["reference"]) / q["scale"], "Standardized aggregate utility")
        else:
            for name, q in quality.items():
                values[name] = _number((values[name] - q["reference"]) / q["scale"], f"Standardized {name}")
            utility = self._sum(values, relative=self.combiner.type == "gated")
        if self.combiner.type == "weighted_sum":
            return {"reward": utility, "objective": utility, "standardized_utility": utility,
                    "type": "weighted_sum", "gate_passed": True, "gates": []}
        diagnostics = self.combiner.gate_details(gate_values)
        for gate in diagnostics:
            scale = self._state["gates"][gate["name"]]["scale"]
            gate["violation_scale"] = scale
            gate["normalized_violation"] = _number(gate["violation"] / scale, "Normalized violation")
            gate["penalty"] = _number(gate["priority"] * gate["normalized_violation"], "Normalized penalty")
        passed = all(g["passed"] for g in diagnostics)
        objective = _number(1.0 + max(utility, 0.0) + math.log1p(math.exp(-abs(utility))), "Passing reward")
        reward = objective if passed else _number(-math.fsum(g["penalty"] for g in diagnostics), "Failure reward")
        return {"reward": reward, "objective": objective, "standardized_utility": utility,
                "type": "gated", "gate_passed": passed, "gates": diagnostics}
