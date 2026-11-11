"""Conservative, paired-seed parameter range diagnosis.

This module schedules measurements through an injected evaluator.  It never
launches training, changes a trial in progress, or treats low scores as proof
that a parameter value is intrinsically unusable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import math
import random
from typing import Any, Callable, Mapping


_EXTENSION_CHOICES = {
    "dqn_batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dqn_target_update_freq": [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000],
    "pg_batch_eps": [1, 2, 4, 8, 16, 32, 64, 128, 256],
}
_NUMERICAL_STATUSES = {"INFEASIBLE_NUMERICAL", "NUMERICAL_INSTABILITY", "NUMERICAL_ERROR", "NONFINITE", "NONFINITE_LOSS"}
_SYSTEM_STATUSES = {"SYSTEM_ERROR", "INFRASTRUCTURE_ERROR"}


def parameter_id(parameters: Mapping[str, Any]) -> str:
    text = json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _choices(spec: Mapping[str, Any]) -> list[Any]:
    values = list(spec["choices"])
    if not values:
        raise ValueError("categorical search range must contain choices")
    try:
        return sorted(set(values))
    except TypeError:
        # String and numeric categories can coexist, but their ordering must not
        # depend on Python comparing unrelated types.
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _cast(value: float, spec: Mapping[str, Any]) -> float | int:
    if spec["type"] == "int":
        step = int(spec.get("step", 1))
        anchor = int(spec["low"])
        return anchor + round((value - anchor) / step) * step
    return float(value)


def _middle(low: float, high: float, spec: Mapping[str, Any]) -> float | int:
    value = math.sqrt(low * high) if spec.get("log", False) else (low + high) / 2
    return _cast(value, spec)


def initial_reference(search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the working reference from ranges, without a user reference value."""
    result = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            choices = _choices(spec)
            result[name] = choices[(len(choices) - 1) // 2]
        elif spec["type"] in {"int", "float"}:
            if spec["low"] > spec["high"] or (spec.get("log") and spec["low"] <= 0):
                raise ValueError(f"invalid range for {name}")
            result[name] = _middle(spec["low"], spec["high"], spec)
        else:
            raise ValueError(f"unsupported range type for {name}: {spec['type']}")
    return result


def _seeds(result: Mapping[str, Any]) -> dict[Any, Mapping[str, Any]]:
    raw = result.get("seed_results", [])
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    return {str(value["seed"]): value for value in raw if value.get("seed") is not None}


def _eligible(result: Mapping[str, Any]) -> bool:
    score = result.get("score")
    return bool(result.get("eligible_for_selection", False)) and isinstance(score, (int, float)) and math.isfinite(score)


def _numerical_count(result: Mapping[str, Any]) -> int:
    return max(int(result.get("numerical_failure_count", 0)), sum(value.get("status") in _NUMERICAL_STATUSES for value in _seeds(result).values()))


def _system_error(result: Mapping[str, Any]) -> bool:
    return result.get("status") in _SYSTEM_STATUSES or any(value.get("status") in _SYSTEM_STATUSES for value in _seeds(result).values())


def _paired_differences(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> list[float] | None:
    current, baseline = _seeds(candidate), _seeds(reference)
    if len(current) < 2 or current.keys() != baseline.keys():
        return None
    differences = []
    for seed in sorted(current):
        if not _eligible(current[seed]) or not _eligible(baseline[seed]):
            return None
        differences.append(float(current[seed]["score"]) - float(baseline[seed]["score"]))
    return differences


def _improved(candidate: Mapping[str, Any], reference: Mapping[str, Any], minimum: float) -> bool:
    differences = _paired_differences(candidate, reference)
    return bool(_eligible(candidate) and _eligible(reference) and differences
                and all(value > 0 for value in differences)
                and sum(differences) / len(differences) >= minimum - 1e-12)


def _ambiguous(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    statuses = [value.get("status") in _NUMERICAL_STATUSES for value in _seeds(candidate).values()]
    if any(statuses) and not all(statuses) and _numerical_count(candidate) < 2:
        return True
    differences = _paired_differences(candidate, reference)
    return bool(differences and min(differences) < 0 < max(differences))


def _domain_bounds(name: str, spec: Mapping[str, Any]) -> tuple[float, float]:
    low, high = -math.inf, math.inf
    if spec.get("log") or name in {"warmup_budget_fraction", "epsilon_anneal_budget_ratio", "dqn_lr", "pg_lr_actor", "pg_lr_critic"}:
        low = math.nextafter(0.0, math.inf)
    if name == "pg_entropy_coef":
        low = 0.0
    if name in {"dqn_eps_min", "pg_entropy_min", "warmup_budget_fraction"}:
        low = 0.0 if name != "warmup_budget_fraction" else math.nextafter(0.0, math.inf)
        high = math.nextafter(1.0, 0.0) if name == "warmup_budget_fraction" else 1.0
    if spec["type"] == "int":
        low = max(low, 1)
    return float(spec.get("hard_low", low)), float(spec.get("hard_high", high))


def _outside(name: str, spec: Mapping[str, Any], current: Any, direction: str, initial: Mapping[str, Any]) -> Any | None:
    if spec["type"] == "categorical":
        extensions = spec.get("extension_choices", _EXTENSION_CHOICES.get(name, []))
        if current not in extensions:
            return None
        choices = sorted(set(extensions))
        index = choices.index(current) + (-1 if direction == "low" else 1)
        return choices[index] if 0 <= index < len(choices) else None
    factor = spec.get("expansion_factor")
    if factor is None:
        if name in {"warmup_budget_fraction", "epsilon_anneal_budget_ratio"}:
            factor = 2.0
        elif spec.get("log"):
            factor = 10.0
    if factor is not None:
        value = current / factor if direction == "low" else current * factor
    else:
        width = (initial["high"] - initial["low"]) / 2
        value = current + (-width if direction == "low" else width)
    if name == "warmup_budget_fraction" and value >= 1:
        # There is no greatest legal value below one. nextafter(1, 0) would
        # round warmup to the full budget in ordinary finite-budget runs.
        return None
    low, high = _domain_bounds(name, spec)
    value = _cast(max(low, min(high, value)), spec)
    if not math.isfinite(value) or value == current or value < low or value > high:
        return None
    return value


def _inward(bad: Any, good: Any, spec: Mapping[str, Any]) -> Any | None:
    if spec["type"] == "categorical":
        choices = _choices(spec)
        first, last = choices.index(bad), choices.index(good)
        if abs(first - last) <= 1:
            return None
        return choices[(first + last) // 2]
    candidate = _middle(bad, good, spec)
    return None if candidate == bad or candidate == good else candidate


def _set_boundary(spec: dict[str, Any], direction: str, value: Any) -> None:
    if spec["type"] == "categorical":
        choices = _choices(spec)
        if value not in choices:
            choices.append(value)
            choices.sort()
        spec["choices"] = [choice for choice in choices if choice >= value] if direction == "low" else [choice for choice in choices if choice <= value]
    else:
        spec[direction] = value


def _endpoints(spec: Mapping[str, Any]) -> tuple[Any, Any]:
    if spec["type"] == "categorical":
        choices = _choices(spec)
        return choices[0], choices[-1]
    return spec["low"], spec["high"]


def run_range_search(plan: Mapping[str, Any], evaluate: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    """Diagnose ranges using ``evaluate(parameters, seeds=[...])`` aggregates.

    The callback may accept only parameters; then uncertain paired comparisons
    remain unresolved rather than manufacturing independent replication.  Results
    should expose ``seed_results`` with seed, score, status and eligibility.
    """
    original = deepcopy(plan["search_space"])
    ranges = deepcopy(original)
    options = plan.get("range_search", {})
    seeds = list(options.get("seeds", [11, 29]))
    confirmation = options.get("confirmation_seed", 47)
    maximum_extra = int(options.get("max_extra_values_per_direction", 2))
    improvement = float(options.get("score_improvement", 0.02))
    if maximum_extra < 0 or improvement < 0 or len(set(seeds)) < 2:
        raise ValueError("range search requires >=2 unique seeds, nonnegative extra values and improvement")
    initial = initial_reference(original)
    reference = dict(initial)
    measurements: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    states = {name: {"low": "unchanged", "high": "unchanged"} for name in original}
    try:
        signature = inspect.signature(evaluate)
        accepts_seeds = "seeds" in signature.parameters or any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values())
    except (TypeError, ValueError):
        accepts_seeds = False

    def measure(parameters: Mapping[str, Any], selected_seeds: list[Any] | None = None) -> dict[str, Any]:
        selected_seeds = list(selected_seeds or seeds)
        identifier = parameter_id(parameters)
        key = (identifier, tuple(selected_seeds))
        if key not in measurements:
            result = dict(evaluate(dict(parameters), seeds=selected_seeds) if accepts_seeds else evaluate(dict(parameters)))
            result.update(parameters=dict(parameters), parameter_id=identifier, requested_seeds=selected_seeds)
            measurements[key] = result
            evaluations.append(result)
        return measurements[key]

    def fresh_evidence() -> list[dict[str, Any]]:
        """Never let an old successful subset replace a larger failed cohort."""
        latest = {}
        for result in evaluations:
            identifier = result["parameter_id"]
            previous = latest.get(identifier)
            if previous is None or len(result["requested_seeds"]) >= len(previous["requested_seeds"]):
                latest[identifier] = result
        return list(latest.values())

    def latest_result(result: dict[str, Any]) -> dict[str, Any]:
        return next(item for item in fresh_evidence() if item["parameter_id"] == result["parameter_id"])

    def outcome(status: str, reference_result: Mapping[str, Any]) -> dict[str, Any]:
        if status == "COMPLETED":
            for name, entry in list(ranges.items()):
                if entry["type"] != "categorical" and entry["low"] == entry["high"]:
                    ranges[name] = {"type": "categorical", "choices": [entry["low"]]}
        return {"status": status, "ranges": ranges if status == "COMPLETED" else None,
                "initial_ranges": original, "reference": reference, "reference_result": dict(latest_result(reference_result)),
                "evaluations": fresh_evidence(), "measurement_history": evaluations,
                "decisions": decisions, "boundary_status": states,
                "range_status": "unresolved" if any(value in {"unknown", "unresolved", "conditional_discrete"} for item in states.values() for value in item.values()) else "diagnosed",
                "legal_combination_conditions": ["Runtime validates every complete configuration at the common diagnostic budget.",
                                                   "Single-parameter diagnoses are conditional on the recorded reference configuration."]}

    reference_result = measure(reference)
    if _system_error(reference_result):
        return outcome("SYSTEM_ERROR", reference_result)
    # The original 1+2d measurements also provide candidate replacement references.
    endpoints = {}
    for name, spec in original.items():
        for direction, value in zip(("low", "high"), _endpoints(spec)):
            parameters = {**reference, name: value}
            endpoints[(name, direction)] = measure(parameters)
            if _system_error(endpoints[(name, direction)]):
                return outcome("SYSTEM_ERROR", reference_result)
    if not _eligible(reference_result):
        candidates = [result for result in endpoints.values() if _eligible(result)]
        if not candidates:
            generator = random.Random(options.get("random_seed", 0))
            for _ in range(2 * len(original)):
                parameters = {}
                for name, spec in original.items():
                    if spec["type"] == "categorical":
                        parameters[name] = generator.choice(_choices(spec))
                    else:
                        value = math.exp(generator.uniform(math.log(spec["low"]), math.log(spec["high"]))) if spec.get("log") else generator.uniform(spec["low"], spec["high"])
                        parameters[name] = _cast(value, spec)
                result = measure(parameters)
                if _system_error(result):
                    return outcome("SYSTEM_ERROR", reference_result)
                if _eligible(result):
                    candidates.append(result)
                    break
        if not candidates:
            decisions.append({"action": "stop", "reason": "no feasible reference after endpoints and bounded joint proposals"})
            return outcome("NO_FEASIBLE_CONFIGURATION", reference_result)
        reference_result = max(candidates, key=lambda result: (float(result["score"]), result["parameter_id"]))
        reference = dict(reference_result["parameters"])
        decisions.append({"action": "replace_reference", "old_reference": initial, "reference": reference,
                          "reason": "initial reference ineligible; redo comparisons with a runnable reference"})
        for name, spec in original.items():
            for direction, value in zip(("low", "high"), _endpoints(spec)):
                endpoints[(name, direction)] = measure({**reference, name: value})
                if _system_error(endpoints[(name, direction)]):
                    return outcome("SYSTEM_ERROR", reference_result)

    def confirm(result: dict[str, Any], compared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result, compared = latest_result(result), latest_result(compared)
        shared_seeds = list(dict.fromkeys([*result["requested_seeds"], *compared["requested_seeds"]]))
        if accepts_seeds and (result["requested_seeds"] != shared_seeds or compared["requested_seeds"] != shared_seeds):
            result = measure(result["parameters"], shared_seeds)
            compared = measure(compared["parameters"], shared_seeds)
            decisions.append({"action": "paired_alignment", "parameter_id": result["parameter_id"], "seeds": shared_seeds})
        if _ambiguous(result, compared):
            if accepts_seeds and confirmation not in seeds:
                expanded = [*seeds, confirmation]
                result = measure(result["parameters"], expanded)
                compared = measure(compared["parameters"], expanded)
                decisions.append({"action": "paired_confirmation", "parameter_id": result["parameter_id"], "seeds": expanded})
            else:
                decisions.append({"action": "unresolved", "parameter_id": result["parameter_id"], "reason": "independent confirmation seed unavailable"})
        return result, compared

    for name, initial_spec in original.items():
        for direction, endpoint in zip(("low", "high"), _endpoints(initial_spec)):
            compared = latest_result(reference_result)
            if not _eligible(compared):
                states[name][direction] = "unresolved"
                decisions.append({"parameter": name, "direction": direction, "action": "retain",
                                  "range": deepcopy(ranges[name]), "reason": "latest reference cohort is ineligible; do not use an older successful subset as stability evidence"})
                continue
            result, compared = confirm(endpoints[(name, direction)], compared)
            if _system_error(result) or _system_error(compared):
                return outcome("SYSTEM_ERROR", reference_result)
            before = deepcopy(ranges[name])
            if not _eligible(compared):
                states[name][direction] = "unresolved"
                decisions.append({"parameter": name, "direction": direction, "action": "retain",
                                  "range": before, "reason": "reference failed the added paired seed; boundary evidence is unresolved"})
                continue
            if _numerical_count(result) >= 2:
                # Keep the closest already measured eligible point on the inward
                # side; no efficiency failure is used as numerical instability.
                available = []
                for item in fresh_evidence():
                    parameters = item["parameters"]
                    if _eligible(item) and all(parameters[key] == reference[key] for key in reference if key != name):
                        value = parameters[name]
                        if (direction == "low" and value > endpoint) or (direction == "high" and value < endpoint):
                            available.append(value)
                if not available:
                    states[name][direction] = "unresolved"
                    continue
                good = min(available) if direction == "low" else max(available)
                bad = endpoint
                reference_lost = False
                for _ in range(maximum_extra):
                    value = _inward(bad, good, initial_spec)
                    if value is None:
                        break
                    tested, baseline = confirm(measure({**reference, name: value}), compared)
                    decisions.append({"parameter": name, "direction": direction, "action": "test_inward", "value": value, "from_unstable": bad, "towards_stable": good, "parameter_id": tested["parameter_id"]})
                    if _system_error(tested) or _system_error(baseline):
                        return outcome("SYSTEM_ERROR", reference_result)
                    if not _eligible(baseline):
                        reference_lost = True
                        break
                    if _eligible(tested):
                        good = value
                        break
                    if _numerical_count(tested) >= 2:
                        bad = value
                    else:
                        states[name][direction] = "unresolved"
                        break
                if reference_lost:
                    states[name][direction] = "unresolved"
                    decisions.append({"parameter": name, "direction": direction, "action": "retain",
                                      "range": before, "reason": "reference failed inward-probe confirmation; retain the previous boundary"})
                    continue
                _set_boundary(ranges[name], direction, good)
                if states[name][direction] != "unresolved":
                    states[name][direction] = "conditional_stability_boundary"
                decisions.append({"parameter": name, "direction": direction, "action": "contract", "before": before, "after": deepcopy(ranges[name]), "reference": reference, "reason": "repeated numerical instability; conservative nearest tested eligible boundary"})
            elif _improved(result, compared, improvement):
                current, current_result = endpoint, result
                states[name][direction] = "unknown"
                for _ in range(maximum_extra):
                    probe_spec = dict(ranges[name])
                    if name in plan.get("extension_choices", {}):
                        probe_spec["extension_choices"] = plan["extension_choices"][name]
                    value = _outside(name, probe_spec, current, direction, initial_spec)
                    if value is None:
                        states[name][direction] = "legal_boundary_reached"
                        break
                    tested, baseline = confirm(measure({**reference, name: value}), current_result)
                    decisions.append({"parameter": name, "direction": direction, "action": "test_outward", "value": value, "parameter_id": tested["parameter_id"]})
                    if _system_error(tested) or _system_error(baseline):
                        return outcome("SYSTEM_ERROR", reference_result)
                    if not _eligible(baseline):
                        ranges[name] = before
                        states[name][direction] = "unresolved"
                        decisions.append({"parameter": name, "direction": direction, "action": "retain",
                                          "range": before, "reason": "outward baseline failed the added paired seed; discard tentative expansion"})
                        break
                    if not _improved(tested, baseline, improvement):
                        states[name][direction] = "retained_after_outward_test"
                        break
                    _set_boundary(ranges[name], direction, value)
                    decisions.append({"parameter": name, "direction": direction, "action": "expand", "old_boundary": current, "new_boundary": value, "reference": reference, "reason": "all paired seeds improve and mean improvement meets threshold"})
                    current, current_result = value, tested
            else:
                reason = "stable low score or insufficient paired improvement; retain range"
                if not _eligible(result):
                    reason = "ineligible complete configuration; do not globally remove a single parameter value"
                    states[name][direction] = "unresolved"
                elif _ambiguous(result, compared) or _paired_differences(result, compared) is None:
                    states[name][direction] = "unresolved"
                decisions.append({"parameter": name, "direction": direction, "action": "retain", "range": deepcopy(ranges[name]), "reason": reason, "parameter_id": result["parameter_id"]})

        # An unstable interior with eligible values on both sides violates a
        # contiguous-stability interpretation.  Export tested eligible choices.
        points: dict[Any, dict[str, Any]] = {}
        for item in evaluations:
            if all(item["parameters"][key] == reference[key] for key in reference if key != name):
                value = item["parameters"][name]
                previous = points.get(value)
                if previous is None or len(_seeds(item)) >= len(_seeds(previous)):
                    points[value] = item
        ordered = sorted(points)
        eligible_values = [value for value in ordered if _eligible(points[value])]
        interleaved = any(_numerical_count(points[value]) >= 2 and any(good < value for good in eligible_values) and any(good > value for good in eligible_values) for value in ordered)
        if interleaved:
            ranges[name] = {"type": "categorical", "choices": eligible_values}
            states[name] = {"low": "conditional_discrete", "high": "conditional_discrete"}
            decisions.append({"parameter": name, "action": "tested_discrete_only", "choices": eligible_values, "initial_range": initial_spec, "reference": reference, "reason": "interleaved measured numerical stability; no contiguous interval claim"})
    return outcome("COMPLETED", reference_result)
