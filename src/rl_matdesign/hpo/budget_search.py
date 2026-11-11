"""Finite, bidirectional budget calibration independent of an RL runtime.

All comparisons are on completed paired-seed aggregates.  A selected budget is
an economical choice within the measured grid; it is not a proof of convergence.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import json
import math
from typing import Any, Mapping, Sequence


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    decimal = _decimal(value, name)
    if decimal != decimal.to_integral_value() or decimal < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(decimal)


def minimum_unique_for_preparation(preparation_attempts: int, ratio_limit: float = 1.5) -> int:
    """Smallest B with (preparation_attempts + B) / B < ratio_limit."""
    preparation = _integer(preparation_attempts, "preparation_attempts", minimum=0)
    ratio = _decimal(ratio_limit, "ratio_limit")
    if ratio <= 1:
        raise ValueError("ratio_limit must be > 1")
    return int((Decimal(preparation) / (ratio - 1)).to_integral_value(rounding=ROUND_FLOOR)) + 1


def _phase_config(plan: Mapping[str, Any], phase: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if phase not in {"train", "generation"}:
        raise ValueError("phase must be train or generation")
    budget = plan["budget"]
    return budget, budget[phase]


def _grid_values(low: Decimal, initial: Decimal, high: Decimal, factor: Decimal) -> list[Decimal]:
    if low <= 0 or not low <= initial <= high:
        raise ValueError("budget bounds must satisfy 0 < minimum <= initial <= maximum")
    if factor <= 1:
        raise ValueError("budget factor must be > 1")
    values = {low, initial, high}
    for multiply in (False, True):
        value = initial
        for _ in range(10000):
            following = value * factor if multiply else value / factor
            if (multiply and following >= high) or (not multiply and following <= low):
                break
            if following == value:
                raise ValueError("budget factor does not produce distinct grid points")
            values.add(following)
            value = following
        else:
            raise ValueError("budget grid exceeds 10000 steps; increase factor or narrow bounds")
    return sorted(values)


def budget_grid(
    plan: Mapping[str, Any], phase: str, space_size: int | None = None,
    minimum_unique: int = 1,
) -> list[int]:
    """Return ascending unique integer budgets including both configured bounds.

    ``fraction_of_space`` requires an exact canonical-candidate count.  Runtime
    prerequisites filter the grid; they never silently increase its upper bound.
    ``absolute_unique`` uses min_unique / initial_unique / max_unique.
    """
    budget, config = _phase_config(plan, phase)
    minimum = max(_integer(minimum_unique, "minimum_unique"), _integer(config.get("min_unique", 1), "min_unique"))
    factor = _decimal(config.get("factor", 2), "factor")
    mode = budget.get("mode", "fraction_of_space")
    if space_size is not None:
        space_size = _integer(space_size, "exact space_size")
    if mode == "fraction_of_space":
        if space_size is None:
            raise ValueError("fraction_of_space requires an exact finite canonical-candidate space_size")
        low = _decimal(config["min_fraction"], "min_fraction")
        initial = _decimal(config["initial_fraction"], "initial_fraction")
        high = _decimal(config["max_fraction"], "max_fraction")
        if high > 1:
            raise ValueError("max_fraction cannot exceed 1")
        scale = Decimal(space_size)
    elif mode == "absolute_unique":
        low = Decimal(_integer(config["min_unique"], "min_unique"))
        initial = Decimal(_integer(config["initial_unique"], "initial_unique"))
        high = Decimal(_integer(config["max_unique"], "max_unique"))
        if space_size is not None and high > space_size:
            raise ValueError("max_unique exceeds the exact design space")
        scale = Decimal(1)
    else:
        raise ValueError(f"unsupported budget mode: {mode}")
    raw = _grid_values(low, initial, high, factor)
    maximum = int((high * scale).to_integral_value(rounding=ROUND_CEILING))
    if space_size is not None:
        maximum = min(maximum, space_size)
    result = sorted({
        int((value * scale).to_integral_value(rounding=ROUND_CEILING))
        for value in raw
        if minimum <= int((value * scale).to_integral_value(rounding=ROUND_CEILING)) <= maximum
    })
    if not result:
        raise ValueError(f"no legal {phase} budget: required minimum {minimum}, configured maximum {maximum}")
    return result


def budget_schedule(
    plan: Mapping[str, Any], phase: str, space_size: int | None = None,
    minimum_unique: int = 1,
) -> list[int]:
    """Initial-nearest budget first, then alternate lower and higher grid points."""
    budgets = budget_grid(plan, phase, space_size, minimum_unique)
    budget, config = _phase_config(plan, phase)
    if budget.get("mode", "fraction_of_space") == "fraction_of_space":
        initial = int((_decimal(config["initial_fraction"], "initial_fraction") * Decimal(space_size)).to_integral_value(rounding=ROUND_CEILING))
    else:
        initial = int(config["initial_unique"])
    index = min(range(len(budgets)), key=lambda position: (abs(budgets[position] - initial), budgets[position]))
    order = [budgets[index]]
    for offset in range(1, len(budgets)):
        if index - offset >= 0:
            order.append(budgets[index - offset])
        if index + offset < len(budgets):
            order.append(budgets[index + offset])
    return order


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) and math.isfinite(value)


def _parameter_id(result: Mapping[str, Any]) -> str:
    if result.get("parameter_id") is not None:
        return str(result["parameter_id"])
    return json.dumps(result.get("parameters", {}), sort_keys=True, separators=(",", ":"), default=str)


def _tolerances(plan: Mapping[str, Any], phase: str) -> dict[str, float]:
    budget, phase_config = _phase_config(plan, phase)
    configured = {**budget.get("tolerances", {}), **phase_config.get("tolerances", {})}
    defaults: dict[str, Any] = {
        "score": 0.01, "qualified_fraction": 0.02, "attempts_per_unique": 0.10,
        "best_fraction_mean": {"kind": "threshold_fraction", "value": 0.005},
        "best_value": {"kind": "threshold_fraction", "value": 0.005},
    }
    result = {}
    for key, fallback in defaults.items():
        value = configured.get(key, fallback)
        if isinstance(value, Mapping):
            kind = value.get("kind", "absolute")
            magnitude = _decimal(value["value"], key)
            if kind == "threshold_fraction":
                magnitude *= abs(_decimal(plan["objective"]["threshold"], "objective.threshold"))
            elif kind != "absolute":
                raise ValueError(f"unsupported tolerance kind: {kind}")
        else:
            magnitude = _decimal(value, key)
        if magnitude < 0:
            raise ValueError(f"{key} tolerance must be nonnegative")
        result[key] = float(magnitude)
    return result


def select_budget(results: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Select the smallest measured sufficient budget, and report its evidence.

    Controller supplies ``plan['_budget_grids'][phase]`` for completeness checks.
    Failed jobs and absent scientific metrics can never establish a plateau.
    """
    tolerances = _tolerances(plan, phase)
    direction = plan["objective"]["direction"]
    if direction not in {"minimize", "maximize"}:
        raise ValueError("objective.direction must be minimize or maximize")
    objective_sign = 1 if direction == "minimize" else -1
    metrics = ["qualified_fraction", "best_fraction_mean", "best_value"]
    if phase == "train":
        metrics.append("attempts_per_unique")
    all_results = [dict(result) for result in results]
    measured_budgets = sorted({int(result["budget"]) for result in all_results if result.get("budget") is not None})
    declared = plan.get("_budget_grids", {}).get(phase)
    expected = sorted(set(map(int, declared))) if declared is not None else measured_budgets
    output: dict[str, Any] = {
        "selected_budget": None, "selected": None, "reference": None,
        "convergence_status": "NO_FEASIBLE_CONFIGURATION", "lower_bound_reached": False,
        "upper_bound_reached": False,
        "evidence": {"expected_budgets": expected, "grid_source": "controller" if declared is not None else "observed_results",
                     "tolerances": tolerances, "excluded": [], "comparisons": []},
    }

    efficiency = plan.get("efficiency", {})
    ratio_limit = float(efficiency.get(phase, {}).get("max_attempts_per_unique", efficiency.get("max_attempts_per_unique", 1.5)))
    valid = []
    for result in all_results:
        raw = result.get("raw_metrics", {})
        is_complete = bool(result.get("eligible_for_selection", False)) and _finite(result.get("score"))
        is_complete = is_complete and all(_finite(raw.get(key)) for key in ("qualified_fraction", "attempts_per_unique"))
        is_complete = is_complete and raw.get("attempts_per_unique", 0) >= 1
        is_complete = is_complete and raw.get("attempts_per_unique", math.inf) < ratio_limit
        if not is_complete:
            output["evidence"]["excluded"].append({"parameter_id": _parameter_id(result), "budget": result.get("budget"), "reason": result.get("status", "INELIGIBLE_OR_INVALID_METRICS")})
            continue
        valid.append(result)
    if not valid:
        return output
    scientific = [result for result in valid if all(_finite(result.get("raw_metrics", {}).get(key)) for key in ("best_fraction_mean", "best_value"))]
    if not scientific:
        output["convergence_status"] = "NO_FEASIBLE_MATERIAL_EVIDENCE"
        return output

    def tie_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
        raw = result["raw_metrics"]
        return (int(result["budget"]), float(raw["attempts_per_unique"]), objective_sign * float(raw["best_fraction_mean"]), _parameter_id(result))

    highest_score = max(float(result["score"]) for result in valid)
    score_band = [result for result in scientific if highest_score - float(result["score"]) <= tolerances["score"] + 1e-12]
    if not score_band:
        output["convergence_status"] = "EVIDENCE_INCOMPLETE"
        output["evidence"]["reason"] = "highest-score band has no finite scientific quality evidence"
        return output
    reference = min(score_band, key=tie_key)
    output["reference"] = reference
    output["evidence"]["highest_score"] = highest_score

    def degradation(candidate: Mapping[str, Any], compared: Mapping[str, Any], key: str) -> float:
        if key == "score":
            return float(compared["score"]) - float(candidate["score"])
        current, other = candidate["raw_metrics"].get(key), compared["raw_metrics"].get(key)
        if not _finite(current):
            return math.inf
        if not _finite(other):
            return -math.inf
        sign = -1 if key == "qualified_fraction" else (1 if key == "attempts_per_unique" else objective_sign)
        return sign * (float(current) - float(other))

    sufficient = []
    for result in scientific:
        differences = {key: degradation(result, reference, key) for key in metrics}
        acceptable = highest_score - float(result["score"]) <= tolerances["score"] + 1e-12 and all(value <= tolerances[key] + 1e-12 for key, value in differences.items())
        output["evidence"]["comparisons"].append({"parameter_id": _parameter_id(result), "budget": result["budget"], "sufficient": acceptable, "one_way_degradation": differences})
        if acceptable:
            sufficient.append(result)
    selected = min(sufficient, key=lambda result: (int(result["budget"]), -float(result["score"]), *tie_key(result)[1:]))
    selected_budget = int(selected["budget"])
    parameter_id = _parameter_id(selected)
    output.update(selected_budget=selected_budget, selected=selected,
                  lower_bound_reached=bool(expected and selected_budget == expected[0]),
                  upper_bound_reached=bool(expected and selected_budget == expected[-1]))
    expected_higher = [value for value in expected if value > selected_budget]
    same_configuration = {int(result["budget"]): result for result in all_results if _parameter_id(result) == parameter_id}
    missing = [value for value in expected_higher if value not in same_configuration or same_configuration[value] not in scientific]
    output["evidence"]["missing_or_ineligible_higher_budgets"] = missing
    changes = []
    for value in expected_higher:
        result = same_configuration.get(value)
        if result is None or result not in scientific:
            continue
        differences = {key: degradation(result, selected, key) for key in ["score", *metrics]}
        changes.append({"budget": value, "changes": differences,
                        "within_tolerance": all(abs(change) <= tolerances[key] + 1e-12 for key, change in differences.items()),
                        "degraded": any(change > tolerances[key] + 1e-12 for key, change in differences.items()),
                        "improved": any(change < -tolerances[key] - 1e-12 for key, change in differences.items())})
    output["evidence"]["higher_budget_changes"] = changes
    # The selected configuration may plateau while a different measured
    # configuration still improves at a larger budget. Keep the economical
    # selection rule separate, but do not call the complete grid a plateau in
    # that case. Every phase uses the same quality metrics as its local check.
    global_improvements = []
    for result in scientific:
        if int(result["budget"]) <= selected_budget:
            continue
        differences = {key: degradation(result, selected, key) for key in ["score", *metrics]}
        improved = {key: -value for key, value in differences.items()
                    if value < -tolerances[key] - 1e-12}
        if improved:
            global_improvements.append({"parameter_id": _parameter_id(result),
                                        "budget": int(result["budget"]),
                                        "improvements": improved,
                                        "one_way_degradation": differences})
    output["evidence"]["global_higher_budget_improvements"] = global_improvements
    # Every scheduled (configuration, budget) cell must be terminal.  A terminal
    # failure is evidence, but a missing job cannot support a complete scan.
    parameter_ids = {_parameter_id(result) for result in all_results}
    measured_cells = {(_parameter_id(result), int(result["budget"])) for result in all_results}
    missing_cells = [{"parameter_id": key, "budget": value} for key in sorted(parameter_ids) for value in expected if (key, value) not in measured_cells]
    output["evidence"]["missing_cells"] = missing_cells
    nonterminal_cells = [{"parameter_id": _parameter_id(result), "budget": result["budget"]}
                         for result in all_results if result.get("status") in {"PENDING", "QUEUED", "RUNNING", "RETRYING", "PARTIAL", "INCOMPLETE"}]
    output["evidence"]["nonterminal_cells"] = nonterminal_cells
    required_levels = int(plan["budget"].get("upper_confirmation_levels", 2))
    if missing or missing_cells or nonterminal_cells:
        status = "EVIDENCE_INCOMPLETE"
    elif any(change["degraded"] for change in changes):
        status = "BUDGET_DEGRADATION"
    elif global_improvements:
        status = "BUDGET_LIMIT_REACHED"
    elif len(changes) >= required_levels and all(change["within_tolerance"] for change in changes):
        status = "OBSERVED_PLATEAU"
    else:
        status = "BUDGET_LIMIT_REACHED"
    output["convergence_status"] = status
    return output
