"""Direction-aware HPO scores from complete candidate archives, never reward CSVs."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math

from .spec import phase_efficiency


def quality(value: float, objective: Mapping) -> float:
    """Apply the campaign's fixed transform; do not normalize against a trial."""
    threshold = objective["threshold"]
    minimize = objective["direction"] == "minimize"
    norm = objective["normalization"]
    if norm["kind"] == "positive_threshold_ratio":
        if value <= 0 or threshold <= 0:
            raise ValueError("positive_threshold_ratio requires positive property values and threshold")
        # Rescale before adding, avoiding overflow at very large finite values.
        scale = max(threshold, value)
        t, v = threshold / scale, value / scale
        return (t if minimize else v) / (t + v)
    if norm["kind"] != "sigmoid":
        raise ValueError(f"Unknown normalization kind {norm['kind']!r}")
    scale = norm["scale"]
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Sigmoid quality scale must be calibrated and positive")
    difference = (threshold - value) if minimize else (value - threshold)
    # Opposite finite extremes can overflow on subtraction while their scaled
    # difference is finite. Divide first only in that case.
    z = difference / scale if math.isfinite(difference) else (
        (threshold / scale - value / scale) if minimize else (value / scale - threshold / scale))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _identity(value):
    # Preserve scalar types (1 != '1' != True), mapping names and sequence order.
    # Environment-provided canonical identities determine material equivalence.
    if value is None:
        raise ValueError("Missing canonical candidate identity")
    if isinstance(value, Mapping):
        return ("mapping", tuple(sorted((_identity(k), _identity(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_identity(v) for v in value))
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("Candidate identity must be JSON serializable")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Candidate identity cannot contain nonfinite values")
    return (type(value).__name__, value)


def _count(ledger, name):
    value = ledger.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ledger.{name} must be a nonnegative integer")
    return value


def score_records(records, ledger: Mapping, plan: Mapping, phase: str) -> dict:
    """Score one seed/job; partial or invalid jobs always receive finite -1.

    Records are deduplicated by canonical identity. Conflicting repeated
    snapshots are rejected rather than cherry-picked. The successful-unique
    denominator must agree with the ledger, including constraint failures.
    """
    if phase == "training":
        phase = "train"
    efficiency = phase_efficiency(plan, phase)
    result = {"status": str(ledger.get("status", "INCOMPLETE")).upper(),
              "eligible_for_selection": False, "score": -1.0,
              "raw_metrics": {}, "normalized_metrics": {}}
    objective = plan["objective"]
    field = objective["field"]
    try:
        attempts = _count(ledger, "attempts_started")
        unique = _count(ledger, "unique_evaluated")
        target = _count(ledger, "target_unique")
        if target <= 0 or unique > target or attempts < unique:
            raise ValueError("Inconsistent candidate ledger counts")
        candidates = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Candidate records must be mappings")
            # Failure and preparation diagnostics carry no evaluated candidate.
            evaluation_status = record.get("evaluation_status", record.get("status", "evaluated"))
            if str(evaluation_status).lower() in {"failed", "invalid", "preparation_only", "evaluation_failed", "rollout_failed"}:
                continue
            key = _identity(record.get("identity"))
            properties = record.get("properties")
            if not isinstance(properties, Mapping) or field not in properties:
                raise ValueError(f"Evaluated candidate lacks objective property {field!r}")
            value = properties[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Objective property {field!r} must be finite numeric data")
            if objective["normalization"]["kind"] == "positive_threshold_ratio" and value <= 0:
                raise ValueError("positive_threshold_ratio requires positive property values; choose sigmoid for this domain")
            passed = record.get("constraint_passed")
            if not isinstance(passed, bool):
                raise ValueError("constraint_passed must be an explicit boolean")
            units = record.get("units", {})
            if objective.get("unit") is not None and field in units and units[field] != objective["unit"]:
                raise ValueError(f"Objective property unit mismatch for {field!r}")
            snapshot = (float(value), passed)
            if key in candidates and candidates[key] != snapshot:
                raise ValueError("Conflicting evaluated snapshots for one canonical candidate")
            candidates[key] = snapshot
        if len(candidates) != unique:
            raise ValueError(f"Archive has {len(candidates)} evaluated identities, ledger reports {unique}")
        minimize = objective["direction"] == "minimize"
        inclusive = objective.get("threshold_inclusive", False)
        threshold = objective["threshold"]
        feasible_values = sorted((v for v, passed in candidates.values() if passed), reverse=not minimize)
        def qualified(value):
            if minimize:
                return value <= threshold if inclusive else value < threshold
            return value >= threshold if inclusive else value > threshold
        qualified_count = sum(qualified(v) for v in feasible_values)
        slots = max(1, math.ceil(plan["metrics"]["best_fraction"] * target))
        best_values = feasible_values[:slots]
        # Dividing first avoids overflowing the sum of finite positive values.
        mean = math.fsum(v / len(best_values) for v in best_values) if best_values else None
        best = feasible_values[0] if feasible_values else None
        ratio = attempts / unique if unique else None
        raw = {"qualified_fraction": qualified_count / unique if unique else 0.0,
               "qualified_count": qualified_count, "constraint_passed_count": len(feasible_values),
               "best_fraction_mean": mean, "best_value": best, "attempts_per_unique": ratio,
               "best_fraction_slots": slots, "best_fraction_filled": len(best_values),
               "unique_evaluated": unique, "attempts_started": attempts, "target_unique": target}
        result["raw_metrics"] = raw
        if (objective["normalization"].get("calibration") == "reference"
                and objective["normalization"].get("scale") is None):
            if result["status"] == "COMPLETED":
                if unique != target:
                    result["status"] = "INCOMPLETE"
                elif Decimal(attempts) >= Decimal(str(efficiency["max_attempts_per_unique"])) * unique:
                    result["status"] = "INFEASIBLE_EFFICIENCY"
                else:
                    result.update(status="QUALITY_CALIBRATION_REQUIRED",
                                  error="No fixed quality scale is available for this campaign")
            return result
        normalized = {"qualified_fraction": raw["qualified_fraction"],
                      "best_fraction_quality": quality(mean, objective) * len(best_values) / slots if mean is not None else 0.0,
                      "best_value_quality": quality(best, objective) if best is not None else 0.0,
                      "sampling_efficiency": unique / attempts if attempts else 0.0}
        result["normalized_metrics"] = normalized
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result.update(status="INVALID_RECORDS", error=str(exc))
        return result
    if result["status"] != "COMPLETED" or unique != target:
        if result["status"] == "COMPLETED":
            result["status"] = "INCOMPLETE"
        return result
    if Decimal(attempts) >= Decimal(str(efficiency["max_attempts_per_unique"])) * unique:
        result["status"] = "INFEASIBLE_EFFICIENCY"
        return result
    weights = plan["metrics"]["training_weights" if phase == "train" else "generation_weights"]
    score = math.fsum(weights[name] * normalized[name] for name in weights)
    if not math.isfinite(score) or score < -1e-12 or score > 1 + 1e-12:
        result.update(status="INVALID_RECORDS", error="Normalized score is outside [0, 1]")
        return result
    result.update(eligible_for_selection=True, score=min(1.0, max(0.0, score)))
    return result
