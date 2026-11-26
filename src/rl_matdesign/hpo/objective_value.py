"""Read an HPO measurement from one completed candidate evaluation."""
from __future__ import annotations

from collections.abc import Mapping
import math


def objective_value(record: Mapping, objective: Mapping) -> float:
    """Select the exact recorded reward or a named physical property.

    Reward belongs to the evaluation snapshot, including any configured
    uncertainty, scalarization, calibration and gate penalties. Never infer it
    from a property of the same name or reconstruct it from a newer predictor.
    Both reference calibration and final scoring must read this same value.
    """
    metric = objective.get("metric", "property")
    if metric == "reward":
        prediction = record.get("prediction_record")
        if not isinstance(prediction, Mapping) or "reward" not in prediction:
            raise ValueError("Evaluated candidate lacks recorded reward in prediction_record.reward")
        value = prediction["reward"]
        label = "Recorded reward"
    elif metric == "property":
        field = objective["field"]
        properties = record.get("properties")
        if not isinstance(properties, Mapping) or field not in properties:
            raise ValueError(f"Evaluated candidate lacks objective property {field!r}")
        value = properties[field]
        label = f"Objective property {field!r}"
    else:
        raise ValueError(f"Unknown HPO evaluation metric {metric!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return value
