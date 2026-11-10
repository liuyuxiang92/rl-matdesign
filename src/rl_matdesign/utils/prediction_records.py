"""Snapshot completed predictions without making another oracle call."""
from __future__ import annotations

import copy
import json


def prediction_record(predictor, candidate, prediction, *, k: float = 1.0) -> dict:
    """Use a cache-only plug-in hook, or the tuple already returned by predict.

    A predictor exposing transformed/raw interfaces must implement the optional
    ``prediction_record(candidate, prediction, *, k)`` hook to expose its raw
    statistics without rerunning inference. Missing diagnostics are explicit;
    they never cause a successfully scored candidate to be retried.
    """
    mean, std = map(float, prediction)
    row = {"reward": mean}
    try:
        row["candidate_json"] = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        row["candidate_repr"] = repr(candidate)
    hook = getattr(predictor, "prediction_record", None)
    try:
        if callable(hook):
            fields = hook(candidate, prediction, k=k)
            if not isinstance(fields, dict):
                raise TypeError("prediction_record must return a dict")
            row.update(copy.deepcopy(fields))
        elif hasattr(predictor, "predict_raw") or hasattr(predictor, "per_objective_stats"):
            # A second call could change a stochastic result or rebuild an MD
            # structure. Do not label the transformed reward a raw property.
            row.update(predict_mean=mean, predict_std=std,
                       prediction_record_error="Raw properties unavailable: predictor needs a cache-only prediction_record hook")
        else:
            row.update(dp_mean=mean, dp_std=std, dp_mean_minus_std=mean - k * std)
    except Exception as exc:
        row.update(predict_mean=mean, predict_std=std,
                   prediction_record_error=f"{type(exc).__name__}: {exc}")
    # The reward belongs to this particular evaluation, even for repeated input.
    row["reward"] = mean
    return row


def objective_record(raw_mean, stats, details) -> dict:
    """The generated.csv property and gate column names, from captured values."""
    row = {"dp_mean": float(raw_mean)}
    for name, (mean, std) in stats.items():
        row[f"obj_{name}_mean"] = float(mean)
        row[f"obj_{name}_std"] = float(std)
    if details["type"] == "gated":
        row["reward_objective"] = float(details["objective"])
        row["gate_passed"] = bool(details["gate_passed"])
        for gate in details["gates"]:
            prefix = f"gate_{gate['name']}"
            row[f"{prefix}_value"] = float(gate["score"])
            row[f"{prefix}_passed"] = bool(gate["passed"])
            row[f"{prefix}_violation"] = float(gate["violation"])
    return row
