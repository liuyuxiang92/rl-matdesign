"""CSV views of completed predictions, without a second oracle evaluation."""
from __future__ import annotations

import copy
import json


def metadata_record(metadata: dict) -> dict:
    """Use generated.csv's JSON and scalar metadata columns."""
    if not isinstance(metadata, dict):
        raise TypeError("candidate metadata must be a mapping")
    row = {"metadata_json": json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False)}
    for name, value in metadata.items():
        if value is None or isinstance(value, (str, bool, int, float)):
            row[f"meta_{name}"] = value
    return row


def objective_record(raw_mean, stats, details) -> dict:
    """Format selected scientific uncertainties and gates like generated.csv.

    The compatibility statistics select a specific spread for each output.
    An unavailable spread remains None; mixed property units never produce a
    fabricated joint standard deviation.
    """
    row = {"dp_mean": float(raw_mean)}
    for name, (mean, std) in stats.items():
        row[f"obj_{name}_mean"] = float(mean)
        row[f"obj_{name}_std"] = None if std is None else float(std)
    if details and details.get("type") == "gated":
        row["reward_objective"] = float(details["objective"])
        row["gate_passed"] = bool(details["gate_passed"])
        for gate in details["gates"]:
            prefix = f"gate_{gate['name']}"
            row[f"{prefix}_value"] = float(gate["score"])
            row[f"{prefix}_passed"] = bool(gate["passed"])
            row[f"{prefix}_violation"] = float(gate["violation"])
    return row


def prediction_record(predictor, candidate, prediction, *, k: float = 1.0) -> dict:
    """Capture a cache-only hook or the tuple returned by this evaluation.

    Predictors with separate scientific/raw interfaces should expose
    ``prediction_record(candidate, prediction, *, k)``. That optional hook must
    read the completed evaluation only, including its paired metadata, and
    must not infer, build structures, advance random generators or alter timing.
    A missing or broken diagnostics hook never requests a replacement sample.
    """
    mean, std = prediction
    mean = float(mean)
    std = None if std is None else float(std)
    row = {"reward": mean, **metadata_record({})}
    try:
        row["candidate_json"] = json.dumps(candidate, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False)
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
            # A reward tuple does not identify the physical outputs or selected
            # uncertainty. None is unknown, not zero, and cannot be inverted.
            row.update(predict_mean=mean, predict_std=std,
                       prediction_record_error="Raw properties unavailable: predictor needs a cache-only prediction_record hook")
        else:
            # Match generation's tuple fallback. A native output interface alone
            # does not change that view (for example direct LookupPredictor).
            row.update(dp_mean=mean, dp_std=std,
                       dp_mean_minus_std=None if std is None else mean - k * std)
    except Exception as exc:
        row.update(predict_mean=mean, predict_std=std,
                   prediction_record_error=f"{type(exc).__name__}: {exc}")
    # The reward belongs to this evaluation, even for repeated/stochastic input.
    row["reward"] = mean
    return row
