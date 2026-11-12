"""One recorded, immutable quality scale shared by all HPO v2 measurements."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

from .artifacts import atomic_json, file_digest, read_json
from .scoring import _identity, score_records


class QualityCalibrationError(ValueError):
    """Reference observations cannot determine a meaningful physical scale."""


def _objective_contract(objective):
    return {name: objective.get(name) for name in ("field", "direction", "threshold", "unit")}


def _quantile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def fit_quality_scale(records, objective):
    """Use canonical distinct observations, without discarding gate failures.

    IQR is preferred; a nonzero full range or distance from the user threshold
    handles constant/mostly constant reference predictions. A wholly constant
    reference at the threshold provides no scale, so we never invent a unit.
    """
    distinct = {}
    for row in records:
        status = str(row.get("evaluation_status", row.get("status", "evaluated"))).lower()
        if status in {"failed", "invalid", "preparation_only", "evaluation_failed", "rollout_failed"}:
            continue
        value = row.get("properties", {}).get(objective["field"])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise QualityCalibrationError("Reference archive has a missing or nonfinite objective property")
        key = _identity(row.get("identity"))
        if key in distinct and distinct[key] != value:
            raise QualityCalibrationError("Reference archive has conflicting duplicate predictions")
        distinct[key] = float(value)
    if not distinct:
        raise QualityCalibrationError("Reference archive has no evaluated distinct candidates")
    values = sorted(distinct.values())
    threshold = float(objective["threshold"])
    extent = max(abs(threshold), max(abs(value) for value in values))
    if extent == 0:
        raise QualityCalibrationError("Reference values all equal the threshold; cannot determine quality scale")
    scaled = [value / extent for value in values]
    candidates = [
        ("reference_iqr", _quantile(scaled, .75) - _quantile(scaled, .25)),
        ("reference_range", scaled[-1] - scaled[0]),
        ("reference_threshold_distance", statistics.median(abs(value - threshold / extent) for value in scaled)),
    ]
    for method, magnitude in candidates:
        if magnitude > 0:
            # The full spread of opposite floating point extremes may exceed
            # the largest representable scale; the recorded cap is deterministic.
            scale = min(magnitude, sys.float_info.max / extent) * extent
            if math.isfinite(scale) and scale > 0:
                return {"scale": scale, "method": method, "sample_count": len(values),
                        "minimum": values[0], "maximum": values[-1]}
    raise QualityCalibrationError("Reference values all equal the threshold; cannot determine quality scale")


def restore_quality_scale(plan, out):
    norm = plan["objective"]["normalization"]
    path = Path(out) / "quality_calibration.json"
    if norm.get("calibration") != "reference":
        return None
    if not path.exists():
        # A finished stage can be reused without invoking its jobs again. Do
        # not silently accept its scores, or calibrate a new scale, if the
        # shared scale those scores depended on has been removed.
        for result_path in (Path(out) / "jobs").glob("*/result.json"):
            previous = read_json(result_path)
            if previous.get("quality_normalization", {}).get("scale") is not None:
                raise QualityCalibrationError("Quality calibration is missing for existing scored jobs")
        return None
    record = read_json(path)
    if record.get("objective") != _objective_contract(plan["objective"]):
        raise QualityCalibrationError("Saved quality calibration does not match the campaign objective")
    source = Path(record["source_records"])
    if not source.is_file() or file_digest(source) != record["source_records_sha256"]:
        raise QualityCalibrationError("Quality calibration reference archive changed or is missing")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    fitted = fit_quality_scale(rows, plan["objective"])
    if any(record.get(name) != value for name, value in fitted.items()):
        raise QualityCalibrationError("Saved quality calibration was changed; use a new campaign")
    norm["scale"] = record["scale"]
    return record


def calibrate_quality_scale(plan, out, records, ledger, record_path, job_id, phase):
    norm = plan["objective"]["normalization"]
    if norm.get("calibration") != "reference" or norm.get("scale") is not None:
        return None
    if phase != "train" or ledger.get("status") != "COMPLETED":
        return None
    # Check the full archive, units, ledger and completion/efficiency contract
    # before making these observations the immutable campaign reference.
    if score_records(records, ledger, plan, phase)["status"] != "QUALITY_CALIBRATION_REQUIRED":
        return None
    fitted = fit_quality_scale(records, plan["objective"])
    record = {"version": 1, "objective": _objective_contract(plan["objective"]),
              "source_job_id": job_id, "source_records": str(Path(record_path).resolve()),
              "source_records_sha256": file_digest(record_path), **fitted}
    atomic_json(Path(out) / "quality_calibration.json", record)
    norm["scale"] = record["scale"]
    return record
