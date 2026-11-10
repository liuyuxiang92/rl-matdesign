"""Named scientific outputs with explicit units, representations and sample axes.

This contract has no reward direction, weight, feasibility threshold or registry.
Model spread and configuration spread are descriptive population standard
deviations along different axes; neither is automatically a confidence interval.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Protocol, runtime_checkable


OUTPUT_SCHEMA_VERSION = 2
_REPRESENTATIONS = frozenset({"physical", "ln", "log10", "pIC50"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _spread(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative or None.")
    return number


def _fields(value: Any, fields: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}.")
    if "schema_version" in fields and (
        type(value["schema_version"]) is not int
        or value["schema_version"] != OUTPUT_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported {name} schema_version.")
    return value


@dataclass(frozen=True)
class OutputSchema:
    """The identity and representation of one scientific scalar.

    ``unit`` describes the represented scalar, for example ``K``,
    ``log10(K^-1)`` or ``1``. None explicitly means the unit is unknown; it
    does not mean dimensionless. It is metadata, not a unit-conversion expression.
    ``representation`` is explicit: physical, ln, log10, or pIC50. A pIC50 value
    stays pIC50 unless an adapter explicitly implements the correct scientific
    conversion; it is not treated as a generic log10 concentration here.
    """

    name: str
    unit: str | None
    representation: str = "physical"

    def __post_init__(self) -> None:
        _text(self.name, "Output name")
        if self.unit is not None:
            _text(self.unit, "Output unit")
        if not isinstance(self.representation, str) or self.representation not in _REPRESENTATIONS:
            raise ValueError(f"Output representation must be one of {sorted(_REPRESENTATIONS)}.")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": OUTPUT_SCHEMA_VERSION, "name": self.name,
                "unit": self.unit, "representation": self.representation}

    @classmethod
    def from_dict(cls, value: Mapping) -> OutputSchema:
        data = _fields(value, {"schema_version", "name", "unit", "representation"}, "output schema")
        return cls(data["name"], data["unit"], data["representation"])


@dataclass(frozen=True)
class ModelSamples:
    """One model's predictions on explicitly identified, ordered configurations.

    Configuration IDs identify the same evaluated objects across models, not
    merely row numbers created after dropping failures. A composition-only
    predictor can use one sample ID for its candidate. Inputs are defensively
    copied to tuples, so later caller mutations cannot change an output record.
    """

    model_id: str
    configuration_ids: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.model_id, "Model ID")
        for value, name in ((self.configuration_ids, "configuration_ids"), (self.values, "values")):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
                raise ValueError(f"{name} must be a nonempty sequence.")
        ids = tuple(_text(value, "Configuration ID") for value in self.configuration_ids)
        values = tuple(_finite(value, "Scientific sample") for value in self.values)
        if len(ids) != len(values):
            raise ValueError("Each scientific sample needs one configuration ID.")
        if len(set(ids)) != len(ids):
            raise ValueError("Configuration IDs must be unique within a model.")
        object.__setattr__(self, "configuration_ids", ids)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "configuration_ids": list(self.configuration_ids),
                "values": list(self.values)}

    @classmethod
    def from_dict(cls, value: Mapping) -> ModelSamples:
        data = _fields(value, {"model_id", "configuration_ids", "values"}, "model samples")
        return cls(data["model_id"], data["configuration_ids"], data["values"])


def _aligned(samples: Sequence[ModelSamples]) -> tuple[ModelSamples, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise ValueError("Samples must be a nonempty sequence of ModelSamples.")
    rows = tuple(samples)
    if any(not isinstance(row, ModelSamples) for row in rows):
        raise ValueError("Every model row must be ModelSamples with explicit configuration IDs.")
    if len({row.model_id for row in rows}) != len(rows):
        raise ValueError("Model IDs must be unique.")
    if any(row.configuration_ids != rows[0].configuration_ids for row in rows[1:]):
        raise ValueError("Models must evaluate the same ordered configuration IDs; ragged/misaligned samples are invalid.")
    return rows


def _mean(values: Sequence[float]) -> float:
    # Sum first to preserve subnormal values. Only overflow requires dividing
    # first; a finite mean can exist even when its intermediate sum overflows.
    try:
        mean = math.fsum(values) / len(values)
    except OverflowError:
        mean = math.fsum(value / len(values) for value in values)
    return _finite(mean, "Sample mean")


def _population_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    # Subtract a shared reference before scaling so a large common offset does
    # not erase a small, representable spread. Opposite extreme values can make
    # that subtraction overflow; scale the original values only in that case.
    shifted = [value - values[0] for value in values]
    centered_values = shifted if all(math.isfinite(value) for value in shifted) else values
    scale = max(abs(value) for value in centered_values)
    if scale == 0:
        return 0.0
    scaled = [value / scale for value in centered_values]
    center = _mean(scaled)
    result = scale * math.sqrt(math.fsum((value - center) ** 2 for value in scaled) / len(scaled))
    return _finite(result, "Population standard deviation")


def _statistics(samples: tuple[ModelSamples, ...]) -> tuple[float, float | None, float | None]:
    model_means = [_mean(row.values) for row in samples]
    config_means = [_mean(column) for column in zip(*(row.values for row in samples))]
    return _mean(model_means), _population_std(model_means), _population_std(config_means)


@dataclass(frozen=True)
class ScientificOutput:
    """One named mean and explicitly distinguished optional spread estimates.

    ``model_spread`` = std_m(mean_c Y[m,c], ddof=0).
    ``config_spread`` = std_c(mean_m Y[m,c], ddof=0).
    Either is None if its axis has fewer than two samples. These two summaries
    do not decompose total variance or quantify model/configuration interaction.
    ``pooled_spread`` is derived from every retained model/configuration sample
    with ddof=0; it is None without samples or with fewer than two total samples.
    It is not a native uncertainty or an axis-specific variance decomposition.
    ``native_uncertainty`` is a backend-provided standard deviation in the same
    representation and unit, if available; no calibration claim is inferred.

    Point-only outputs may carry declared spreads without raw samples, but
    nonlinear inverse transformations then cannot be performed. When raw samples
    are retained, mean and axis spreads must agree with those samples exactly.
    """

    schema: OutputSchema
    mean: float
    model_spread: float | None = None
    config_spread: float | None = None
    native_uncertainty: float | None = None
    samples: tuple[ModelSamples, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, OutputSchema):
            raise ValueError("Scientific output requires an OutputSchema.")
        object.__setattr__(self, "mean", _finite(self.mean, "Output mean"))
        for name in ("model_spread", "config_spread", "native_uncertainty"):
            object.__setattr__(self, name, _spread(getattr(self, name), name))
        if self.samples is not None:
            samples = _aligned(self.samples)
            object.__setattr__(self, "samples", samples)
            if (self.mean, self.model_spread, self.config_spread) != _statistics(samples):
                raise ValueError("Output statistics do not match the retained model/configuration samples.")

    @property
    def pooled_spread(self) -> float | None:
        if self.samples is None:
            return None
        return _population_std(tuple(value for row in self.samples for value in row.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION, "schema": self.schema.to_dict(),
            "mean": self.mean, "model_spread": self.model_spread,
            "config_spread": self.config_spread, "native_uncertainty": self.native_uncertainty,
            "pooled_spread": self.pooled_spread,
            "samples": None if self.samples is None else [row.to_dict() for row in self.samples],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> ScientificOutput:
        data = _fields(value, {"schema_version", "schema", "mean", "model_spread", "config_spread",
                               "native_uncertainty", "pooled_spread", "samples"}, "scientific output")
        raw_samples = data["samples"]
        if raw_samples is not None and not isinstance(raw_samples, list):
            raise ValueError("Serialized samples must be a list or null.")
        result = cls(
            OutputSchema.from_dict(data["schema"]), data["mean"], data["model_spread"],
            data["config_spread"], data["native_uncertainty"],
            None if raw_samples is None else tuple(ModelSamples.from_dict(row) for row in raw_samples),
        )
        if _spread(data["pooled_spread"], "pooled_spread") != result.pooled_spread:
            raise ValueError("Serialized pooled_spread does not match the retained samples.")
        return result


def aggregate_samples(
    schema: OutputSchema,
    samples: Sequence[ModelSamples],
    *,
    native_uncertainty: float | None = None,
) -> ScientificOutput:
    """Average configurations first, then models; compute two population spreads.

    Rows must be rectangular and carry identical ordered configuration IDs.
    Missing model results are errors, not a reason to align the remaining rows
    by position or to treat models as independent configurations. Callers must
    supply every configured model: this function cannot detect a completely
    omitted model without access to the adapter's expected model catalog.
    """
    rows = _aligned(samples)
    mean, model_spread, config_spread = _statistics(rows)
    return ScientificOutput(schema, mean, model_spread, config_spread, native_uncertainty, rows)


def inverse_transform_output(
    output: ScientificOutput, transform: str = "identity", *, unit: str | None,
) -> ScientificOutput:
    """Apply identity/exp/pow10 to individual samples, then recompute statistics.

    exp requires representation='ln'; pow10 requires representation='log10'.
    The caller must explicitly pass the destination unit, including None when
    unknown. Nonlinear transforms require raw samples and strictly positive,
    finite results; numerical underflow is not a physical zero. Native
    uncertainty cannot be transformed from a single standard deviation, so a
    non-null native uncertainty also prevents nonlinear conversion here.
    Identity preserves representation and all uncertainties; pIC50 stays pIC50.
    """
    if not isinstance(output, ScientificOutput):
        raise ValueError("Expected ScientificOutput.")
    if transform == "identity":
        if unit != output.schema.unit:
            raise ValueError("Identity transformation cannot change units.")
        return output
    if transform not in ("exp", "pow10"):
        raise ValueError("Inverse transform must be identity, exp or pow10.")
    expected = "ln" if transform == "exp" else "log10"
    if output.schema.representation != expected:
        raise ValueError(f"{transform} requires representation={expected!r}; no automatic pIC50 conversion is defined.")
    if unit is not None:
        _text(unit, "Physical output unit")
    if output.samples is None:
        raise ValueError("Nonlinear inverse requires raw samples; transforming only mean/std is not valid.")
    if output.native_uncertainty is not None:
        raise ValueError("Nonlinear inverse cannot propagate native_uncertainty without its raw distribution.")
    function = math.exp if transform == "exp" else lambda value: 10.0 ** value
    def transform_sample(value: float) -> float:
        transformed = function(value)
        if not math.isfinite(transformed) or transformed <= 0:
            raise ValueError("Nonlinear inverse produced a nonpositive or nonfinite scientific sample (underflow/overflow).")
        return transformed

    try:
        samples = [ModelSamples(row.model_id, row.configuration_ids,
                                tuple(transform_sample(value) for value in row.values)) for row in output.samples]
    except OverflowError as exc:
        raise ValueError("Nonlinear inverse produced a nonfinite scientific sample.") from exc
    return aggregate_samples(OutputSchema(output.schema.name, unit, "physical"), samples)


def outputs_to_json(outputs: Mapping[str, ScientificOutput]) -> str:
    """Serialize a shared multi-output result deterministically without NaN."""
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("Named outputs must be a nonempty mapping.")
    records = {}
    for name, output in outputs.items():
        if not isinstance(output, ScientificOutput) or name != output.schema.name:
            raise ValueError("Each output mapping key must match its scientific output name.")
        records[name] = output.to_dict()
    return json.dumps({"schema_version": OUTPUT_SCHEMA_VERSION, "outputs": records},
                      ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def outputs_from_json(payload: str) -> dict[str, ScientificOutput]:
    """Load strict versioned output records; reject duplicate JSON keys."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate scientific output JSON key {key!r}.")
            result[key] = value
        return result

    data = _fields(json.loads(payload, object_pairs_hook=unique_object),
                   {"schema_version", "outputs"}, "named outputs")
    if not isinstance(data["outputs"], dict) or not data["outputs"]:
        raise ValueError("Named outputs must be a nonempty mapping.")
    outputs = {name: ScientificOutput.from_dict(value) for name, value in data["outputs"].items()}
    outputs_to_json(outputs)  # Validate names against schemas, including after loading.
    return outputs


@runtime_checkable
class NamedOutputPredictor(Protocol):
    """One shared adapter instance can return several scientific quantities.

    This is a structural protocol, not a registry or a legacy reward interface.
    Units/representations are declared once per output. Implementations control
    model loading, caching and batching; requesting multiple names must not
    imply loading a separate copy of a shared model for each property.
    """

    @property
    def output_schema(self) -> Mapping[str, OutputSchema]: ...

    def predict_outputs(self, candidate: Any) -> Mapping[str, ScientificOutput]: ...

    def batch_predict_outputs(self, candidates: Sequence[Any]) -> list[Mapping[str, ScientificOutput]]: ...
