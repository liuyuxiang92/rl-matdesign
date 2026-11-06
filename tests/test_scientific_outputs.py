"""Scientific output contracts and independent numerical expectations."""
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from rl_matdesign.predictors.outputs import (
    ModelSamples,
    NamedOutputPredictor,
    OutputSchema,
    ScientificOutput,
    aggregate_samples,
    inverse_transform_output,
    outputs_from_json,
    outputs_to_json,
)


SCHEMA = OutputSchema("Tg", "K")


def rows(matrix):
    return [ModelSamples(f"model-{index}", [f"config-{j}" for j in range(len(values))], values)
            for index, values in enumerate(matrix)]


def test_means_and_axis_population_spreads_have_distinct_definitions():
    result = aggregate_samples(SCHEMA, rows([[1, 3], [5, 7]]))
    assert result.mean == 4  # Model means are 2 and 6.
    assert result.model_spread == 2
    assert result.config_spread == 1  # Ensemble config means are 3 and 5.
    assert result.native_uncertainty is None
    assert result.samples[0].configuration_ids == ("config-0", "config-1")


def test_interaction_is_not_relabelled_as_configuration_or_model_spread():
    result = aggregate_samples(SCHEMA, rows([[0, 10], [10, 0]]))
    assert result.mean == 5
    # Both axis marginal means are [5, 5]. The pooled sample std/RMS of
    # within-model std would instead be 5; it is intentionally not either axis.
    assert result.model_spread == result.config_spread == 0


@pytest.mark.parametrize("matrix,mean,model,config", [
    ([[3]], 3, None, None),
    ([[1, 3]], 2, None, 1),
    ([[1], [3]], 2, 1, None),
    ([[2, 2], [2, 2]], 2, 0, 0),
])
def test_singleton_axis_has_no_spread_estimate_not_fabricated_zero(matrix, mean, model, config):
    result = aggregate_samples(SCHEMA, rows(matrix))
    assert (result.mean, result.model_spread, result.config_spread) == (mean, model, config)


def test_native_uncertainty_is_preserved_as_a_distinct_backend_quantity():
    result = aggregate_samples(SCHEMA, rows([[1, 3], [5, 7]]), native_uncertainty=.25)
    assert result.native_uncertainty == .25
    assert result.model_spread == 2 and result.config_spread == 1
    point = ScientificOutput(SCHEMA, 800)
    assert point.model_spread is point.config_spread is point.native_uncertainty is None


def test_raw_sample_alignment_is_required_including_order():
    a = ModelSamples("model-a", ["config-a", "config-b"], [1, 3])
    for b in (
        ModelSamples("model-b", ["config-a"], [5]),
        ModelSamples("model-b", ["config-b", "config-a"], [7, 5]),
        ModelSamples("model-b", ["config-a", "different-config"], [5, 7]),
    ):
        with pytest.raises(ValueError, match="same ordered configuration IDs"):
            aggregate_samples(SCHEMA, [a, b])
    with pytest.raises(ValueError, match="Model IDs must be unique"):
        aggregate_samples(SCHEMA, [a, a])
    with pytest.raises(ValueError, match="explicit configuration IDs"):
        aggregate_samples(SCHEMA, [a, None])  # Failed model result cannot be silently skipped.


def test_point_samples_are_defensively_copied_and_results_are_immutable():
    ids, values = ["c0", "c1"], [1, 3]
    sample = ModelSamples("m0", ids, values)
    result = aggregate_samples(SCHEMA, [sample])
    ids[0], values[0] = "wrong", 1000
    assert sample.configuration_ids == ("c0", "c1")
    assert sample.values == (1.0, 3.0)
    with pytest.raises(FrozenInstanceError):
        result.mean = 999
    serialized = result.to_dict()
    serialized["samples"][0]["values"][0] = 999
    assert result.samples[0].values == (1.0, 3.0)


def test_exp_transforms_each_sample_before_recomputing_mean_and_std():
    logged = aggregate_samples(OutputSchema("rate", "ln(s^-1)", "ln"), rows([[0, math.log(9)]]))
    physical = inverse_transform_output(logged, "exp", unit="s^-1")
    assert physical.schema == OutputSchema("rate", "s^-1", "physical")
    assert physical.mean == pytest.approx(5)
    assert physical.config_spread == pytest.approx(4)
    assert physical.model_spread is None
    assert physical.mean != pytest.approx(math.exp(logged.mean))  # That would be 3.
    assert logged.schema.representation == "ln"  # Original object is retained.


def test_pow10_recomputes_both_axes_and_preserves_identifiers():
    logged = aggregate_samples(OutputSchema("cte", "log10(K^-1)", "log10"), rows([[0, 1], [2, 3]]))
    physical = inverse_transform_output(logged, "pow10", unit="K^-1")
    assert physical.mean == pytest.approx(277.75)
    assert physical.model_spread == pytest.approx(272.25)
    assert physical.config_spread == pytest.approx(227.25)
    assert physical.samples[0].values == (1, 10)
    assert physical.samples[1].values == (100, 1000)
    assert [row.model_id for row in physical.samples] == [row.model_id for row in logged.samples]
    assert physical.samples[0].configuration_ids == logged.samples[0].configuration_ids


def test_nonlinear_inverse_rejects_mean_std_without_raw_distribution():
    point = ScientificOutput(OutputSchema("x", "ln(K)", "ln"), 2, model_spread=.5)
    with pytest.raises(ValueError, match="raw samples"):
        inverse_transform_output(point, "exp", unit="K")
    sampled = aggregate_samples(point.schema, rows([[1, 3]]), native_uncertainty=.5)
    with pytest.raises(ValueError, match="native_uncertainty"):
        inverse_transform_output(sampled, "exp", unit="K")


def test_pic50_remains_pic50_and_identity_cannot_change_units():
    output = ScientificOutput(OutputSchema("potency", "1", "pIC50"), 6.2, native_uncertainty=.3)
    assert inverse_transform_output(output, unit="1") is output
    assert inverse_transform_output(output, unit="1").native_uncertainty == .3
    with pytest.raises(ValueError, match="no automatic pIC50"):
        inverse_transform_output(output, "pow10", unit="mol/L")
    with pytest.raises(ValueError, match="cannot change units"):
        inverse_transform_output(output, "identity", unit="mol/L")


@pytest.mark.parametrize("transform", ["exp", "pow10"])
def test_nonlinear_inverse_overflow_is_an_error_not_an_infinite_property(transform):
    representation = "ln" if transform == "exp" else "log10"
    result = aggregate_samples(OutputSchema("x", "1", representation), rows([[1000]]))
    with pytest.raises(ValueError, match="nonfinite"):
        inverse_transform_output(result, transform, unit="1")


def test_large_finite_samples_do_not_overflow_intermediate_sum_or_squared_deviation():
    large = aggregate_samples(SCHEMA, rows([[1e308, 1e308], [-1e308, -1e308]]))
    assert large.mean == 0
    assert large.model_spread == 1e308
    assert large.config_spread == 0


@pytest.mark.parametrize("mean", [math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_point_means_are_rejected(mean):
    with pytest.raises(ValueError):
        ScientificOutput(SCHEMA, mean)


@pytest.mark.parametrize("field", ["model_spread", "config_spread", "native_uncertainty"])
@pytest.mark.parametrize("value", [-.1, math.nan, math.inf, True])
def test_invalid_uncertainties_are_rejected(field, value):
    with pytest.raises(ValueError):
        ScientificOutput(SCHEMA, 1, **{field: value})


@pytest.mark.parametrize("kwargs", [
    {"model_id": "", "configuration_ids": ["c"], "values": [1]},
    {"model_id": "m", "configuration_ids": [], "values": []},
    {"model_id": "m", "configuration_ids": ["c", "c"], "values": [1, 2]},
    {"model_id": "m", "configuration_ids": ["c"], "values": [1, 2]},
    {"model_id": "m", "configuration_ids": ["c"], "values": [math.nan]},
    {"model_id": "m", "configuration_ids": ["c"], "values": [None]},
])
def test_invalid_sample_records_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelSamples(**kwargs)


def test_sample_statistics_cannot_be_forged_or_marked_zero_for_a_singleton_axis():
    sample = rows([[2]])
    with pytest.raises(ValueError, match="do not match"):
        ScientificOutput(SCHEMA, 2, model_spread=0, samples=sample)
    with pytest.raises(ValueError, match="do not match"):
        ScientificOutput(SCHEMA, 3, samples=sample)


def test_multiple_outputs_and_schemas_have_stable_strict_json_round_trip():
    tg = aggregate_samples(SCHEMA, rows([[800, 820], [810, 830]]))
    pic50 = ScientificOutput(OutputSchema("potency", "1", "pIC50"), 6.5, native_uncertainty=.2)
    first = outputs_to_json({"Tg": tg, "potency": pic50})
    second = outputs_to_json({"potency": pic50, "Tg": tg})
    assert first == second
    assert outputs_from_json(first) == {"Tg": tg, "potency": pic50}
    assert outputs_to_json(outputs_from_json(first)) == first
    assert OutputSchema.from_dict(json.loads(json.dumps(SCHEMA.to_dict()))) == SCHEMA
    assert json.loads(first)["outputs"]["potency"]["model_spread"] is None


def test_json_rejects_mismatched_names_versions_unknown_fields_and_changed_samples():
    result = aggregate_samples(SCHEMA, rows([[800, 820]]))
    with pytest.raises(ValueError, match="match its scientific output name"):
        outputs_to_json({"wrong": result})
    for change in ("version", "extra", "samples", "inner_version", "name"):
        data = json.loads(outputs_to_json({"Tg": result}))
        if change == "version":
            data["schema_version"] = 999
        elif change == "extra":
            data["unexpected"] = 1
        elif change == "samples":
            data["outputs"]["Tg"]["samples"][0]["values"][0] = 900
        elif change == "inner_version":
            data["outputs"]["Tg"]["schema"]["schema_version"] = True
        else:
            data["outputs"]["Tg"]["schema"]["name"] = "wrong"
        with pytest.raises(ValueError):
            outputs_from_json(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        outputs_from_json('{"schema_version":1,"schema_version":1,"outputs":{}}')


def test_named_predictor_protocol_describes_shared_multi_output_without_registry():
    class SharedFakePredictor:
        """A fake adapter demonstrates the protocol, not real scientific inference."""

        output_schema = {"Tg": SCHEMA, "T3": OutputSchema("T3", "K")}

        def __init__(self):
            self.shared_evaluations = 0

        def predict_outputs(self, candidate):
            self.shared_evaluations += 1
            return {"Tg": ScientificOutput(SCHEMA, 800),
                    "T3": ScientificOutput(self.output_schema["T3"], 1400)}

        def batch_predict_outputs(self, candidates):
            return [self.predict_outputs(candidate) for candidate in candidates]

    predictor = SharedFakePredictor()
    assert isinstance(predictor, NamedOutputPredictor)
    assert set(predictor.predict_outputs({"SiO2": 1})) == {"Tg", "T3"}
    assert predictor.shared_evaluations == 1


def test_unknown_unit_is_explicit_and_round_trips_without_becoming_dimensionless():
    schema = OutputSchema('overpotential', None)
    result = ScientificOutput(schema, 230)
    encoded = outputs_to_json({'overpotential': result})
    assert json.loads(encoded)['outputs']['overpotential']['schema']['unit'] is None
    assert outputs_from_json(encoded)['overpotential'].schema.unit is None
    assert schema != OutputSchema('overpotential', '1')
    assert inverse_transform_output(result, unit=None) is result
    with pytest.raises(ValueError, match='cannot change units'):
        inverse_transform_output(result, unit='mV')
    with pytest.raises(ValueError, match='cannot change units'):
        inverse_transform_output(ScientificOutput(SCHEMA, 800), unit=None)


@pytest.mark.parametrize('transform,representation', [('exp', 'ln'), ('pow10', 'log10')])
def test_inverse_requires_explicit_destination_unit_but_allows_explicit_unknown(transform, representation):
    logged = aggregate_samples(OutputSchema('x', None, representation), rows([[0, 1]]))
    with pytest.raises(TypeError, match='unit'):
        inverse_transform_output(logged, transform)
    converted = inverse_transform_output(logged, transform, unit=None)
    assert converted.schema.unit is None and converted.schema.representation == 'physical'
    assert converted.mean > 0
    with pytest.raises(TypeError, match='unit'):
        inverse_transform_output(converted)


@pytest.mark.parametrize('transform,representation', [('exp', 'ln'), ('pow10', 'log10')])
def test_inverse_rejects_underflow_instead_of_claiming_a_zero_physical_property(transform, representation):
    logged = aggregate_samples(OutputSchema('x', None, representation), rows([[-1000, 0]]))
    with pytest.raises(ValueError, match='underflow'):
        inverse_transform_output(logged, transform, unit=None)


def test_large_common_offset_does_not_corrupt_small_population_spread():
    result = aggregate_samples(SCHEMA, rows([[1e16, 1e16 + 2]]))
    assert result.mean == 1e16  # The exact mean lies halfway between float64 values.
    assert result.model_spread is None
    assert result.config_spread == 1
    transposed = aggregate_samples(SCHEMA, rows([[1e16], [1e16 + 2]]))
    assert transposed.model_spread == 1 and transposed.config_spread is None


def test_subnormal_means_survive_sum_before_division():
    smallest = math.ulp(0.0)
    result = aggregate_samples(SCHEMA, rows([[smallest, smallest]]))
    assert result.mean == smallest
    assert result.config_spread == 0
    positive_spread = aggregate_samples(SCHEMA, rows([[0, smallest * 2]]))
    assert positive_spread.mean == smallest
    assert positive_spread.config_spread == smallest


def test_extreme_opposite_values_use_finite_fallback_for_mean_and_spread():
    import sys
    largest = sys.float_info.max
    positive = aggregate_samples(SCHEMA, rows([[largest, largest]]))
    assert positive.mean == largest and positive.config_spread == 0
    opposite = aggregate_samples(SCHEMA, rows([[largest, -largest]]))
    assert opposite.mean == 0 and opposite.config_spread == largest
    cancellation = aggregate_samples(SCHEMA, rows([[largest, largest, -largest, -largest]]))
    assert cancellation.mean == 0 and cancellation.config_spread == largest


def test_statistics_match_560_independent_random_matrix_expectations():
    import random
    rng = random.Random(17)
    for model_count in range(1, 8):
        for config_count in range(1, 9):
            for _ in range(10):
                values = [[rng.uniform(-100, 100) for _ in range(config_count)]
                          for _ in range(model_count)]
                result = aggregate_samples(SCHEMA, rows(values))
                model_means = [math.fsum(row) / config_count for row in values]
                config_means = [math.fsum(values[i][j] for i in range(model_count)) / model_count
                                for j in range(config_count)]
                def independent_std(data):
                    if len(data) == 1:
                        return None
                    mean = math.fsum(data) / len(data)
                    return math.sqrt(math.fsum((value - mean) ** 2 for value in data) / len(data))
                assert result.mean == pytest.approx(math.fsum(model_means) / model_count, abs=1e-12)
                for actual, expected in [(result.model_spread, independent_std(model_means)),
                                         (result.config_spread, independent_std(config_means))]:
                    assert actual is None if expected is None else actual == pytest.approx(expected, abs=1e-12)


def test_pooled_spread_is_derived_from_samples_and_is_not_native_uncertainty():
    output = aggregate_samples(SCHEMA, rows([[0, 10], [10, 0]]))
    assert output.model_spread == output.config_spread == 0
    assert output.pooled_spread == 5
    assert output.native_uncertainty is None
    assert output.to_dict()['pooled_spread'] == 5
    assert ScientificOutput.from_dict(output.to_dict()) == output
    assert ScientificOutput(SCHEMA, 1, model_spread=2).pooled_spread is None
    assert aggregate_samples(SCHEMA, rows([[1]])).pooled_spread is None
    changed = output.to_dict()
    changed['pooled_spread'] = 4
    with pytest.raises(ValueError, match='pooled_spread'):
        ScientificOutput.from_dict(changed)
