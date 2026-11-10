"""Training snapshots preserve scientific units, missingness and evaluation identity."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from rl_matdesign.predictors.outputs import ModelSamples, OutputSchema, aggregate_samples
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.prediction_records import prediction_record
from tests.test_ooh_scientific_outputs import COMP, oracle_factory, slab_poscar
from tests.test_reward_integration import _cfg


class NamedLeaf:
    """A native output with one model and two configurations, plus paired metadata."""

    output_schema = {"activity": OutputSchema("activity", "mol")}

    def __init__(self, cfg, *, seed=None):
        self.calls = []
        self.last_metadata = {}

    def predict_outputs(self, candidate, **conditions):
        self.calls.append(deepcopy(candidate))
        value = float(candidate.get("value", .04))
        condition = (conditions.get("fparam") or [None])[0]
        if condition is not None:
            value = 5.0 - abs(condition - 2)
        self.last_metadata = {"label": "测试", "condition": condition,
                              "candidate": deepcopy(candidate), "positions": [1, 3],
                              "formula": "metadata only", "reward": -999, "optional": None}
        return {"activity": aggregate_samples(
            self.output_schema["activity"],
            [ModelSamples("model", ("c1", "c2"), (value - .02, value + .02))],
        )}

    def candidate_metadata(self, candidate):
        return deepcopy(self.last_metadata)


def _named_predictor(**updates):
    cfg = {"properties": [{"name": "delta", "predictor": f"{__name__}:NamedLeaf",
                           "output": "activity", "direction": "max", "objective": "mean"}]}
    cfg["properties"][0].update(updates.pop("property_settings", {}))
    cfg.update(updates)
    return StructureScorePredictor(cfg, seed=3)


def _deny_inference(monkeypatch, predictor):
    def forbidden(*args, **kwargs):
        pytest.fail("snapshot attempted a second prediction")
    for name in ("predict", "predict_raw", "predict_outputs", "per_objective_stats",
                 "candidate_metadata", "_evaluation", "_raw_stats"):
        monkeypatch.setattr(predictor, name, forbidden)


@pytest.mark.parametrize("selected,expected", [(None, None), ("config_spread", .02)])
def test_native_scientific_record_preserves_selected_missing_spread_and_metadata(monkeypatch, selected, expected):
    predictor = _named_predictor(property_settings={} if selected is None else {"uncertainty": selected})
    candidate = {"peptide": ["A", "C", "A"], "field": {"zero": 0.0}, "value": .04}
    prediction = predictor.predict(candidate)
    assert prediction == (pytest.approx(.04), None)
    output = predictor.cached_outputs(candidate)["delta"]
    assert output.schema.unit == "mol" and output.model_spread is None
    assert output.config_spread == pytest.approx(.02)
    before_timing = deepcopy(predictor.last_evaluation_timings)
    leaf = predictor.properties[0]["instance"][0]
    _deny_inference(monkeypatch, predictor)
    monkeypatch.setattr(predictor, "composition_formula", lambda _: "FullHostO3")
    row = prediction_record(predictor, candidate, prediction)
    assert row["formula"] == "FullHostO3"
    assert row["reward"] == row["dp_mean"] == pytest.approx(.04)
    assert row["obj_delta_mean"] == pytest.approx(.04)
    assert row["obj_delta_std"] is None if expected is None else row["obj_delta_std"] == pytest.approx(expected)
    assert "dp_std" not in row and "dp_mean_minus_std" not in row
    assert json.loads(row["candidate_json"]) == candidate
    assert json.loads(row["metadata_json"])["candidate"] == candidate
    assert row["meta_label"] == "测试" and row["meta_optional"] is None
    assert row["meta_formula"] == "metadata only" and row["meta_reward"] == -999
    assert "meta_positions" not in row and "meta_candidate" not in row
    assert predictor.last_evaluation_timings == before_timing
    assert len(leaf.calls) == 1


def test_batch_records_use_paired_outputs_metadata_after_lru_eviction(monkeypatch):
    predictor = _named_predictor(predict_cache_size=1)
    candidates = [{"value": value} for value in (.1, .2, .3)]
    predictions = predictor.batch_predict(candidates)
    assert len(predictor._stats_cache) == 1
    leaf = predictor.properties[0]["instance"][0]
    assert len(leaf.calls) == 3
    _deny_inference(monkeypatch, predictor)
    rows = [prediction_record(predictor, c, p) for c, p in zip(candidates, predictions)]
    assert [row["obj_delta_mean"] for row in rows] == pytest.approx([.1, .2, .3])
    assert [json.loads(row["metadata_json"])["candidate"] for row in rows] == candidates
    assert all(row["obj_delta_std"] is None for row in rows)
    assert len(leaf.calls) == 3


def test_sweep_snapshot_uses_winning_condition_metadata_and_missing_uncertainty(monkeypatch):
    predictor = _named_predictor(
        sweep={"name": "condition", "values": [1, 2, 3]},
        property_settings={"fparam": [None]},
    )
    candidate = {"value": 0}
    prediction = predictor.predict(candidate)
    leaf = predictor.properties[0]["instance"][0]
    assert leaf.last_metadata["condition"] == 3
    _deny_inference(monkeypatch, predictor)
    row = prediction_record(predictor, candidate, prediction)
    assert row["obj_delta_mean"] == 5
    assert row["obj_condition_mean"] == 2
    assert row["obj_condition_std"] is None
    assert row["meta_condition"] == 2
    assert len(leaf.calls) == 3


def test_gated_record_keeps_physical_delta_when_reward_and_raw_composite_are_penalties(monkeypatch):
    predictor = StructureScorePredictor(_cfg())
    candidate = {"delta": .07, "pure": .4, "cost": 0.0}
    prediction = predictor.predict(candidate)
    _deny_inference(monkeypatch, predictor)
    row = prediction_record(predictor, candidate, prediction)
    assert row["reward"] == row["dp_mean"] == pytest.approx(-10.8)
    assert row["obj_delta_mean"] == row["reward_objective"] == .07
    assert row["obj_delta_std"] is None
    assert row["gate_passed"] is False
    assert row["gate_synthesis_violation"] == pytest.approx(.4)


@pytest.mark.parametrize("matrix,uncertainty,expected", [
    (((2.,),), "models", None),
    (((1., 3.),), "configs", 1.),
    (((1.,), (3.,)), "models", 1.),
])
def test_ooh_scientific_cache_preserves_positive_raw_value_and_nullable_spread(
    oracle_factory, monkeypatch, matrix, uncertainty, expected,
):
    predictor, calls, _ = oracle_factory(matrix, uncertainty=uncertainty)
    assert predictor.cached_outputs(COMP) is None
    assert calls["evals"] == 0
    prediction = predictor.predict(COMP)
    before = dict(calls)
    monkeypatch.setattr(predictor, "predict_outputs", lambda _: pytest.fail("OOH reevaluated"))
    monkeypatch.setattr(predictor, "predict_raw", lambda _: pytest.fail("OOH raw reevaluated"))
    monkeypatch.setattr(predictor, "check_phase", lambda _: (True, "NiFe"))
    row = prediction_record(predictor, COMP, prediction, k=2)
    assert row["reward"] == -2 and row["dp_mean"] == 2
    assert row["dp_std"] == expected
    assert row["dp_mean_minus_std"] == (None if expected is None else 2 - 2 * expected)
    assert row["metadata_json"] == "{}" and row["primary_label"] == "NiFe"
    assert calls == before


def test_missing_cached_record_is_explicit_and_does_not_infer():
    predictor = _named_predictor()
    row = prediction_record(predictor, {"value": .2}, (.2, None))
    assert "Completed prediction missing" in row["prediction_record_error"]
    assert row["reward"] == row["predict_mean"] == .2
    assert row["predict_std"] is None and "dp_mean" not in row
    assert predictor.properties[0]["instance"][0].calls == []


def test_predict_only_and_unadapted_scientific_interfaces_do_not_fabricate_spread():
    class Bare:
        def predict(self, candidate):
            pytest.fail("must reuse completed tuple")
    plain = prediction_record(Bare(), {"A": 1}, (2, None))
    assert plain["dp_mean"] == 2 and plain["dp_std"] is None
    assert plain["dp_mean_minus_std"] is None
    class Scientific(Bare):
        def predict_raw(self, candidate):
            pytest.fail("must not rerun raw outputs")
    native = prediction_record(Scientific(), {"A": 1}, (-2, None))
    assert native["predict_mean"] == -2 and native["predict_std"] is None
    assert "dp_mean" not in native and "prediction_record_error" in native


@pytest.mark.parametrize("uncertainty", [None, .25])
def test_direct_lookup_with_native_outputs_matches_generation_tuple_fallback(tmp_path, monkeypatch, uncertainty):
    from rl_matdesign.predictors.lookup import LookupPredictor
    from tests.test_lookup_predictor import _write_model

    path = _write_model(tmp_path, **({} if uncertainty is None else {"std": uncertainty}))
    predictor = LookupPredictor({"model": path})
    candidate = {"conditions": {"temperature": 150.0, "solvent": "01"}}
    prediction = predictor.predict(candidate)
    scientific = predictor.predict_outputs(candidate)["response"]
    assert scientific.mean == 4 and scientific.native_uncertainty == uncertainty
    monkeypatch.setattr(predictor, "predict", lambda _: pytest.fail("lookup reevaluated"))
    monkeypatch.setattr(predictor, "predict_outputs", lambda _: pytest.fail("lookup outputs reevaluated"))
    row = prediction_record(predictor, candidate, prediction, k=2)
    assert row["reward"] == row["dp_mean"] == 4
    assert row["dp_std"] == uncertainty
    assert row["dp_mean_minus_std"] == (None if uncertainty is None else 3.5)
    assert "prediction_record_error" not in row
    assert json.loads(row["candidate_json"]) == candidate


def test_metadata_json_is_detached_and_hook_cannot_overwrite_evaluated_reward():
    fields = {"reward": 999, "meta_nested": {"x": [1]}}
    class Predictor:
        def prediction_record(self, *args, **kwargs):
            return fields
    row = prediction_record(Predictor(), {"A": 1}, (.5, None))
    fields["meta_nested"]["x"].append(2)
    assert row["reward"] == .5 and row["meta_nested"] == {"x": [1]}
