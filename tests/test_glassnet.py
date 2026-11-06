"""GlassNet adapter contract tests, using explicitly fake model outputs.

The optional real-model test requires an already prepared pinned environment.
Normal unit tests neither import GlassPy nor download/train a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rl_matdesign.predictors import glassnet
from rl_matdesign.predictors.glassnet import GlassNetPredictor


# Small integer ratios and an explicit zero exercise input semantics only;
# this fixture is not a researched composition or a recommended glass recipe.
TOY_COMPOSITION = {"SiO2": 1, "B2O3": 1, "Na2O": 0}


class FakeGlassNet:
    """Deliberately artificial predictions; never scientific validation."""

    def __init__(self):
        self.calls = []

    def predict(self, rows, *, input_cols, return_dataframe):
        assert return_dataframe is True
        self.calls.append((rows, input_cols))
        compositions = [dict(zip(input_cols, row)) for row in rows]
        return {
            "CTEbelowTg": [-5.0 for _ in rows],
            "Tg": [700.0 + c.get("SiO2", 0) for c in compositions],
            "T3": [1100.0 + c.get("B2O3", 0) for c in compositions],
        }


def fake_predictor(**kwargs):
    model = FakeGlassNet()
    loads = []

    def factory():
        loads.append(True)
        return model

    return GlassNetPredictor(_model_factory=factory, **kwargs), model, loads


def test_named_units_conversion_and_one_shared_persistent_model():
    predictor, model, loads = fake_predictor()
    assert not loads
    assert predictor.identity["backend"] == "injected_unvalidated"
    assert not loads
    result = predictor.predict_outputs(TOY_COMPOSITION)
    assert {name: output.mean for name, output in result.items()} == {
        "CTEbelowTg": pytest.approx(1e-5), "Tg": 750.0, "T3": 1150.0}
    assert all(output.model_spread is output.config_spread is output.native_uncertainty
               is output.pooled_spread is output.samples is None for output in result.values())
    predictor.predict_outputs({"SiO2": 60, "B2O3": 40})
    assert len(loads) == 1
    assert len(model.calls) == 2
    schema = predictor.output_schema
    assert predictor.identity['output_metadata']["CTEbelowTg"]["source_unit"] == "log10(K^-1)"
    assert schema["CTEbelowTg"].unit == "K^-1"
    assert schema["Tg"].unit == schema["T3"].unit == "K"
    assert all(item.representation == 'physical' for item in schema.values())
    schema["Tg"] = None
    assert predictor.output_schema["Tg"].unit == "K"
    # Provenance remains stable JSON despite the typed public schema.
    json.dumps(predictor.identity, allow_nan=False)
    assert not hasattr(predictor, "predict")  # Do not impersonate reward/std protocol.


def test_key_and_input_ignore_mapping_order_zero_labels_and_mole_scaling():
    predictor, model, _ = fake_predictor()
    fractional = {name: value / 10 for name, value in reversed(list(TOY_COMPOSITION.items())) if value}
    assert predictor.canonical_key(TOY_COMPOSITION) == predictor.canonical_key(fractional)
    first = predictor.predict_outputs(TOY_COMPOSITION)
    assert predictor.predict_outputs(fractional) == first
    assert predictor.cache_info()["cache_hits"] == 1
    rows, names = model.calls[0]
    assert names == sorted(predictor.oxide_basis)
    assert dict(zip(names, rows[0])) == {
        name: 50.0 if name in {"SiO2", "B2O3"} else 0.0 for name in names
    }
    first["Tg"] = -500
    assert predictor.predict_outputs(TOY_COMPOSITION)["Tg"].mean == 750


def test_no_rounding_collision_between_close_compositions():
    predictor, _, _ = fake_predictor()
    assert predictor.canonical_key({"SiO2": 0.7, "B2O3": 0.3}) != predictor.canonical_key(
        {"SiO2": 0.700000000000001, "B2O3": 0.3}
    )


def test_mutating_caller_input_cannot_reuse_a_stale_prediction():
    predictor, model, _ = fake_predictor()
    composition = {"SiO2": 70, "B2O3": 30}
    assert predictor.predict_outputs(composition)["Tg"].mean == 770
    composition.update(SiO2=60, B2O3=40)
    assert predictor.predict_outputs(composition)["Tg"].mean == 760
    assert len(model.calls) == 2


@pytest.mark.parametrize("name,value", [
    ("oxide_basis", ("Li2O", "SiO2")), ("cache_size", 0), ("batch_size", 1),
])
def test_oracle_protocol_is_immutable_after_construction(name, value):
    predictor, _, _ = fake_predictor()
    with pytest.raises(AttributeError):
        setattr(predictor, name, value)


def test_batch_order_duplicate_coalescing_and_lru_eviction():
    predictor, model, loads = fake_predictor(cache_size=1, batch_size=4)
    a, b = {"SiO2": 70, "B2O3": 30}, {"SiO2": 60, "B2O3": 40}
    results = predictor.batch_predict_outputs([a, b, a])
    assert [result["Tg"].mean for result in results] == [770, 760, 770]
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 2
    assert predictor.cache_info() == {
        "requests": 3, "cache_hits": 0, "batch_duplicates": 1,
        "model_batches": 1, "model_candidates": 2, "cache_entries": 1, "cache_capacity": 1,
    }
    assert predictor.predict_outputs(a) == results[0]  # Last request was a.
    assert len(model.calls) == 1
    predictor.predict_outputs(b)  # Evicted prediction is recalculated, not invented.
    assert len(model.calls) == 2
    predictor.clear_cache()
    assert predictor.cache_info()["cache_entries"] == 0
    predictor.predict_outputs(b)
    assert len(loads) == 1


def test_bounded_batches_and_cache_disabled():
    predictor, model, _ = fake_predictor(cache_size=0, batch_size=2)
    inputs = [{"SiO2": value, "B2O3": 100 - value} for value in range(60, 65)]
    outputs = predictor.batch_predict_outputs(inputs)
    assert [output["Tg"].mean for output in outputs] == list(range(760, 765))
    assert [len(rows) for rows, _ in model.calls] == [2, 2, 1]
    assert predictor.cache_info()["cache_entries"] == 0
    assert predictor.predict_outputs(inputs[0]) == outputs[0]
    assert len(model.calls) == 4
    assert predictor.batch_predict_outputs([]) == []


@pytest.mark.parametrize("batch", ["SiO2", TOY_COMPOSITION, None])
def test_invalid_batch_container_is_rejected_before_loading(batch):
    predictor, _, loads = fake_predictor()
    with pytest.raises(ValueError, match="sequence"):
        predictor.batch_predict_outputs(batch)
    assert not loads


@pytest.mark.parametrize("composition", [
    {}, {"SiO2": 0}, {"SiO2": -1}, {"SiO2": math.nan}, {"SiO2": math.inf},
    {"SiO2": True}, {"SiO2": "70"}, {"Si": 1}, {"SiO2": 90, "unknown": 0}, [70, 30],
])
def test_invalid_inputs_fail_before_model_load(composition):
    predictor, _, loads = fake_predictor()
    with pytest.raises(ValueError):
        predictor.predict_outputs(composition)
    assert not loads


@pytest.mark.parametrize("kwargs", [
    {"oxide_basis": []}, {"oxide_basis": "SiO2"}, {"oxide_basis": ["SiO2", "SiO2"]},
    {"oxide_basis": ["Si"]}, {"oxide_basis": ["O2"]}, {"oxide_basis": [1]},
    {"cache_size": -1}, {"cache_size": True}, {"batch_size": 0}, {"batch_size": 1.5},
])
def test_invalid_runtime_settings(kwargs):
    with pytest.raises(ValueError):
        GlassNetPredictor(**kwargs)


@pytest.mark.parametrize("bad_frame", [
    {"Tg": [800], "T3": [1200]},
    {"CTEbelowTg": [-5], "Tg": [], "T3": [1200]},
    {"CTEbelowTg": [math.nan], "Tg": [800], "T3": [1200]},
    {"CTEbelowTg": [400], "Tg": [800], "T3": [1200]},
    {"CTEbelowTg": [-400], "Tg": [800], "T3": [1200]},
    {"CTEbelowTg": [-5], "Tg": [-1], "T3": [1200]},
])
def test_failed_prediction_is_not_cached_or_mapped_to_reward_penalty(bad_frame):
    model = SimpleNamespace(predict=lambda *args, **kwargs: bad_frame)
    predictor = GlassNetPredictor(_model_factory=lambda: model)
    with pytest.raises(ValueError):
        predictor.predict_outputs(TOY_COMPOSITION)
    assert predictor.cache_info()["cache_entries"] == 0
    assert predictor.cache_info()["model_batches"] == 1


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    """Only exercise preflight bookkeeping; these are not loadable model files."""
    import platformdirs

    source = tmp_path / "source" / "glasspy"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# Fake, never imported.\n")
    monkeypatch.setattr(glassnet.util, "find_spec", lambda name: SimpleNamespace(origin=str(source / "__init__.py")))
    monkeypatch.setattr(glassnet, "_SOURCE_SHA256", glassnet._source_digest(source))
    monkeypatch.setattr(glassnet.metadata, "version", lambda name: "0.6.0" if name == "glasspy" else "test-version")
    data = tmp_path / "data"
    (data / "models" / "st-nn").mkdir(parents=True)
    (data / "versions.json").write_text(json.dumps({"glasspy": "0.6.0", "sklearn": "test-version"}))
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda name: str(data))
    weights = {}
    for name in (*glassnet._WEIGHT_SHA256, *glassnet._AUDITED_SCALER_SHA256):
        content = f"FAKE: {name}".encode()
        (data / "models" / name).write_bytes(content)
        if name in glassnet._WEIGHT_SHA256:
            weights[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(glassnet, "_WEIGHT_SHA256", weights)
    return data, source


def test_asset_identity_records_actual_scalers_without_loading_model(fake_assets):
    identity, data = glassnet._stock_assets()
    assert data == fake_assets[0]
    assert identity["matches_audited_scalers"] is False
    assert len(identity["assets_sha256"]) == 6
    assert identity["assets_sha256"]["GlassNet_scaler_x.joblib"] == glassnet._sha256(
        data / "models" / "GlassNet_scaler_x.joblib"
    )


def test_scaler_change_after_manifest_is_rejected_before_glasspy_import(fake_assets):
    data, _ = fake_assets
    predictor = GlassNetPredictor()
    saved = predictor.identity
    (data / "models" / "GlassNet_scaler_x.joblib").write_bytes(b"DIFFERENT SCALER")
    with pytest.raises(RuntimeError, match="assets changed after identity"):
        predictor.predict_outputs(TOY_COMPOSITION)
    assert predictor.identity == saved
    assert predictor.cache_info()["model_batches"] == 0
    assert predictor.cache_info()["cache_entries"] == 0


def test_stock_constructor_arguments_and_loaded_model_survives_disk_changes(fake_assets, monkeypatch):
    """Fake stock modules check loader wiring; these are not real predictions."""
    data, _ = fake_assets
    model = FakeGlassNet()
    constructors = []

    def construct(*, st_models):
        constructors.append(st_models)
        return model

    package = ModuleType("glasspy")
    predict = ModuleType("glasspy.predict")
    base = ModuleType("glasspy.predict.base")
    predict.GlassNet = construct
    base._BASEMODELPATH = data / "models"
    for name, module in (("glasspy", package), ("glasspy.predict", predict), ("glasspy.predict.base", base)):
        monkeypatch.setitem(sys.modules, name, module)

    predictor = GlassNetPredictor()
    first = predictor.predict_outputs(TOY_COMPOSITION)
    identity = predictor.identity
    (data / "models" / "GlassNet_scaler_x.joblib").write_bytes(b"NEW DISK SCALER")
    predictor.clear_cache()
    assert predictor.predict_outputs(TOY_COMPOSITION) == first
    assert predictor.identity == identity  # The in-memory oracle did not reload.
    assert constructors == [["Tg"]]
    assert len(model.calls) == 2


@pytest.mark.parametrize("failure", ["version", "source", "weight", "missing_scaler", "missing_versions"])
def test_preflight_rejects_unsafe_or_incomplete_cache_before_import(fake_assets, failure):
    data, source = fake_assets
    if failure == "version":
        (data / "versions.json").write_text('{"glasspy":"0.5","sklearn":"test-version"}')
    elif failure == "source":
        (source / "__init__.py").write_text("# Changed model source\n")
    elif failure == "weight":
        (data / "models" / "GlassNetMH.pth").write_bytes(b"WRONG WEIGHTS")
    elif failure == "missing_scaler":
        (data / "models" / "GlassNet_scaler_x.joblib").unlink()
    else:
        (data / "versions.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        glassnet._stock_assets()


@pytest.mark.skipif(os.environ.get("RL_MATDESIGN_GLASSNET_REAL") != "1", reason="Requires prepared, pinned real GlassPy assets.")
def test_real_glassnet_matches_stock_conversion_and_input_order():
    """Compare toy-input inference with stock GlassPy, not a research result."""
    predictor = GlassNetPredictor(oxide_basis=["SiO2", "B2O3", "Na2O"], cache_size=0)
    other = {"SiO2": 3, "B2O3": 1}
    reordered = dict(reversed(list(TOY_COMPOSITION.items())))
    outputs = predictor.batch_predict_outputs([TOY_COMPOSITION, other, reordered])
    assert predictor.identity["backend"] == "stock_glasspy"
    # Asset preflight has already passed before importing the stock package.
    # Spell the normalized rows independently and use a different column order
    # from the adapter, so an input normalization/order bug cannot self-validate.
    from glasspy.predict import GlassNet
    stock = GlassNet(st_models=["Tg"])
    raw = stock.predict([[50, 50, 0], [75, 25, 0]],
                        input_cols=["SiO2", "B2O3", "Na2O"], return_dataframe=True)
    for record, row in zip(outputs, [0, 1, 0]):
        assert set(record) == {"CTEbelowTg", "Tg", "T3"}
        for name, unit in [("CTEbelowTg", "K^-1"), ("Tg", "K"), ("T3", "K")]:
            expected = float(raw[name].iloc[row])
            if name == "CTEbelowTg":
                expected = 10 ** expected
            assert math.isfinite(record[name].mean) and record[name].mean > 0
            assert record[name].schema.unit == unit
            assert record[name].mean == pytest.approx(expected, rel=1e-5,
                                                      abs=1e-10 if name == "CTEbelowTg" else 0.01)
    scalar = predictor.predict_outputs(reordered)
    for name in outputs[0]:
        assert scalar[name].mean == pytest.approx(outputs[0][name].mean, rel=1e-5,
                                                 abs=1e-10 if name == "CTEbelowTg" else 0.01)
    assert predictor.cache_info()["cache_entries"] == 0

def test_named_scientific_contract_roundtrips_without_fabricated_uncertainty():
    from rl_matdesign.predictors.outputs import NamedOutputPredictor, outputs_from_json, outputs_to_json
    predictor, model, _ = fake_predictor()
    assert isinstance(predictor, NamedOutputPredictor)
    outputs = predictor.predict_outputs(TOY_COMPOSITION)
    assert outputs_from_json(outputs_to_json(outputs)) == outputs
    assert all(isinstance(value, float) for values in predictor._cache.values() for value in values.values())
    assert len(model.calls) == 1
