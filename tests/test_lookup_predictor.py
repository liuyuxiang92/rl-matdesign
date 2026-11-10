"""Dense lookup leaves use the existing property engine without training access."""
from __future__ import annotations

import numpy as np
import pytest

from rl_matdesign.configuration import build_predictor
from rl_matdesign.predictors.lookup import LookupPredictor
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError


FQN = "rl_matdesign.predictors.lookup:LookupPredictor"


def _write_model(tmp_path, *, filename="table.npz", remove=(), **updates):
    arrays = {
        "field_names": np.asarray(["temperature", "solvent"]),
        "axis_0": np.asarray([100.0, 150.0]),
        "axis_1": np.asarray(["01", "1", "water"]),
        "mean": np.asarray([[1., 2., 3.], [4., 5., 6.]]),
    }
    arrays.update(updates)
    for key in remove:
        del arrays[key]
    path = tmp_path / filename
    np.savez(path, **arrays)
    return path


def _engine(*paths, objective="mean", uncertainty=None, **settings):
    return build_predictor({
        "properties": [
            {"name": f"p{i}", "predictor": FQN, "model": str(path), "direction": "max",
             "objective": objective, "uncertainty": uncertainty}
            for i, path in enumerate(paths)
        ],
        **settings,
    }, seed=42)


def test_single_property_without_std_accepts_order_and_one_group(tmp_path):
    predictor = _engine(_write_model(tmp_path))
    first = dict({"temperature": 150, "solvent": "01"})
    reordered = dict({"solvent": "01", "temperature": 150.0})
    assert predictor.predict(first) == (4., None)
    assert predictor.predict(reordered) == (4., None)
    assert predictor.predict(dict({"conditions": reordered})) == (4., None)
    assert predictor.per_objective_stats(first) == {"p0": (4., None)}
    assert predictor.predict(dict({"solvent": "1", "temperature": 150})) == (5., None)


def test_multiple_properties_preserve_table_and_scalar_uncertainty(tmp_path):
    first = _write_model(tmp_path, std=np.full((2, 3), .25))
    second = _write_model(tmp_path, filename="second.npz", mean=np.full((2, 3), 10.), std=2.)
    predictor = _engine(first, second, k=2., objective="mean_minus_kstd",
                        uncertainty="native_uncertainty")
    candidate = {"conditions": {"solvent": "water", "temperature": 100}}
    assert predictor.per_objective_stats(candidate) == {"p0": (3., .25), "p1": (10., 2.)}
    # The scientific standard deviations remain per property; a sum of rewards
    # does not invent an uncertainty by averaging unrelated native estimates.
    assert predictor.predict(candidate) == (8.5, None)


def test_self_folded_lookup_ensemble_requires_raw_samples_without_discarding_native_std(tmp_path):
    first = _write_model(tmp_path, std=.5)
    second = _write_model(tmp_path, filename="second.npz", mean=np.full((2, 3), 5.), std=1.5)
    predictor = build_predictor({"properties": [{
        "name": "response", "predictor": FQN,
        "model": [str(first), str(second)], "direction": "max", "objective": "mean",
    }]})
    # Means plus native std do not define aligned underlying model/configuration
    # samples. Combining them must not silently drop std or relabel it model spread.
    with pytest.raises(EvaluationProtocolError, match="raw samples.*native uncertainty"):
        predictor.per_objective_stats({"temperature": 100, "solvent": "water"})


@pytest.mark.parametrize("std", [None, 0., .5])
def test_named_output_preserves_missing_and_native_uncertainty(tmp_path, std):
    path = _write_model(tmp_path, **({} if std is None else {"std": std}))
    predictor = LookupPredictor({"model": path})
    candidate = {"temperature": 100, "solvent": "water"}
    output = predictor.predict_outputs(candidate)["response"]
    assert output.schema == predictor.output_schema["response"]
    assert output.schema.unit is None and output.schema.representation == "physical"
    assert output.mean == 3.
    assert output.native_uncertainty == std
    assert output.model_spread is output.config_spread is output.pooled_spread is None
    assert output.samples is None
    assert predictor.predict(candidate) == (3., std)
    assert predictor.batch_predict_outputs([candidate, candidate]) == [{"response": output}] * 2


def test_missing_uncertainty_cannot_support_native_penalty(tmp_path):
    predictor = _engine(_write_model(tmp_path), objective="mean_minus_kstd",
                        uncertainty="native_uncertainty")
    with pytest.raises(EvaluationProtocolError, match="uncertainty is unavailable"):
        predictor.predict({"temperature": 100, "solvent": "water"})


def test_loads_once_and_retains_read_only_arrays(tmp_path, monkeypatch):
    path = _write_model(tmp_path, std=.5)
    original = np.load
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    predictor = LookupPredictor({"model": path})
    path.unlink()
    candidate = {"solvent": "water", "temperature": np.int64(100)}
    assert predictor.batch_predict([candidate, candidate]) == [(3., .5)] * 2
    assert len(loads) == 1 and loads[0][1] == {"allow_pickle": False}
    assert not hasattr(predictor, "score")
    for array in (*predictor._axes, predictor._mean, predictor._std):
        assert not array.flags.writeable


@pytest.mark.parametrize("candidate", [
    {}, {"temperature": 100}, {"temperature": 100, "solvent": "water", "extra": 0},
    {"conditions": {"temperature": 100, "solvent": "water"}, "other": {}},
    {"temperature": 151, "solvent": "water"},
    {"temperature": 100.0000000001, "solvent": "water"},
    {"temperature": "100", "solvent": "water"},
    {"temperature": True, "solvent": "water"},
    {"temperature": np.bool_(False), "solvent": "water"},
    {"temperature": np.nan, "solvent": "water"},
    {"temperature": np.inf, "solvent": "water"},
    {"temperature": 100, "solvent": 1},
    {"temperature": 100, "solvent": "unknown"},
    {"temperature": [100], "solvent": "water"},
    {"temperature": 100, "solvent": {"x": 1}},
    [100, "water"],
])
def test_invalid_candidates_raise_instead_of_returning_zero(tmp_path, candidate):
    predictor = _engine(_write_model(tmp_path))
    with pytest.raises((ValueError, TypeError)):
        predictor.predict(candidate)


@pytest.mark.parametrize("updates, message", [
    ({"field_names": np.asarray(["temperature", "temperature"])}, "unique names"),
    ({"field_names": np.asarray(["", "solvent"])}, "empty names"),
    ({"field_names": np.asarray([], dtype=str)}, "Unicode"),
    ({"field_names": np.asarray([b"temperature", b"solvent"])}, "Unicode"),
    ({"field_names": np.asarray([["temperature", "solvent"]])}, "Unicode"),
    ({"axis_0": np.asarray([100., 100.])}, "unique values"),
    ({"axis_0": np.asarray([100., np.nan])}, "finite"),
    ({"axis_0": np.asarray([100., np.inf])}, "finite"),
    ({"axis_0": np.asarray([])}, "nonempty"),
    ({"axis_0": np.asarray([[100., 150.]])}, "1-D"),
    ({"axis_0": np.asarray([False, True])}, "numeric or string"),
    ({"axis_0": np.asarray([100., 150.], dtype=object)}, "without pickle"),
    ({"mean": np.ones((3, 2))}, "mean shape"),
    ({"mean": np.full((2, 3), np.nan)}, "finite"),
    ({"mean": np.full((2, 3), np.inf)}, "finite"),
    ({"mean": np.full((2, 3), "1")}, "numeric"),
    ({"mean": np.full((2, 3), 1j)}, "numeric"),
    ({"std": np.asarray([.5, .5])}, "std must be scalar"),
    ({"std": -.1}, "nonnegative"),
    ({"std": np.full((2, 3), -.1)}, "nonnegative"),
    ({"std": np.inf}, "finite"),
    ({"std": np.nan}, "finite"),
])
def test_invalid_archives_raise_at_construction(tmp_path, updates, message):
    path = _write_model(tmp_path, **updates)
    with pytest.raises(ValueError, match=message):
        _engine(path)


@pytest.mark.parametrize("missing", ["field_names", "axis_0", "axis_1", "mean"])
def test_required_archive_arrays_cannot_be_missing(tmp_path, missing):
    with pytest.raises(ValueError, match=missing):
        _engine(_write_model(tmp_path, remove=[missing]))


def test_missing_path_and_non_archive_are_errors(tmp_path):
    with pytest.raises(ValueError, match="model"):
        LookupPredictor({})
    with pytest.raises(FileNotFoundError):
        _engine(tmp_path / "missing.npz")
    path = tmp_path / "array.npy"
    np.save(path, np.asarray([1., 2.]))
    with pytest.raises(ValueError, match="NPZ archive"):
        _engine(path)


def test_byte_string_axes_decode_without_collapsing_distinct_labels(tmp_path):
    predictor = _engine(_write_model(tmp_path, axis_1=np.asarray([b"01", b"1", b"water"])))
    assert predictor.predict(dict({"temperature": 100, "solvent": "01"})) == (1., None)
    assert predictor.predict(dict({"temperature": 100, "solvent": "1"})) == (2., None)


def test_leaf_plain_dicts_keep_exact_types_and_numeric_values(tmp_path):
    path = _write_model(tmp_path, axis_0=np.asarray([1., 1.00000000001]))
    predictor = LookupPredictor({"model": path})
    assert predictor.predict({"temperature": 1, "solvent": "01"}) == (1., None)
    assert predictor.predict({"solvent": "1", "temperature": 1.}) == (2., None)
    assert predictor.predict({"conditions": {"temperature": 1.00000000001, "solvent": "01"}}) == (4., None)
    with pytest.raises(ValueError, match="not bool"):
        predictor.predict({"temperature": True, "solvent": "01"})
    with pytest.raises(ValueError, match="not on the model grid"):
        predictor.predict({"temperature": 1.000000000001, "solvent": "01"})


def test_engine_cache_keeps_exact_field_values_and_does_not_mask_invalid_types(tmp_path):
    first, second = 1.0000000000001, 1.0000000000002
    predictor = _engine(_write_model(tmp_path, axis_0=np.asarray([first, second])))
    assert predictor.predict({"temperature": first, "solvent": "01"}) == (1., None)
    assert predictor.predict({"temperature": second, "solvent": "01"}) == (4., None)
    numeric = _engine(_write_model(tmp_path, axis_0=np.asarray([0, 1])))
    assert numeric.predict({"temperature": 1, "solvent": "01"}) == (4., None)
    assert numeric.predict({"temperature": 1., "solvent": "01"}) == (4., None)
    with pytest.raises(ValueError, match="not bool"):
        numeric.predict({"temperature": True, "solvent": "01"})
    with pytest.raises(ValueError, match="finite real number"):
        numeric.predict({"temperature": "1", "solvent": "01"})
