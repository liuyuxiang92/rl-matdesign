"""DeepMD scalar selection, output contracts, and ensemble orientation."""
from __future__ import annotations

import numpy as np
import pytest


def test_pick_scalar_default_index_0():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0])) == 1.0


def test_pick_scalar_explicit_index():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_index=2) == 3.0


def test_pick_scalar_mean_aggregator():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_aggregator="mean") == 2.0


def test_pick_scalar_max_aggregator():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_aggregator="max") == 3.0


def test_pick_scalar_handles_2d_input():
    """Multi-output models often return shape (1, K) — must be flattened."""
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([[10.0, 20.0]]), output_index=1) == 20.0


def test_pick_scalar_index_out_of_range_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(IndexError) as info:
        pick_scalar(np.array([1.0]), output_index=5)
    msg = str(info.value)
    assert "5" in msg and "1" in msg          # surfaces both the index and the vector length


def test_pick_scalar_unknown_aggregator_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(ValueError):
        pick_scalar(np.array([1.0, 2.0]), output_aggregator="median")


def test_pick_scalar_empty_vector_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(ValueError):
        pick_scalar(np.array([]))


# --------------------------------------------------------------------------- #
# Fold orientation: both helpers must return (n_models, n_structures).
#
# They used to iterate in OPPOSITE orders and return flat lists — eval_energy_ase
# structure-major, eval_property_ensemble model-major — so a caller reshaping either
# one blindly would average the wrong axis and silently report configurational
# scatter as model uncertainty.
# --------------------------------------------------------------------------- #

class _FakeCalc:
    """Minimal ASE-like calculator: energy = offset * natoms."""

    def __init__(self, offset):
        self.offset = offset

    def get_potential_energy(self, atoms=None):
        return self.offset * len(self._atoms)


class _FakeAtoms:
    def __init__(self, n, tag):
        self.n, self.tag, self.calc = n, tag, None

    def __len__(self):
        return self.n

    def copy(self):
        return _FakeAtoms(self.n, self.tag)

    def get_potential_energy(self):
        return float(self.calc.offset * 100 + self.tag)


def test_eval_energy_ase_returns_models_by_structures():
    from rl_matdesign.utils.dp_eval import eval_energy_ase

    structures = [_FakeAtoms(1, 1), _FakeAtoms(1, 2), _FakeAtoms(1, 3)]
    calcs = [_FakeCalc(1), _FakeCalc(2)]
    out = eval_energy_ase(structures, calcs, energy_per_atom=False)
    assert out.shape == (2, 3)                     # (n_models, n_structures)
    # row = model, column = structure
    assert list(out[0]) == [101.0, 102.0, 103.0]
    assert list(out[1]) == [201.0, 202.0, 203.0]


def test_fold_averages_structures_then_spreads_over_models():
    from rl_matdesign.predictors.structure_score import _fold_models_structures

    # 2 models x 3 structures. Per-model means are 10 and 20 regardless of how
    # scattered the structures are within each model.
    arr = np.array([[0.0, 10.0, 20.0],
                    [19.0, 20.0, 21.0]])
    mean, std = _fold_models_structures(arr)
    assert mean == pytest.approx(15.0)             # mean(10, 20)
    assert std == pytest.approx(5.0)               # population std of (10, 20)


def test_fold_with_one_model_has_zero_std_however_scattered():
    from rl_matdesign.predictors.structure_score import _fold_models_structures

    mean, std = _fold_models_structures(np.array([[-100.0, 0.0, 100.0]]))
    assert mean == pytest.approx(0.0)
    assert std == 0.0


class _PredictionAtoms:
    """Structure inputs without requiring ASE or DeepMD in helper tests."""

    def __init__(self, symbols=("H", "O")):
        self.symbols = symbols

    def __len__(self):
        return len(self.symbols)

    def get_positions(self):
        return np.zeros((len(self), 3))

    def get_cell(self):
        from types import SimpleNamespace
        return SimpleNamespace(array=np.eye(3))

    def get_chemical_symbols(self):
        return list(self.symbols)


class _PropertyModel:
    def __init__(self, result):
        self.result = result

    def eval(self, **kwargs):
        return self.result


class _ClassificationModel:
    def __init__(self, probabilities, num_classes=None):
        self.probabilities = np.asarray(probabilities)
        self.num_classes = (
            num_classes if num_classes is not None else self.probabilities.shape[-1]
        )
        self.calls = []

    def eval(self, **kwargs):
        raise AssertionError("Classification must call predict_proba, not raw eval.")

    def predict_proba(self, **kwargs):
        self.calls.append(kwargs)
        return self.probabilities

    def get_num_classes(self):
        return self.num_classes


def test_eval_property_unwraps_tuple_without_losing_configuration_rows():
    from rl_matdesign.utils.dp_eval import eval_property_ensemble

    values = np.array([[101., 102.], [201., 202.], [301., 302.]])
    models = [_PropertyModel((values,)), _PropertyModel(values + 1000)]
    result = eval_property_ensemble(
        [_PredictionAtoms()] * 3, models, {"H": 0, "O": 1}, output_index=1,
    )
    np.testing.assert_array_equal(result, [[102., 202., 302.], [1102., 1202., 1302.]])


@pytest.mark.parametrize("result", [np.array([7., 9.]), (np.array([[7., 9.]]),)])
def test_eval_property_supports_single_frame_tuple_and_legacy_array(result):
    from rl_matdesign.utils.dp_eval import eval_property_ensemble

    actual = eval_property_ensemble(
        [_PredictionAtoms()], [_PropertyModel(result)], {"H": 0, "O": 1}, output_index=1,
    )
    np.testing.assert_array_equal(actual, [[9.]])


def test_eval_property_supports_legacy_scalar_per_frame():
    from rl_matdesign.utils.dp_eval import eval_property_ensemble

    actual = eval_property_ensemble(
        [_PredictionAtoms()] * 2, [_PropertyModel(np.array([7., 9.]))], {"H": 0, "O": 1},
    )
    np.testing.assert_array_equal(actual, [[7., 9.]])


@pytest.mark.parametrize("nframes,result", [
    (2, (np.ones((1, 3)),)),
    (2, np.ones((3, 2))),
    (1, np.ones((2, 3))),
    (2, np.ones((1, 2, 3))),
    (2, np.ones(3)),
    (1, ()),
])
def test_eval_property_rejects_wrong_frame_count_and_shape(nframes, result):
    from rl_matdesign.utils.dp_eval import eval_property_ensemble

    with pytest.raises(ValueError, match="shape|empty tuple"):
        eval_property_ensemble(
            [_PredictionAtoms()] * nframes, [_PropertyModel(result)], {"H": 0, "O": 1},
        )


def test_classification_binary_ensemble_preserves_probabilities_axes_and_params():
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    models = [_ClassificationModel([[0.2, 0.8], [0.7, 0.3]]),
              _ClassificationModel([[0.9, 0.1], [0.4, 0.6]])]
    actual = eval_classification_ensemble(
        [_PredictionAtoms(), _PredictionAtoms(("O", "H"))],
        models, {"H": 0, "O": 1}, output_index=1, fparam=[300.], aparam=[1., 2.],
    )
    np.testing.assert_array_equal(actual, [[0.8, 0.3], [0.1, 0.6]])
    for model in models:
        assert len(model.calls) == 1
        inputs = model.calls[0]
        assert inputs["mixed_type"] is True
        assert inputs["coords"].shape == (2, 2, 3)
        assert inputs["cells"].shape == (2, 3, 3)
        np.testing.assert_array_equal(inputs["atom_types"], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(inputs["fparam"], [[300.], [300.]])
        np.testing.assert_array_equal(inputs["aparam"], [[[1.], [2.]], [[1.], [2.]]])


def test_classification_ternary_single_frame_selects_pure_probability_directly():
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    actual = eval_classification_ensemble(
        [_PredictionAtoms()], [_ClassificationModel([[0.05, 0.15, 0.8]])],
        {"H": 0, "O": 1}, output_index=2,
    )
    np.testing.assert_array_equal(actual, [[0.8]])


@pytest.mark.parametrize("aggregator,expected", [("mean", 1 / 3), ("max", 0.8)])
def test_classification_preserves_property_selector_interface(aggregator, expected):
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    actual = eval_classification_ensemble(
        [_PredictionAtoms()], [_ClassificationModel([[0.05, 0.15, 0.8]])],
        {"H": 0, "O": 1}, output_aggregator=aggregator,
    )
    assert actual[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("output_index", [-1, 2])
def test_classification_rejects_invalid_binary_class_index(output_index):
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(IndexError, match="out of range"):
        eval_classification_ensemble(
            [_PredictionAtoms()], [_ClassificationModel([[0.2, 0.8]])],
            {"H": 0, "O": 1}, output_index=output_index,
        )


@pytest.mark.parametrize("probabilities,match", [
    ([[np.nan, 0.8]], "non-finite"),
    ([[np.inf, 0.8]], "non-finite"),
    ([[-0.1, 1.1]], "lie in"),
    ([[0.2, 0.7]], "sum to 1"),
    ([[1.]], "n_classes >= 2"),
    ([0.2, 0.8], "shape"),
    ([[[0.2, 0.8]]], "shape"),
    ([[0.2, 0.8], [0.3, 0.7]], "shape"),
])
def test_classification_rejects_invalid_single_frame_output(probabilities, match):
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(ValueError, match=match):
        eval_classification_ensemble(
            [_PredictionAtoms()], [_ClassificationModel(probabilities)], {"H": 0, "O": 1},
        )


@pytest.mark.parametrize("rows", [1, 3])
def test_classification_rejects_wrong_batched_frame_count(rows):
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(ValueError, match="shape"):
        eval_classification_ensemble(
            [_PredictionAtoms()] * 2, [_ClassificationModel([[0.2, 0.8]] * rows)],
            {"H": 0, "O": 1},
        )


def test_classification_rejects_different_class_counts_within_ensemble():
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(ValueError, match="same class count"):
        eval_classification_ensemble(
            [_PredictionAtoms()],
            [_ClassificationModel([[0.2, 0.8]]), _ClassificationModel([[0.2, 0.3, 0.5]])],
            {"H": 0, "O": 1},
        )


def test_classification_checks_reported_class_count():
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(ValueError, match="class count disagrees"):
        eval_classification_ensemble(
            [_PredictionAtoms()], [_ClassificationModel([[0.2, 0.8]], num_classes=3)],
            {"H": 0, "O": 1},
        )


def test_classification_reports_missing_probability_api():
    from rl_matdesign.utils.dp_eval import eval_classification_ensemble

    with pytest.raises(TypeError, match="runtime supporting classification"):
        eval_classification_ensemble(
            [_PredictionAtoms()], [_PropertyModel(np.array([[0.2, 0.8]]))],
            {"H": 0, "O": 1},
        )


@pytest.mark.parametrize("helper_name", ["eval_property_ensemble", "eval_classification_ensemble"])
def test_dp_helpers_report_unknown_species_before_inference(helper_name):
    import rl_matdesign.utils.dp_eval as dp_eval

    with pytest.raises(RuntimeError, match="not in the DP model type_map"):
        getattr(dp_eval, helper_name)([_PredictionAtoms()], [object()], {"H": 0})
