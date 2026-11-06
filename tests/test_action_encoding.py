"""Stage 2B preparation: action identity and representation, without a trainer."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rl_matdesign.action_encoding import (
    ActionEncoding,
    LabelEncoding,
    ValueEncoding,
    ValueSpec,
    typed_identity,
)


def test_composition_values_retain_absolute_physical_amounts():
    encoding = ActionEncoding(["Ni", "Fe"], [0.2, 0.6])
    label_20, value_20 = encoding.encode(0, 0)
    label_60, value_60 = encoding.encode(0, 1)
    np.testing.assert_array_equal(label_20, label_60)
    np.testing.assert_array_equal(value_20, [0.2])
    np.testing.assert_array_equal(value_60, [0.6])
    assert encoding.values.feature_dim == 1


def test_monomer_values_have_identity_without_ordinal_distance():
    encoding = ValueEncoding(["A", "noncanonical-138", "C"], kind="categorical")
    np.testing.assert_array_equal(encoding.features, np.eye(3))
    distances = [np.linalg.norm(encoding.encode(i) - encoding.encode(j))
                 for i in range(3) for j in range(i)]
    np.testing.assert_allclose(distances, np.sqrt(2))


def test_complete_position_monomer_pair_uses_two_independent_feature_axes():
    encoding = ActionEncoding(
        [("peptide", "position", 1), ("peptide", "position", 3)],
        ["A", "C", "D"], value_kind="categorical",
    )
    labels, values = encoding.encode_many([(1, 0), (0, 2)])
    np.testing.assert_array_equal(labels, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(values, [[1, 0, 0], [0, 0, 1]])
    assert labels.shape[1] + values.shape[1] == 5  # no Cartesian-product vocabulary


def test_categorical_scalar_types_remain_distinct():
    values = [True, 1, 1.0, "1", "01", False, None]
    encoding = ValueEncoding(values, kind="categorical")
    assert [encoding.index_of(value) for value in values] == list(range(len(values)))
    np.testing.assert_array_equal(encoding.features, np.eye(len(values)))
    restored = ValueEncoding.from_schema(json.loads(json.dumps(encoding.to_schema())))
    assert [type(spec.value) for spec in restored.specs] == [type(value) for value in values]
    assert restored.to_schema() == encoding.to_schema()


def test_numeric_axis_treats_equal_number_spellings_as_duplicate_amounts():
    with pytest.raises(ValueError, match="unique identities"):
        ValueEncoding([1, 1.0], kind="numeric")
    with pytest.raises(ValueError, match="unique identities"):
        ValueEncoding([0.0, -0.0], kind="numeric")
    assert ValueEncoding([1.0]).index_of(1) == 0


def test_label_namespaces_do_not_merge_identical_position_numbers():
    encoding = LabelEncoding([("A_chain", 1), ("B_chain", 1)])
    assert encoding.index_of(("A_chain", 1)) == 0
    assert encoding.index_of(("B_chain", 1)) == 1
    assert typed_identity(("A_chain", 1)) != typed_identity(("B_chain", 1))


def test_mapping_labels_ignore_construction_order_but_keep_types():
    first = {"group": "peptide", "position": 3}
    second = {"position": 3, "group": "peptide"}
    assert typed_identity(first) == typed_identity(second)
    encoding = LabelEncoding([first, {"group": "peptide", "position": "3"}])
    assert encoding.index_of(second) == 0
    assert encoding.index_of({"group": "peptide", "position": "3"}) == 1
    with pytest.raises(ValueError, match="unique typed identities"):
        LabelEncoding([first, second])


def test_declared_index_order_is_not_silently_changed():
    first, second = LabelEncoding(["Ni", "Fe"]), LabelEncoding(["Fe", "Ni"])
    assert first.index_of("Ni") == 0
    assert second.index_of("Ni") == 1
    assert first.to_schema() != second.to_schema()
    assert typed_identity(first.labels[0]) == typed_identity(second.labels[1])


def test_mixed_axis_preserves_numeric_magnitude_and_typed_categories():
    specs = [ValueSpec(0.2, "numeric", "composition"),
             ValueSpec("A", "categorical", "peptide"),
             ValueSpec(0.6, "numeric", "composition"),
             ValueSpec(1, "categorical", "vessel"),
             ValueSpec("1", "categorical", "vessel")]
    encoding = ValueEncoding.from_specs(specs)
    np.testing.assert_array_equal(encoding.features,
                                  [[.2, 0, 0, 0], [0, 1, 0, 0], [.6, 0, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1]])
    assert encoding.index_of("A", kind="categorical", namespace="peptide") == 1
    assert encoding.index_of(.6, kind="numeric", namespace="composition") == 2
    with pytest.raises(ValueError, match="mixed value axis"):
        encoding.index_of("A", namespace="peptide")


def test_identical_token_names_in_distinct_domains_have_distinct_channels():
    encoding = ValueEncoding.from_specs([
        ValueSpec("A", namespace="peptide"), ValueSpec("A", namespace="process"),
    ])
    np.testing.assert_array_equal(encoding.features, np.eye(2))
    assert encoding.index_of("A", namespace="process") == 1


def test_schema_roundtrip_reconstructs_features_and_indices():
    original = ActionEncoding(
        [("peptide", 1), ("process", "temperature")],
        ValueEncoding.from_specs([ValueSpec("C", namespace="peptide"),
                                  ValueSpec(320, "numeric", "temperature")]),
    )
    schema = json.loads(json.dumps(original.to_schema()))
    restored = ActionEncoding.from_schema(schema)
    assert restored.to_schema() == schema
    np.testing.assert_array_equal(restored.labels.features, original.labels.features)
    np.testing.assert_array_equal(restored.values.features, original.values.features)
    assert restored.labels.index_of(("peptide", 1)) == 0
    assert restored.values.index_of(320, kind="numeric", namespace="temperature") == 1


def test_feature_tables_and_public_schemas_cannot_mutate_encoding():
    labels = [{"position": [1]}]
    encoding = ActionEncoding(labels, ["A", "C"], value_kind="categorical")
    original = deepcopy(encoding.to_schema())
    labels[0]["position"].append(2)
    encoding.labels.labels[0]["position"].append(3)
    schema = encoding.to_schema()
    schema["labels"]["labels"].clear()
    with pytest.raises(ValueError):
        encoding.labels.features[0, 0] = 0
    with pytest.raises(ValueError):
        encoding.values.encode(0)[0] = 0
    assert encoding.to_schema() == original


def test_empty_batch_has_correct_widths():
    labels, values = ActionEncoding(["Ni", "Fe"], [0.2]).encode_many([])
    assert labels.shape == (0, 2)
    assert values.shape == (0, 1)


@pytest.mark.parametrize("index", [-1, 2, 1.1, True, "0"])
def test_invalid_indices_cannot_select_a_different_action(index):
    encoding = ActionEncoding(["Ni", "Fe"], [0.2, 0.6])
    with pytest.raises((TypeError, IndexError)):
        encoding.encode(index, 0)
    with pytest.raises((TypeError, IndexError)):
        encoding.encode(0, index)


@pytest.mark.parametrize("value", [True, "0.2", None, float("nan"), float("inf")])
def test_numeric_values_reject_nonphysical_or_nonfinite_scalars(value):
    with pytest.raises((TypeError, ValueError)):
        ValueEncoding([value], kind="numeric")


@pytest.mark.parametrize("value", [float("nan"), {1, 2}, object()])
def test_categorical_identity_rejects_unserializable_values(value):
    with pytest.raises((TypeError, ValueError)):
        ValueEncoding([value], kind="categorical")


def test_numpy_scalars_have_the_same_declared_identity_as_python_scalars():
    assert typed_identity(np.int64(1)) == typed_identity(1)
    assert typed_identity(np.float64(1.0)) == typed_identity(1.0)
    assert typed_identity(np.bool_(True)) == typed_identity(True)
    assert ValueEncoding([np.float64(.2)]).index_of(.2) == 0


@pytest.mark.parametrize("schema", [
    {"version": 2, "encoding": "label_one_hot", "labels": [["str", "a"]]},
    {"version": True, "encoding": "label_one_hot", "labels": [["str", "a"]]},
    {"version": 1, "encoding": "label_one_hot", "labels": [["str", 1]]},
    {"version": 1, "encoding": "label_one_hot", "labels": [["int", True]]},
    {"version": 1, "encoding": "wrong", "labels": [["str", "a"]]},
])
def test_schema_rejects_unsupported_version_or_corrupted_identity(schema):
    with pytest.raises(ValueError):
        LabelEncoding.from_schema(schema)


def test_unknown_value_is_not_coerced_or_clipped():
    encoding = ValueEncoding(["01", "1"], kind="categorical")
    with pytest.raises(KeyError, match="Unknown categorical"):
        encoding.index_of(1)
    with pytest.raises(KeyError, match="Unknown numeric"):
        ValueEncoding([.2, .6]).index_of(.4)
    with pytest.raises(KeyError, match="Unknown action label"):
        LabelEncoding(["Ni"]).index_of("Fe")
