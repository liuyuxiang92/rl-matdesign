import numpy as np
import pytest
from types import SimpleNamespace

from rl_matdesign.action_encoding import ValueSpec
from rl_matdesign.action_features import value_alphabet, value_features


def test_numeric_values_keep_real_amplitude():
    env = SimpleNamespace(fraction_set=["0.20", "0.60"])
    values = value_alphabet(env)
    np.testing.assert_array_equal(value_features(values, [1, 0]), [[0.6], [0.2]])
    np.testing.assert_array_equal(value_features(env.fraction_set, [1, 0]), [[0.6], [0.2]])
    assert values.feature_dim == 1


def test_opaque_monomer_codes_use_identity_not_float_conversion():
    env = SimpleNamespace(fraction_set=['token:A', 'token:C'],
                          value_specs=[ValueSpec('A', 'categorical', 'monomer'),
                                       ValueSpec('C', 'categorical', 'monomer')])
    values = value_alphabet(env)
    np.testing.assert_array_equal(value_features(values, [1, 0]), [[0, 1], [1, 0]])
    assert values.to_schema()['codes'] == env.fraction_set
    with pytest.raises(TypeError):
        values[0] = 'other'


def test_mixed_group_values_preserve_type_and_scientific_numeric_axis():
    env = SimpleNamespace(fraction_set=['frac:0.2', 'field:1', 'field:true'], value_specs=[
        ValueSpec(.2, 'numeric', 'amount'), ValueSpec(1, 'categorical', 'field'),
        ValueSpec(True, 'categorical', 'field')])
    values = value_alphabet(env)
    matrix = value_features(values, [0, 1, 2])
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == .2
    assert not np.array_equal(matrix[1], matrix[2])
