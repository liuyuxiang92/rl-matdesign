"""Absolute assignment state must retain resources and ignore choice order."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.env import CompositionEnv


def _descriptor(_formula):
    # Deliberately contains no amount information. The environment must provide
    # the necessary design state independently of any chemistry descriptor.
    return np.array([2.0, 5.0])


def _env(*, zero=False, species=("Ni", "Fe", "Co", "Cu"), descriptor=_descriptor):
    return CompositionEnv(
        species_set=species,
        fraction_set=[f"{i / 10:.2f}" for i in range(0 if zero else 1, 9)],
        total_units=10, n_components=3, state_featurizer=descriptor,
    )


def _pick(env, element, amount):
    for action in env.allowed_actions():
        if (decode_one_hot(action[0], env.species_set) == element
                and float(decode_one_hot(action[1], env.fraction_set)) == amount):
            env.step(action)
            return
    raise AssertionError(f"No legal action for {element}={amount}")


def _decoded(env):
    return [(decode_one_hot(a, env.species_set), decode_one_hot(b, env.fraction_set))
            for a, b in env.allowed_actions()]


def _explicit(env):
    return env.current_state_features()[env.state_feature_schema()["descriptor_dim"]:]


def test_absolute_amount_is_visible_even_when_descriptor_is_identical():
    low, high = _env(), _env()
    _pick(low, "Ni", 0.2)
    _pick(high, "Ni", 0.6)
    assert low.state_labels == high.state_labels == ("Co", "Cu", "Fe", "Ni")
    np.testing.assert_array_equal(low.current_state_features()[:2], [2, 5])
    np.testing.assert_array_equal(high.current_state_features()[:2], [2, 5])
    np.testing.assert_array_equal(_explicit(low), [0, 0, 0, 0.2, 0.8, 2])
    np.testing.assert_array_equal(_explicit(high), [0, 0, 0, 0.6, 0.4, 2])
    assert low.current_state_features().size == 2 + 4 + 2


def test_default_magpie_descriptor_is_supplemented_with_absolute_state():
    from rl_matdesign.featurization import featurize_formula

    low, high = _env(descriptor=featurize_formula), _env(descriptor=featurize_formula)
    _pick(low, "Ni", 0.2)
    _pick(high, "Ni", 0.6)
    dimension = low.state_feature_schema()["descriptor_dim"]
    np.testing.assert_allclose(low.current_state_features()[:dimension],
                               high.current_state_features()[:dimension], atol=1e-12, rtol=1e-12)
    assert not np.array_equal(low.current_state_features(), high.current_state_features())


def test_equal_assignments_have_exactly_equal_state_and_canonical_descriptor_input():
    calls = []

    def record(formula):
        calls.append(formula)
        # An order-sensitive descriptor exposes whether the ENV input is stable.
        return np.array([sum((i + 1) * ord(c) for i, c in enumerate(formula))])

    a, b = _env(descriptor=record), _env(descriptor=record)
    for element, amount in [("Ni", 0.2), ("Fe", 0.4)]:
        _pick(a, element, amount)
    for element, amount in [("Fe", 0.4), ("Ni", 0.2)]:
        _pick(b, element, amount)
    assert a.state == b.state == "Fe0.40Ni0.20"
    assert a.cation_fractions() == b.cation_fractions() == {"Fe": 0.4, "Ni": 0.2}
    assert a.remaining_units == b.remaining_units == 4
    np.testing.assert_array_equal(a.current_state_features(), b.current_state_features())
    assert calls[-2:] == ["Fe0.40Ni0.20", "Fe0.40Ni0.20"]
    assert _decoded(a) == _decoded(b) == [("Co", "0.40"), ("Cu", "0.40")]


def test_zero_value_has_assignment_status_without_redundant_mutable_state():
    a, b = _env(zero=True), _env(zero=True)
    blank = a.current_state_features().copy()
    _pick(a, "Ni", 0)
    _pick(b, "Fe", 0)
    assert a.state == b.state == ""
    assert a.counter == b.counter == 1
    assert a.remaining_units == b.remaining_units == 10
    np.testing.assert_array_equal(_explicit(a), [0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(_explicit(b), [0, 0, 0, 0, 0, 0, 1, 0, 1, 2])
    assert not np.array_equal(a.current_state_features(), blank)
    assert not np.array_equal(a.current_state_features(), b.current_state_features())
    assert all(element != "Ni" for element, _ in _decoded(a))
    assert all(element != "Fe" for element, _ in _decoded(b))
    assert a.cation_fractions() == {"Ni": 0.0}
    assert "_selected" not in vars(a) and "_used_units" not in vars(a)


def test_positive_only_grid_derives_assignment_from_value_without_mask():
    env = _env()
    schema = env.state_feature_schema()
    assert not schema["assignment_mask"]
    assert not any(name.startswith("assigned:") for name in schema["explicit_feature_labels"])


def test_transition_stores_full_pre_action_state_and_does_not_alias_future_state():
    env = _env(zero=True)
    expected = env.current_state_features().copy()
    _pick(env, "Ni", 0.2)
    np.testing.assert_array_equal(env.path[0].state_material_features, expected)
    next_expected = env.current_state_features().copy()
    _pick(env, "Fe", 0.4)
    np.testing.assert_array_equal(env.path[1].state_material_features, next_expected)
    np.testing.assert_array_equal(env.path[0].state_material_features, expected)
    _pick(env, "Co", 0.4)
    assert env.counter == 3  # The fourth candidate species remains unassigned.
    assert env.remaining_units == 0
    assert env.terminal_formula == "Co0.40Fe0.40Ni0.20"
    assert "Cu" not in env.cation_fractions()
    assert _explicit(env)[-2:].tolist() == [0, 0]


def test_formula_is_derived_and_cannot_override_assignment_map():
    env = _env()
    _pick(env, "Ni", 0.2)
    with pytest.raises(AttributeError):
        env.state = "Ni0.60"
    exported = env.cation_fractions()
    exported["Ni"] = 0.6
    assert env.state == "Ni0.20"
    assert env.remaining_units == 8


def test_schema_does_not_depend_on_episode_progress_and_labels_are_stable():
    a = _env(zero=True)
    before = a.state_feature_schema()
    json.dumps(before)
    for element, amount in [("Ni", 0), ("Fe", 0.4), ("Co", 0.6)]:
        _pick(a, element, amount)
        assert a.state_feature_schema() == before
        assert a.current_state_features().size == before["feature_dim"]
    a.initialize()
    assert a.state_feature_schema() == before
    assert a.counter == 0 and a.remaining_units == 10

    b = _env(zero=True, species=("Cu", "Co", "Fe", "Ni"))
    np.testing.assert_array_equal(a.current_state_features(), b.current_state_features())
    assert before["labels"] == b.state_feature_schema()["labels"]
    assert before["action_species_order"] != b.state_feature_schema()["action_species_order"]


def test_descriptor_contract_rejects_nondimensional_vectors():
    with pytest.raises(ValueError, match="one-dimensional"):
        _env(descriptor=lambda _: np.zeros((2, 2)))


def test_descriptor_dimension_cannot_change_with_episode_progress():
    env = _env(descriptor=lambda formula: np.zeros(3 if formula else 2))
    initial_schema = env.state_feature_schema()
    _pick(env, "Ni", 0.2)
    assert env.state_feature_schema() == initial_schema
    with pytest.raises(ValueError, match="dimension must remain fixed"):
        env.current_state_features()


@pytest.mark.parametrize("left,right", [
    ("Ni0.2Fe0.4", "Fe0.4Ni0.2"),
    ("Fe2Ti3", "Ti3Fe2"),
    ("Fe3Fe4O2", "O2Fe7"),
    ("Ni0Fe0.4", "Fe0.4"),
])
def test_minimal_descriptor_backend_is_order_invariant_and_merges_repeats(monkeypatch, left, right):
    import rl_matdesign.featurization as features

    monkeypatch.setattr(features, "Composition", None)
    monkeypatch.setattr(features, "feature_calculators", None)
    monkeypatch.setattr(features, "_NUM_FEATURES", 8)
    monkeypatch.setattr(features, "_featurize_cache", {})
    a, b = features.featurize_formula(left), features.featurize_formula(right)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and np.any(a)
    assert len(features._featurize_cache) == 1


def test_minimal_descriptor_invalid_and_empty_inputs_are_zero(monkeypatch):
    import rl_matdesign.featurization as features

    monkeypatch.setattr(features, "Composition", None)
    monkeypatch.setattr(features, "feature_calculators", None)
    monkeypatch.setattr(features, "_NUM_FEATURES", 8)
    monkeypatch.setattr(features, "_featurize_cache", {})
    for formula in ("", "Ni0Fe0", "not a formula", "Xx0.5", "Ni-1"):
        np.testing.assert_array_equal(features.featurize_formula(formula), np.zeros(8))
