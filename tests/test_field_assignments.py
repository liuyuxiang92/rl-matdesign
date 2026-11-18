"""Typed labelled fields and group-scoped action identities."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rl_matdesign.action_encoding import ValueEncoding
from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.categorical import CategoricalGroup, _field_values
from rl_matdesign.utils.composition import composition_key


def local_action(group, label, value):
    label_index = group.species_set.index(label)
    selected = next(code for candidate, code in group._slot_codes[label_index]
                    if composition_key(candidate) == composition_key(value))
    return (tuple(np.eye(len(group.species_set))[label_index]),
            tuple(np.eye(len(group.action_values))[group.action_values.index(selected)]))


def choose_field(env, label, value):
    group = env._active()
    env.step(env._to_union(local_action(group, label, value), group))






def test_field_filling_order_is_free_and_state_is_canonical():
    choices = [{"name": "temperature", "values": [0, 300, 1200]},
               {"name": "vessel", "values": ["glass", "metal"]},
               {"name": "catalyst", "values": [True, 1, "1"]}]
    first, second = CategoricalGroup({"choices": choices}), CategoricalGroup({"choices": choices})
    assert len(first.allowed_actions()) == 8
    for group, fills in [(first, [("temperature", 300), ("catalyst", True)]),
                         (second, [("catalyst", True), ("temperature", 300)])]:
        for label, value in fills:
            group.step(local_action(group, label, value))
    np.testing.assert_array_equal(first.current_state_features(), second.current_state_features())
    assert first.allowed_actions() == second.allowed_actions()
    for group in (first, second):
        group.step(local_action(group, "vessel", "glass"))
    assert first.terminal_comp_key() == second.terminal_comp_key()
    assert first.cation_fractions() == {"temperature": 300, "vessel": "glass", "catalyst": True}


def test_numeric_fields_keep_real_magnitude_and_filled_zero_differs_from_empty():
    group = CategoricalGroup({"choices": [{"name": "temperature", "values": [0, 300, 1200]}]})
    np.testing.assert_array_equal(group.current_state_features()[:2], [0, 0])
    table = ValueEncoding.from_specs(group.value_specs).features
    np.testing.assert_array_equal(table[:, 0], [0, 300, 1200])
    group.step(local_action(group, "temperature", 0))
    np.testing.assert_array_equal(group.current_state_features()[:2], [0, 1])
    group.initialize()
    group.step(local_action(group, "temperature", 1200))
    np.testing.assert_array_equal(group.current_state_features()[:2], [1200, 1])
    assert group.cation_fractions()["temperature"] == 1200


def test_boolean_numeric_and_string_field_values_have_distinct_actions_state_and_keys():
    choices = [{"name": "value", "values": [True, 1, "1", "01"]}]
    states, keys = [], []
    for value in [True, 1, "1", "01"]:
        group = CategoricalGroup({"choices": choices})
        group.step(local_action(group, "value", value))
        states.append(tuple(group.current_state_features()))
        keys.append(group.terminal_comp_key())
        assert type(group.cation_fractions()["value"]) is type(value)
    assert len(set(states)) == len(set(keys)) == 4


def test_equal_numeric_spellings_are_one_physical_identity():
    assert composition_key({"amount": 1}) == composition_key({"amount": 1.0})
    assert composition_key({"amount": 2**53 + 1}) != composition_key({"amount": float(2**53 + 1)})
    for values in [[1, 1.0], [True, 1, 1.0, "1"]]:
        with pytest.raises(ValueError, match="equal numeric"):
            CategoricalGroup({"choices": [{"name": "amount", "values": values}]})


def test_mapping_order_ignores_assignment_order_but_sequence_positions_remain_significant():
    assert composition_key({"Ni": .2, "Fe": .4}) == composition_key({"Fe": .4, "Ni": .2})
    assert composition_key({"chain": ["01", "1"]}) != composition_key({"chain": ["1", "01"]})
    assert composition_key({"chain": ["01"]}) != composition_key({"chain": [1]})
    assert composition_key({"chain": ["01"]}) != composition_key({"chain": ["1"]})
    assert composition_key({"a": True}) != composition_key({"a": 1})
    assert composition_key({"a": "0.123456789"}, decimals=6) == composition_key({"a": "0.123456789"})


def test_two_groups_with_same_label_and_value_have_independent_action_axes():
    spec = [{"name": name, "kind": "categorical", "choices": [{"name": "temperature", "values": [300, 600]}]}
            for name in ["synthesis", "anneal"]]
    env = MultiGroupEnv(groups=spec)
    assert len(env.species_set) == len(set(env.species_set)) == 2
    assert len(env.action_values) == len(set(env.action_values)) == 4
    assert len(env.value_specs) == len(env.action_values)
    before = env.allowed_actions()[0]
    choose_field(env, "temperature", 300)
    assert all(action[0] != before[0] and action[1] != before[1] for action in env.allowed_actions())
    state = env.current_state_features()
    with pytest.raises(ValueError, match="another group"):
        env.step(before)
    np.testing.assert_array_equal(env.current_state_features(), state)
    choose_field(env, "temperature", 600)
    assert env.terminal_cation_fractions() == {"synthesis": {"temperature": 300}, "anneal": {"temperature": 600}}
    assert json.loads(env.terminal_formula) == env.terminal_cation_fractions()
    with pytest.raises(TypeError, match="builder to determine physical composition"):
        env.assembled_composition()




def test_direct_field_step_cannot_bypass_an_empty_filter_result():
    class ExcludeAll:
        def filter_actions(self, *, actions, **_):
            return []
    group = CategoricalGroup({"choices": [{"name": "x", "values": [1]}]}, filters=[ExcludeAll()])
    before = group.current_state_features()
    assert group.allowed_actions() == []
    with pytest.raises(ValueError, match="excluded by the field constraints"):
        group.step(local_action(group, "x", 1))
    np.testing.assert_array_equal(group.current_state_features(), before)
    assert group.counter == 0 and group.cation_fractions() == {}


def test_wrapper_revalidation_supplies_completed_groups_to_field_constraints():
    observed = []
    class RequiresPrior:
        def filter_actions(self, *, actions, prior_groups=None, **_):
            observed.append(deepcopy(prior_groups))
            return actions if prior_groups == [{"temperature": 300}] else []
    env = MultiGroupEnv(groups=[
        {"name": "conditions", "kind": "categorical", "choices": [{"name": "temperature", "values": [300]}]},
        {"name": "flags", "kind": "categorical", "choices": [{"name": "flag", "values": [True]}]},
    ])
    env._inners[1].filters = [RequiresPrior()]
    choose_field(env, "temperature", 300)
    choose_field(env, "flag", True)
    assert env.is_terminal and len(observed) >= 2
    assert all(prior == [{"temperature": 300}] for prior in observed)








def test_mixed_label_features_preserve_chemistry_and_never_parse_positions_or_fields():
    calls = []
    def featurizer(label):
        calls.append(label)
        if label == "":
            return np.zeros(2)
        assert label in ("Fe", "Ni"), f"Nonchemical label reached chemical featurizer: {label}"
        return {"Fe": [2., 5.], "Ni": [4., 9.]}[label]
    env = MultiGroupEnv(groups=[
        {"name": "metal", "kind": "composition", "species_set": ["Fe", "Ni"],
         "amount_set": [1], "n_steps": 1, "total": 1},
        {"name": "second", "kind": "composition", "species_set": ["Fe"],
         "amount_set": [1], "n_steps": 1, "total": 1},
        {"name": "peptide", "kind": "categorical", "labels": [1], "values": ["Ni"], "n_steps": 1},
        {"name": "conditions", "kind": "categorical", "choices": [{"name": "Fe", "values": [300]}]},
    ], state_featurizer=featurizer)
    calls.clear()
    table = env.action_features()
    assert calls == ["Fe", "Ni"]
    np.testing.assert_array_equal(table[:, :5], np.eye(5))
    np.testing.assert_array_equal(table[:, 5:], [[-1, -1], [1, 1], [-1, -1], [0, 0], [0, 0]])
    assert env.action_feature_schema()["chemical_labels"] == ["Fe", "Ni", "Fe", None, None]
    assert env.action_feature_schema()["shape"] == [5, 7]
    assert calls == ["Fe", "Ni"]  # cached: schema checks do not re-featurize
    with pytest.raises(ValueError):
        table[0, 0] = 9


def test_single_group_action_chemistry_matches_flat_feature_table():
    from rl_matdesign.training import _precompute_elem_features
    featurizer = lambda label: np.asarray({"": [0., 0.], "Fe": [2., 5.], "Ni": [4., 9.]}[label])
    env = MultiGroupEnv(groups=[{"name": "metal", "species_set": ["Fe", "Ni"],
                                 "amount_set": [1], "n_steps": 1, "total": 1}], state_featurizer=featurizer)
    flat, _ = _precompute_elem_features(["Fe", "Ni"], featurizer)
    np.testing.assert_array_equal(env.action_features(), flat)






def test_state_and_design_schema_record_typed_domains_and_mutation_budget():
    base = {"name": "fields", "kind": "categorical", "choices": [{"name": "x", "values": [True, 1, "1"]}]}
    env = MultiGroupEnv(groups=[base])
    original = env.design_schema()
    assert original["version"] == 3
    assert env.action_features().shape == (1, 1)
    alternative = deepcopy(base)
    alternative["choices"][0]["values"] = [1, True, "1"]
    assert json.dumps(original, sort_keys=True) != json.dumps(MultiGroupEnv(groups=[alternative]).design_schema(), sort_keys=True)
    original["groups"].clear()
    assert env.design_schema()["groups"]


def test_typed_field_grid_and_rejections():
    assert _field_values({"min": 0, "max": 2, "step": 1}) == [0, 1, 2]
    assert all(type(value) is int for value in _field_values({"min": 0, "max": 2, "step": 1}))
    assert _field_values({"min": .01, "max": .03, "step": .01}) == [.01, .02, .03]
    for values in [[], [float("nan")], [object()], {"min": 0, "max": 1, "step": .3},
                   {"min": False, "max": 1, "step": 1}]:
        with pytest.raises(ValueError):
            _field_values(values)


def test_removed_output_configuration_is_rejected_at_load(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("generated_columns: {score: reward}\ngroups: []\n")
    with pytest.raises(ValueError, match="generated_columns"):
        load_config(path)


def test_element_spelling_in_a_field_label_never_makes_it_a_chemical_count():
    env = MultiGroupEnv(groups=[{"name": "settings", "kind": "categorical", "choices": [
        {"name": "O", "values": [2]}, {"name": "Cl", "values": [1]}]}])
    choose_field(env, "O", 2)
    choose_field(env, "Cl", 1)
    assert env.terminal_cation_fractions() == {"settings": {"O": 2, "Cl": 1}}
    assert env.action_feature_schema()["chemical_labels"] == [None, None]
    with pytest.raises(TypeError, match="builder"):
        env.assembled_composition()
