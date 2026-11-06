"""Typed labelled fields and group-scoped action identities."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rl_matdesign.action_encoding import ValueEncoding
from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.env_multigroup import CategoricalGroup, MultiGroupEnv, _field_values
from rl_matdesign.sequence import SequenceGroup
from rl_matdesign.utils.composition import composition_key


def local_action(group, label, value):
    label_index = group.species_set.index(label)
    selected = next(code for candidate, code in group._slot_codes[label_index]
                    if composition_key(candidate) == composition_key(value))
    return (tuple(np.eye(len(group.species_set))[label_index]),
            tuple(np.eye(len(group.fraction_set))[group.fraction_set.index(selected)]))


def choose_field(env, label, value):
    group = env._active()
    env.step(env._to_union(local_action(group, label, value), group))


def choose_monomer(env, position, token):
    group = env._active()
    label = group.encoding.labels.index_of(("position", position))
    value = group.encoding.values.index_of(token, namespace=(group.name, "monomer"))
    env.step(env._to_union(group.encoding.encode(label, value), group))


def stop_sequence(env):
    group = env._active()
    label = group.encoding.labels.index_of(("control", "stop"))
    value = group.encoding.values.index_of("stop", namespace=(group.name, "control"))
    env.step(env._to_union(group.encoding.encode(label, value), group))


def test_field_filling_order_is_free_and_state_is_canonical():
    choices = [{"name": "temperature", "values": [0, 300, 1200]},
               {"name": "vessel", "values": ["glass", "metal"]},
               {"name": "catalyst", "values": [True, 1, "1"]}]
    first, second = CategoricalGroup(choices), CategoricalGroup(choices)
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
    group = CategoricalGroup([{"name": "temperature", "values": [0, 300, 1200]}])
    np.testing.assert_array_equal(group.current_state_features(), [0, 0])
    table = ValueEncoding.from_specs(group.value_specs).features
    np.testing.assert_array_equal(table[:, 0], [0, 300, 1200])
    group.step(local_action(group, "temperature", 0))
    np.testing.assert_array_equal(group.current_state_features(), [0, 1])
    group.initialize()
    group.step(local_action(group, "temperature", 1200))
    np.testing.assert_array_equal(group.current_state_features(), [1200, 1])
    assert group.cation_fractions()["temperature"] == 1200


def test_boolean_numeric_and_string_field_values_have_distinct_actions_state_and_keys():
    choices = [{"name": "value", "values": [True, 1, "1", "01"]}]
    states, keys = [], []
    for value in [True, 1, "1", "01"]:
        group = CategoricalGroup(choices)
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
            CategoricalGroup([{"name": "amount", "values": values}])


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
    assert len(env.fraction_set) == len(set(env.fraction_set)) == 4
    assert len(env.value_specs) == len(env.fraction_set)
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
    with pytest.raises(TypeError, match="not an elemental composition"):
        env.assembled_composition()


def test_aliased_chemical_flags_keep_structured_output_for_the_builder():
    env = MultiGroupEnv(groups=[
        {"name": "S_site", "kind": "categorical", "choices": [
            {"name": "O_a", "element": "O", "values": [0, 1]},
            {"name": "O_b", "element": "O", "values": [0, 1]},
            {"element": "Cl", "values": [0.6, 1.0]},
        ]},
    ])
    choose_field(env, "O_b", 1)
    choose_field(env, "O_a", 1)
    choose_field(env, "Cl", 0.6)
    with pytest.raises(TypeError, match="builder.*physical composition"):
        env.assembled_composition()
    assert json.loads(env.terminal_formula) == {"S_site": {"O_a": 1, "O_b": 1, "Cl": 0.6}}
    # The flags are not interpreted as O2, and their labels are not elements.
    ordinary = MultiGroupEnv(groups=[{"name": "counts", "kind": "categorical", "choices": [
        {"element": "O", "values": [2]}, {"element": "Cl", "values": [1]},
    ]}])
    choose_field(ordinary, "O", 2)
    choose_field(ordinary, "Cl", 1)
    assert ordinary.assembled_composition() == {"O": 2.0, "Cl": 1.0}
    assert ordinary.terminal_formula == "O2Cl1"


def test_direct_field_step_cannot_bypass_an_empty_filter_result():
    class ExcludeAll:
        def filter_actions(self, *, actions, **_):
            return []
    group = CategoricalGroup([{"name": "x", "values": [1]}], phase_filter=ExcludeAll())
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
        {"name": "flags", "kind": "categorical", "choices": [{"name": "flag", "values": [True]}],
         "constraint_filter": RequiresPrior()},
    ])
    choose_field(env, "temperature", 300)
    choose_field(env, "flag", True)
    assert env.is_terminal and len(observed) >= 2
    assert all(prior == [{"temperature": 300}] for prior in observed)


def test_parent_stop_advances_to_next_group_without_horizon_padding_and_rewards_once():
    received = []
    env = MultiGroupEnv(groups=[
        {"name": "chain", "kind": "sequence", "tokens": ["A", "B", "C"],
         "parent": ["A", "B", "C"], "num_mutations": 3},
        {"name": "caps", "kind": "categorical", "choices": [{"name": "N", "values": ["H", "Ac"]}]},
    ], reward_fn=lambda candidate: received.append(deepcopy(candidate)) or 7)
    assert env.max_steps == 4
    choose_monomer(env, 3, "A")
    stop_sequence(env)
    assert env.counter == 2 and env._gi == 1 and not env.is_terminal
    choose_field(env, "N", "Ac")
    assert env.is_terminal and env.is_done and env._is_terminal()
    assert env.counter == 3 < env.n_components and env.allowed_actions() == []
    assert [step.reward for step in env.path] == [0, 0, 7]
    assert received == [{"chain": ["A", "B", "A"], "caps": {"N": "Ac"}}]
    assert env.terminal_comp_key() == (("caps", composition_key({"N": "Ac"})),
                                       ("chain", ("A", "B", "A")))
    assert env.path[-1].state_step_onehot.size == 4
    env.initialize()
    assert env.counter == 0 and not env.is_terminal
    assert isinstance(env._active(), SequenceGroup)


def test_two_sequence_groups_do_not_alias_their_positions_or_monomers():
    env = MultiGroupEnv(groups=[{"name": name, "kind": "sequence", "tokens": ["A", "C"], "length": 1}
                               for name in ["left", "right"]])
    assert len(env.species_set) == 2 and len(env.fraction_set) == 4
    encoding = ValueEncoding.from_specs(env.value_specs)
    assert encoding.feature_dim == 4
    choose_monomer(env, 1, "A")
    choose_monomer(env, 1, "C")
    assert env.terminal_cation_fractions() == {"left": ["A"], "right": ["C"]}


def test_integer_subgroup_keys_preserve_gcd_equivalence():
    groups = [{"name": "oxide", "kind": "integer", "species_set": ["Fe", "O"],
               "ratio_set": ["1", "2", "4"], "n_components": 2},
              {"name": "conditions", "kind": "categorical", "choices": [{"name": "temperature", "values": [300]}]}]
    keys = []
    for amounts in [(1, 2), (2, 4)]:
        env = MultiGroupEnv(groups=groups)
        for symbol, amount in zip(["Fe", "O"], amounts):
            inner = env._active()
            local = (tuple(np.eye(2)[inner.species_set.index(symbol)]),
                     tuple(np.eye(3)[inner.fraction_set.index(str(amount))]))
            env.step(env._to_union(local, inner))
        choose_field(env, "temperature", 300)
        keys.append(env.terminal_comp_key())
    assert keys[0] == keys[1]


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
         "fraction_set": ["1.00"], "n_components": 1},
        {"name": "second", "kind": "composition", "species_set": ["Fe"],
         "fraction_set": ["1.00"], "n_components": 1},
        {"name": "peptide", "kind": "sequence", "tokens": ["Ni"], "length": 1},
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
                                 "fraction_set": ["1.00"], "n_components": 1}], state_featurizer=featurizer)
    flat, _ = _precompute_elem_features(["Fe", "Ni"], featurizer)
    np.testing.assert_array_equal(env.action_features(), flat)


def test_predictor_intrinsic_constraints_include_parent_only_ids_and_are_recorded():
    class Predictor:
        def sequence_constraints(self, name, tokens):
            assert name == "chain" and tokens == ["A", "B", "cap-only"]
            return [{"tokens": ["cap-only"], "min": 1, "max": 1}]

        def predict(self, candidate):
            assert candidate == {"chain": ["cap-only", "B"]}
            return 5, 0

    cfg = {"groups": [{"name": "chain", "kind": "sequence", "tokens": ["A", "B"],
                       "parent": ["cap-only", "A"], "num_mutations": 1}]}
    authored = deepcopy(cfg)
    env, kind, _ = build_env(cfg, Predictor())
    assert cfg == authored and kind == "multi_group"
    assert env.design_spec["groups"][0]["count_constraints"] == [{"tokens": ["cap-only"], "min": 1, "max": 1}]
    choose_monomer(env, 2, "B")
    assert env.path[-1].reward == 5


@pytest.mark.parametrize("result", [None, {}, ["not a mapping"]])
def test_bad_predictor_constraints_fail_before_any_reward(result):
    class Predictor:
        def sequence_constraints(self, *_):
            return result
    with pytest.raises(ValueError, match="sequence_constraints"):
        build_env({"groups": [{"name": "chain", "kind": "sequence", "tokens": ["A"], "length": 1}]}, Predictor())


def test_state_and_design_schema_record_typed_domains_and_mutation_budget():
    base = {"name": "fields", "kind": "categorical", "choices": [{"name": "x", "values": [True, 1, "1"]}]}
    env = MultiGroupEnv(groups=[base])
    original = env.design_schema()
    assert original["version"] == 2
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


def test_generated_columns_validation_precedes_environment_work(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("generated_columns: {score: reward}\ngroups: []\n")
    with pytest.raises(ValueError, match="preserve formula"):
        load_config(path)
    with pytest.raises(ValueError, match="preserve formula"):
        build_env({"generated_columns": {"score": "reward"}}, object())
