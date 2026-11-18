"""Exact composition domains and free assignment-order integration."""
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup, _code
from rl_matdesign.encoding import decode_one_hot, encode_choice


def _env(**kwargs):
    return ChemicalAmountsGroup(total=1, unique=True, state_featurizer=lambda _formula: np.zeros(2), **kwargs)


def _actions(env):
    return {(decode_one_hot(element, env.species_set), decode_one_hot(amount, env.action_values))
            for element, amount in env.allowed_actions()}


def _action(env, label, code):
    return (tuple(encode_choice(label, env.species_set)), tuple(encode_choice(_code(Fraction(code)), env.action_values)))


def _terminal_keys(env):
    keys = set()
    def visit(prefix):
        env.initialize()
        for action in prefix:
            env.step(action)
        if env.is_terminal:
            assert env.remaining_units == 0
            keys.add(env.terminal_comp_key())
            return
        actions = env.allowed_actions()
        assert actions, f"dead end after {env.cation_fractions()}"
        for action in actions:
            visit(prefix + [action])
    visit([])
    return keys


def test_regular_range_never_exceeds_authored_endpoint_in_environment():
    env = _env(species_set=["Ni", "Fe"], n_steps=2,
               amount_set={"Ni": {"min": 0, "max": 0.15, "step": 0.02},
                             "Fe": {"min": 0.86, "max": 1, "step": 0.02}})
    assert "0.16" not in env._species_amount_codes["Ni"]
    assert ("Ni", "0.14") in _actions(env)
    assert all(float(code) <= 0.15 for label, code in _actions(env) if label == "Ni")


def test_species_menus_derive_one_lattice_without_cross_assigning_values():
    menus = {"Ni": [0, 0.25, 0.5, 0.75, 1], "Fe": [0, 0.125, 0.25, 0.375],
             "Co": [0, 0.2, 0.4, 0.6]}
    env = _env(species_set=list(menus), amount_set=menus, n_steps=3)
    assert env.amount_scale == 40
    assert set(env._species_amount_codes["Ni"]) == {"0", "0.25", "0.5", "0.75", "1"}
    assert "0.125" in env.action_values
    assert ("Co", "0.125") not in _actions(env)
    with pytest.raises(ValueError, match="infeasible"):
        env.step(_action(env, "Co", "0.125"))
    assert env.counter == 0


@pytest.mark.parametrize("menus", [{"Ni": [0, 1]}, {"Ni": [0, 1], "Fe": [0, 1], "Co": [0, 1]}])
def test_species_menu_keys_must_match_declared_species(menus):
    with pytest.raises(ValueError, match="keys must match"):
        _env(species_set=["Ni", "Fe"], amount_set=menus, n_steps=2)


@pytest.mark.parametrize("bounds, expected", [
    ({"B": [0.05, 0.25]}, {"0.06", "0.08", "0.1", "0.12", "0.14", "0.16", "0.18", "0.2", "0.22", "0.24"}),
    ({"B": [0, 0.15]}, {"0", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.14"}),
])
def test_physical_bounds_never_round_outwards(bounds, expected):
    env = _env(species_set=["B", "Si"], n_steps=2,
               amount_set={"min": 0, "max": 1, "step": 0.02}, element_bounds=bounds)
    assert {code for label, code in _actions(env) if label == "B"} == expected


def test_tiny_positive_lower_bound_remains_mandatory():
    env = _env(species_set=["Zr", "Ni", "Fe"], n_steps=2,
               amount_set={"min": 0, "max": 1, "step": 0.05},
               element_bounds={"Zr": [0.01, 0.10]})
    assert env.element_bounds["Zr"] == [.01, .1]
    keys = _terminal_keys(env)
    assert keys and all(dict(key)["Zr"] > 0 for key in keys)


def test_sparse_per_species_completion_does_not_admit_dead_end_prefix():
    env = _env(species_set=["Ni", "Fe", "Co"], n_steps=3,
               amount_set=[0, 0.2, 0.5, 0.8, 1],
               element_bounds={"Ni": [0, 0.2], "Fe": [0.2, 0.8], "Co": [0.2, 0.8]})
    assert ("Ni", "0.2") not in _actions(env)
    assert ("Ni", "0") in _actions(env)
    with pytest.raises(ValueError, match="infeasible"):
        env.step(_action(env, "Ni", "0.2"))
    assert env.counter == 0
    assert _terminal_keys(env) == {
        (("Co", .8), ("Fe", .2)), (("Co", .5), ("Fe", .5)), (("Co", .2), ("Fe", .8)),
    }


def test_sparse_initial_domain_without_a_completion_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        _env(species_set=["Ni", "Fe"], n_steps=2,
             amount_set={"Ni": [0.2, 0.5], "Fe": [0.2, 0.8]},
             element_bounds={"Ni": [0.5, 0.5], "Fe": [0.2, 0.8]})


def test_optional_species_with_no_legal_menu_can_be_omitted():
    env = _env(species_set=["Ni", "Fe"], n_steps=1,
               amount_set={"Ni": [0.8], "Fe": [1]}, element_bounds={"Ni": [0, 0.5]})
    assert _actions(env) == {("Fe", "1")}
    env.step(env.allowed_actions()[0])
    assert env.terminal_cation_fractions() == {"Fe": 1.0}


def test_explicit_zero_assignment_consumes_a_slot_and_has_state_identity():
    env = _env(species_set=["Ni", "Fe", "Co"], n_steps=2, amount_set=[0, 0.5, 1])
    before = env.current_state_features().copy()
    env.step(_action(env, "Ni", "0"))
    assert env.counter == 1 and env.remaining_units == env.amount_scale
    assert env.cation_fractions() == {"Ni": 0}
    assert not np.array_equal(before, env.current_state_features())
    assert all(label != "Ni" for label, _ in _actions(env))
    env.step(_action(env, "Fe", "1"))
    assert env.terminal_comp_key() == (("Fe", 1.0),)
    assert env.terminal_cation_fractions() == {"Fe": 1.0}
    assert env.cation_fractions()["Ni"] == 0
    assert len(env.path) == 2


def test_explicit_unique_stage_keeps_all_targets_and_free_assignment_order():
    env = _env(species_set=["Ni", "Fe", "Co"], n_steps=3, amount_set=[0, .5, 1])
    assert {label for label, _ in _actions(env)} == {"Ni", "Fe", "Co"}
    for label, value in [("Co", ".5"), ("Ni", "0"), ("Fe", ".5")]:
        env.step(_action(env, label, value))
    assert env.counter == 3
    assert env.terminal_cation_fractions() == {"Co": .5, "Fe": .5}
    assert env.cation_fractions()["Ni"] == 0


def test_positive_only_menus_do_not_make_all_species_mandatory():
    env = _env(species_set=["Ni", "Fe", "Co"], n_steps=2,
               amount_set={"Ni": [0.5], "Fe": [0.5], "Co": [0.5]})
    assert len(_terminal_keys(env)) == 3


def test_exact_slot_domain_matches_independent_small_enumeration():
    menus = {"Ni": [0, 0.2, 0.5], "Fe": [0, 0.2, 0.5, 0.8],
             "Co": [0, 0.5, 1], "Zr": [0.2, 0.5]}
    bounds = {"Zr": (0.2, 0.5), "Fe": (0, 0.5)}
    env = _env(species_set=list(menus), amount_set=menus, element_bounds=bounds, n_steps=3)
    expected = set()
    for labels in combinations(menus, 3):
        if "Zr" not in labels:
            continue
        for values in product(*(menus[label] for label in labels)):
            exact = {label: Fraction(str(value)) for label, value in zip(labels, values)}
            if sum(exact.values()) != 1:
                continue
            if any(not Fraction(str(bounds.get(label, (0, 1))[0])) <= value <= Fraction(str(bounds.get(label, (0, 1))[1]))
                   for label, value in exact.items()):
                continue
            expected.add(tuple(sorted((label, float(value))
                                      for label, value in exact.items() if value)))
    assert _terminal_keys(env) == expected


def test_schema_records_species_menus_bounds_and_exact_budget():
    env = _env(species_set=["Ni", "Zr"], n_steps=2, amount_set=[0, .5, 1],
               element_bounds={"Zr": [.5, 1]})
    schema = env.design_schema()
    assert schema["amount_menus"]["Zr"] == ["0", "0.5", "1"]
    assert schema["element_bounds"]["Zr"] == [.5, 1]
    assert schema["n_steps"] == 2 and schema["unique"]


@pytest.mark.parametrize("components", [0, 4, True, 1.5])
def test_invalid_slot_budgets_are_rejected(components):
    with pytest.raises(ValueError, match="n_steps|unique group"):
        _env(species_set=["Ni", "Fe", "Co"], n_steps=components, amount_set=[0, 0.5, 1])


def test_unknown_bounds_are_not_ignored():
    with pytest.raises(ValueError, match="species_set labels"):
        _env(species_set=["Ni", "Fe"], n_steps=2, amount_set=[0, 0.5, 1],
             element_bounds={"Nii": [0.5, 1]})


def test_step_checks_constraints_with_the_same_prior_group_context():
    class ContextFilter:
        def filter_actions(self, *, actions, units_map, species_set, prior_groups=None, **_):
            if units_map:
                return actions
            selected = "Fe" if prior_groups == [{"host": "Ni"}] else "Ni"
            return [action for action in actions if decode_one_hot(action[0], species_set) == selected]

    env = _env(species_set=["Ni", "Fe"], n_steps=2, amount_set=[0.5],
               filters=[ContextFilter()])
    action = _action(env, "Fe", "0.5")
    with pytest.raises(ValueError, match="infeasible or excluded"):
        env.step(action)
    assert env.counter == 0
    env.step(action, prior_groups=[{"host": "Ni"}])
    assert env.cation_fractions() == {"Fe": 0.5}
    assert env.path[0].allowed_actions == [action]


def test_empty_constraints_reject_prefix_without_oracle_work():
    from rl_matdesign.utils.rollout_errors import InfeasibleEpisode

    class RejectAll:
        def filter_actions(self, *, actions, **_):
            return []

    env = _env(species_set=["Ni", "Fe"], n_steps=2, amount_set=[0.5],
               filters=[RejectAll()])
    assert env.allowed_actions() == []
    with pytest.raises(InfeasibleEpisode):
        env.sample_random_action()
    with pytest.raises(ValueError, match="infeasible or excluded"):
        env.step(_action(env, "Fe", "0.5"))
    assert not env.path and not env.cation_fractions()


def test_sampling_after_termination_is_misuse_not_retriable_infeasibility():
    env = _env(species_set=["Ni"], n_steps=1, amount_set=[1])
    env.step(env.allowed_actions()[0])
    with pytest.raises(RuntimeError, match="completed"):
        env.sample_random_action()
