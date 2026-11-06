"""Exact composition domains and free assignment-order integration."""
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from rl_matdesign.env import CompositionEnv, derive_total_units, expand_fraction_set
from rl_matdesign.encoding import decode_one_hot, encode_choice


def _env(**kwargs):
    return CompositionEnv(state_featurizer=lambda _formula: np.zeros(2), **kwargs)


def _actions(env):
    return {(decode_one_hot(element, env.species_set), decode_one_hot(amount, env.fraction_set))
            for element, amount in env.allowed_actions()}


def _action(env, label, code):
    return (tuple(encode_choice(label, env.species_set)), tuple(encode_choice(code, env.fraction_set)))


def _terminal_keys(env):
    keys = set()

    def visit():
        if env.counter == env.n_components:
            assert env.remaining_units == 0
            keys.add(env.terminal_comp_key())
            return
        actions = env.allowed_actions()
        assert actions, f"dead end after {env.cation_fractions()}"
        assignments, path = dict(env._units_map), list(env.path)
        for action in actions:
            env.step(action)
            visit()
            env._units_map, env.path = dict(assignments), list(path)

    visit()
    return keys


def test_regular_range_never_exceeds_authored_endpoint_in_environment():
    env = _env(species_set=["Ni", "Fe"], n_components=2,
               fraction_set={"Ni": {"min": 0, "max": 0.15, "step": 0.02},
                             "Fe": {"min": 0.86, "max": 1, "step": 0.02}})
    assert "0.16" not in env._species_fraction_codes["Ni"]
    assert ("Ni", "0.14") in _actions(env)
    assert all(float(code) <= 0.15 for label, code in _actions(env) if label == "Ni")


def test_species_menus_derive_one_lattice_without_cross_assigning_values():
    menus = {"Ni": [0, 0.25, 0.5, 0.75, 1], "Fe": [0, 0.125, 0.25, 0.375],
             "Co": [0, 0.2, 0.4, 0.6]}
    expanded = expand_fraction_set(menus)
    assert derive_total_units(expanded) == 40
    env = _env(species_set=list(menus), fraction_set=menus, n_components=3)
    assert env._total_units == 40
    assert set(env._species_fraction_codes["Ni"]) == {"0.000", "0.250", "0.500", "0.750", "1.000"}
    assert "0.125" in env.fraction_set
    assert ("Co", "0.125") not in _actions(env)
    with pytest.raises(ValueError, match="fraction menu"):
        env.step(_action(env, "Co", "0.125"))
    assert env.counter == 0


@pytest.mark.parametrize("menus", [{"Ni": [0, 1]}, {"Ni": [0, 1], "Fe": [0, 1], "Co": [0, 1]}])
def test_species_menu_keys_must_match_declared_species(menus):
    with pytest.raises(ValueError, match="species keys"):
        _env(species_set=["Ni", "Fe"], fraction_set=menus, n_components=2)


@pytest.mark.parametrize("bounds, expected", [
    ({"B": [0.05, 0.25]}, {"0.06", "0.08", "0.10", "0.12", "0.14", "0.16", "0.18", "0.20", "0.22", "0.24"}),
    ({"B": [0, 0.15]}, {"0.00", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.14"}),
])
def test_physical_bounds_never_round_outwards(bounds, expected):
    env = _env(species_set=["B", "Si"], n_components=2,
               fraction_set={"min": 0, "max": 1, "step": 0.02}, element_bounds=bounds)
    assert {code for label, code in _actions(env) if label == "B"} == expected


def test_tiny_positive_lower_bound_remains_mandatory():
    env = _env(species_set=["Zr", "Ni", "Fe"], n_components=2,
               fraction_set={"min": 0, "max": 1, "step": 0.05},
               element_bounds={"Zr": [0.01, 0.10]})
    assert env._element_unit_bounds_by_sym["Zr"] == (1, 2)
    keys = _terminal_keys(env)
    assert keys and all(dict(key)["Zr"] > 0 for key in keys)


def test_sparse_per_species_completion_does_not_admit_dead_end_prefix():
    env = _env(species_set=["Ni", "Fe", "Co"], n_components=3,
               fraction_set=[0, 0.2, 0.5, 0.8, 1],
               element_bounds={"Ni": [0, 0.2], "Fe": [0.2, 0.8], "Co": [0.2, 0.8]})
    assert ("Ni", "0.20") not in _actions(env)
    assert ("Ni", "0.00") in _actions(env)
    with pytest.raises(ValueError, match="impossible"):
        env.step(_action(env, "Ni", "0.20"))
    assert env.counter == 0
    assert _terminal_keys(env) == {
        (("Co", 8), ("Fe", 2)), (("Co", 5), ("Fe", 5)), (("Co", 2), ("Fe", 8)),
    }


def test_sparse_initial_domain_without_a_completion_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        _env(species_set=["Ni", "Fe"], n_components=2,
             fraction_set={"Ni": [0.2, 0.5], "Fe": [0.2, 0.8]},
             element_bounds={"Ni": [0.5, 0.5], "Fe": [0.2, 0.8]})


def test_optional_species_with_no_legal_menu_can_be_omitted():
    env = _env(species_set=["Ni", "Fe"], n_components=1,
               fraction_set={"Ni": [0.8], "Fe": [1]}, element_bounds={"Ni": [0, 0.5]})
    assert _actions(env) == {("Fe", "1.00")}
    env.step(env.allowed_actions()[0])
    assert env.terminal_cation_fractions() == {"Fe": 1.0}


def test_explicit_zero_assignment_consumes_a_slot_and_has_state_identity():
    env = _env(species_set=["Ni", "Fe", "Co"], n_components=2, fraction_set=[0, 0.5, 1])
    before = env.current_state_features().copy()
    env.step(_action(env, "Ni", "0.00"))
    assert env.counter == 1 and env.remaining_units == env._total_units
    assert env.cation_fractions() == {"Ni": 0}
    assert not np.array_equal(before, env.current_state_features())
    assert all(label != "Ni" for label, _ in _actions(env))
    env.step(_action(env, "Fe", "1.00"))
    assert env.terminal_comp_key() == (("Fe", 2),)
    assert env.terminal_cation_fractions() == {"Fe": 1.0}
    assert env.cation_fractions()["Ni"] == 0
    assert len(env.path) == 2


def test_fixed_order_alias_keeps_all_slots_but_frees_assignment_order():
    with pytest.warns(DeprecationWarning, match="assignment order is now free"):
        env = _env(species_set=["Ni", "Fe", "Co"], n_components=1,
                   fraction_set=[0, 0.5, 1], episode_style="fixed_order_amount")
    assert env.n_components == 3 and env.episode_style == "element_then_amount"
    assert {label for label, _ in _actions(env)} == {"Ni", "Fe", "Co"}
    env.step(_action(env, "Co", "0.50"))
    env.step(_action(env, "Ni", "0.00"))
    env.step(_action(env, "Fe", "0.50"))
    assert env.counter == 3 and env.terminal_cation_fractions() == {"Co": 0.5, "Fe": 0.5}
    assert env.cation_fractions()["Ni"] == 0


def test_positive_only_menus_do_not_make_all_species_mandatory():
    env = _env(species_set=["Ni", "Fe", "Co"], n_components=2,
               fraction_set={"Ni": [0.5], "Fe": [0.5], "Co": [0.5]})
    assert len(_terminal_keys(env)) == 3


def test_exact_slot_domain_matches_independent_small_enumeration():
    menus = {"Ni": [0, 0.2, 0.5], "Fe": [0, 0.2, 0.5, 0.8],
             "Co": [0, 0.5, 1], "Zr": [0.2, 0.5]}
    bounds = {"Zr": (0.2, 0.5), "Fe": (0, 0.5)}
    env = _env(species_set=list(menus), fraction_set=menus, element_bounds=bounds, n_components=3)
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
            expected.add(tuple(sorted((label, int(value * env._total_units))
                                      for label, value in exact.items() if value)))
    assert _terminal_keys(env) == expected


def test_schema_records_species_menu_bounds_and_mandatory_semantics():
    env = _env(species_set=["Ni", "Zr"], n_components=2, fraction_set=[0, 0.5, 1],
               element_bounds={"Zr": [0.5, 1]})
    schema = env.state_feature_schema()
    assert schema["schema_version"] == 3
    assert schema["species_allowed_units"]["Zr"] == [1, 2]
    assert schema["element_unit_bounds"]["Zr"] == [1, 2]
    assert schema["mandatory_species"] == ["Zr"]


@pytest.mark.parametrize("components", [0, 4, True, 1.5])
def test_invalid_slot_budgets_are_rejected(components):
    with pytest.raises(ValueError, match="n_components"):
        _env(species_set=["Ni", "Fe", "Co"], n_components=components, fraction_set=[0, 0.5, 1])


def test_unknown_bounds_are_not_ignored():
    with pytest.raises(ValueError, match="not in species_set"):
        _env(species_set=["Ni", "Fe"], n_components=2, fraction_set=[0, 0.5, 1],
             element_bounds={"Nii": [0.5, 1]})


def test_step_checks_constraints_with_the_same_prior_group_context():
    class ContextFilter:
        def filter_actions(self, *, actions, units_map, species_set, prior_groups=None, **_):
            if units_map:
                return actions
            selected = "Fe" if prior_groups == [{"host": "Ni"}] else "Ni"
            return [action for action in actions if decode_one_hot(action[0], species_set) == selected]

    env = _env(species_set=["Ni", "Fe"], n_components=2, fraction_set=[0.5],
               phase_filter=ContextFilter())
    action = _action(env, "Fe", "0.50")
    with pytest.raises(ValueError, match="current composition constraints"):
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

    calls = []
    env = _env(species_set=["Ni", "Fe"], n_components=2, fraction_set=[0.5],
               phase_filter=RejectAll(), reward_fn=lambda payload: calls.append(payload) or 0)
    assert env.allowed_actions() == []
    with pytest.raises(InfeasibleEpisode):
        env.sample_random_action()
    with pytest.raises(ValueError, match="current composition constraints"):
        env.step(_action(env, "Fe", "0.50"))
    assert not calls and not env.path and not env.cation_fractions()


def test_sampling_after_termination_is_misuse_not_retriable_infeasibility():
    env = _env(species_set=["Ni"], n_components=1, fraction_set=[1])
    env.step(env.allowed_actions()[0])
    with pytest.raises(RuntimeError, match="terminated"):
        env.sample_random_action()
