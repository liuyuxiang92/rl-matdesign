"""Exact fraction encoding and isolation of the constraint plug-in boundary."""
from copy import deepcopy

import numpy as np
import pytest

from rl_matdesign.encoding import encode_choice
from fractions import Fraction
from rl_matdesign.amounts import ChemicalAmountsGroup, _code


def _env(**kwargs):
    return ChemicalAmountsGroup(species_set=["Ni", "Fe"], n_steps=2, total=1, unique=True,
                          state_featurizer=lambda _: np.zeros(1), **kwargs)


def _action(env, species, code):
    return (tuple(encode_choice(species, env.species_set)),
            tuple(encode_choice(_code(Fraction(code)), env.action_values)))


@pytest.mark.parametrize("places", [18, 30])
@pytest.mark.parametrize("per_species", [False, True])
def test_padded_exact_decimals_keep_actions_and_display_exact(places, per_species):
    tenth, remainder = "0.1" + "0" * (places - 1), "0.9" + "0" * (places - 1)
    menu = {"Ni": [tenth], "Fe": [remainder]} if per_species else [tenth, remainder]
    env = _env(amount_set=menu)
    assert env.amount_scale == 10
    first = _action(env, "Ni", tenth)
    assert first in env.allowed_actions()
    env.step(first)
    assert env.state == "Ni0.1"
    env.step(_action(env, "Fe", remainder))
    assert env.terminal_formula == "Fe0.9Ni0.1"
    assert env.terminal_comp_key() == (("Fe", .9), ("Ni", .1))
    assert env.terminal_cation_fractions() == {"Fe": .9, "Ni": .1}


def test_exact_amount_code_keeps_non_decimal_step_precision():
    assert _code(Fraction(1, 8)) == "0.125"
    assert _code(Fraction(1, 10000000)) == "0.0000001"


def test_filter_cannot_mutate_environment_or_prior_group_context():
    class MutatingReject:
        def filter_actions(self, *, actions, units_map, allowed_units,
                           possible_sums_by_k, species_set, action_values,
                           prior_groups, **_):
            units_map["Ni"] = 1
            allowed_units.clear()
            possible_sums_by_k[0].clear()
            possible_sums_by_k.clear()
            species_set.reverse()
            action_values.append("1.00")
            prior_groups[0]["nested"]["host"] = "Co"
            prior_groups.clear()
            actions.clear()
            return actions

    env = _env(amount_set=[.5], filters=[MutatingReject()])
    prior = [{"nested": {"host": "Ni"}}]
    schema, sums = deepcopy(env.state_feature_schema()), deepcopy(env._possible_sums_by_k)
    with pytest.raises(ValueError, match="infeasible or excluded"):
        env.step(_action(env, "Ni", "0.50"), prior_groups=prior)
    assert env.counter == 0 and env.path == [] and env.cation_fractions() == {}
    assert env.state_feature_schema() == schema
    assert env._possible_sums_by_k == sums
    assert prior == [{"nested": {"host": "Ni"}}]


def test_filter_exception_cannot_leave_a_partial_assignment():
    class MutatingFailure:
        def filter_actions(self, *, units_map, **_):
            units_map["Ni"] = 1
            raise RuntimeError("filter failed")

    env = _env(amount_set=[.5], filters=[MutatingFailure()])
    with pytest.raises(RuntimeError, match="filter failed"):
        env.step(_action(env, "Ni", "0.50"))
    assert env.counter == 0 and env.path == [] and env.cation_fractions() == {}


@pytest.mark.parametrize("kind, message", [
    ("none", "list or tuple"),
    ("iterator", "list or tuple"),
    ("mapping", "list or tuple"),
    ("unhashable", "invalid action"),
    ("duplicate", "duplicate"),
    ("non_onehot", "subset"),
    ("mutating_insert", "subset"),
])
def test_invalid_filter_outputs_fail_before_assignment(kind, message):
    class InvalidFilter:
        def filter_actions(self, *, actions, **_):
            if kind == "none":
                return None
            if kind == "iterator":
                return iter(actions)
            if kind == "mapping":
                return {"actions": actions}
            if kind == "unhashable":
                return [list(actions[0])]
            if kind == "duplicate":
                return [actions[0], actions[0]]
            malformed = ((.5, .5), (1.,))
            if kind == "mutating_insert":
                actions.append(malformed)
                return actions
            return [malformed]

    env = _env(amount_set=[.5], filters=[InvalidFilter()])
    with pytest.raises(ValueError, match=message):
        env.step(_action(env, "Ni", "0.50"))
    assert env.counter == 0 and env.path == [] and env.cation_fractions() == {}


@pytest.mark.parametrize("as_tuple", [False, True])
def test_filter_can_prune_in_place_and_return_list_or_tuple(as_tuple):
    class Prune:
        def filter_actions(self, *, actions, **_):
            if len(actions) > 1:
                actions.pop()
            return tuple(actions) if as_tuple else actions

    env = _env(amount_set=[.5], filters=[Prune()])
    first = _action(env, "Ni", "0.50")
    assert env.allowed_actions() == [first]
    env.step(first)
    env.step(_action(env, "Fe", "0.50"))
    assert env.terminal_cation_fractions() == {"Fe": .5, "Ni": .5}
