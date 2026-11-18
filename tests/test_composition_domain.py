"""Exact amount menus and completion masks, independently checked by enumeration."""
from fractions import Fraction
from itertools import product
import random

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup, _code, _menu
from rl_matdesign.encoding import decode_one_hot


def make(**kwargs):
    return ChemicalAmountsGroup(state_featurizer=lambda _: np.zeros(1), **kwargs)


def codes(spec):
    return [_code(value) for value in _menu(spec)]


def test_range_does_not_round_non_grid_maximum_outward():
    assert codes({"min": 0, "max": .15, "step": .02}) == ["0", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.14"]
    assert max(_menu({"min": 0, "max": .15, "step": .02})) < Fraction(".15")


@pytest.mark.parametrize("spec, expected", [
    ({"min": ".005", "max": ".05", "step": ".02"}, ["0.005", "0.025", "0.045"]),
    ({"min": "1e-3", "max": "5e-3", "step": "2e-3"}, ["0.001", "0.003", "0.005"]),
    ({"min": .07, "max": .10, "step": .01}, ["0.07", "0.08", "0.09", "0.1"]),
    ({"min": .15, "max": .15, "step": .02}, ["0.15"]),
])
def test_range_preserves_exact_origin_step_and_endpoint(spec, expected):
    assert codes(spec) == expected


@pytest.mark.parametrize("step, scale", [(".05", 20), (".01", 100), (".125", 8), (".03125", 32)])
def test_regular_lattice_preserves_exact_amounts(step, scale):
    env = make(species_set=["Ni"], n_steps=1, amount_set={"min": 0, "max": 1, "step": step})
    assert [Fraction(code) for code in env.action_values] == [Fraction(i, scale) for i in range(scale + 1)]
    assert env.amount_scale == scale


def test_sparse_menu_preserves_gaps_and_authored_order():
    env = make(species_set=["Ni"], n_steps=1, amount_set=[0, .11, .18, .24, .72, .84, .96],
               element_bounds={"Ni": [.70, .85]})
    assert env.action_values == ["0", "0.11", "0.18", "0.24", "0.72", "0.84", "0.96"]
    assert {float(decode_one_hot(a[1], env.action_values)) for a in env.allowed_actions()} == {.72, .84}
    assert codes([.8, 0, .2]) == ["0.8", "0", "0.2"]


@pytest.mark.parametrize("spec", [[], "0.05", [0, "0.00"], [False, .5], ["nan"], ["Infinity"],
    {"min": 0, "max": 1, "step": 0}, {"min": 0, "max": 1, "step": -.1},
    {"min": .5, "max": .4, "step": .1}, {"min": 0, "max": 1},
    {"min": 0, "max": 1, "step": .1, "extra": True}])
def test_malformed_menus_fail_before_rollout(spec):
    with pytest.raises(ValueError):
        make(species_set=["Ni"], n_steps=1, amount_set=spec)


def test_signed_amounts_and_counts_above_one_are_valid():
    env = make(species_set=["Ni"], n_steps=1, amount_set=[-2, 0, 3], element_bounds={"Ni": [-2, 3]})
    assert len(env.allowed_actions()) == 3


def test_total_is_never_rounded_to_the_action_lattice():
    with pytest.raises(ValueError, match="infeasible"):
        make(species_set=["Ni"], n_steps=1, amount_set=[.125], total=.15)


@pytest.mark.parametrize("bounds, scale, expected", [
    ([.13, .62], 8, {2, 3, 4}), ([0, .62], 8, {0, 1, 2, 3, 4}),
    ([.25, .75], 8, {2, 3, 4, 5, 6}), ([.51, .99], 4, {3}),
    ([.01, .10], 20, {1, 2}), ([.07, .14], 100, set(range(7, 15)))])
def test_bounds_intersect_lattice_without_widening(bounds, scale, expected):
    env = make(species_set=["Ni"], n_steps=1,
               amount_set=[i / scale for i in range(scale + 1)], element_bounds={"Ni": bounds})
    values = {Fraction(decode_one_hot(a[1], env.action_values)) for a in env.allowed_actions()}
    assert values == {Fraction(i, scale) for i in expected}


@pytest.mark.parametrize("bounds", [[.2, .1], [0, "nan"], [0], [False, 1]])
def test_malformed_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        make(species_set=["Ni"], n_steps=1, amount_set=[0, 1], element_bounds={"Ni": bounds})


@pytest.mark.parametrize("menu,bounds", [([0, .05, .1], [.01, .01]), ([0, .2, .5, .8, 1], [.3, .4])])
def test_empty_exact_intersection_is_infeasible(menu, bounds):
    with pytest.raises(ValueError, match="infeasible"):
        make(species_set=["Ni"], n_steps=1, amount_set=menu, element_bounds={"Ni": bounds})


def test_sparse_bounded_completion_rejects_false_positive_window():
    kwargs = dict(species_set=["Ni", "Fe"], amount_set=[.2, .5, .8], n_steps=2, unique=True,
                  element_bounds={"Ni": [.2, .8], "Fe": [.2, .8]})
    with pytest.raises(ValueError, match="infeasible"):
        make(total=.8, **kwargs)
    assert make(total=.7, **kwargs).allowed_actions()
    assert make(total=1, **kwargs).allowed_actions()


def test_zero_selection_and_omission_use_different_action_budgets():
    kwargs = dict(species_set=["Ni", "Fe"], amount_set={"Ni": [0], "Fe": [.5]}, total=.5, unique=True,
                  element_bounds={"Fe": [.5, .5]})
    one, two = make(n_steps=1, **kwargs), make(n_steps=2, **kwargs)
    assert {decode_one_hot(a[0], one.species_set) for a in one.allowed_actions()} == {"Fe"}
    assert {decode_one_hot(a[0], two.species_set) for a in two.allowed_actions()} == {"Ni", "Fe"}


def test_positive_menu_does_not_make_unselected_species_mandatory():
    kwargs = dict(species_set=["Ni", "Fe"], amount_set={"Ni": [.2, .5], "Fe": [1]}, total=1, unique=True,
                  element_bounds={"Fe": [1, 1]})
    assert len(make(n_steps=1, **kwargs).allowed_actions()) == 1
    with pytest.raises(ValueError, match="infeasible"):
        make(n_steps=2, **kwargs)


def test_completion_matches_exhaustive_random_domains_and_species_permutations():
    rng = random.Random(121)
    labels = ["Ni", "Fe", "Co", "Cu"]
    for _ in range(50):
        menus = {label: list(rng.choice([(0,), (2,), (0, 2, 5), (2, 5, 8), (0, 5, 10)])) for label in labels}
        minimums = {label: rng.choice([0, 1, 2]) for label in labels}
        steps, total = rng.randint(1, 4), rng.randint(0, 20)
        expected = []
        for selected in product(*[(None, *menus[label]) for label in labels]):
            if sum(value is not None for value in selected) != steps or sum(value or 0 for value in selected) != total:
                continue
            if all((value or 0) >= minimums[label] for label, value in zip(labels, selected)):
                expected.append(dict(zip(labels, selected)))
        expected_first = {(label, value) for candidate in expected for label, value in candidate.items() if value is not None}
        for order in (labels, labels[::-1]):
            kwargs = dict(species_set=order, amount_set=menus, n_steps=steps, total=total, unique=True,
                          element_bounds={label: [minimums[label], 10] for label in labels})
            if not expected:
                with pytest.raises(ValueError, match="infeasible"):
                    make(**kwargs)
                continue
            env = make(**kwargs)
            actual = {(decode_one_hot(a[0], env.species_set), int(decode_one_hot(a[1], env.action_values)))
                      for a in env.allowed_actions()}
            assert actual == expected_first
