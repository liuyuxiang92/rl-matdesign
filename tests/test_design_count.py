"""Exact nominal design-space counts; no oracle or environment dependency."""
from collections import Counter
from itertools import product
from math import comb
import random

import pytest

from rl_matdesign.utils.design_count import count_assignments, count_compositions


def enumerate_small(menus, total, required, slots):
    """Independent reference enumerates assignment choices and canonical keys."""
    labels = list(menus)
    choices = [list(menus[label]) + ([] if label in required else [None]) for label in labels]
    compositions = set()
    assignments = 0
    for row in product(*choices):
        if sum(value for value in row if value is not None) != total:
            continue
        if slots is not None and sum(value is not None for value in row) != slots:
            continue
        assignments += 1
        compositions.add(tuple(sorted(
            (label, value) for label, value in zip(labels, row) if value
        )))
    return len(compositions), assignments


def test_count_preserves_sparse_menus_and_species_identity():
    menus = {"A": [2, 5, 8], "B": [2, 5, 8]}
    assert count_compositions(menus, 8) == 0  # No invented interior menu values.
    assert count_compositions(menus, 7) == 2  # A2B5 and A5B2 are distinct.
    assert count_compositions(dict(reversed(list(menus.items()))), 7) == 2


def test_optional_zero_assignment_multiplicity_is_not_material_multiplicity():
    menus = {"A": [0], "B": [0], "C": [0]}
    assert count_compositions(menus, 0, required=[]) == 1
    assert count_assignments(menus, 0, required=[]) == 8
    for slots in range(4):
        assert count_compositions(menus, 0, required=[], slots=slots) == 1
        assert count_assignments(menus, 0, required=[], slots=slots) == comb(3, slots)


def test_optional_positive_menu_can_be_omitted_but_cannot_pad_zero_slots():
    menus = {"A": [1], "B": [1]}
    assert count_compositions(menus, 1) == 0  # Default: all species required.
    assert count_compositions(menus, 1, required=[]) == 2
    assert count_compositions(menus, 1, required=[], slots=1) == 2
    assert count_compositions(menus, 1, required=[], slots=2) == 0
    assert count_compositions({"A": [0, 1], "B": [1]}, 1, required=[], slots=2) == 1


def test_required_zero_still_consumes_a_slot():
    menus = {"A": [0], "B": [0, 1]}
    assert count_compositions(menus, 0, required=["A"], slots=0) == 0
    assert count_compositions(menus, 0, required=["A"], slots=1) == 1
    assert count_compositions(menus, 0, required=["A"], slots=2) == 1
    assert count_compositions(menus, 1, required=["A"], slots=1) == 0
    assert count_compositions(menus, 1, required=["A"], slots=2) == 1


def test_empty_and_unreachable_domains():
    assert count_compositions({}, 0) == 1
    assert count_compositions({}, 0, slots=0) == 1
    assert count_compositions({}, 0, slots=1) == 0
    assert count_compositions({}, 1) == 0
    assert count_compositions({"A": []}, 0) == 0
    assert count_compositions({"A": []}, 0, required=[], slots=0) == 1
    assert count_compositions({"A": []}, 0, required=[], slots=1) == 0
    assert count_compositions({"A": [20]}, 1) == 0
    assert count_assignments({}, 0) == 1


def test_arbitrary_precision_counts_not_fixed_width_numpy_integers():
    menus = {str(index): [0, 1] for index in range(100)}
    assert count_compositions(menus, 50) == comb(100, 50)
    assert count_compositions(menus, 50) > 2**64


def test_dp_agrees_with_independent_exhaustive_small_domains():
    rng = random.Random(20261105)
    for _ in range(60):
        menus = {
            label: rng.sample(range(5), rng.randrange(5))
            for label in ("A", "B", "C", "D")
        }
        required = rng.sample(list(menus), rng.randrange(5))
        total = rng.randrange(9)
        for slots in (None, 0, 1, 2, 3, 4, 5):
            unique, assigned = enumerate_small(menus, total, required, slots)
            assert count_compositions(menus, total, required=required, slots=slots) == unique
            assert count_assignments(menus, total, required=required, slots=slots) == assigned


@pytest.mark.parametrize("menus,total,kwargs", [
    ({"A": [0, 0]}, 1, {}), ({"A": [True]}, 1, {}),
    ({"A": [0.5]}, 1, {}), ({"A": [-1]}, 1, {}),
    ({"A": "01"}, 1, {}), ({"A": {0, 1}}, 1, {}),
    ({"": [0]}, 1, {}), ({1: [0]}, 1, {}),
    ({"A": [0]}, True, {}), ({"A": [0]}, -1, {}),
    ({"A": [0]}, 1.0, {}), ({"A": [0]}, 1, {"slots": True}),
    ({"A": [0]}, 1, {"slots": -1}),
    ({"A": [0]}, 1, {"required": "A"}),
    ({"A": [0]}, 1, {"required": ["A", "A"]}),
    ({"A": [0]}, 1, {"required": ["unknown"]}),
])
def test_malformed_counting_domains_fail(menus, total, kwargs):
    for count in (count_compositions, count_assignments):
        with pytest.raises(ValueError):
            count(menus, total, **kwargs)


def independent_partial_sum_count(menus, total):
    """Meet in the middle, rather than repeating the implementation's DP."""
    values = list(menus.values())
    midpoint = len(values) // 2
    left = Counter(map(sum, product(*values[:midpoint])))
    right = Counter(map(sum, product(*values[midpoint:])))
    return sum(count * right.get(total - subtotal, 0) for subtotal, count in left.items())


@pytest.mark.parametrize("step", [4, 2, 1])
def test_nested_integer_simplex_counts_match_independent_reference(step):
    # An abstract four-coordinate simplex; no material recipe or model inputs.
    menus = {label: list(range(0, 9, step)) for label in ["A", "B", "C", "D"]}
    expected = comb(8 // step + 3, 3)
    assert count_compositions(menus, 8, slots=4) == expected
    assert independent_partial_sum_count(menus, 8) == expected


def test_uniform_grid_refinement_preserves_each_coarse_assignment():
    grids = [{label: set(range(0, 9, step)) for label in ["A", "B", "C", "D"]}
             for step in [4, 2, 1]]
    for label in grids[0]:
        assert grids[0][label] < grids[1][label] < grids[2][label]


@pytest.mark.parametrize("coordinates", [8, 16, 32])
def test_count_scales_without_enumerating_large_simplexes(coordinates):
    # Stars and bars independently gives the number of nonnegative integer
    # coordinate vectors summing to 8; the largest case has >60 million points.
    menus = {f"coordinate-{index}": list(range(9)) for index in range(coordinates)}
    expected = comb(8 + coordinates - 1, coordinates - 1)
    assert count_compositions(menus, 8, slots=coordinates) == expected
    if coordinates == 32:
        assert expected > 60_000_000
