"""Physical bounds and exact completion for regular and sparse fraction menus."""
from fractions import Fraction
from itertools import product
import random

import pytest

from rl_matdesign.utils.composition_domain import (
    allowed_unit_menu,
    can_complete,
    completion_sums,
    derive_total_units,
    expand_fraction_codes,
    fraction_units,
    normalize_unit_bounds,
)


def test_range_does_not_round_non_grid_maximum_outward():
    codes = expand_fraction_codes({"min": 0, "max": 0.15, "step": 0.02})
    assert codes == ["0.00", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.14"]
    assert all(Fraction(code) <= Fraction("0.15") for code in codes)


@pytest.mark.parametrize("spec, expected", [
    ({"min": "0.005", "max": "0.05", "step": "0.02"}, ["0.005", "0.025", "0.045"]),
    ({"min": "1e-3", "max": "5e-3", "step": "2e-3"}, ["0.001", "0.003", "0.005"]),
    ({"min": 0.07, "max": 0.10, "step": 0.01}, ["0.07", "0.08", "0.09", "0.10"]),
    ({"min": 0.15, "max": 0.15, "step": 0.02}, ["0.15"]),
])
def test_range_preserves_exact_origin_step_and_inclusive_endpoint(spec, expected):
    assert expand_fraction_codes(spec) == expected


@pytest.mark.parametrize("step, total, places", [
    ("0.05", 20, 2), ("0.01", 100, 2), ("0.125", 8, 3), ("0.03125", 32, 5),
])
def test_existing_regular_scenario_grids_keep_their_codes(step, total, places):
    codes = expand_fraction_codes({"min": 0, "max": 1, "step": step})
    assert codes == [f"{i / total:.{places}f}" for i in range(total + 1)]
    assert derive_total_units(codes) == total
    assert fraction_units(codes, total) == list(range(total + 1))


def test_sparse_explicit_union_keeps_gaps_and_authored_order():
    menu = [0, 0.11, 0.18, 0.24, 0.72, 0.84, 0.96]
    codes = expand_fraction_codes(menu)
    assert codes == [f"{value:.2f}" for value in menu]
    assert derive_total_units(codes) == 100
    units = fraction_units(codes, 100)
    assert 1 not in units and 13 not in units and 50 not in units
    assert allowed_unit_menu(units, normalize_unit_bounds([0.70, 0.85], 100)) == (72, 84)
    assert expand_fraction_codes([0.8, 0, 0.2]) == ["0.80", "0.00", "0.20"]


@pytest.mark.parametrize("spec", [
    [], "0.05", [0, "0.00"], [False, 0.5], [-0.01, 0.5], [0, 1.01],
    ["nan"], ["Infinity"],
    {"min": 0, "max": 1, "step": 0},
    {"min": 0, "max": 1, "step": -0.1},
    {"min": 0.5, "max": 0.4, "step": 0.1},
    {"min": 0, "max": 1},
    {"min": 0, "max": 1, "step": 0.1, "extra": True},
])
def test_malformed_grids_fail_before_environment_construction(spec):
    with pytest.raises(ValueError):
        expand_fraction_codes(spec)


def test_incompatible_integer_resolution_does_not_round_compositions():
    with pytest.raises(ValueError, match="not representable"):
        fraction_units(["0.125"], 20)
    assert fraction_units(["0.125"], 8) == [1]


@pytest.mark.parametrize("bounds, total, expected", [
    ([0.13, 0.62], 8, (2, 4)),     # Both endpoints round inward on an eighth grid.
    ([0, 0.62], 8, (0, 4)),        # Zero remains an allowed amount.
    ([0.25, 0.75], 8, (2, 6)),     # Aligned endpoints stay inclusive.
    ([0.51, 0.99], 4, (3, 3)),     # A narrow interval can contain one grid point.
    ([0.01, 0.10], 20, (1, 2)),     # A tiny positive bound never becomes zero.
    ([0.07, 0.14], 100, (7, 14)),   # Binary float products must not trim endpoints.
])
def test_bounds_intersect_the_lattice_without_widening(bounds, total, expected):
    assert normalize_unit_bounds(bounds, total) == expected
    menu = allowed_unit_menu(range(total + 1), expected)
    assert all(Fraction(str(bounds[0])) <= Fraction(value, total) <= Fraction(str(bounds[1]))
               for value in menu)


@pytest.mark.parametrize("bounds", [[0.2, 0.1], [-0.1, 1], [0, 1.1], [0, "nan"], [0]])
def test_invalid_physical_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        normalize_unit_bounds(bounds, 100)


def test_empty_lattice_interval_is_reported_and_sparse_menu_can_be_empty():
    with pytest.raises(ValueError, match="no representable"):
        normalize_unit_bounds([0.01, 0.01], 20)
    # The integer lattice has units 3..4, but this actual menu does not.
    assert allowed_unit_menu([0, 2, 5, 8, 10], (3, 4)) == ()
    assert not can_complete(4, 1, [((), True)], 10)
    assert can_complete(5, 1, [((), False), ((5,), True)], 10)


def test_sparse_bounded_completion_rejects_false_positive_window():
    # Both remaining species must use 2, 5, or 8 units. Eight is in the
    # sum-of-bounds window [4,16] and is a two-pick sum on the global menu
    # {0,2,5,8,10}, but no two legal species values actually sum to eight.
    pool = [((2, 5, 8), True), ((2, 5, 8), True)]
    assert not can_complete(8, 2, pool, 20)
    assert can_complete(7, 2, pool, 20)
    assert can_complete(10, 2, pool, 20)


def test_zero_selection_and_omitted_species_consume_different_slot_budgets():
    pool = [((0,), False), ((5,), True)]
    assert can_complete(5, 1, pool, 10)  # Skip the first species.
    assert can_complete(5, 2, pool, 10)  # Select its zero value, using a slot.
    assert not can_complete(5, 0, pool, 10)
    assert can_complete(0, 0, [], 0)
    assert not can_complete(0, 0, [((0,), True)], 0)


def test_positive_only_menu_does_not_make_an_optional_species_mandatory():
    pool = [((2, 5), False), ((10,), True)]
    assert can_complete(10, 1, pool, 10)
    assert not can_complete(10, 2, pool, 10)


def _enumerated_sums(pool, slots, maximum):
    choices = [tuple(menu) if mandatory else (None, *menu) for menu, mandatory in pool]
    sums = set()
    for selected in product(*choices):
        if sum(value is not None for value in selected) != slots:
            continue
        total = sum(value for value in selected if value is not None)
        if total <= maximum:
            sums.add(total)
    return sums


def test_completion_matches_exhaustive_small_domains_and_species_permutations():
    rng = random.Random(121)
    menus = [(), (0,), (2,), (0, 2, 5), (2, 5, 8), (0, 5, 10)]
    for _ in range(50):
        pool = [(rng.choice(menus), rng.choice([False, True])) for _ in range(4)]
        for slots in range(6):
            expected = _enumerated_sums(pool, slots, 20)
            bits = completion_sums(pool, slots, 20)
            assert {value for value in range(21) if bits & (1 << value)} == expected
            assert completion_sums(pool[::-1], slots, 20) == bits
    assert not can_complete(21, 1, [((21,), True)], 20)
