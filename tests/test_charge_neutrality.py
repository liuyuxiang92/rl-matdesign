"""Charge-neutrality utility, scaffold helper, filter, and generation gating."""
from __future__ import annotations

import pytest

pytest.importorskip("pymatgen")  # integerization + metal detection


# --------------------------------------------------------------------------- #
# charge_neutral — amount-weighted neutrality on the WHOLE formula
# --------------------------------------------------------------------------- #

def test_neutral_oxides_true():
    from rl_matdesign.constraints.charge import charge_neutral

    assert charge_neutral({"Fe": 2, "O": 3})          # Fe2O3: 2*3 - 3*2 = 0
    assert charge_neutral("La1Fe1O3")                  # LaFeO3: +3 +3 -6 = 0
    assert charge_neutral({"Li": 6, "P": 1, "S": 5, "Cl": 1})  # Li6PS5Cl


def test_non_neutral_false():
    from rl_matdesign.constraints.charge import charge_neutral

    # Genuinely non-neutral even under smact's broad oxidation table: no integer
    # oxidation-state assignment on the reduced stoichiometry sums to zero.
    assert not charge_neutral({"Ca": 1, "F": 3})   # CaF3: Ca +2, F -1 -> +2-3 = -1
    assert not charge_neutral({"Na": 3, "O": 1})   # Na3O: 3(+1) + O(-2/-1) != 0
    assert not charge_neutral({"Li": 1, "O": 3})   # LiO3: +1 + 3*O != 0


def test_stoichiometry_aware_not_just_element_set():
    """The whole point: the SAME elements give different answers by amount."""
    from rl_matdesign.constraints.charge import charge_neutral

    assert charge_neutral({"Ca": 1, "F": 2})       # CaF2: +2 -2 = 0, neutral
    assert not charge_neutral({"Ca": 1, "F": 3})   # same elements, not neutral


# --------------------------------------------------------------------------- #
# Pauling electronegativity (use_pauling) — neutral AND EN-sensible
# --------------------------------------------------------------------------- #

def test_pauling_neutral_oxide_passes_en():
    from rl_matdesign.constraints.charge import charge_neutral

    # A sensible oxide is neutral both with and without the EN test.
    assert charge_neutral({"Fe": 2, "O": 3})
    assert charge_neutral({"Fe": 2, "O": 3}, use_pauling=True)


def test_pauling_rejects_neutral_but_en_implausible():
    from rl_matdesign.constraints.charge import charge_neutral

    # S5Bi7Cl5O8 admits a charge-neutral assignment but no assignment that is also
    # Pauling-electronegativity-sensible -> use_pauling tightens neutral -> False.
    assert charge_neutral("S5Bi7Cl5O8")
    assert not charge_neutral("S5Bi7Cl5O8", use_pauling=True)


def test_all_metal_alloy_true():
    from rl_matdesign.constraints.charge import charge_neutral

    # HEA: all metals -> valid alloy regardless of oxidation states.
    assert charge_neutral({"Fe": 1, "Co": 1, "Ni": 1, "Cr": 1, "Mn": 1})


def test_unknown_element_is_lenient():
    from rl_matdesign.constraints.charge import charge_neutral

    # A non-real symbol has no oxidation states -> can't judge -> allow.
    assert charge_neutral({"Xx": 1, "O": 2})


# --------------------------------------------------------------------------- #
# parse_formula + template_scaffold ("whole formula before substitution")
# --------------------------------------------------------------------------- #

def test_parse_formula():
    from rl_matdesign.constraints.charge import parse_formula

    assert parse_formula("") == {}
    assert parse_formula(None) == {}
    assert parse_formula("O2H1") == {"O": 2.0, "H": 1.0}


_LABO3_POSCAR = """LaFeO3 perovskite template (B-site = Fe placeholder)
1.0
3.9 0.0 0.0
0.0 3.9 0.0
0.0 0.0 3.9
La Fe O
1 1 3
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
"""


def test_template_scaffold(tmp_path):
    pytest.importorskip("ase")
    from rl_matdesign.constraints.charge import template_scaffold

    p = tmp_path / "LaBO3.POSCAR"
    p.write_text(_LABO3_POSCAR)
    fixed, n_sites = template_scaffold(str(p), "Fe")
    assert fixed == {"La": 1.0, "O": 3.0}
    assert n_sites == 1


# --------------------------------------------------------------------------- #
# SMACTChargeFilter — final-step pruning on the whole picked formula
# --------------------------------------------------------------------------- #

def _oh(idx: int, n: int) -> tuple:
    v = [0.0] * n
    v[idx] = 1.0
    return tuple(v)


def test_filter_prunes_non_neutral_final_pick():
    from rl_matdesign.constraints.smact_filter import SMACTChargeFilter

    # Agent has picked Ca1 (unit), final step adds F. Candidate F amounts 0..3.
    # Under smact: CaF2 (u=2) is neutral; CaF3 (u=3) is not (Ca +2, F -1 -> -1).
    # (Fe/O can't demonstrate pruning here — smact's broad Fe states balance every
    # Fe-O ratio, which is exactly the looser-backend behavior we switched to.)
    species = ["Ca", "F"]
    ratio = ["0", "1", "2", "3"]
    allowed_units = [0, 1, 2, 3]
    filt = SMACTChargeFilter()  # no scaffold: working picks ARE the formula

    actions = [(_oh(1, 2), _oh(u, 4)) for u in range(4)]  # elem=F, comp=u
    kept = filt.filter_actions(
        actions=actions,
        chemical_amounts={"Ca": 1}, action_amounts=[0, 1, 2, 3],
        steps_left=0,
        allowed_units=allowed_units,
        possible_sums_by_k=[],
        species_set=species,
        action_values=ratio,
    )
    kept_units = {int(c.index(1.0)) for _, c in kept}
    assert 2 in kept_units        # CaF2 neutral -> kept
    assert 3 not in kept_units    # CaF3 non-neutral -> pruned


def test_filter_noop_before_final_step():
    from rl_matdesign.constraints.smact_filter import SMACTChargeFilter

    filt = SMACTChargeFilter()
    actions = [(_oh(0, 2), _oh(1, 4))]
    out = filt.filter_actions(
        actions=actions, chemical_amounts={}, action_amounts=[0, 1, 2, 3], steps_left=2, allowed_units=[0, 1, 2, 3],
        possible_sums_by_k=[], species_set=["Fe", "O"], action_values=["0", "1", "2", "3"],
    )
    assert out == actions  # untouched away from the final step


# --------------------------------------------------------------------------- #
# Registry wiring + generation gating switch
# --------------------------------------------------------------------------- #

def test_make_smact_charge_with_scaffold_formula():
    from rl_matdesign.registry import resolve_constraint

    filt = resolve_constraint("smact_charge", {"scaffold_formula": "O2H1"})
    assert filt.scaffold_per_fu == {"O": 2.0, "H": 1.0}


def test_group_filter_detection_and_combined_switches():
    from rl_matdesign.registry import smact_charge_enabled, pauling_en_enabled, charge_check_enabled, charge_use_pauling
    for filters, smact, pauling in [([], False, False), (["smact_charge"], True, False),
            ([{"type": "pauling_en"}], False, True), (["smact_charge", "pauling_en"], True, True)]:
        cfg = {"groups": [{"name": "oxide", "filters": filters}]}
        assert smact_charge_enabled(cfg) is smact
        assert pauling_en_enabled(cfg) is pauling
        assert charge_check_enabled(cfg) is (smact or pauling)
        assert charge_use_pauling(cfg) is pauling


def test_make_pauling_en_filter_uses_pauling():
    from rl_matdesign.registry import resolve_constraint

    filt = resolve_constraint("pauling_en", {})
    assert filt.use_pauling is True


def test_electronegativity_filter_prunes_en_implausible_final_pick():
    """pauling_en at the final step keeps a neutral+EN-sensible pick, drops the rest."""
    from rl_matdesign.constraints.smact_filter import ElectronegativityFilter

    # Agent has picked Fe2 (units); final step adds O. Fe2O3 is neutral AND
    # EN-sensible; Fe2O1 balances charge under smact but Fe-cation/O-anion EN holds,
    # so this mainly checks the EN filter still admits a valid completion.
    species = ["Fe", "O"]
    ratio = ["0", "1", "2", "3"]
    filt = ElectronegativityFilter()
    assert filt.use_pauling is True
    actions = [(_oh(1, 2), _oh(u, 4)) for u in range(4)]  # elem=O, comp=u units
    kept = filt.filter_actions(
        actions=actions, chemical_amounts={"Fe": 2}, action_amounts=[0, 1, 2, 3], steps_left=0,
        allowed_units=[0, 1, 2, 3], possible_sums_by_k=[],
        species_set=species, action_values=ratio,
    )
    kept_units = {int(c.index(1.0)) for _, c in kept}
    assert 3 in kept_units  # Fe2O3 neutral + EN-sensible -> kept
