"""Constraints remove known-invalid actions; missing chemistry data stays lenient."""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from rl_matdesign.constraints.smact_filter import ElectronegativityFilter, SMACTChargeFilter


def _oh(index, size):
    return tuple(float(i == index) for i in range(size))


def _context(species, codes, *, units=None, steps_left=0):
    return dict(
        units_map=units or {}, steps_left=steps_left,
        allowed_units=[float(c) for c in codes], possible_sums_by_k=[],
        species_set=species, action_values=codes, chemical_amounts=units or {},
        action_amounts=[float(c) for c in codes],
    )


@pytest.mark.parametrize("filter_type", [SMACTChargeFilter, ElectronegativityFilter])
def test_known_invalid_charge_screen_never_restores_actions(monkeypatch, filter_type):
    from rl_matdesign.constraints import charge

    monkeypatch.setattr(charge, "charge_neutral", lambda *args, **kwargs: False)
    context = _context(["Na", "O"], ["1", "2"], units={"Na": 1})
    actions = [(_oh(1, 2), _oh(i, 2)) for i in range(2)]
    assert filter_type(allow_alloys=False).filter_actions(actions=actions, **context) == []


@pytest.mark.parametrize("filter_type", [SMACTChargeFilter, ElectronegativityFilter])
def test_charge_screen_retains_only_matching_stoichiometry(monkeypatch, filter_type):
    from rl_matdesign.constraints import charge

    screened = []

    def check(comp, **kwargs):
        screened.append((comp, kwargs["use_pauling"]))
        return comp["Na"] == 2 * comp["O"]

    monkeypatch.setattr(charge, "charge_neutral", check)
    actions = [(_oh(1, 2), _oh(i, 2)) for i in range(2)]
    assert filter_type(allow_alloys=False).filter_actions(
        actions=actions, **_context(["Na", "O"], ["1", "2"], units={"Na": 2}),
    ) == actions[:1]
    assert screened == [
        ({"Na": 2.0, "O": 1.0}, filter_type is ElectronegativityFilter),
        ({"Na": 2.0, "O": 2.0}, filter_type is ElectronegativityFilter),
    ]






def _mock_chemistry(monkeypatch, states):
    from rl_matdesign.constraints import charge

    monkeypatch.setattr(charge, "_integerize", lambda amounts: {e: int(n) for e, n in amounts.items()})
    monkeypatch.setattr(charge, "_oxidation_states", lambda e: states.get(e, ()))
    return charge


def test_missing_oxidation_states_keeps_existing_lenient_policy(monkeypatch):
    charge = _mock_chemistry(monkeypatch, {"O": (-2,)})
    assert charge.charge_neutral({"Xx": 1, "O": 1}, allow_alloys=False)


def test_missing_electronegativity_skips_only_pauling_check(monkeypatch):
    charge = _mock_chemistry(monkeypatch, {"Na": (1,), "O": (-2,)})
    monkeypatch.setattr(charge, "_pauling_eneg", lambda e: None)
    assert charge.charge_neutral({"Na": 2, "O": 1}, allow_alloys=False, use_pauling=True)
    assert not charge.charge_neutral({"Na": 1, "O": 1}, allow_alloys=False, use_pauling=True)


def test_combination_cap_keeps_warning_and_lenient_policy(monkeypatch):
    charge = _mock_chemistry(monkeypatch, {"Na": (1, 2), "O": (-2,)})
    monkeypatch.setattr(charge, "_MAX_COMBOS", 1)
    with pytest.warns(RuntimeWarning, match="exceeds the cap"):
        assert charge.charge_neutral({"Na": 1, "O": 1}, allow_alloys=False)


def test_pauling_exception_keeps_neutrality_only_policy(monkeypatch):
    charge = _mock_chemistry(monkeypatch, {"Na": (1,), "O": (-2,)})
    monkeypatch.setattr(charge, "_pauling_eneg", lambda e: 1.0)
    screening = ModuleType("smact.screening")

    def unavailable(*args):
        raise RuntimeError("Pauling backend unavailable")

    screening.pauling_test = unavailable
    monkeypatch.setitem(sys.modules, "smact", ModuleType("smact"))
    monkeypatch.setitem(sys.modules, "smact.screening", screening)
    assert charge.charge_neutral({"Na": 2, "O": 1}, allow_alloys=False, use_pauling=True)
    assert not charge.charge_neutral({"Na": 1, "O": 1}, allow_alloys=False, use_pauling=True)
