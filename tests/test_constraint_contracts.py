"""Constraints remove known-invalid actions; missing chemistry data stays lenient."""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from rl_matdesign.constraints.chain import ChainConstraintFilter
from rl_matdesign.constraints.last_step_element import LastStepElementFilter
from rl_matdesign.constraints.smact_filter import ElectronegativityFilter, SMACTChargeFilter
from rl_matdesign.constraints.sse_doping import SSEDopingFilter


def _oh(index, size):
    return tuple(float(i == index) for i in range(size))


def _context(species, codes, *, units=None, steps_left=0):
    return dict(
        units_map=units or {}, steps_left=steps_left,
        allowed_units=[float(c) for c in codes], possible_sums_by_k=[],
        species_set=species, fraction_set=codes,
    )


def _chain(*children):
    chain = ChainConstraintFilter.__new__(ChainConstraintFilter)
    chain.children = children
    return chain


class _Returns:
    def __init__(self, result):
        self.result = result

    def filter_actions(self, *, actions, **kwargs):
        return self.result


@pytest.mark.parametrize("result", [["a", "c"], ["a", "a"], None, "a", {"a": 1}])
def test_chain_rejects_invalid_child_results(result):
    with pytest.raises(ValueError, match="subset|duplicate|list or tuple"):
        _chain(_Returns(result)).filter_actions(actions=["a", "b"])


def test_chain_cannot_restore_action_removed_by_previous_child():
    with pytest.raises(ValueError, match="subset"):
        _chain(_Returns(["b"]), _Returns(["a"])).filter_actions(actions=["a", "b"])


def test_chain_rejects_actions_inserted_by_mutating_its_input():
    class _Insert:
        def filter_actions(self, *, actions, **kwargs):
            actions.append("new")
            return actions

    original = ["a", "b"]
    with pytest.raises(ValueError, match="subset"):
        _chain(_Insert()).filter_actions(actions=original)
    assert original == ["a", "b"]


def test_chain_allows_in_place_pruning_without_mutating_caller():
    class _Pop:
        def filter_actions(self, *, actions, **kwargs):
            actions.pop()
            return actions

    original = ["a", "b"]
    assert _chain(_Pop()).filter_actions(actions=original) == ["a"]
    assert original == ["a", "b"]


def test_chain_empty_input_cannot_be_restored():
    assert _chain(_Returns(["new"])).filter_actions(actions=[]) == []


@pytest.mark.parametrize("zero", ["0", "0.00", "0e0", "-0.0", "+0.000e10"])
def test_last_step_zero_is_numeric_not_string_identity(zero):
    codes = [zero, "0.25"]
    actions = [(_oh(0, 1), _oh(i, 2)) for i in range(2)]
    f = LastStepElementFilter(["O"])
    assert f.filter_actions(actions=actions, **_context(["O"], codes)) == actions[1:]
    assert f.filter_actions(actions=actions[:1], **_context(["O"], codes)) == []


def test_last_step_uses_decoded_typed_value_when_supplied():
    actions = [(_oh(0, 1), _oh(i, 2)) for i in range(2)]
    context = _context(["O"], ["0", "1"])
    context["fraction_set"] = ["code0", "code1"]
    assert LastStepElementFilter(["O"]).filter_actions(
        actions=actions, code_to_value={"code0": 0.0, "code1": 0.5}, **context,
    ) == actions[1:]


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "oxygen", True])
def test_last_step_rejects_invalid_numeric_amounts(value):
    with pytest.raises(ValueError, match="numeric|finite"):
        LastStepElementFilter(["O"]).filter_actions(
            actions=[(_oh(0, 1), _oh(0, 1))], code_to_value={"0": value},
            **_context(["O"], ["0"]),
        )


def test_last_step_reservation_and_nonzero_disabled_have_exact_meaning():
    action = [(_oh(0, 1), _oh(0, 1))]
    f = LastStepElementFilter(["O"], nonzero_ratio=False)
    assert f.filter_actions(actions=action, **_context(["O"], ["0.00"])) == action
    assert f.filter_actions(
        actions=action, **_context(["O"], ["0.00"], steps_left=1),
    ) == []


@pytest.mark.parametrize("filter_type", [SMACTChargeFilter, ElectronegativityFilter])
@pytest.mark.parametrize("categorical", [False, True])
def test_known_invalid_charge_screen_never_restores_actions(monkeypatch, filter_type, categorical):
    from rl_matdesign.constraints import charge

    monkeypatch.setattr(charge, "charge_neutral", lambda *args, **kwargs: False)
    context = _context(["Na", "O"], ["1", "2"], units={"Na": 1})
    if categorical:
        context.update(code_to_value={"1": "O", "2": "F"}, prior_groups=[{"A": "Na"}])
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


@pytest.mark.parametrize("policy,codes", [("metal_only", ["0.1", "1.0"]), ("oxide_only", ["0", "0.00"])])
def test_sse_all_disallowed_forms_remain_empty(policy, codes):
    f = SSEDopingFilter({policy: ["Ru"]})
    actions = [(_oh(0, 1), _oh(i, 2)) for i in range(2)]
    assert f.filter_actions(
        actions=actions, prior_groups=[{"P": 0.8, "Ru": 0.2}],
        species_set=["O"], fraction_set=codes,
    ) == []


def test_sse_preserves_unaffected_slots_and_sorted_dopant_pairing():
    f = SSEDopingFilter({
        "choices": [{"name": "O_a", "element": "O"}, {"name": "O_b", "element": "O"}],
        "metal_only": ["Ru"], "oxide_only": ["Al"],
    })
    species, codes = ["O_a", "O_b", "Cl"], ["0.0", "0.2"]
    for prior in [{"Ru": 0.1, "Al": 0.1, "P": 0.8}, {"P": 0.8, "Al": 0.1, "Ru": 0.1}]:
        for slot, retained in [(0, [1]), (1, [0]), (2, [0, 1])]:
            actions = [(_oh(slot, 3), _oh(i, 2)) for i in range(2)]
            assert f.filter_actions(
                actions=actions, prior_groups=[prior], species_set=species, fraction_set=codes,
            ) == [actions[i] for i in retained]


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


def test_chain_preserves_phase_pattern_rejection():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter

    pattern = PhasePatternFilter(patterns=[{"required": {"Ni": [0.8, 0.9]}}])
    actions = [(_oh(1, 2), _oh(0, 1))]
    context = _context(["Ni", "Fe"], ["1"], units={"Ni": 1})
    assert pattern.filter_actions(actions=actions, **context) == []
    assert _chain(pattern).filter_actions(actions=actions, **context) == []
