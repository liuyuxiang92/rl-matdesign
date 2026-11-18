"""Optional filters are ordered lists whose intersection preserves group rules."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.categorical import CategoricalGroup
from rl_matdesign.constraints.base import apply_filters
from rl_matdesign.registry import build_constraints


def _oh(index, size):
    return tuple(float(i == index) for i in range(size))


ACTIONS = [(_oh(0, 2), _oh(index, 3)) for index in range(3)]


class Returns:
    def __init__(self, result):
        self.result = result

    def filter_actions(self, *, actions, **context):
        return self.result


def test_public_filter_list_resolves_real_filters_and_intersects_chemistry(monkeypatch):
    from rl_matdesign.constraints import charge
    observed = []

    def charge_check(amounts, *, use_pauling, **kwargs):
        observed.append((dict(amounts), use_pauling))
        return amounts["Ca"] * 2 == amounts["F"] and (not use_pauling or amounts["F"] < 3)

    monkeypatch.setattr(charge, "charge_neutral", charge_check)
    filters = build_constraints({"filters": ["smact_charge", {"type": "pauling_en"}]})
    assert type(filters) is list
    assert [type(item).__name__ for item in filters] == ["SMACTChargeFilter", "ElectronegativityFilter"]
    actions = [(_oh(1, 2), _oh(i, 2)) for i in range(2)]
    assert apply_filters(filters, actions=actions, chemical_amounts={"Ca": 1},
        steps_left=0, species_set=["Ca", "F"], action_values=["2", "3"], action_amounts=[2, 3]) == actions[:1]
    assert observed == [({"Ca": 1., "F": 2.}, False), ({"Ca": 1., "F": 3.}, False),
                        ({"Ca": 1., "F": 2.}, True)]


def test_later_filter_receives_only_actions_accepted_by_previous_filter():
    seen = []
    class Observe:
        def filter_actions(self, *, actions, **context):
            seen.append(list(actions))
            return actions[1:]
    assert apply_filters([Returns(ACTIONS[1:]), Observe()], actions=ACTIONS) == ACTIONS[2:]
    assert seen == [ACTIONS[1:]]


def test_empty_intersection_short_circuits_and_cannot_restore_actions():
    class MustNotRun:
        def filter_actions(self, **context):
            pytest.fail("No later filter may restore an empty action set")
    assert apply_filters([Returns([]), MustNotRun()], actions=ACTIONS) == []
    assert apply_filters([MustNotRun()], actions=[]) == []


@pytest.mark.parametrize("result,message", [
    (None, "list or tuple"), ("action", "list or tuple"), ({"action": 1}, "list or tuple"),
    ([list(ACTIONS[0])], "invalid action encoding"),
    ([ACTIONS[0], ACTIONS[0]], "duplicate"),
    ([(_oh(1, 2), _oh(0, 3))], "subset"),
])
def test_invalid_filter_outputs_reject(result, message):
    with pytest.raises(ValueError, match=message):
        apply_filters([Returns(result)], actions=ACTIONS)


def test_later_filter_cannot_restore_a_previously_removed_action():
    with pytest.raises(ValueError, match="subset"):
        apply_filters([Returns(ACTIONS[1:]), Returns(ACTIONS[:1])], actions=ACTIONS)


def test_mutating_insert_cannot_hide_an_added_action_or_modify_caller():
    class Insert:
        def filter_actions(self, *, actions, **context):
            actions.append((_oh(1, 2), _oh(0, 3)))
            return actions
    original = list(ACTIONS)
    with pytest.raises(ValueError, match="subset"):
        apply_filters([Insert()], actions=original)
    assert original == ACTIONS


@pytest.mark.parametrize("as_tuple", [False, True])
def test_in_place_pruning_is_valid_and_does_not_modify_caller(as_tuple):
    class Prune:
        def filter_actions(self, *, actions, **context):
            actions.pop()
            return tuple(actions) if as_tuple else actions
    original = list(ACTIONS)
    assert apply_filters([Prune()], actions=original) == ACTIONS[:2]
    assert original == ACTIONS


@pytest.mark.parametrize("cfg", [{}, {"filters": []}])
def test_absent_and_empty_filters_are_plain_empty_lists(cfg):
    filters = build_constraints(cfg)
    assert type(filters) is list and filters == []
    assert apply_filters(filters, actions=ACTIONS) == ACTIONS


@pytest.mark.parametrize("spec,message", [
    (None, "must be a list"), ({"type": "pauling_en"}, "must be a list"),
    ("pauling_en", "must be a list"), ([42], "dict or a bare string"),
    ([{}], "missing 'type'"), ([{"type": 42}], "nonempty string"),
])
def test_invalid_filter_declarations_fail(spec, message):
    with pytest.raises(ValueError, match=message):
        build_constraints({"filters": spec})


def test_custom_filter_factory_must_provide_filter_actions(monkeypatch):
    import rl_matdesign.registry as registry
    monkeypatch.setattr(registry, "resolve_constraint", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="implement filter_actions"):
        build_constraints({"filters": ["custom"]})


def test_public_oxide_configs_resolve_to_two_filters_without_wrapper():
    from rl_matdesign.configuration import load_config
    repo = Path(__file__).resolve().parents[1]
    for name in ("oxides_sinter.yaml", "oxides_calcine.yaml"):
        cfg = load_config(repo / "configs" / name)
        group = cfg["groups"][0]
        assert group["element_bounds"]["O"] == [1, 9]
        assert group["n_steps"] == 5 and "amount_set" in group
        filters = build_constraints(group)
        assert type(filters) is list
        assert [type(item).__name__ for item in filters] == ["SMACTChargeFilter", "ElectronegativityFilter"]


@pytest.mark.parametrize("configured", [False, True])
def test_chemical_filter_list_preserves_intrinsic_oxygen_bounds(configured):
    class OxygenOnly:
        def filter_actions(self, *, actions, species_set, **context):
            return [a for a in actions if species_set[int(np.argmax(a[0]))] == "O"]
    class AmountTwo:
        def filter_actions(self, *, actions, action_amounts, **context):
            return [a for a in actions if action_amounts[int(np.argmax(a[1]))] == 2]
    group = ChemicalAmountsGroup({"species_set": ["Fe", "O"], "amount_set": [1, 2], "n_steps": 2,
                                  "element_bounds": {"O": [1, 4]}},
                                 filters=[OxygenOnly(), AmountTwo()] if configured else [],
                                 state_featurizer=lambda _: np.zeros(1))
    def decoded():
        return {(group.species_set[int(np.argmax(a))], Fraction(group.action_values[int(np.argmax(b))]))
                for a, b in group.allowed_actions()}
    assert len(group.filters) == (2 if configured else 0)
    assert decoded() == ({("O", 2)} if configured else {("Fe", 1), ("Fe", 2), ("O", 1), ("O", 2)})
    group.step((_oh(1 if configured else 0, 2), _oh(1, 2)))
    assert decoded() == ({("O", 2)} if configured else {("O", 1), ("O", 2)})
    group.step(group.allowed_actions()[0])
    assert group.terminal_cation_fractions()["O"] >= 1


@pytest.mark.parametrize("configured", [False, True])
def test_categorical_filter_list_preserves_unique_assignments(configured):
    class NextLabel:
        def filter_actions(self, *, actions, current_fields, **context):
            index = 1 if current_fields else 0
            return [a for a in actions if a[0][index] == 1]
    class OnlyA:
        def filter_actions(self, *, actions, action_values, code_to_value, **context):
            return [a for a in actions if code_to_value[action_values[int(np.argmax(a[1]))]] == "A"]
    group = CategoricalGroup({"labels": [1, 2], "values": ["A", "C"], "n_steps": 2},
                             filters=[NextLabel(), OnlyA()] if configured else [])
    assert len(group.filters) == (2 if configured else 0)
    assert len(group.allowed_actions()) == (1 if configured else 4)
    first_label = 0 if configured else 1
    group.step(next(a for a in group.allowed_actions() if a[0][first_label] == 1))
    assert len(group.allowed_actions()) == (1 if configured else 2)
    assert all(a[0][first_label] == 0 for a in group.allowed_actions())
    group.step(group.allowed_actions()[0])
    assert group.is_terminal and group.allowed_actions() == []
