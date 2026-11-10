"""Standalone Stage 2B contract: complete position/value assignments.

The finite-domain oracle below enumerates candidates independently of the
environment's feasibility solver, including every reachable partial assignment.
"""
from copy import deepcopy
from itertools import combinations, product
import json

import numpy as np
import pytest

from rl_matdesign.action_encoding import ActionEncoding, ValueEncoding
from rl_matdesign.sequence import SequenceGroup


STOP = ("stop", None)


def action(group, position, token):
    if position == "stop":
        label = group.encoding.labels.index_of(("control", "stop"))
        value = group.encoding.values.index_of("stop", namespace=(group.name, "control"))
    else:
        label = group.encoding.labels.index_of(("position", position))
        value = group.encoding.values.index_of(token, namespace=(group.name, "monomer"))
    return group.encoding.encode(label, value)


def decoded_actions(group):
    result = set()
    for labels, values in group.allowed_actions():
        label = group.encoding.labels.labels[int(np.argmax(labels))]
        value = group.value_specs[int(np.argmax(values))].value
        result.add(STOP if label == ("control", "stop") else (label[1], value))
    return result


def finite_candidates(spec):
    """Brute-force complete molecular candidates, not action trajectories."""
    parent = tuple(spec["parent"]) if "parent" in spec else None
    length = len(parent) if parent else spec["length"]
    choices = [list(dict.fromkeys(spec["tokens"] + ([parent[p]] if parent else [])))
               for p in range(length)]
    candidates = set()
    for candidate in product(*choices):
        if parent and not 1 <= sum(a != b for a, b in zip(candidate, parent)) <= spec["num_mutations"]:
            continue
        if not spec.get("allow_repeats", True) and len(set(candidate)) != length:
            continue
        if any(not bound.get("min", 0) <= sum(token in bound["tokens"] for token in candidate)
               <= bound.get("max", length) for bound in spec.get("count_constraints", [])):
            continue
        candidates.add(candidate)
    return candidates


ORACLE_CASES = [
    {"tokens": ["A", "B", "C"], "length": 3},
    {"tokens": ["A", "B", "C"], "length": 3, "allow_repeats": False},
    {"tokens": ["A", "B", "C"], "length": 3,
     "count_constraints": [{"tokens": ["A", "B"], "min": 2, "max": 2},
                           {"tokens": ["B", "C"], "min": 2, "max": 3}]},
    {"tokens": ["A", "B", "C", "D"], "length": 3, "allow_repeats": False,
     "count_constraints": [{"tokens": ["A", "B"], "min": 1, "max": 2},
                           {"tokens": ["B", "C"], "min": 2, "max": 2}]},
    {"tokens": ["A", "B", "C"], "parent": ["A", "A", "B"], "num_mutations": 1},
    {"tokens": ["A", "B", "C"], "parent": ["A", "B", "C"], "num_mutations": 2},
    {"tokens": ["A", "B", "C"], "parent": ["A", "B", "C"], "num_mutations": 3,
     "allow_repeats": False},
    {"tokens": ["A", "B", "C"], "parent": ["A", "A", "B"], "num_mutations": 2,
     "allow_repeats": False},
    {"tokens": ["A", "B", "C"], "parent": ["A", "B", "C"], "num_mutations": 2,
     "count_constraints": [{"tokens": ["A"], "min": 1, "max": 1},
                           {"tokens": ["B", "C"], "min": 2, "max": 2}]},
    {"tokens": ["A", "B", "C"], "parent": ["X", "A", "A"], "num_mutations": 2,
     "allow_repeats": False,
     "count_constraints": [{"tokens": ["X", "A"], "min": 1, "max": 2},
                           {"tokens": ["A", "B"], "min": 1, "max": 2}]},
    {"tokens": ["A", "B"], "parent": ["X", "X", "A", "B"], "num_mutations": 3,
     "count_constraints": [{"tokens": ["X"], "min": 1, "max": 1}]},
    {"tokens": ["A", "B", "C", "D"], "parent": ["A", "A", "A", "B"], "num_mutations": 3,
     "allow_repeats": False,
     "count_constraints": [{"tokens": ["A", "C"], "min": 2, "max": 2}]},
]


@pytest.mark.parametrize("spec", ORACLE_CASES)
def test_all_reachable_partial_action_masks_match_independent_finite_oracle(spec):
    candidates = finite_candidates(spec)
    assert candidates
    parent = tuple(spec["parent"]) if "parent" in spec else None
    length = len(parent) if parent else spec["length"]
    budget = spec["num_mutations"] if parent else length
    partials = {()}
    for candidate in candidates:
        edits = tuple((p, token) for p, token in enumerate(candidate)
                      if not parent or token != parent[p])
        for count in range(len(edits) + 1):
            partials.update(combinations(edits, count))
    terminal_candidates = set()
    for fixed in sorted(partials):
        group = SequenceGroup(spec)
        # The same canonical state is reachable through reverse position order.
        for position, token in reversed(fixed):
            group.step(action(group, position + 1, token))
        assert group.is_terminal == (len(fixed) == budget)
        if group.is_terminal:
            assert group.allowed_actions() == []
            terminal_candidates.add(group.terminal_comp_key())
            continue
        fixed_map = dict(fixed)
        compatible = [candidate for candidate in candidates
                      if all(candidate[p] == token for p, token in fixed)]
        expected = {(p + 1, token) for candidate in compatible for p, token in enumerate(candidate)
                    if p not in fixed_map and (not parent or token != parent[p])}
        current = tuple(fixed_map.get(p, parent[p] if parent else None) for p in range(length))
        if parent and fixed and current in candidates:
            expected.add(STOP)
        assert decoded_actions(group) == expected, (spec, fixed)
        if STOP in expected:
            group.step(action(group, *STOP))
            terminal_candidates.add(group.terminal_comp_key())
            assert group.is_terminal and group.counter <= budget
    assert terminal_candidates == candidates


def test_assignment_execution_order_has_identical_state_mask_and_candidate():
    spec = {"name": "peptide", "tokens": ["A", "B", "C"], "length": 3}
    first, second = SequenceGroup(spec), SequenceGroup(spec)
    for position, token in [(1, "A"), (3, "C")]:
        first.step(action(first, position, token))
    for position, token in [(3, "C"), (1, "A")]:
        second.step(action(second, position, token))
    np.testing.assert_array_equal(first.current_state_features(), second.current_state_features())
    assert first.allowed_actions() == second.allowed_actions()
    first.step(action(first, 2, "B"))
    second.step(action(second, 2, "B"))
    assert first.terminal_comp_key() == second.terminal_comp_key() == ("A", "B", "C")
    reverse = SequenceGroup(spec)
    for position, token in [(3, "A"), (1, "C"), (2, "B")]:
        reverse.step(action(reverse, position, token))
    assert reverse.terminal_comp_key() == ("C", "B", "A")
    assert reverse.terminal_comp_key() != first.terminal_comp_key()


def test_parent_blank_same_complete_pair_contract_and_distinct_initial_states():
    blank = SequenceGroup({"tokens": ["A", "B"], "length": 2})
    parent = SequenceGroup({"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 2})
    blank.step(action(blank, 2, "A"))
    parent.step(action(parent, 2, "A"))
    assert blank.cation_fractions() == [None, "A"]
    assert parent.cation_fractions() == ["A", "A"]
    assert blank.counter == parent.counter == 1
    assert STOP not in decoded_actions(blank)
    assert STOP in decoded_actions(parent)
    assert parent.max_steps == parent.n_components == 2


def test_parent_can_stop_after_one_edit_without_padding_to_maximum():
    group = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B", "C"], "num_mutations": 3})
    assert STOP not in decoded_actions(group)
    group.step(action(group, 3, "A"))
    before = group.current_state_features()
    group.step(action(group, *STOP))
    assert group.counter == 2 < group.max_steps
    assert group.is_terminal and group._is_terminal() and group.is_done
    assert group.terminal_comp_key() == ("A", "B", "A")
    assert group.allowed_actions() == []
    np.testing.assert_array_equal(group.current_state_features()[:-1], before[:-1])
    assert before[-1] == 0 and group.current_state_features()[-1] == 1
    with pytest.raises(RuntimeError, match="after termination"):
        group.step(action(group, 1, "C"))


def test_budget_one_auto_terminates_without_stop_step():
    group = SequenceGroup({"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 1})
    group.step(action(group, 1, "B"))
    assert group.counter == group.max_steps == 1
    assert group.is_terminal


def test_intrinsic_constraints_allow_repairable_intermediate_duplicates():
    group = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B", "C"],
                           "num_mutations": 2, "allow_repeats": False})
    group.constraints_enabled = False  # A training filter toggle cannot bypass molecular validity.
    group.step(action(group, 1, "B"))
    assert group.cation_fractions() == ["B", "B", "C"]
    assert decoded_actions(group) == {(2, "A")}
    group.step(action(group, 2, "A"))
    assert group.terminal_comp_key() == ("B", "A", "C")


def test_parent_only_ids_are_retained_and_counted_but_never_new_choices():
    group = SequenceGroup({"tokens": ["01", "1"], "parent": ["cap-X", "01", "1"],
                           "num_mutations": 2,
                           "count_constraints": [{"tokens": ["cap-X"], "min": 1, "max": 1}]})
    assert (1, "01") not in decoded_actions(group)
    assert not any(token == "cap-X" for _, token in decoded_actions(group))
    group.step(action(group, 2, "1"))
    group.step(action(group, *STOP))
    assert group.terminal_comp_key() == ("cap-X", "1", "1")
    assert group.state_feature_schema()["tokens"] == ["01", "1", "cap-X"]


def test_exact_monomer_ids_do_not_collide_with_stop_or_numeric_spellings():
    tokens = ["01", "1", "1.0", "stop", '["control","stop"]', " cap-A "]
    group = SequenceGroup({"tokens": tokens, "parent": ["01", "1"], "num_mutations": 2})
    assert len(group.fraction_set) == len(set(group.fraction_set)) == len(tokens) + 1
    assert all(spec.kind == "categorical" for spec in group.value_specs)
    np.testing.assert_array_equal(ValueEncoding.from_specs(group.value_specs).features, np.eye(len(tokens) + 1))
    group.step(action(group, 2, " cap-A "))
    group.step(action(group, *STOP))
    assert group.terminal_comp_key() == ("01", " cap-A ")


@pytest.mark.parametrize("wrong_action", [
    ((0, 0, 0), (1, 0, 0)), ((1, 1, 0), (1, 0, 0)),
    ((.9, .1, 0), (1, 0, 0)), ((1, 0), (1, 0, 0)),
    ((1, 0, 0), (float("nan"), 0, 0)), ((1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0)), ((0, 0, 1), (0, 0, 1)), None,
])
def test_invalid_actions_do_not_mutate_state(wrong_action):
    group = SequenceGroup({"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 2})
    before = group.current_state_features()
    with pytest.raises(ValueError):
        group.step(wrong_action)
    np.testing.assert_array_equal(group.current_state_features(), before)
    assert group.counter == 0


def test_noop_or_repeat_edit_is_rejected_and_initialization_restores_parent():
    group = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B"], "num_mutations": 2})
    with pytest.raises(ValueError, match="not allowed"):
        group.step(action(group, 1, "A"))
    group.step(action(group, 1, "C"))
    before = group.current_state_features()
    with pytest.raises(ValueError, match="not allowed"):
        group.step(action(group, 1, "B"))
    np.testing.assert_array_equal(group.current_state_features(), before)
    assert group.counter == 1
    group.initialize()
    assert group.cation_fractions() == ["A", "B"]
    assert group.counter == 0 and not group.is_terminal
    with pytest.raises(RuntimeError, match="completed sequence"):
        group.terminal_comp_key()


@pytest.mark.parametrize("spec", [
    {"tokens": ["A"], "length": 2, "allow_repeats": False},
    {"tokens": ["A"], "parent": ["A"], "num_mutations": 1},
    {"tokens": ["A", "B"], "parent": ["A", "A", "A"], "num_mutations": 1,
     "allow_repeats": False},
    {"tokens": ["A", "B"], "length": 2,
     "count_constraints": [{"tokens": ["A"], "min": 2}, {"tokens": ["B"], "min": 1}]},
])
def test_infeasible_candidate_domains_fail_during_construction(spec):
    assert finite_candidates(spec) == set()
    with pytest.raises(ValueError, match="no complete candidate"):
        SequenceGroup(spec)


@pytest.mark.parametrize("changes", [
    {"tokens": [1, "A"]}, {"tokens": ["A", "A"]}, {"tokens": []}, {"tokens": {"csv": "a.csv"}},
    {"length": True}, {"length": 1.5}, {"length": 0}, {"allow_repeats": "false"},
    {"parent": ["A"]}, {"num_mutations": 1},
    {"parent": ["A", "B"], "num_mutations": False},
    {"parent": ["A", "B"], "num_mutations": 3},
    {"count_constraints": [{"tokens": ["unknown"]}]},
    {"count_constraints": [{"tokens": ["A"], "max": True}]},
    {"count_constraints": [{"tokens": ["A"], "min": 2, "max": 1}]},
    {"count_constraints": [{"tokens": ["A"], "max": 3}]},
    {"count_constraints": [{"where": {"cap": True}}]},
    {"filters": ["anything"]},
])
def test_invalid_specs_fail_without_silent_coercion(changes):
    with pytest.raises(ValueError):
        SequenceGroup({"tokens": ["A", "B"], "length": 2, **changes})


def test_schema_records_action_semantics_budget_and_exact_vocabulary():
    spec = {"name": "chain", "tokens": ["A", "B", "C"], "parent": ["A", "B", "C"], "num_mutations": 2}
    group = SequenceGroup(spec)
    schema = group.design_schema()
    assert schema["version"] == 2 and schema["action_protocol"] == "position_value_assignment"
    assert schema["max_steps"] == 2
    restored = ActionEncoding.from_schema(json.loads(json.dumps(schema["actions"])))
    assert restored.to_schema() == group.encoding.to_schema()
    assert sum(np.prod(block["shape"]) for block in schema["state"]["blocks"]) == len(group.current_state_features())
    for change in ({"tokens": ["B", "A", "C"]}, {"parent": ["B", "A", "C"]},
                   {"num_mutations": 3}, {"allow_repeats": False},
                   {"count_constraints": [{"tokens": ["A"], "max": 1}]}):
        assert SequenceGroup({**spec, **change}).design_schema() != schema
    detached = deepcopy(schema)
    schema["initial"][0] = "changed"
    schema["state"]["tokens"].clear()
    assert group.design_schema() == detached
    group.cation_fractions()[0] = "changed"
    assert group.cation_fractions() == ["A", "B", "C"]
