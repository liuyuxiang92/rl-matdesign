"""Parent sequence replacement actions and exact terminal feasibility."""
import json
from itertools import combinations, product

import numpy as np
import pytest

from rl_matdesign.sequence import SequenceGroup


def action(env, kind, value):
    name = json.dumps([kind, value], separators=(",", ":"))
    assert name in env.species_set
    return (tuple(float(token == name) for token in env.species_set), (1.0,))


def allowed(env):
    return {tuple(json.loads(env.species_set[int(np.argmax(a[0]))])) for a in env.allowed_actions()}


def edit(env, position, token):
    env.step(action(env, "position", position))
    env.step(action(env, "token", token))


def test_two_decisions_per_edit_and_full_parent_state():
    env = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B", "A"], "num_mutations": 2})
    assert env.length == 3
    assert env.n_components == 4
    assert env.cation_fractions() == ["A", "B", "A"]
    before = env.current_state_features().copy()
    assert before.shape == (15,)
    assert before.sum() == 3
    assert allowed(env) == {("position", 1), ("position", 2), ("position", 3)}
    env.step(action(env, "position", 1))
    assert env.cation_fractions() == ["A", "B", "A"]
    assert env.current_state_features()[-3:].tolist() == [1, 0, 0]
    assert allowed(env) == {("token", "B"), ("token", "C")}
    env.step(action(env, "token", "C"))
    assert env.current_state_features()[-6:].tolist() == [1, 0, 0, 0, 0, 0]
    assert allowed(env) == {("position", 2), ("position", 3)}
    edit(env, 2, "A")
    assert env.cation_fractions() == ["C", "A", "A"]
    assert env.terminal_comp_key() == ("C", "A", "A")
    assert env.allowed_actions() == []
    env.initialize()
    assert env.counter == 0
    np.testing.assert_array_equal(env.current_state_features(), before)
    assert env.cation_fractions() == ["A", "B", "A"]


def test_replacement_is_chosen_independently_and_can_repeat_at_different_sites():
    env = SequenceGroup({"tokens": ["X", "Y"], "parent": ["A", "B", "C"], "num_mutations": 2})
    edit(env, 1, "X")
    assert allowed(env) == {("position", 2), ("position", 3)}
    edit(env, 3, "X")
    assert env.cation_fractions() == ["X", "B", "X"]
    env.initialize()
    edit(env, 3, "Y")
    edit(env, 1, "X")
    assert env.cation_fractions() == ["X", "B", "Y"]


def test_final_identity_does_not_depend_on_edit_order():
    env = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B", "A"], "num_mutations": 2})
    edit(env, 1, "C")
    edit(env, 3, "B")
    key, features = env.terminal_comp_key(), env.current_state_features()
    env.initialize()
    edit(env, 3, "B")
    edit(env, 1, "C")
    assert env.terminal_comp_key() == key
    np.testing.assert_array_equal(env.current_state_features(), features)


@pytest.mark.parametrize("first,second", [(1, 3), (3, 1)])
def test_count_constraint_can_be_temporarily_violated_to_move_membership(first, second):
    env = SequenceGroup({
        "tokens": ["A", "C"], "parent": ["C", "C", "A"], "num_mutations": 2,
        "count_constraints": [{"tokens": ["C"], "min": 2, "max": 2}],
    })
    replacements = {1: "A", 3: "C"}
    edit(env, first, replacements[first])
    assert env.cation_fractions().count("C") != 2
    assert ("position", second) in allowed(env)
    edit(env, second, replacements[second])
    assert env.cation_fractions() == ["A", "C", "C"]


def test_parent_only_names_can_be_retained_and_constrained_but_not_inserted():
    env = SequenceGroup({
        "tokens": ["A", "B"], "parent": ["legacy", "A", "A"], "num_mutations": 1,
        "count_constraints": [{"tokens": ["legacy"], "min": 1, "max": 1}],
    })
    assert allowed(env) == {("position", 2), ("position", 3)}
    edit(env, 2, "B")
    assert env.cation_fractions() == ["legacy", "B", "A"]
    assert env.current_state_features().shape == (15,)
    schema = env.design_schema()
    assert schema["tokens"] == ["A", "B"]
    assert schema["parent"] == ["legacy", "A", "A"]
    assert schema["length"] == 3
    assert schema["num_mutations"] == 1
    assert schema["action_order"] == ["position", "replacement"]
    schema["parent"].clear()
    assert env.design_schema()["parent"] == ["legacy", "A", "A"]


def test_unique_final_sequence_can_swap_tokens_through_a_duplicate_intermediate():
    env = SequenceGroup({"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 2, "allow_repeats": False})
    edit(env, 1, "B")
    assert env.cation_fractions() == ["B", "B"]
    edit(env, 2, "A")
    assert env.cation_fractions() == ["B", "A"]


def test_unique_final_sequence_can_repair_duplicate_parent_but_not_reuse_finalized_tokens():
    env = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "A", "A"], "num_mutations": 2, "allow_repeats": False})
    edit(env, 1, "B")
    env.step(action(env, "position", 2))
    assert allowed(env) == {("token", "C")}
    env.step(action(env, "token", "C"))
    assert env.cation_fractions() == ["B", "C", "A"]


@pytest.mark.parametrize("allow_repeats", [True, False])
@pytest.mark.parametrize("parent,tokens,n,constraints", [
    (["A", "B", "C"], ["A", "B", "C", "D"], 2, [
        {"tokens": ["A", "B"], "min": 1, "max": 2},
        {"tokens": ["B", "C"], "min": 2, "max": 2},
        {"tokens": ["A", "D"], "min": 1, "max": 1},
    ]),
    (["A", "A", "A"], ["A", "B", "C"], 2, []),
    (["legacy", "A", "B"], ["B", "C", "D"], 2, [
        {"tokens": ["legacy", "C"], "min": 1, "max": 1},
        {"tokens": ["B", "C"], "min": 1, "max": 2},
    ]),
    (["A", "B", "A", "C"], ["A", "B", "C", "D"], 3, [
        {"tokens": ["A", "B"], "min": 2, "max": 2},
        {"tokens": ["B", "C"], "min": 2, "max": 3},
    ]),
])
def test_every_reachable_mask_matches_exhaustive_final_sequence_oracle(parent, tokens, n, constraints, allow_repeats):
    valid = []
    for candidate in product(set(parent) | set(tokens), repeat=len(parent)):
        if sum(a != b for a, b in zip(parent, candidate)) != n:
            continue
        if any(new != old and new not in tokens for old, new in zip(parent, candidate)):
            continue
        if not allow_repeats and len(set(candidate)) != len(candidate):
            continue
        if not all(c["min"] <= sum(t in c["tokens"] for t in candidate) <= c["max"] for c in constraints):
            continue
        valid.append(candidate)
    assert valid
    env = SequenceGroup({"tokens": tokens, "parent": parent, "num_mutations": n, "allow_repeats": allow_repeats, "count_constraints": constraints})
    # All distinct partial finalized edits (not just one path through each
    # final sequence), including alternatives sharing a constraint signature.
    partials = {()}
    for candidate in valid:
        differences = tuple((i, token) for i, token in enumerate(candidate) if token != parent[i])
        for size in range(n + 1):
            partials.update(combinations(differences, size))
    for fixed in partials:
        env.initialize()
        for i, token in fixed:
            edit(env, i + 1, token)
        completions = [candidate for candidate in valid if all(candidate[i] == token for i, token in fixed)]
        edited = {i for i, _ in fixed}
        expected_positions = {
            ("position", i + 1) for i in range(len(parent)) if i not in edited
            and any(candidate[i] != parent[i] for candidate in completions)
        } if len(fixed) < n else set()
        assert allowed(env) == expected_positions, (fixed, completions)
        for _, position in expected_positions:
            env.initialize()
            for i, token in fixed:
                edit(env, i + 1, token)
            env.step(action(env, "position", position))
            expected_tokens = {
                ("token", candidate[position - 1]) for candidate in completions
                if candidate[position - 1] != parent[position - 1]
            }
            assert allowed(env) == expected_tokens, (fixed, position, completions)


@pytest.mark.parametrize("extra,match", [
    ({"parent": None}, "non-empty list"),
    ({"parent": []}, "non-empty list"),
    ({"parent": ["A", 1]}, "string IDs"),
    ({"num_mutations": True}, "integer"),
    ({"num_mutations": 1.5}, "integer"),
    ({"num_mutations": 0}, "between 1"),
    ({"num_mutations": 3}, "between 1"),
    ({"length": 3}, "parent length"),
])
def test_invalid_parent_edit_configuration_fails_explicitly(extra, match):
    with pytest.raises(ValueError, match=match):
        SequenceGroup({"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 1, **extra})


@pytest.mark.parametrize("spec", [
    {"tokens": ["A", "B"], "parent": ["A", "B"]},
    {"tokens": ["A", "B"], "num_mutations": 1},
])
def test_parent_and_edit_budget_are_required_together(spec):
    with pytest.raises(ValueError, match="provided together"):
        SequenceGroup(spec)


@pytest.mark.parametrize("spec", [
    {"tokens": ["A"], "parent": ["A"], "num_mutations": 1},
    {"tokens": ["A", "B"], "parent": ["A", "A"], "num_mutations": 2, "allow_repeats": False},
    {"tokens": ["A", "B"], "parent": ["A", "B"], "num_mutations": 1, "allow_repeats": False},
    {"tokens": ["A", "B"], "parent": ["A", "A"], "num_mutations": 1,
     "count_constraints": [{"tokens": ["A"], "min": 0, "max": 0}]},
])
def test_impossible_edit_spaces_fail_at_construction(spec):
    with pytest.raises(ValueError, match="no complete sequence"):
        SequenceGroup(spec)


def test_phase_noop_edited_position_and_malformed_actions_are_rejected_atomically():
    env = SequenceGroup({"tokens": ["A", "B", "C"], "parent": ["A", "B"], "num_mutations": 2})
    for bad in [action(env, "token", "C"), ((0.0,) * len(env.species_set), (1.0,)), (1.0, 0.0)]:
        with pytest.raises(ValueError):
            env.step(bad)
        assert env.counter == 0
        assert env.cation_fractions() == ["A", "B"]
    env.step(action(env, "position", 1))
    for bad in [action(env, "position", 2), action(env, "token", "A")]:
        with pytest.raises(ValueError, match="not allowed"):
            env.step(bad)
        assert env.counter == 1
        assert env.cation_fractions() == ["A", "B"]
    env.step(action(env, "token", "C"))
    with pytest.raises(ValueError, match="not allowed"):
        env.step(action(env, "position", 1))
    assert env.counter == 2
    edit(env, 2, "A")
    with pytest.raises(RuntimeError, match="termination"):
        env.step(action(env, "position", 1))


def test_arbitrary_token_names_cannot_collide_with_position_actions():
    weird = '["position",1]'
    env = SequenceGroup({"tokens": [weird, "A"], "parent": ["A"], "num_mutations": 1})
    assert len(set(env.species_set)) == 3
    edit(env, 1, weird)
    assert env.cation_fractions() == [weird]
