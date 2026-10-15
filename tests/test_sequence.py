"""Ordered token identity and feasibility of generic sequence groups."""
from itertools import product

import numpy as np
import pytest

from rl_matdesign.sequence import SequenceGroup


def _action(env, token):
    return (tuple(float(value == token) for value in env.species_set), (1.0,))


def _allowed(env):
    return {env.species_set[int(np.argmax(action[0]))] for action in env.allowed_actions()}


def _walk(env, sequence):
    env.initialize()
    for token in sequence:
        env.step(_action(env, token))


def test_order_and_repetition_are_preserved_in_candidates_keys_and_features():
    env = SequenceGroup({"tokens": ["A", "B"], "length": 3})
    assert not env.current_state_features().any()
    _walk(env, ["A", "B", "A"])
    first_features = env.current_state_features()
    first_key = env.terminal_comp_key()
    assert env.cation_fractions() == ["A", "B", "A"]
    assert first_key == ("A", "B", "A")
    copy = env.cation_fractions()
    copy[0] = "B"
    assert env.cation_fractions()[0] == "A"
    _walk(env, ["B", "A", "A"])
    assert env.terminal_comp_key() != first_key
    assert not np.array_equal(first_features, env.current_state_features())
    assert env.allowed_actions() == []


def test_exact_count_bounds_force_remaining_positions_and_cannot_be_disabled():
    env = SequenceGroup({
        "tokens": ["A", "C"], "length": 11,
        "count_constraints": [{"tokens": ["C"], "min": 2, "max": 2}],
    })
    env.constraints_enabled = False
    _walk(env, ["A"] * 9)
    assert _allowed(env) == {"C"}
    env.step(_action(env, "C"))
    assert _allowed(env) == {"C"}
    env.step(_action(env, "C"))
    assert env.cation_fractions().count("C") == 2
    _walk(env, ["C", "C"])
    assert _allowed(env) == {"A"}
    with pytest.raises(ValueError, match="not allowed"):
        env.step(_action(env, "C"))
    assert env.counter == 2


@pytest.mark.parametrize("allow_repeats", [True, False])
def test_joint_overlap_mask_matches_exhaustive_completion_oracle(allow_repeats):
    tokens = ["A", "B", "C", "D"]
    constraints = [
        {"tokens": ["A", "B"], "min": 2, "max": 2},
        {"tokens": ["B", "C"], "min": 2, "max": 2},
        {"tokens": ["A", "D"], "min": 1, "max": 2},
    ]
    length = 4
    valid = [
        seq for seq in product(tokens, repeat=length)
        if (allow_repeats or len(set(seq)) == length)
        and all(c["min"] <= sum(token in c["tokens"] for token in seq) <= c["max"] for c in constraints)
    ]
    assert valid
    env = SequenceGroup({
        "tokens": tokens, "length": length,
        "allow_repeats": allow_repeats, "count_constraints": constraints,
    })
    for size in range(length):
        for prefix in {sequence[:size] for sequence in valid}:
            _walk(env, prefix)
            expected = {sequence[size] for sequence in valid if sequence[:size] == prefix}
            assert _allowed(env) == expected


@pytest.mark.parametrize("spec", [
    {"tokens": ["A", "B"], "length": 3, "count_constraints": [
        {"tokens": ["A"], "min": 2}, {"tokens": ["B"], "min": 2},
    ]},
    {"tokens": ["A", "B", "C"], "length": 3, "allow_repeats": False,
     "count_constraints": [{"tokens": ["C"], "min": 2, "max": 2}]},
    {"tokens": ["A", "B"], "length": 3, "allow_repeats": False},
])
def test_impossible_spaces_fail_at_construction(spec):
    with pytest.raises(ValueError, match="no complete sequence"):
        SequenceGroup(spec)


def test_csv_ids_keep_leading_zeros_and_selectors_use_explicit_annotations(tmp_path):
    csv_path = tmp_path / "tokens.csv"
    csv_path.write_text("id,has_free_thiol,smiles\n001,true,S\n002,false,CS\n003,false,C\n")
    env = SequenceGroup({
        "tokens": {"csv": str(csv_path), "id_column": "id"}, "length": 3,
        "count_constraints": [{"where": {"column": "has_free_thiol", "equals": "true"}, "min": 2, "max": 2}],
    })
    assert env.species_set == ["001", "002", "003"]
    assert env.token_metadata["002"]["smiles"] == "CS"
    schema = env.design_schema()
    assert schema == {
        "kind": "sequence", "tokens": ["001", "002", "003"],
        "length": 3, "allow_repeats": True,
        "count_constraints": [{"tokens": ["001"], "min": 2, "max": 2}],
    }
    schema["tokens"].reverse()
    assert env.design_schema()["tokens"] == ["001", "002", "003"]
    _walk(env, ["002"])
    assert _allowed(env) == {"001"}
    env.step(_action(env, "001"))
    env.step(_action(env, "001"))
    assert env.cation_fractions() == ["002", "001", "001"]


@pytest.mark.parametrize("text,match", [
    ("id,tag\n,yes\n", "missing token ID"),
    ("id,tag\n001,yes\n001,no\n", "duplicates token ID"),
    ("wrong,tag\n001,yes\n", "missing ID column"),
    ("id,id\n001,002\n", "unique, non-empty column"),
    ("id,tag\n001\n", "wrong number of fields"),
    ("id,tag\n", "no catalog entries"),
])
def test_csv_rejects_missing_duplicate_and_malformed_entries(tmp_path, text, match):
    path = tmp_path / "tokens.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        SequenceGroup({"tokens": {"csv": str(path), "id_column": "id"}, "length": 2})


@pytest.mark.parametrize("action", [
    ((0.0, 0.0), (1.0,)),
    ((1.0, 1.0), (1.0,)),
    ((0.3, 0.7), (1.0,)),
    ((float("nan"), 1.0), (1.0,)),
    ((1.0,), (1.0,)),
    ((1.0, 0.0), (0.0,)),
    ((1.0, 0.0), (1.0, 0.0)),
    (1.0, 0.0),
])
def test_strict_action_validation_is_atomic(action):
    env = SequenceGroup({"tokens": ["A", "B"], "length": 1})
    with pytest.raises(ValueError, match="one-hot"):
        env.step(action)
    assert env.counter == 0
    assert env.cation_fractions() == []
    env.step(_action(env, "A"))
    with pytest.raises(RuntimeError, match="termination"):
        env.step(_action(env, "B"))


@pytest.mark.parametrize("extra,match", [
    ({"filters": [{"type": "smact_charge"}]}, "do not support composition filters"),
    ({"constraint_filter": object()}, "do not support composition filters"),
    ({"length": 0}, "positive"),
    ({"length": 2.5}, "integer"),
    ({"allow_repeats": "false"}, "boolean"),
    ({"tokens": ["A", "A"]}, "duplicate"),
    ({"count_constraints": [{"tokens": ["Z"], "min": 1}]}, "unknown tokens"),
    ({"count_constraints": [{"tokens": ["A"], "min": 2, "max": 1}]}, "Count bounds"),
    ({"count_constraints": [{"where": {"column": "tag", "equals": "true"}, "min": 1}]}, "require a CSV"),
])
def test_invalid_configuration_fails_explicitly(extra, match):
    with pytest.raises(ValueError, match=match):
        SequenceGroup({"tokens": ["A", "B"], "length": 2, **extra})
