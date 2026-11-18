"""Ordered peptide identity is assembled from categorical assignment stages."""
from copy import deepcopy
from itertools import combinations, product, permutations

import numpy as np
import pytest

from rl_matdesign.categorical import CategoricalGroup
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.peptide_design import compile_groups


def config(parent=None, budget=2):
    structure = {"builder": "peptide", "parent": parent, "require_change": ["peptide"]} if parent else {"builder": "peptide", "length": 3}
    ordinary = {"name": "peptide", "kind": "categorical", "labels": [1, 2, 3],
                "values": ["A", "B", "__STOP__", "1"], "n_steps": budget}
    groups = [{"name": "n_terminal", "kind": "categorical", "choices": [{"name": "treatment", "values": ["H", "Ac"]}]}, ordinary]
    if parent is None:
        groups.append({"name": "remaining", "kind": "categorical", "labels": {"unassigned_from": "peptide"}, "values": ["C"], "n_steps": 3-budget})
    return {"structure": structure, "groups": groups}


def env_for(cfg):
    groups, adapter = compile_groups(cfg)
    return MultiGroupEnv(groups=groups, candidate_adapter=adapter)


def action(group, label, value):
    li = group.species_set.index(label)
    code = next(code for candidate, code in group._slot_codes[li] if candidate == value)
    return (tuple(np.eye(len(group.species_set))[li]), tuple(np.eye(len(group.action_values))[group.action_values.index(code)]))


def choose(env, label, value):
    inner = env._active()
    env.step(env._to_union(action(inner, label, value), inner))


def test_exact_edit_budget_has_no_early_stop_and_reserved_looking_ids_are_values():
    env = env_for(config(["A", "B", "A"]))
    choose(env, "treatment", "H")
    choose(env, 1, "__STOP__")
    assert not env.is_terminal
    choose(env, 2, "1")
    assert env.is_terminal and env.counter == env.max_steps == 3
    assert env.terminal_cation_fractions()["peptide"] == ["__STOP__", "1", "A"]


def test_parent_and_novel_share_label_value_actions_but_expose_initial_state():
    parent = env_for(config(["A", "B", "A"]))
    novel = env_for(config())
    assert parent.candidate_adapter._sequence(parent._raw_group_payload(), complete=False) == ["A", "B", "A"]
    assert novel.candidate_adapter._sequence(novel._raw_group_payload(), complete=False) == [None]*3
    assert parent.candidate_adapter.state_feature_schema()["initial_sequence"] == ["A", "B", "A"]
    for env in (parent, novel):
        choose(env, "treatment", "H")
        choose(env, 1, "B")
        assert env._inners[1].assigned_labels == {1}


def test_noop_and_repeat_edit_fail_atomically_and_initialize_restores_parent():
    env = env_for(config(["A", "B", "A"]))
    choose(env, "treatment", "H")
    original = env.current_state_features().copy()
    with pytest.raises(ValueError):
        choose(env, 1, "A")
    np.testing.assert_array_equal(env.current_state_features(), original)
    choose(env, 1, "B")
    after = env.current_state_features().copy()
    with pytest.raises(ValueError):
        choose(env, 1, "A")
    np.testing.assert_array_equal(env.current_state_features(), after)
    env.initialize()
    assert env.candidate_adapter._sequence(env._raw_group_payload(), complete=True) == ["A", "B", "A"]


@pytest.mark.parametrize("vector", [(0, 0, 0), (1, 1, 0), (.5, .5, 0), (float("nan"), 0, 1), (1, 0), (2, 0, 0)])
def test_invalid_target_encoding_does_not_mutate_ordered_state(vector):
    env = env_for(config(["A", "B", "A"]))
    inner = env._inners[1]
    before = inner.current_state_features().copy()
    valid = action(inner, 1, "B")
    with pytest.raises(ValueError, match="one-hot"):
        inner.step((vector, valid[1]))
    np.testing.assert_array_equal(inner.current_state_features(), before)
    assert inner.counter == 0


def test_parent_only_identity_is_retained_at_its_original_target_only():
    cfg = config(["parent-only", "A", "B"])
    cfg["structure"]["require_change"] = []
    env = env_for(cfg)
    inner = env._inners[1]
    assert "parent-only" in dict(inner.slots)[1]
    assert all("parent-only" not in values for position, values in inner.slots if position != 1)


@pytest.mark.parametrize("parent", [None, ["A", "B", "A"]])
def test_plugin_count_range_matches_independent_enumeration(parent):
    cfg = config(parent)
    cfg["groups"][1].pop("labels")
    cfg["groups"][1].pop("values")
    cfg["groups"][1]["choices"] = [{"name": 1, "values": ["A", "B"]},
                                      {"name": 2, "values": ["A", "C"]},
                                      {"name": 3, "values": ["B", "C"]}]
    _, adapter = compile_groups(cfg)
    counts = []
    menus = {1: ["A", "B"], 2: ["A", "C"], 3: ["B", "C"]}
    for selected in combinations(range(1,4), 2):
        for values in product(*(menus[position] for position in selected)):
            if parent and any(value == parent[position-1] for position, value in zip(selected,values)):
                continue
            candidate = list(parent) if parent else ["C"]*3
            for position, value in zip(selected, values):
                candidate[position-1] = value
            counts.append(sum(token in {"A", "C"} for token in candidate))
    assert adapter._count_range({"A", "C"}) == (min(counts), max(counts))


def test_assignment_order_commutes_while_position_order_changes_identity():
    keys, states = [], []
    for edits in permutations([(1,"B"), (3,"1")]):
        env = env_for(config(["A", "B", "A"]))
        choose(env, "treatment", "H")
        for target, value in edits:
            choose(env, target, value)
        keys.append(env.terminal_comp_key())
        states.append(env.current_state_features())
    assert keys[0] == keys[1]
    np.testing.assert_array_equal(states[0], states[1])
    swapped = env_for(config(["A", "B", "A"]))
    choose(swapped, "treatment", "H")
    choose(swapped, 1, "1")
    choose(swapped, 3, "B")
    assert swapped.terminal_comp_key() != keys[0]
