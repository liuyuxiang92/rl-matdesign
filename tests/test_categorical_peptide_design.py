"""Application-owned parent state and complete (target, replacement) actions."""
from copy import deepcopy

import numpy as np
import pytest

from rl_matdesign.categorical import CategoricalGroup
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.peptide_design import compile_groups


PARENT = ["A", "HCys", "F", "K", "V", "W", "W", "A", "P", "R", "b-HCys"]


def config(parent=True):
    return {
        "groups": [
            {"name": "n_terminal", "kind": "categorical", "choices": [
                {"name": "treatment", "values": ["H", "Ac"]}]},
            {"name": "peptide", "kind": "categorical",
             "labels": ["A", "F", "K", "V", "W", "W", "A", "P", "R"] if parent else list(range(1, 12)),
             "values": ["A", "F", "K", "V", "W", "P", "R", "G"], "n_steps": 5 if parent else 9},
            {"name": "thiols", "kind": "categorical",
             "labels": ["HCys", "b-HCys"] if parent else {"unassigned_from": "peptide"},
             "values": ["HCys", "C", "D-HCys"], "n_steps": 2},
        ],
        "structure": {"builder": "peptide", "parent": PARENT, "require_change": ["peptide"]}
        if parent else {"builder": "peptide", "length": 11},
    }


def make_env(cfg=None, reward=None):
    groups, adapter = compile_groups(cfg or config())
    return MultiGroupEnv(groups=groups, candidate_adapter=adapter, reward_fn=reward)


def action(group, label, value):
    li = group.species_set.index(label)
    code = next(code for candidate, code in group._slot_codes[li] if candidate == value)
    return (tuple(np.eye(len(group.species_set))[li]),
            tuple(np.eye(len(group.action_values))[group.action_values.index(code)]))


def choose(env, label, value):
    inner = env._active()
    env.step(env._to_union(action(inner, label, value), inner))


def decoded(group):
    return {(group.species_set[int(np.argmax(a[0]))], group.value_specs[int(np.argmax(a[1]))].value)
            for a in group.allowed_actions()}


def finish_parent(env, edits):
    choose(env, "treatment", "H")
    for label, value in edits:
        choose(env, label, value)
    choose(env, 2, "HCys")
    choose(env, 11, "b-HCys")


def test_parent_occurrences_are_immutable_and_five_edits_are_exact():
    received = []
    env = make_env(reward=lambda payload: received.append(deepcopy(payload)) or 2.0)
    assert env.max_steps == 8
    assert env._inners[1].species_set == [1, 3, 4, 5, 6, 7, 8, 9, 10]
    assert env._inners[2].species_set == [2, 11]
    assert env.candidate_adapter._sequence(env._raw_group_payload(), complete=True) == PARENT
    assert np.any(env.current_state_features())
    choose(env, "treatment", "H")
    ordinary = env._active()
    assert (7, "W") not in decoded(ordinary)
    choose(env, 1, "W")  # This new W must not retarget either original W.
    assert (7, "F") in decoded(ordinary)
    choose(env, 7, "F")
    assert all(label not in (1, 7) for label, _ in decoded(ordinary))
    for label in (3, 4, 5):
        choose(env, label, "G")
    assert env._active().name == "thiols"
    assert (2, "b-HCys") not in decoded(env._active())
    assert (11, "b-HCys") in decoded(env._active())
    choose(env, 2, "HCys")
    choose(env, 11, "b-HCys")
    assert env.is_terminal and env.counter == 8
    candidate = env.terminal_cation_fractions()
    assert set(candidate) == {"n_terminal", "peptide"}
    assert candidate["peptide"] == ["W", "HCys", "G", "G", "G", "W", "F", "A", "P", "R", "b-HCys"]
    assert sum(a != b for a, b in zip(candidate["peptide"], PARENT)) == 5
    assert received == [candidate] and env.path[-1].reward == 2.0
    env.initialize()
    assert env.candidate_adapter._sequence(env._raw_group_payload(), complete=True) == PARENT
    assert env.counter == 0


def test_order_commutes_but_final_positions_change_candidate_identity():
    edits = [(1, "G"), (3, "G"), (4, "G"), (5, "G"), (7, "F")]
    first, second, different = make_env(), make_env(), make_env()
    finish_parent(first, edits)
    finish_parent(second, list(reversed(edits)))
    finish_parent(different, edits[:-1] + [(6, "F")])
    assert first.terminal_comp_key() == second.terminal_comp_key()
    np.testing.assert_array_equal(first.current_state_features(), second.current_state_features())
    assert first.terminal_comp_key() != different.terminal_comp_key()


def test_novel_thiols_fill_only_the_two_unassigned_positions():
    env = make_env(config(parent=False))
    choose(env, "treatment", "Ac")
    for label in (11, 10, 9, 8, 6, 5, 4, 3, 1):
        choose(env, label, "A")
    assert env._active().name == "thiols"
    assert {label for label, _ in decoded(env._active())} == {2, 7}
    choose(env, 7, "C")
    choose(env, 2, "C")
    assert env.is_terminal and env.counter == 12
    assert env.terminal_cation_fractions() == {"n_terminal": {"treatment": "Ac"},
                                              "peptide": ["A", "C", "A", "A", "A", "A", "C", "A", "A", "A", "A"]}
    env.initialize()
    assert env._inners[2]._eligible == set(range(1, 12))


def test_partial_targets_keep_the_complete_parent_visible():
    cfg = config()
    cfg["groups"][1].update(labels=["A"], n_steps=1)
    env = make_env(cfg)
    assert env._inners[1].species_set == [1]
    assert env.candidate_adapter._sequence(env._raw_group_payload(), complete=True) == PARENT
    cfg["groups"][1]["labels"] = [8]
    assert make_env(cfg)._inners[1].species_set == [8]


@pytest.mark.parametrize("labels", [["A", "A", "A"], ["A", 8], [0], [12]])
def test_invalid_parent_selectors_fail_before_rollout(labels):
    cfg = config()
    cfg["groups"][1].update(labels=labels, n_steps=1)
    with pytest.raises(ValueError):
        make_env(cfg)


def test_overlap_and_unfillable_novel_stages_are_rejected():
    cfg = config()
    cfg["groups"][2]["labels"] = ["A", "b-HCys"]
    with pytest.raises(ValueError, match="overlapping"):
        make_env(cfg)
    cfg = config(False)
    cfg["groups"][2]["n_steps"] = 1
    with pytest.raises(ValueError, match="remaining"):
        make_env(cfg)


def test_generic_partial_categorical_stage_preserves_zero_and_typed_values():
    group = CategoricalGroup({"name": "fields", "labels": ["x", "y", "z"],
                              "values": [0, True, "0"], "n_steps": 2})
    initial = group.current_state_features()
    group.step(action(group, "x", 0))
    assert not np.array_equal(group.current_state_features(), initial)
    group.step(action(group, "y", "0"))
    assert group.is_terminal and group.cation_fractions() == {"x": 0, "y": "0"}
    assert group.allowed_actions() == []


def test_compilation_does_not_mutate_the_authored_configuration():
    cfg = config()
    before = deepcopy(cfg)
    make_env(cfg)
    assert cfg == before


def test_misspelled_parent_change_requirement_fails_before_rollout():
    cfg = config()
    cfg["structure"]["require_changes"] = cfg["structure"].pop("require_change")
    with pytest.raises(ValueError, match="Unsupported peptide structure fields.*require_changes"):
        make_env(cfg)


@pytest.mark.parametrize("field", ["value", "n_steps", "require_change"])
def test_unused_categorical_choice_fields_fail_before_rollout(field):
    with pytest.raises(ValueError, match=f"Unsupported categorical choice fields.*{field}"):
        CategoricalGroup({"name": "settings", "choices": [
            {"name": "temperature", "values": [300, 400], field: 1}]})


@pytest.mark.parametrize("removed", ["element", "sites", "n_components", "total_units", "episode_style"])
def test_removed_categorical_parameters_are_rejected(removed):
    spec = {"name": "settings", "choices": [{"name": "temperature", "values": [300]}]}
    if removed == "element":
        spec["choices"][0][removed] = "O"
    else:
        spec[removed] = 1
    with pytest.raises(ValueError, match="Unsupported categorical"):
        CategoricalGroup(spec)


def test_categorical_list_api_and_removed_group_kinds_are_rejected():
    with pytest.raises(ValueError, match="mapping"):
        CategoricalGroup([{"name": "x", "values": [0]}])
    for kind in ("sequence", "integer"):
        with pytest.raises(ValueError, match="Unknown group kind"):
            MultiGroupEnv(groups=[{"name": "removed", "kind": kind}])
