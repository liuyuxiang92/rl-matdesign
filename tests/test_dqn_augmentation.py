"""DQN trajectory-permutation augmentation: correctness + dedup + validity rules."""
from __future__ import annotations

import collections
import math
import random
from typing import List, Tuple

import numpy as np
from rl_matdesign.action_features import value_alphabet
import pytest


# ============================================================================
# Helpers
# ============================================================================

def _oh(idx: int, n: int) -> Tuple[float, ...]:
    v = [0.0] * n
    v[idx] = 1.0
    return tuple(v)


def _build_env_simple(reward_fn=None):
    from rl_matdesign.env_multigroup import MultiGroupEnv
    return MultiGroupEnv(groups=[dict(name="cation", kind="composition",
        species_set=["Fe", "Co", "Ni", "Mn", "Cr"],
        amount_set=[i / 20 for i in range(1, 9)], n_steps=5, total=1, unique=True)],
        reward_fn=reward_fn or (lambda _: 17.0), state_featurizer=lambda _: np.zeros(2))


def _build_oxide_env():
    from rl_matdesign.env_multigroup import MultiGroupEnv
    return MultiGroupEnv(groups=[dict(name="oxide", kind="composition",
        species_set=["O", "Fe", "Co", "Ni", "Mn", "Cr"], amount_set=list(range(10)),
        n_steps=5, element_bounds={"O": [1, 45]})], reward_fn=lambda _: 42.0,
        state_featurizer=lambda _: np.zeros(2))


def _walk(env, action_seq):
    env.initialize()
    inner = env._inners[0]
    for elem, amount in action_seq:
        action = next(action for action in env.allowed_actions()
                      if inner.species_set[int(np.argmax(action[0]))] == elem
                      and float(inner.action_values[int(np.argmax(action[1]))]) == float(amount))
        env.step(action)


def _make_elem_feats(env):
    """Tiny stand-in for the Magpie elem_feats_scaled array."""
    return np.eye(len(env.species_set), dtype=float)


# ============================================================================
# Baseline behavior: K=0 is byte-for-byte identical to add_episode_to_buffer
# ============================================================================

def test_k_zero_is_no_op_identical_to_original():
    """K=0 must produce exactly the same rows as the unaugmented path."""
    from rl_matdesign.training import (
        _augment_episode_in_buffer, add_episode_to_buffer,
    )

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    action_values = value_alphabet(env)
    elem_feats = _make_elem_feats(env)

    buf_a: collections.deque = collections.deque()
    add_episode_to_buffer(env.path, buf_a, elem_feats, action_values)

    buf_b: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf_b, elem_feats, action_values, K=0,
    )

    assert len(buf_a) == len(buf_b) == 5
    for ra, rb in zip(buf_a, buf_b):
        assert ra["a_elem_idx"] == rb["a_elem_idx"]
        assert ra["a_comp_val"] == rb["a_comp_val"]
        assert ra["reward"] == rb["reward"]
        assert ra["done"] == rb["done"]
        np.testing.assert_array_equal(ra["s_mat_raw"], rb["s_mat_raw"])


# ============================================================================
# Buffer row count with augmentation (bounded by (K+1)*N, dedup may drop some)
# ============================================================================

def test_buffer_row_count_within_bound():
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    action_values = value_alphabet(env)
    elem_feats = _make_elem_feats(env)

    K = 3
    N = 5
    buf: collections.deque = collections.deque()
    random.seed(0)
    _augment_episode_in_buffer(
        env, env.path, buf, elem_feats, action_values, K=K,
    )
    # With no last-position pin and 5 distinct elements, all 5! orderings exist.
    # Step 0 row varies by first element so augmented step-0 rows are NEW;
    # late-step rows may collide. Total is in [N, (K+1)*N].
    assert N <= len(buf) <= (K + 1) * N


# ============================================================================
# Terminal reward + bootstrap target invariance
# ============================================================================

def test_terminal_reward_matches_original_for_all_augmented_rows():
    """Every terminal-row reward across original + augmented copies equals R."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple(reward_fn=lambda _f: 5.5)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    action_values = value_alphabet(env)
    elem_feats = _make_elem_feats(env)

    buf: collections.deque = collections.deque()
    random.seed(1)
    _augment_episode_in_buffer(
        env, env.path, buf, elem_feats, action_values, K=4,
    )

    terminal_rewards = [r["reward"] for r in buf if r["done"]]
    # At least the original's terminal row is there; augmented terminal rows
    # are dedup'd when last position is pinned, but here there's no pin so
    # the augmented terminal rows differ in (s_mat, action) — survive dedup.
    assert terminal_rewards, "expected at least one terminal row"
    for r in terminal_rewards:
        assert r == pytest.approx(5.5)


def test_non_terminal_rows_have_zero_immediate_reward():
    """All non-terminal rows carry reward=0; only the terminal row gets R."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple(reward_fn=lambda _f: 7.7)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(2)
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=3,
    )
    for r in buf:
        if r["done"]:
            assert r["reward"] == pytest.approx(7.7)
        else:
            assert r["reward"] == 0.0


# ============================================================================
# ============================================================================

def test_oxygen_bound_does_not_pin_action_order():
    from rl_matdesign.training import _augment_episode_in_buffer
    env = _build_oxide_env()
    _walk(env, [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")])
    terminal_key = env.terminal_comp_key()
    buf = collections.deque()
    random.seed(3)
    _augment_episode_in_buffer(env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=119)
    terminal_actions = {row["a_elem_idx"] for row in buf if row["done"]}
    assert len(terminal_actions) > 1
    assert env.terminal_comp_key() == terminal_key
    assert env.terminal_cation_fractions()["oxide"]["O"] >= 1


def test_unique_species_augmentation_preserves_terminal_candidate(capsys):
    """Distinct species can be chosen in any feasible order."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple(reward_fn=lambda _: 11.0)
    seq = [("Fe", "0.30"), ("Co", "0.30"), ("Ni", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)
    terminal_key = env.terminal_comp_key()
    buf: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=5,
    )
    assert len(buf) > len(seq)
    assert env.terminal_comp_key() == terminal_key
    assert all(row["reward"] == 11.0 for row in buf if row["done"])
    assert "no-op" not in capsys.readouterr().out.lower()


# ============================================================================
# Dedup: within-episode duplicates are skipped
# ============================================================================

def test_dedup_within_episode_drops_duplicate_terminal_rows():
    """Equivalent state-action rows within augmentation are stored once."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_oxide_env()
    seq = [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(4)
    K = 3
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=K,
    )

    # No two row dicts within this single episode's augmentation pass share
    # the same (s_mat_raw, s_step, a_elem_idx, a_comp_val, done) fingerprint.
    fingerprints = set()
    for r in buf:
        fp = (
            tuple(np.round(r["s_mat_raw"], 6).tolist()),
            tuple(r["s_step"].tolist()),
            int(r["a_elem_idx"]),
            float(r["a_comp_val"]),
            bool(r["done"]),
        )
        assert fp not in fingerprints, (
            "found duplicate fingerprint within a single episode's "
            "augmentation pass — dedup should have skipped it"
        )
        fingerprints.add(fp)



def test_dedup_scope_is_per_episode_not_global():
    """Identical-state rows across two separate episode augmentations are kept,
    not deduped. They represent legitimate independent observations."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env_a = _build_oxide_env()
    seq = [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")]
    _walk(env_a, seq)

    env_b = _build_oxide_env()
    _walk(env_b, seq)  # second episode reaches the same multiset

    buf: collections.deque = collections.deque()
    random.seed(5)
    K = 0
    _augment_episode_in_buffer(
        env_a, env_a.path, buf, _make_elem_feats(env_a),
        value_alphabet(env_a), K=K,
    )
    rows_after_a = len(buf)
    _augment_episode_in_buffer(
        env_b, env_b.path, buf, _make_elem_feats(env_b),
        value_alphabet(env_b), K=K,
    )
    # Both episodes produced 5 rows even though they're row-for-row identical.
    assert len(buf) == 2 * rows_after_a == 10


# ============================================================================
# Augmentation truly produces new (state, action) coverage at early steps
# ============================================================================

def test_augmentation_produces_distinct_first_step_action():
    """With no last-position pin and K > 0, at least one augmented step-0 row
    differs from the original's step-0 action (new first-element coverage)."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(6)
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=3,
    )

    # Original step-0 has Fe (idx 0) as the first action.
    step0_actions = []
    for r in buf:
        # Step-0 rows have s_step == [1, 0, 0, 0, 0] (one-hot of step 1).
        if int(np.argmax(r["s_step"])) == 0 and r["a_comp_val"] > 0.0:
            step0_actions.append(r["a_elem_idx"])

    assert len(set(step0_actions)) >= 2, (
        "expected augmentation to produce at least one new first-step element"
    )


# ============================================================================
# K capping when alternatives_count < K
# ============================================================================

def test_K_capped_when_alternatives_run_out():
    from rl_matdesign.env_multigroup import MultiGroupEnv
    from rl_matdesign.training import _augment_episode_in_buffer
    env = MultiGroupEnv(groups=[dict(name="oxide", kind="composition", species_set=["Fe", "O"],
        amount_set=[1], n_steps=2, unique=True)], state_featurizer=lambda _: np.zeros(1))
    _walk(env, [("Fe", 1), ("O", 1)])
    buf = collections.deque()
    _augment_episode_in_buffer(env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=10)
    assert len(buf) == 4


def test_reward_fn_restored_after_augmentation():
    """env.reward_fn is the original callable after augmentation completes."""
    from rl_matdesign.training import _augment_episode_in_buffer

    sentinel = lambda f: 12.5
    env = _build_env_simple(reward_fn=sentinel)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    assert env.reward_fn is sentinel

    buf: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), value_alphabet(env), K=3,
    )
    assert env.reward_fn is sentinel, (
        "augmentation should restore env.reward_fn even on the happy path"
    )
