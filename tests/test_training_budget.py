"""Training budgets count canonical compositions independently of learning."""
from __future__ import annotations

import copy
import io

import pytest

from rl_matdesign.utils.training_budget import TrainingBudget


KEY_A = (("A", (("Ba", 32),)), ("B", (("Co", 26), ("Mn", 6))))
KEY_B = (("A", (("Ba", 32),)), ("B", (("Co", 24), ("Mn", 8))))


def test_default_budget_preserves_every_episode_and_tracks_duplicates():
    budget = TrainingBudget(max_duplicate_attempts=1)
    assert [budget.record(key) for key in (KEY_A, KEY_A, KEY_B, KEY_A)] == [True] * 4
    budget.check_can_sample()  # Default mode has no duplicate-streak guard.
    assert budget.snapshot() == {
        "training_attempts": 4,
        "training_budget_used": 4,
        "training_unique_candidates": 2,
        "training_duplicate_episodes": 2,
    }


def test_unique_budget_counts_composition_keys_across_training_phases():
    budget = TrainingBudget(count_duplicates=False)
    assert budget.record(KEY_A)  # Warmup.
    assert not budget.record(copy.deepcopy(KEY_A))  # Same composition in training.
    assert budget.record(KEY_B)
    assert not budget.record(KEY_A)
    assert (budget.attempted, budget.counted, budget.unique, budget.duplicates) == (4, 2, 2, 2)


def test_guard_allows_last_duplicate_to_train_before_next_sampling_fails():
    budget = TrainingBudget(count_duplicates=False, max_duplicate_attempts=2)
    budget.record(KEY_A)
    trained = []
    for key in (KEY_A, KEY_A):
        budget.check_can_sample()
        assert not budget.record(key)
        trained.append(key)
    assert len(trained) == 2
    assert budget.consecutive_duplicates == 2
    with pytest.raises(RuntimeError, match="2 consecutive duplicate episodes"):
        budget.check_can_sample()
    assert budget.counted == 1


def test_new_composition_resets_consecutive_duplicate_guard():
    budget = TrainingBudget(count_duplicates=False, max_duplicate_attempts=2)
    for key in (KEY_A, KEY_A, KEY_B, KEY_A):
        budget.check_can_sample()
        budget.record(key)
    assert budget.consecutive_duplicates == 1
    budget.check_can_sample()


@pytest.mark.parametrize("mode", ["true", "false", 0, 1, None, [], {}])
def test_counting_mode_requires_real_yaml_boolean(mode):
    with pytest.raises(ValueError, match="count_duplicates must be a YAML boolean"):
        TrainingBudget(count_duplicates=mode)


@pytest.mark.parametrize("limit", [0, -1, 1.5, "10", True, False, None])
def test_duplicate_limit_requires_positive_integer(limit):
    with pytest.raises(ValueError, match="max_duplicate_attempts"):
        TrainingBudget(max_duplicate_attempts=limit)


def test_unhashable_key_does_not_change_bookkeeping():
    budget = TrainingBudget(count_duplicates=False)
    before = budget.state_dict()
    with pytest.raises(ValueError, match="keys must be hashable"):
        budget.record(["Ba", 32])
    assert budget.state_dict() == before


@pytest.mark.parametrize("mode", [True, False])
def test_torch_checkpoint_roundtrip_preserves_nested_keys_and_counters(mode):
    torch = pytest.importorskip("torch")
    budget = TrainingBudget(count_duplicates=mode, max_duplicate_attempts=7)
    for key in (KEY_A, KEY_B, KEY_A):
        budget.record(key)
    buffer = io.BytesIO()
    torch.save(budget.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    resumed = TrainingBudget(count_duplicates=mode, max_duplicate_attempts=7, state=state)
    assert resumed.state_dict() == budget.state_dict()
    assert isinstance(state["seen_keys"][0], tuple)
    assert resumed.record(KEY_B) is mode
    assert resumed.duplicates == 2
    assert resumed.consecutive_duplicates == 2


def test_checkpoint_key_container_is_detached_from_live_budget():
    budget = TrainingBudget()
    budget.record(KEY_A)
    state = budget.state_dict()
    state["seen_keys"].append(KEY_B)
    assert budget.unique == 1
    resumed = TrainingBudget(state=budget.state_dict())
    resumed.record(KEY_B)
    assert budget.unique == 1


@pytest.mark.parametrize("mode", [True, False])
def test_nonempty_resume_rejects_changed_counting_mode(mode):
    budget = TrainingBudget(count_duplicates=mode)
    budget.record(KEY_A)
    with pytest.raises(ValueError, match="Cannot change train_count_duplicates"):
        TrainingBudget(count_duplicates=not mode, state=budget.state_dict())


def test_empty_resume_can_change_counting_mode():
    resumed = TrainingBudget(count_duplicates=False, state=TrainingBudget().state_dict())
    assert resumed.record(KEY_A)
    assert not resumed.record(KEY_A)


def test_resume_can_raise_duplicate_limit_without_losing_streak():
    budget = TrainingBudget(count_duplicates=False, max_duplicate_attempts=1)
    budget.record(KEY_A)
    budget.record(KEY_A)
    resumed = TrainingBudget(
        count_duplicates=False, max_duplicate_attempts=2, state=budget.state_dict()
    )
    resumed.check_can_sample()
    resumed.record(KEY_A)
    with pytest.raises(RuntimeError, match="2 consecutive"):
        resumed.check_can_sample()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2), ("version", True), ("count_duplicates", "false"),
        ("max_duplicate_attempts", 0), ("attempted", -1), ("counted", True),
        ("unique", 1.0), ("duplicates", "0"), ("consecutive_duplicates", None),
        ("seen_keys", {}), ("seen_keys", [["Ba", 32]]),
    ],
)
def test_restore_rejects_invalid_state_types(field, value):
    state = TrainingBudget().state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        TrainingBudget(state=state)


@pytest.mark.parametrize(
    "updates",
    [
        {"attempted": 2}, {"counted": 0}, {"unique": 2}, {"duplicates": 1},
        {"consecutive_duplicates": 1}, {"seen_keys": []},
        {"seen_keys": [KEY_A, KEY_A]},
        {"unique": 0, "seen_keys": [], "duplicates": 1},
    ],
)
def test_restore_rejects_inconsistent_history(updates):
    budget = TrainingBudget()
    budget.record(KEY_A)
    state = budget.state_dict()
    state.update(updates)
    with pytest.raises(ValueError):
        TrainingBudget(state=state)


def test_restore_validates_unique_mode_counted_counter():
    budget = TrainingBudget(count_duplicates=False)
    budget.record(KEY_A)
    budget.record(KEY_A)
    state = budget.state_dict()
    state["counted"] = 2
    with pytest.raises(ValueError, match="inconsistent"):
        TrainingBudget(count_duplicates=False, state=state)


@pytest.mark.parametrize("state", [{}, {"version": 1}, [], 1])
def test_restore_rejects_missing_or_nonmapping_state(state):
    with pytest.raises(ValueError, match="training budget state"):
        TrainingBudget(state=state)
