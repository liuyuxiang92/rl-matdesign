"""Exercise unique episode budgets with real DQN replay and optimizer updates."""
from __future__ import annotations

import collections
import copy

import numpy as np
import pytest
import torch

from rl_matdesign.env import EpisodeStep
from rl_matdesign.training import CandidateEvaluationFailed, train_dqn_online


class ScriptedEnv:
    """Choose prescribed terminal candidates while retaining two-step RL paths."""

    n_components = 2

    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.starts = 0
        self.counter = 0
        self.path = []

    def initialize(self):
        self.candidate = next(self.candidates)
        self.starts += 1
        self.counter = 0
        self.path = []

    def current_state_features(self):
        return np.array([float(self.counter), 1.0])

    def allowed_actions(self):
        return [((1.0,), (1.0,))]

    def sample_random_action(self):
        return self.allowed_actions()[0]

    def step(self, action):
        if self.counter == 1 and self.candidate == "failed":
            raise CandidateEvaluationFailed("deliberate failed evaluation")
        reward = (ord(self.candidate) - ord("A") + 1) if self.counter == 1 else 0
        self.path.append(EpisodeStep(
            state_material_features=self.current_state_features(),
            state_step_onehot=tuple(float(i == self.counter) for i in range(2)),
            action_elem_onehot=action[0],
            action_comp_onehot=action[1],
            reward=float(reward),
            allowed_actions=self.allowed_actions(),
        ))
        self.counter += 1

    def terminal_comp_key(self):
        assert self.counter == self.n_components
        return (("group", ((self.candidate, 1),)),)


def run(env, tmp_path, **overrides):
    kwargs = dict(
        env=env,
        elem_feats_scaled=np.array([[0.1, 0.2]]),
        fraction_set=["0.5"],
        device=torch.device("cpu"),
        hidden_dim=4,
        n_warmup_eps=2,
        dqn_num_train_eps=2,
        dqn_buffer_size=100,
        batch_size=1,
        dqn_grad_steps_per_ep=1,
        dqn_target_update_freq=2,
        dqn_eps_anneal_eps=10,
        checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 1},
    )
    kwargs.update(overrides)
    return train_dqn_online(**kwargs)


def checkpoint(tmp_path):
    return torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)


def optimizer_steps(state):
    return {int(item["step"].item()) for item in state["opt_state"]["state"].values()}


def reconstruct_resume(state, qnet, scaler):
    target = copy.deepcopy(qnet)
    target.load_state_dict(state["target_net_state"])
    qnet.load_state_dict(state["qnet_state"])
    optimizer = torch.optim.Adam(qnet.parameters())
    optimizer.load_state_dict(state["opt_state"])
    return {
        "scaler": scaler,
        "qnet": qnet,
        "target_net": target,
        "optimizer": optimizer,
        "buffer": collections.deque(state["buffer"], maxlen=state["dqn_buffer_size"]),
        "start_ep": state["episodes_completed"],
        "training_episodes_completed": state["training_episodes_completed"],
        "training_budget_state": state["training_budget_state"],
        "eps": state["eps"],
    }


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_unique_budget_trains_and_buffers_all_duplicate_rollouts(tmp_path, target_mode):
    env = ScriptedEnv(["A", "A", "B", "A", "C", "C", "D"])
    _, _, rows = run(env, tmp_path, train_count_duplicates=False, dqn_target_mode=target_mode)
    assert env.starts == 7
    assert [row["phase"] for row in rows] == ["dqn_warmup"] * 3 + ["dqn_train"] * 4
    assert [row["budget_counted"] for row in rows] == [True, False, True, False, True, False, True]
    assert rows[-1]["training_attempts"] == 7
    assert rows[-1]["training_budget_used"] == 4
    assert rows[-1]["training_unique_candidates"] == 4
    assert rows[-1]["training_duplicate_episodes"] == 3
    assert [row["iteration"] for row in rows[3:]] == [1, 2, 3, 4]
    assert rows[-1]["epsilon"] == pytest.approx(0.7)
    assert all(np.isfinite(row["train_loss"]) for row in rows[3:])
    state = checkpoint(tmp_path)
    assert len(state["buffer"]) == 14
    assert [row["reward"] for row in state["buffer"][1::2]] == [1, 1, 2, 1, 3, 3, 4]
    assert optimizer_steps(state) == {4}
    assert state["episodes_completed"] == 2
    assert state["training_episodes_completed"] == 4
    assert state["training_budget_state"]["counted"] == 4
    assert len(state["training_budget_state"]["seen_keys"]) == 4
    # Earlier steps have real bootstrap targets and discounted MC returns.
    assert all(not row["done"] for row in state["buffer"][::2])
    assert state["buffer"][0]["mc_return"] == pytest.approx(0.9)


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_default_counts_duplicates_and_preserves_fixed_number_of_updates(tmp_path, target_mode):
    env = ScriptedEnv(["A"] * 4)
    _, _, rows = run(env, tmp_path, train_max_duplicate_attempts=1, dqn_target_mode=target_mode)
    assert env.starts == 4
    assert len(rows) == 4
    assert all(row["budget_counted"] for row in rows)
    assert rows[-1]["training_budget_used"] == 4
    assert rows[-1]["training_unique_candidates"] == 1
    assert rows[-1]["training_duplicate_episodes"] == 3
    assert optimizer_steps(checkpoint(tmp_path)) == {2}


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_resume_restores_seen_candidates_and_actual_optimizer_schedule(tmp_path, target_mode):
    env = ScriptedEnv(["A", "A", "B", "A", "C"])
    qnet, scaler, _ = run(
        env, tmp_path, train_count_duplicates=False, dqn_target_mode=target_mode,
        dqn_num_train_eps=1,
    )
    initial = checkpoint(tmp_path)
    assert (initial["episodes_completed"], initial["training_episodes_completed"]) == (1, 2)
    resumed_env = ScriptedEnv(["A", "C", "D"])
    _, _, rows = run(
        resumed_env, tmp_path, train_count_duplicates=False, dqn_target_mode=target_mode,
        resume_state=reconstruct_resume(initial, qnet, scaler),
    )
    assert resumed_env.starts == 3
    assert [row["budget_counted"] for row in rows] == [False, False, True]
    assert [row["iteration"] for row in rows] == [3, 4, 5]
    assert rows[-1]["training_attempts"] == 8
    assert rows[-1]["training_budget_used"] == 4
    assert rows[-1]["epsilon"] == pytest.approx(0.6)
    resumed = checkpoint(tmp_path)
    assert len(resumed["buffer"]) == 16
    assert optimizer_steps(resumed) == {5}
    assert (resumed["episodes_completed"], resumed["training_episodes_completed"]) == (2, 5)


def test_completed_resume_does_not_sample_or_train_again(tmp_path):
    qnet, scaler, _ = run(
        ScriptedEnv(["A", "B", "C"]), tmp_path,
        train_count_duplicates=False, dqn_num_train_eps=1,
    )
    initial = checkpoint(tmp_path)
    env = ScriptedEnv([])
    _, _, rows = run(
        env, tmp_path, train_count_duplicates=False, dqn_num_train_eps=1,
        resume_state=reconstruct_resume(initial, qnet, scaler),
    )
    assert env.starts == 0
    assert rows == []
    assert optimizer_steps(checkpoint(tmp_path)) == {1}


def test_zero_online_budget_only_collects_warmup(tmp_path):
    env = ScriptedEnv(["A", "A", "B"])
    _, _, rows = run(env, tmp_path, train_count_duplicates=False, dqn_num_train_eps=0)
    assert env.starts == 3
    assert len(rows) == 3
    assert all(row["phase"] == "dqn_warmup" for row in rows)
    assert rows[-1]["training_budget_used"] == 2


def test_failed_evaluations_do_not_enter_budget_buffer_or_optimizer(tmp_path):
    env = ScriptedEnv(["failed", "A", "B", "failed", "A", "C"])
    _, _, rows = run(env, tmp_path, train_count_duplicates=False, dqn_num_train_eps=1)
    assert env.starts == 6
    assert len(rows) == 4
    assert rows[-1]["training_attempts"] == 4
    assert rows[-1]["training_budget_used"] == 3
    assert rows[-1]["training_duplicate_episodes"] == 1
    state = checkpoint(tmp_path)
    assert len(state["buffer"]) == 8
    assert optimizer_steps(state) == {2}
    assert all("failed" not in str(key) for key in state["training_budget_state"]["seen_keys"])


def test_duplicate_guard_preserves_last_allowed_duplicate_update_and_checkpoint(tmp_path):
    env = ScriptedEnv(["A", "A", "A"])
    with pytest.raises(RuntimeError, match="2 consecutive duplicate"):
        run(
            env, tmp_path, train_count_duplicates=False,
            train_max_duplicate_attempts=2, n_warmup_eps=1, dqn_num_train_eps=1,
        )
    state = checkpoint(tmp_path)
    assert env.starts == 3
    assert optimizer_steps(state) == {2}
    assert len(state["buffer"]) == 6
    assert state["episodes_completed"] == 0
    assert state["training_episodes_completed"] == 2
    assert state["training_budget_state"]["consecutive_duplicates"] == 2


def test_legacy_checkpoint_cannot_resume_unique_mode_without_seen_history(tmp_path):
    with pytest.raises(ValueError, match="checkpoint has no training_budget_state"):
        run(ScriptedEnv([]), tmp_path, train_count_duplicates=False, resume_state={})
