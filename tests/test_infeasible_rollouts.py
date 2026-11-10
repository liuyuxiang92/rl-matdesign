"""Rejected prefixes never reach an oracle, replay or successful budget."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign import training
from rl_matdesign.model import PolicyNet, QRegressor, ValueNet
from rl_matdesign.utils.rollout_errors import InfeasibleEpisode
from tests.test_cross_formula_training import Predictor, ScriptedEnv, run_dqn, saved


class PrefixEnv(ScriptedEnv):
    def allowed_actions(self):
        if self.candidate == "blocked" and self.counter == 1:
            return []
        return super().allowed_actions()


class RecordedPredictor(Predictor):
    def __init__(self, size):
        super().__init__(size)
        self.events = []
    def record_episode(self, event):
        self.events.append(event)
    def mark(self, label):
        pass
    def snapshot(self):
        return {}


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("mode", ["mc", "bootstrap"])
def test_dqn_discards_prefix_before_oracle_replay_and_budget(tmp_path, size, mode):
    predictor = RecordedPredictor(size)
    env = PrefixEnv(["blocked", "A", "B", "blocked", "C", "D", "E"], predictor)
    _, _, rows = run_dqn(env, tmp_path, dqn_target_mode=mode, timer=predictor,
                         checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 1})
    state = saved(tmp_path)
    assert state["training_budget_state"]["counted"] == 5
    assert len(state["buffer"]) == 10
    assert [r["terminal_reward"] for r in rows] == [1, 2, 3, 4, 5]
    assert predictor.scalar + [item for wave in predictor.waves for item in wave] == list("ABCDE")
    rejected = [event for event in predictor.events if event["status"] == "infeasible"]
    assert len(rejected) == 2
    assert all(event["eval_wall_share_s"] >= 0 for event in rejected)


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("method", ["a2c", "reinforce"])
def test_pg_discards_prefix_before_learning_and_budget(size, method):
    predictor = RecordedPredictor(size)
    env = PrefixEnv(["blocked", "A", "B", "blocked", "C"], predictor)
    policy = PolicyNet(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=4)
    value = ValueNet(state_dim=2, step_dim=2, hidden_dim=4) if method == "a2c" else None
    rows = training.train_pg(
        policy=policy, value_net=value, env=env, scaler=StandardScaler().fit([[0., 1.], [1., 1.]]),
        elem_feats_scaled=np.array([[.1, .2]]), fraction_set=["0.5"], device=torch.device("cpu"),
        num_iters=1, batch_eps=3, rl_method=method, timer=predictor,
    )
    episodes = [row for row in rows if row["phase"] == "pg_episode"]
    assert [row["terminal_reward"] for row in episodes] == [1, 2, 3]
    assert [row for row in rows if row["phase"] == "pg_train"][-1]["training_budget_used"] == 3
    assert predictor.scalar + [item for wave in predictor.waves for item in wave] == list("ABC")
    assert len([event for event in predictor.events if event["status"] == "infeasible"]) == 2


@pytest.mark.parametrize("size", [1, 4])
def test_generation_counts_rejected_attempts_without_query(size):
    predictor = RecordedPredictor(size)
    env = PrefixEnv(["blocked", "A", "B"], predictor)
    model = QRegressor(state_dim=2, step_dim=2, elem_dim=2, frac_dim=1, hidden_dim=4)
    rows = training.generate_candidates(
        env=env, predictor=predictor, scaler=StandardScaler().fit([[0., 1.], [1., 1.]]),
        device=torch.device("cpu"), elem_feats_scaled=np.array([[.1, .2]]), fraction_set=["0.5"],
        qnet=model, n_exploit=2, max_attempts=3,
    )
    assert len(rows) == 2 and env.starts == ["blocked", "A", "B"]
    assert predictor.scalar + [item for wave in predictor.waves for item in wave] == list("AB")


def test_scaler_warmup_retries_without_any_property_queries():
    predictor = Predictor(1)
    env = PrefixEnv(["blocked", "A", "B"], predictor)
    scaler = training._fit_scaler_from_warmup(env, 2)
    assert scaler.n_samples_seen_ == 4
    assert env.starts == ["blocked", "A", "B"]
    assert not predictor.scalar and not predictor.waves


@pytest.mark.parametrize("size", [1, 4])
def test_impossible_prefix_stops_after_bounded_retries(tmp_path, size):
    predictor = Predictor(size)
    env = PrefixEnv(["blocked"] * 20, predictor)
    with pytest.raises(RuntimeError, match="11 consecutive candidates"):
        run_dqn(env, tmp_path)
    assert len(env.starts) == 11
    assert not predictor.scalar and not predictor.waves


def test_programming_errors_are_not_reclassified_as_infeasible():
    def invalid_plugin():
        raise ValueError("Filter returned an action outside its input")
    with pytest.raises(ValueError, match="outside"):
        training._rollout_with_retry(invalid_plugin)
