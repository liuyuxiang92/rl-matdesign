"""PG duplicate-budget integration with real losses and optimizer updates."""
from __future__ import annotations

import csv

import numpy as np
from rl_matdesign.action_features import value_alphabet
import pytest
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip("torch")

from rl_matdesign import training
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.model import ValueNet
from rl_matdesign.utils.training_samples import TrainingSampleWriter


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))

    def forward(self, _state, _step, elements, _fractions):
        return elements @ self.logits


class _Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state, _step):
        return self.value.expand(state.shape[0], 1)


def _setup(monkeypatch, sequence, *, method="reinforce"):
    """Control sampled actions, retaining real env episodes and gradient terms."""
    env = MultiGroupEnv(groups=[{'species_set': ["Na", "K", "Ca", "Sr"], 'amount_set': ["1.00"], 'n_steps': 1, 'name': 'composition', 'kind': 'composition', 'unique': True, 'total': 1}], state_featurizer=lambda _formula: np.array([0.0]), reward_fn=lambda formula: 1.0 + len(formula))
    samples = iter(sequence)
    sampled = []
    learned = []
    original_terms = training._episode_pg_terms

    def rollout(**kwargs):
        selection = next(samples)
        sampled.append(selection)
        if isinstance(selection, Exception):
            raise selection
        env.initialize()
        actions = env.allowed_actions()
        action = next(a for a in actions if int(np.argmax(a[0])) == selection)
        env.step(action)

    def terms(**kwargs):
        learned.append(int(np.argmax(kwargs["path"][-1].action_elem_onehot)))
        return original_terms(**kwargs)

    monkeypatch.setattr(training, "_rollout_pg_episode", rollout)
    monkeypatch.setattr(training, "_episode_pg_terms", terms)
    policy = _Policy()
    value_net = _Value() if method == "a2c" else None
    kwargs = dict(
        policy=policy, value_net=value_net, env=env,
        scaler=StandardScaler().fit([env.current_state_features()]),
        elem_feats_scaled=np.eye(4), action_values=value_alphabet(env),
        device=torch.device("cpu"), num_iters=2, batch_eps=2,
        exploration_adaptive=False, rl_method=method,
    )
    return kwargs, sampled, learned


@pytest.mark.parametrize("method", ["reinforce", "a2c"])
@pytest.mark.parametrize("count_duplicates", [True, False])
def test_duplicates_keep_learning_with_optional_budget(monkeypatch, method, count_duplicates):
    sequence = [0, 0, 1, 1, 2, 2, 3]
    kwargs, sampled, learned = _setup(monkeypatch, sequence, method=method)
    rows = training.train_pg(**kwargs, train_count_duplicates=count_duplicates)
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    updates = [r for r in rows if r["phase"] == "pg_train"]

    expected_samples = sequence[:4] if count_duplicates else sequence
    assert sampled == learned == expected_samples
    assert len(episodes) == len(expected_samples)
    assert [r["batch_eps_actual"] for r in updates] == (
        [2, 2] if count_duplicates else [2, 2, 2, 1]
    )
    assert updates[-1]["training_budget_used"] == 4
    assert updates[-1]["training_attempts"] == len(expected_samples)
    assert updates[-1]["training_unique_candidates"] == (2 if count_duplicates else 4)
    assert updates[-1]["training_duplicate_episodes"] == (2 if count_duplicates else 3)
    assert [r["budget_counted"] for r in episodes] == (
        [True] * 4 if count_duplicates else [True, False, True, False, True, False, True]
    )
    assert episodes[1]["return_raw"] == episodes[0]["return_raw"]
    assert all("repeat_penalty" not in row and "return_shaped" not in row for row in rows)
    assert not torch.equal(kwargs["policy"].logits.detach(), torch.tensor([1., 0., 0., 0.]))
    if method == "a2c":
        assert kwargs["value_net"].value.item() > 0


def test_explicit_duplicate_counting_has_no_duplicate_guard(monkeypatch):
    kwargs, sampled, learned = _setup(monkeypatch, [0, 0, 0, 0])
    rows = training.train_pg(**kwargs, train_count_duplicates=True, train_max_duplicate_attempts=1)
    assert sampled == learned == [0, 0, 0, 0]
    assert rows[-1]["training_budget_used"] == 4
    assert rows[-1]["training_duplicate_episodes"] == 3


def test_popart_a2c_uses_full_budget_after_prior_property_observations(monkeypatch, tmp_path):
    kwargs, sampled, learned = _setup(monkeypatch, [0, 0, 1, 2, 3], method="a2c")
    env = kwargs["env"]
    # Evaluated before RL, these candidates are still new to the learner.
    for index in (0, 1):
        env.initialize()
        action = next(a for a in env.allowed_actions() if int(np.argmax(a[0])) == index)
        env.step(action)
    kwargs["value_net"] = ValueNet(
        state_dim=kwargs["scaler"].n_features_in_,
        step_dim=len(env.path[0].state_step_onehot),
        value_normalization="popart",
    )
    samples = tmp_path / "training_samples.csv"
    rows = training.train_pg(
        **kwargs, checkpoint_cfg={"total_budget": 4},
        training_samples=TrainingSampleWriter(samples),
    )
    assert sampled == learned == [0, 0, 1, 2, 3]
    assert {r["phase"] for r in rows} == {"pg_episode", "pg_train"}
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    assert [r["budget_counted"] for r in episodes] == [True, False, True, True, True]
    assert [r["training_budget_used"] for r in episodes] == [1, 1, 2, 3, 4]
    updates = [r for r in rows if r["phase"] == "pg_train"]
    assert [r["episode"] for r in updates] == [2, 4, 5]
    assert updates[-1]["training_budget_used"] == 4
    assert updates[-1]["value_popart_targets_seen"] == 5
    with samples.open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert [int(r["training_budget_used"]) for r in exported] == [1, 2, 3, 4]
    assert [r["phase"] for r in exported] == ["pg_episode"] * 4


@pytest.mark.parametrize("stop", ["duplicate_limit", "attempt_limit"])
def test_limit_trains_and_checkpoints_partial_batch_before_raising(monkeypatch, tmp_path, stop):
    kwargs, sampled, learned = _setup(monkeypatch, [0, 0, 0, 0])
    kwargs.update(num_iters=1, batch_eps=4)
    checkpoint = tmp_path / "checkpoint.pt"
    limits = ({"train_max_duplicate_attempts": 2} if stop == "duplicate_limit"
              else {"max_train_attempts": 2})
    match = "consecutive duplicate" if stop == "duplicate_limit" else "budget is incomplete"
    with pytest.raises(RuntimeError, match=match):
        training.train_pg(
            **kwargs, train_count_duplicates=False,
            checkpoint_cfg={"path": str(checkpoint), "freq": 0}, **limits,
        )
    expected_attempts = 3 if stop == "duplicate_limit" else 2
    assert sampled == learned == [0] * expected_attempts
    assert not torch.equal(kwargs["policy"].logits.detach(), torch.tensor([1., 0., 0., 0.]))
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["training_budget_used"] == 1
    assert saved["training_attempts"] == expected_attempts
    assert saved["iteration"] == 1
    assert saved["training_budget_state"]["duplicates"] == expected_attempts - 1
    assert next(iter(saved["opt_actor_state"]["state"].values()))["step"].item() == 1


def test_failed_evaluation_is_retried_without_budget_or_learning(monkeypatch):
    failure = training.CandidateEvaluationFailed("temporary builder failure")
    kwargs, sampled, learned = _setup(monkeypatch, [failure, 0, 0, 1])
    kwargs.update(num_iters=1, batch_eps=2)
    rows = training.train_pg(**kwargs, train_count_duplicates=False)
    assert len(sampled) == 4
    assert learned == [0, 0, 1]
    assert rows[-1]["training_budget_used"] == 2
    assert rows[-1]["training_attempts"] == 3
    assert rows[-1]["training_duplicate_episodes"] == 1


def test_resume_keeps_seen_compositions_and_adds_requested_budget(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    kwargs, _, _ = _setup(monkeypatch, [0, 1])
    kwargs.update(num_iters=1, batch_eps=2)
    training.train_pg(
        **kwargs, train_count_duplicates=False,
        checkpoint_cfg={"path": str(checkpoint), "freq": 0},
    )
    saved = torch.load(checkpoint, weights_only=False)
    # Restore the real policy, optimizer, visits and composition budget; the
    # first two resumed rollouts are duplicates of the preceding invocation.
    kwargs, sampled, learned = _setup(monkeypatch, [0, 1, 2])
    kwargs["policy"].load_state_dict(saved["policy_state"])
    kwargs.update(num_iters=1, batch_eps=1)
    rows = training.train_pg(
        **kwargs, train_count_duplicates=False,
        checkpoint_cfg={
            "path": str(checkpoint), "freq": 1, "is_resume": True,
            "start_episode": saved["episodes_completed"],
            "start_iteration": saved["iteration"],
            "training_budget_state": saved["training_budget_state"],
            "opt_actor_state": saved["opt_actor_state"],
            "exploration_state": saved["exploration_state"],
        },
    )
    assert sampled == learned == [0, 1, 2]
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    assert [r["budget_counted"] for r in episodes] == [False, False, True]
    assert [r["training_budget_used"] for r in episodes] == [2, 2, 3]
    assert [r["iteration"] for r in episodes] == [2, 3, 4]
    assert [r["episode"] for r in episodes] == [3, 4, 5]
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed["training_attempts"] == 5
    assert resumed["training_budget_used"] == 3
    assert resumed["training_duplicate_episodes"] == 2
    assert resumed["iteration"] == 4


@pytest.mark.parametrize("resume_fields", [{"start_episode": 10}, {"is_resume": True}])
def test_legacy_checkpoint_cannot_enable_unique_budget(monkeypatch, resume_fields):
    kwargs, sampled, learned = _setup(monkeypatch, [0])
    with pytest.raises(ValueError, match="no training_budget_state"):
        training.train_pg(**kwargs, train_count_duplicates=False, checkpoint_cfg=resume_fields)
    assert sampled == learned == []


def test_legacy_checkpoint_without_exploration_state_is_rejected(monkeypatch):
    kwargs, sampled, learned = _setup(monkeypatch, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="Exact resume requires exploration_state"):
        training.train_pg(**kwargs, train_count_duplicates=True,
                          checkpoint_cfg={"start_episode": 10})
    assert sampled == learned == []


def test_paid_pilot_is_known_to_controller_but_new_to_learner_budget(monkeypatch):
    from types import SimpleNamespace
    kwargs, _, _ = _setup(monkeypatch, [0, 1])
    kwargs.update(num_iters=1, batch_eps=2)
    env = kwargs["env"]
    env.initialize()
    env.step(next(a for a in env.allowed_actions() if int(np.argmax(a[0])) == 0))
    pilot_key = env.terminal_comp_key()
    rows = training.train_pg(
        **kwargs, gradient_calibration_episodes=[SimpleNamespace(terminal_key=pilot_key)],
    )
    assert rows[-1]["training_budget_used"] == 2
    assert env.exploration_state["attempts"] == 2
    assert env.exploration_state["window"]["unique"] == 1
    assert repr(pilot_key) in env.exploration_state["seen_keys"]
