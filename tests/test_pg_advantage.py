"""Exercise normalized entropy and reward-unit invariance through real PG code."""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from rl_matdesign.domain_runtime import EpisodeStep
from rl_matdesign.training import _episode_pg_terms, train_pg
from tests.test_pg_duplicate_budget import _setup


class LogitPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, state, step, elements, fractions):
        return elements @ self.logits


def _terms(logits, *, selected=0, allowed_indices=None):
    policy = LogitPolicy(logits)
    allowed_indices = list(range(len(logits))) if allowed_indices is None else allowed_indices
    elements = np.eye(len(logits))
    actions = [(elements[i], np.ones(1)) for i in allowed_indices]
    action = actions[selected]
    step = EpisodeStep(
        state_material_features=np.zeros(1), state_step_onehot=np.ones(1),
        action_elem_onehot=action[0], action_comp_onehot=action[1], reward=2.,
        allowed_actions=actions,
    )
    terms = _episode_pg_terms(
        path=[step], returns=[2.], policy=policy, value_net=None,
        scaler=StandardScaler().fit([[0.], [1.]]), elem_feats_scaled=elements,
        action_values=['1.0'], device=torch.device('cpu'),
    )
    return policy, terms


@pytest.mark.parametrize('size', [2, 4, 20])
def test_actor_loss_entropy_is_normalized_for_actual_legal_count(size):
    _, (logp, advantages, entropies, critic) = _terms([0.] * size)
    assert len(entropies) == 1
    assert entropies[0].item() == pytest.approx(1.)
    assert logp[0].item() == pytest.approx(-math.log(size))
    assert advantages == [2.]
    assert critic == []


def test_actor_entropy_and_log_probability_use_identical_mask():
    # Invalid action's enormous logit must not influence either calculation.
    policy, (logp, _, entropies, _) = _terms([0., math.log(3.), 100.], allowed_indices=[0, 1])
    assert logp[0].item() == pytest.approx(math.log(.25))
    expected = -(.25 * math.log(.25) + .75 * math.log(.75)) / math.log(2.)
    assert entropies[0].item() == pytest.approx(expected)
    (-entropies[0]).backward()
    assert policy.logits.grad[2].item() == 0.
    assert policy.logits.grad[0].item() < 0 < policy.logits.grad[1].item()


def test_forced_decisions_have_no_entropy_observation_or_bonus():
    policy, (logp, _, entropies, _) = _terms([3.])
    assert entropies == []
    assert logp[0].item() == 0.
    (-logp[0]).backward()
    assert policy.logits.grad.item() == 0.


@pytest.mark.parametrize('scale,offset', [(1., 0.), (.001, 0.), (100., 0.), (1., 273.15)])
def test_actual_one_step_actor_updates_ignore_positive_reward_unit_changes(monkeypatch, scale, offset):
    def run(factor, shift):
        kwargs, _, _ = _setup(monkeypatch, [0, 1, 2, 3])
        kwargs['env'].reward_fn = lambda formula: factor * len(formula) + shift
        kwargs.update(num_iters=1, batch_eps=4)
        rows = train_pg(**kwargs, train_count_duplicates=True)
        return kwargs['policy'].logits.detach(), rows
    reference, _ = run(1., 0.)
    actual, rows = run(scale, offset)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    assert all('repeat_penalty' not in row and 'return_shaped' not in row for row in rows)


def test_uniform_actor_with_constant_rewards_remains_finite_without_shaping(monkeypatch):
    kwargs, _, _ = _setup(monkeypatch, [0, 0, 0, 0])
    kwargs['env'].reward_fn = lambda _: -530.574
    with torch.no_grad():
        kwargs['policy'].logits.zero_()
    rows = train_pg(**kwargs, train_count_duplicates=True)
    assert torch.equal(kwargs['policy'].logits, torch.zeros(4))
    assert all(np.isfinite(row['actor_loss']) for row in rows if row['phase'] == 'pg_train')
    assert all(row['adv_std_raw'] == 0 for row in rows if row['phase'] == 'pg_train')


@pytest.mark.parametrize('record,expected', [
    ({'gate_passed': True, 'primary_ok': False}, False),
    ({'gate_passed': True, 'constraint_passed': False}, False),
    ({'gate_passed': True, 'gate_custom_passed': False}, False),
    ({'gate_passed': None}, None),
    ({'prediction_record_error': 'diagnostics unavailable'}, None),
    ({}, True),
])
def test_exploration_observation_combines_known_scientific_constraints(record, expected):
    from types import SimpleNamespace
    from rl_matdesign.training import _exploration_observation
    episode = SimpleNamespace(prediction_record=record, terminal_key=('sample', 1),
                              path=[SimpleNamespace(reward=5.)])
    observation = _exploration_observation(episode)
    assert observation['feasible'] is expected
    assert observation['quality'] == 5.


def test_extreme_finite_logits_have_finite_entropy_and_gradients():
    policy, (logp, _, entropies, _) = _terms([3e38, -3e38])
    loss = -logp[0] - entropies[0]
    assert torch.isfinite(loss).item()
    loss.backward()
    assert torch.isfinite(policy.logits.grad).all().item()
    assert entropies[0].item() == 0.


def test_hpo_postconstraint_result_prevents_controller_feasibility_credit():
    from types import SimpleNamespace
    from rl_matdesign.training import _exploration_observation
    episode = SimpleNamespace(prediction_record={'gate_passed': True},
                              metadata={'post_constraint_passed': False},
                              terminal_key=('sample', 1), path=[SimpleNamespace(reward=5.)])
    assert _exploration_observation(episode)['feasible'] is False
