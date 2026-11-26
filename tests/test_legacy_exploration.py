"""The public false toggle restores historical algorithms, not fixed entropy."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from rl_matdesign import training
from rl_matdesign.domain_runtime import EpisodeStep
from rl_matdesign.gradient_calibration import calibrate_pg_gradients
from tests.test_dqn_duplicate_budget import ScriptedEnv, run, checkpoint, reconstruct_resume
from tests.test_pg_duplicate_budget import _setup
from tests.test_pg_advantage import LogitPolicy
from tests.test_gradient_calibration import _episodes, _inputs


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_false_dqn_keeps_historical_epsilon_lag_and_duplicate_clock(tmp_path, target_mode):
    env = ScriptedEnv(["A", "B", "A", "C", "C", "D"])
    _, _, rows = run(env, tmp_path, exploration_adaptive=False, dqn_target_mode=target_mode,
                     dqn_eps_min=.1, dqn_eps_anneal_eps=4)
    online = [row for row in rows if row['phase'] == 'dqn_train']
    assert [row['epsilon_used'] for row in online] == pytest.approx([1., 1., .75, .5])
    assert [row['epsilon'] for row in online] == pytest.approx([1., .75, .5, .25])
    assert [row['budget_counted'] for row in online] == [False, True, False, True]
    saved = checkpoint(tmp_path)
    assert saved['eps'] == .25
    assert saved['exploration_adaptive'] is False
    assert saved['exploration_state'] is None


def test_false_dqn_resume_restores_actual_epsilon_not_unique_budget(tmp_path):
    env = ScriptedEnv(["A", "B", "A", "C"])
    net, scaler, _ = run(env, tmp_path, exploration_adaptive=False,
                         dqn_num_train_eps=1, dqn_eps_min=.1, dqn_eps_anneal_eps=4)
    saved = checkpoint(tmp_path)
    resume = reconstruct_resume(saved, net, scaler)
    resume['eps'] = saved['eps']
    _, _, rows = run(ScriptedEnv(["D"]), tmp_path, exploration_adaptive=False,
                     resume_state=resume, dqn_num_train_eps=2,
                     dqn_eps_min=.1, dqn_eps_anneal_eps=4)
    assert rows[0]['epsilon_used'] == .75
    assert rows[0]['epsilon'] == .5


@pytest.mark.parametrize("include_none", [False, True])
@pytest.mark.parametrize("exact_flag", [None, True])
def test_false_dqn_exact_resume_requires_saved_epsilon_before_sampling(tmp_path, include_none, exact_flag):
    env = ScriptedEnv([])
    resume = {'eps': None} if include_none else {}
    if exact_flag is not None:
        resume['exact_resume'] = exact_flag
    with pytest.raises(ValueError, match='Exact legacy exploration resume requires eps'):
        run(env, tmp_path, exploration_adaptive=False, resume_state=resume,
            train_count_duplicates=True)
    assert env.starts == 0


def test_false_dqn_weights_only_resume_keeps_epsilon_fallback(tmp_path):
    net, scaler, _ = run(ScriptedEnv(['A', 'B', 'C']), tmp_path,
                         exploration_adaptive=False, dqn_num_train_eps=1,
                         dqn_eps_min=.1, dqn_eps_anneal_eps=4)
    resume = reconstruct_resume(checkpoint(tmp_path), net, scaler)
    resume.update(exact_resume=False, eps=None, exploration_adaptive=False)
    _, _, rows = run(ScriptedEnv(['D']), tmp_path, exploration_adaptive=False,
                     resume_state=resume, dqn_num_train_eps=2,
                     dqn_eps_min=.1, dqn_eps_anneal_eps=4)
    assert rows[0]['epsilon_used'] == .75


@pytest.mark.parametrize("adaptive", [False, True])
def test_dqn_resume_rejects_explicit_exploration_mode_mismatch_before_sampling(tmp_path, adaptive):
    env = ScriptedEnv([])
    with pytest.raises(ValueError, match='Checkpoint exploration_adaptive does not match'):
        run(env, tmp_path, exploration_adaptive=adaptive,
            resume_state={'exploration_adaptive': not adaptive}, train_count_duplicates=True)
    assert env.starts == 0


@pytest.mark.parametrize("field", ['entropy_coef_eff', 'visit_counts'])
@pytest.mark.parametrize("include_none", [False, True])
def test_false_pg_exact_resume_requires_saved_entropy_and_visits(monkeypatch, field, include_none):
    args, sampled, _ = _setup(monkeypatch, [])
    args['exploration_adaptive'] = False
    resume = {'is_resume': True, 'entropy_coef_eff': .02, 'visit_counts': {}}
    if include_none:
        resume[field] = None
    else:
        del resume[field]
    with pytest.raises(ValueError, match='Exact legacy exploration resume requires ' + field):
        training.train_pg(**args, train_count_duplicates=True, checkpoint_cfg=resume)
    assert sampled == []


@pytest.mark.parametrize("adaptive", [False, True])
def test_pg_resume_rejects_explicit_exploration_mode_mismatch_before_sampling(monkeypatch, adaptive):
    args, sampled, _ = _setup(monkeypatch, [])
    args['exploration_adaptive'] = adaptive
    with pytest.raises(ValueError, match='Checkpoint exploration_adaptive does not match'):
        training.train_pg(**args, train_count_duplicates=True,
                          checkpoint_cfg={'is_resume': True, 'exploration_adaptive': not adaptive})
    assert sampled == []


def _action_inputs(logits):
    model = LogitPolicy(logits)
    actions = [(np.eye(len(logits))[i], np.ones(1)) for i in range(len(logits))]
    return dict(model=model, device=torch.device('cpu'), s_material=np.zeros(1),
                s_step=np.ones(1), allowed_actions=actions, elem_feats_scaled=np.eye(len(logits)),
                action_values=['1.0'], exploration_adaptive=False)


def test_legacy_greedy_does_not_consume_sampler_rng_and_epsilon_shortcircuits(monkeypatch):
    args = _action_inputs([0., 2., 1.])
    monkeypatch.setattr(np.random, 'choice', lambda *a, **k: pytest.fail('greedy must not sample'))
    chosen = training.choose_action(**args)
    assert np.argmax(chosen[0]) == 1
    monkeypatch.setattr(args['model'], 'forward', lambda *a: pytest.fail('epsilon must skip Q'))
    monkeypatch.setattr(np.random, 'rand', lambda: .01)
    monkeypatch.setattr(np.random, 'randint', lambda n: n - 1)
    chosen = training.choose_action(**args, gen_epsilon=.1, gen_temperature=10.)
    assert np.argmax(chosen[0]) == 2


def test_legacy_temperature_uses_raw_scores_and_top_fraction_uses_round(monkeypatch):
    args = _action_inputs([0., 2., 1.])
    observed = []
    def sample(values, p=None):
        observed.append((values, p))
        return 1
    monkeypatch.setattr(np.random, 'choice', sample)
    training.choose_action(**args, gen_temperature=2.)
    expected = np.exp(np.array([0., 2., 1.]) / 2.)
    np.testing.assert_allclose(observed[-1][1], expected / expected.sum())
    training.choose_action(**args, gen_top_frac=.5)
    assert observed[-1][0] == [1, 2]  # round(.5 * 3) = 2
    assert observed[-1][1] is None


def test_legacy_pg_raw_entropy_includes_forced_steps():
    elements = np.eye(4)
    policy = LogitPolicy([2., 0., 0., 0.])
    path = []
    for n in (4, 2, 1):
        actions = [(elements[i], np.ones(1)) for i in range(n)]
        path.append(EpisodeStep(state_material_features=np.zeros(1), state_step_onehot=np.ones(1),
            action_elem_onehot=actions[0][0], action_comp_onehot=actions[0][1],
            reward=1., allowed_actions=actions))
    args = dict(path=path, returns=[1.]*3, policy=policy, value_net=None,
                scaler=StandardScaler().fit([[0.], [1.]]), elem_feats_scaled=elements,
                action_values=['1.0'], device=torch.device('cpu'))
    _, _, entropy, maximum, _ = training._episode_pg_terms(**args, exploration_adaptive=False)
    assert len(entropy) == 3 and entropy[-1].item() == 0.
    assert maximum == pytest.approx([math.log(4), math.log(2), 0.])
    assert entropy[0].dtype == torch.float32
    adaptive = training._episode_pg_terms(**args)
    assert len(adaptive[2]) == 2
    legacy_ratio = sum(float(h.detach()) for h in entropy) / sum(maximum)
    normalized_mean = np.mean([float(h.detach()) for h in adaptive[2]])
    assert abs(legacy_ratio - normalized_mean) > .01


def test_false_pg_feedback_affects_current_update_and_restores_repeat_penalties(monkeypatch, tmp_path):
    args, _, _ = _setup(monkeypatch, [0, 0, 1, 1])
    args.update(exploration_adaptive=False, pg_entropy_coef=.02, pg_entropy_min=.95,
                pg_repeat_penalty_coef=.3, pg_repeat_penalty_shape='log')
    with torch.no_grad():
        args['policy'].logits.copy_(torch.tensor([5., 0., 0., 0.]))
    monkeypatch.setattr(training, 'ExplorationController', lambda **k: pytest.fail('false must not create Step3 controller'))
    saved_path = tmp_path / 'pg.pt'
    rows = training.train_pg(**args, train_count_duplicates=True,
                            checkpoint_cfg={'path': str(saved_path), 'freq': 1,
                                            'entropy_coef_eff': None, 'visit_counts': None})
    updates = [r for r in rows if r['phase'] == 'pg_train']
    episodes = [r for r in rows if r['phase'] == 'pg_episode']
    expected = training.entropy_coef_update(.02, updates[0]['entropy_norm'], .02, .95)
    assert updates[0]['entropy_coef_eff'] == pytest.approx(expected)
    assert expected > .02
    assert [r['repeat_penalty'] for r in episodes] == pytest.approx([0., .3*math.log(2), 0., .3*math.log(2)])
    saved = torch.load(saved_path, weights_only=False)
    assert saved['exploration_state'] is None
    assert sorted(saved['visit_counts'].values()) == [2, 2]
    assert saved['entropy_coef_eff'] == updates[-1]['entropy_coef_eff']


def test_legacy_gradient_pilot_uses_raw_entropy_feedback_without_updating_live_actor():
    policy = LogitPolicy([5., 0.])
    before = policy.logits.detach().clone()
    inputs = _inputs()
    # LogitPolicy uses the two element features; its logits stay identical across pilot paths.
    _, report = calibrate_pg_gradients(policy=policy, value_net=None, episodes=_episodes(),
                exploration_adaptive=False, entropy_coef=.02, entropy_min=.95, **inputs)
    probs = torch.softmax(before, dim=0)
    raw_entropy = -(probs * torch.log_softmax(before, dim=0)).sum().item()
    expected = training.entropy_coef_update(.02, raw_entropy/math.log(2), .02, .95)
    assert report['actor_entropy_coefficient'] == pytest.approx(expected)
    assert report['actor_entropy_rule'] == 'legacy_raw_entropy_ratio_of_means_feedback'
    assert torch.equal(policy.logits.detach(), before)


def test_legacy_cannot_resume_step3_controller_state():
    with pytest.raises(ValueError, match='cannot resume a Step3'):
        training._restore_exploration(.7, False, {'profile': 'anything'}, resuming=True)
    assert training._restore_exploration(.7, False, None, resuming=True) is None
