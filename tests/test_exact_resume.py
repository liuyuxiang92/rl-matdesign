"""Interruption resumes random streams, optimizer/replay and unique-cost history."""
import json
import random
import shutil

import numpy as np
import pytest
import torch
import yaml

from tests.test_experiment_reproducibility import _config, _popart_config, _run, _scientific_rows
from rl_matdesign.utils.seeding import capture_rng_state, restore_rng_state, isolated_rng
from rl_matdesign.utils.timing import PredictorTimer


def test_inference_cannot_consume_training_rng_or_depend_on_query_order():
    class Stochastic:
        def predict(self, candidate):
            return random.random() + np.random.random() + torch.rand(()).item(), None
    timer = PredictorTimer(Stochastic(), eval_seed=53)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(()).item())
    restore_rng_state(state)
    first = timer.predict({'Ni': .25, 'Fe': .75})
    timer.predict({'Co': 1})
    assert timer.predict({'Fe': .75, 'Ni': .25}) == first
    actual = (random.random(), np.random.random(), torch.rand(()).item())
    assert actual == expected
    with pytest.raises(RuntimeError):
        with isolated_rng(99):
            raise RuntimeError('model initialization failed')


@pytest.mark.parametrize('value_normalization', ['none', 'popart'])
@pytest.mark.parametrize('exploration_adaptive', [True, False])
@pytest.mark.parametrize('method,target,weights,mid', [
    ('dqn', 'mc', 'qnet.pt', 'checkpoint-ep2.pt'),
    ('dqn', 'bootstrap', 'qnet.pt', 'checkpoint-ep2.pt'),
    ('a2c', 'bootstrap', 'policy.pt', 'checkpoint-iter1.pt'),
])
def test_interrupted_runner_matches_uninterrupted_training(
    tmp_path, method, target, weights, mid, value_normalization, exploration_adaptive,
):
    cfg = _popart_config() if value_normalization == 'popart' else _config()
    cfg['value_normalization'] = value_normalization
    cfg['exploration_adaptive'] = exploration_adaptive
    if not exploration_adaptive:
        cfg.update({'dqn_eps_anneal_eps': 5, 'dqn_eps_min': .2} if method == 'dqn'
                   else {'pg_entropy_coef': .03, 'pg_entropy_min': .6})
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    _run(config, full, method, target, '--save-checkpoint-freq', '1')
    resumed.mkdir()
    shutil.copyfile(full / mid, resumed / 'checkpoint.pt')
    shutil.copyfile(full / 'std_scaler.bin', resumed / 'std_scaler.bin')
    if value_normalization == 'popart':
        shutil.copyfile(full / 'reward_calibration.json', resumed / 'reward_calibration.json')
    else:
        assert not (full / 'calibration_checkpoint.pt').exists()
        assert not (full / 'reward_calibration.json').exists()
    _run(config, resumed, method, target, '--resume-training', '--save-checkpoint-freq', '1')
    role = 'qnet_state' if method == 'dqn' else 'policy_state'
    a = torch.load(full / weights, weights_only=True)
    b = torch.load(resumed / weights, weights_only=True)
    assert all(torch.equal(a[role][key], b[role][key]) for key in a[role])
    value_state = a[role]
    if method == 'a2c':
        a = torch.load(full / 'value_net.pt', weights_only=True)['value_net_state']
        b = torch.load(resumed / 'value_net.pt', weights_only=True)['value_net_state']
        assert all(torch.equal(a[key], b[key]) for key in a)
        value_state = a
    updates = value_state.get('value_normalizer.updates')
    if value_normalization == 'popart':
        assert updates is not None and updates.item() > 0
    else:
        assert updates is None
    assert _scientific_rows(full / 'generated.csv') == _scientific_rows(resumed / 'generated.csv')
    left = torch.load(full / 'checkpoint.pt', weights_only=False)
    right = torch.load(resumed / 'checkpoint.pt', weights_only=False)
    assert left['predictor_caches']['_stats_cache']
    assert right['predictor_caches']['_stats_cache']
    assert set(left['predictor_caches']['_stats_cache']) == set(right['predictor_caches']['_stats_cache'])
    assert left['training_budget_state'] == right['training_budget_state']
    assert left['exploration_state'] == right['exploration_state']
    assert left['exploration_adaptive'] is right['exploration_adaptive'] is exploration_adaptive
    if not exploration_adaptive:
        assert left['exploration_state'] is None
        for key in (('eps',) if method == 'dqn' else ('entropy_coef_eff', 'visit_counts')):
            assert left[key] == right[key]
    assert left['predictor_ledger']['phase_costs'] == right['predictor_ledger']['phase_costs']
    assert torch.equal(left['rng_state']['torch_cpu'], right['rng_state']['torch_cpu'])


def test_ledger_unique_history_survives_cache_eviction_resume_and_phase_change(tmp_path):
    class OneEntryCache:
        key = None
        calls = 0
        def predict(self, candidate):
            hit = self.key == candidate
            self.calls += int(not hit)
            self.key = candidate
            self.last_evaluation_timings = [{'cache_hit': hit}]
            return float(candidate['x']), None
    inner = OneEntryCache()
    timer = PredictorTimer(inner, episode_log_path=str(tmp_path / 'episode_timing.csv'))
    for value in (1, 2, 1):
        timer.predict({'x': value})
    assert inner.calls == 3
    saved = timer.state_dict()
    restarted = PredictorTimer(OneEntryCache(), episode_log_path=str(tmp_path / 'episode_timing.csv'), append=True)
    restarted.load_state_dict(saved)
    restarted.predict({'x': 1})
    restarted.set_phase('generate')
    restarted.predict({'x': 99})
    report = restarted.summary()['phases']
    assert report['train']['unique_successes'] == 2
    assert report['train']['requests'] == 4
    assert report['train']['best_reward'] == 2
    assert report['generate']['best_reward'] == 99
    rows = [json.loads(line) for line in (tmp_path / 'evaluations.jsonl').read_text().splitlines()]
    assert rows[-2]['new_successful_candidate'] is False
    assert rows[-1]['phase'] == 'generate'


def test_opaque_batch_does_not_change_candidate_random_stream():
    class RandomBatch:
        def predict(self, candidate):
            return random.random(), None
        def batch_predict(self, candidates):
            return [self.predict(c) for c in candidates]
    timer = PredictorTimer(RandomBatch(), eval_seed=53)
    first = timer.predict({'x': 1})
    assert timer.batch_predict([{'x': 1}, {'x': 2}])[0] == first
    assert timer.batch_predict([{'x': 2}, {'x': 1}])[1] == first


@pytest.mark.parametrize('returned', [True, False])
def test_completed_batch_protocol_error_records_all_requests_before_raising(returned):
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    class BrokenBatch:
        candidate_seeded_batch = True
        def batch_predict(self, candidates, *, return_exceptions=False):
            if returned:
                return [EvaluationProtocolError('bad output'), (4., None)]
            raise EvaluationProtocolError('bad output')
    timer = PredictorTimer(BrokenBatch())
    with pytest.raises(EvaluationProtocolError):
        timer.batch_predict([{'x': 1}, {'x': 2}], return_exceptions=True)
    costs = timer.summary()['phases']['train']
    assert costs['requests'] == 2
    assert costs['failures'] == (1 if returned else 2)
    assert timer.t_predict_s > 0
