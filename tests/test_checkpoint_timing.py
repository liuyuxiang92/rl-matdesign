"""Checkpoint telemetry describes committed work without changing safe stops."""
from __future__ import annotations

import json
import os
import signal

import pytest
import torch
import yaml

from rl_matdesign import training
from rl_matdesign.utils import stage_timing
from rl_matdesign.utils.hpo_stop import cooperative_worker_stop
from tests.test_experiment_reproducibility import _config
from tests.test_hpo_checkpoint_retention import _assert_success, _run_recorded


def _events(directory):
    return [json.loads(line) for line in (directory / 'checkpoint_timing.jsonl').read_text().splitlines()]


def _stoppable_payload():
    return {'weights': torch.tensor([1., 2.]), 'hpo_state': {'budget': {
        'pending': 0, 'attempts_started': 1, 'unique_evaluated': 1,
        'target_unique': 6, 'max_attempts': 9, 'status': 'RUNNING',
    }}}


def test_stage_captures_preparation_and_total_without_changing_measurements(tmp_path, monkeypatch):
    now = [10.]
    monkeypatch.setattr(stage_timing.time, 'perf_counter', lambda: now[0])
    measurements = {'checkpoint_pack_s': 1., 'checkpoint_serialize_s': 2.,
                    'checkpoint_replace_s': .5, 'checkpoint_bytes': 41}
    with stage_timing.checkpoint_stage('dqn_train', 7):
        now[0] = 12.
        stage_timing.checkpoint_prepared()
        now[0] = 16.
        stage_timing.record_checkpoint(tmp_path / 'checkpoint.pt', measurements)
    row = _events(tmp_path)[0]
    assert row['phase'] == 'dqn_train' and row['unit'] == 7
    assert row['checkpoint_prepare_s'] == 2.
    assert row['checkpoint_total_s'] == 6.
    assert all(row[key] == value for key, value in measurements.items())
    assert 'checkpoint_prepare_s' not in measurements


def test_timing_context_restores_outer_scope_and_resets_after_exception(tmp_path):
    with stage_timing.checkpoint_stage('outer', 1):
        with pytest.raises(RuntimeError):
            with stage_timing.checkpoint_stage('inner', 2):
                raise RuntimeError('interrupt preparation')
        stage_timing.checkpoint_prepared()
        stage_timing.record_checkpoint(tmp_path / 'checkpoint.pt', {})
    stage_timing.record_checkpoint(tmp_path / 'checkpoint.pt', {})
    inside, outside = _events(tmp_path)
    assert inside['phase'] == 'outer' and inside['unit'] == 1
    assert inside['checkpoint_prepare_s'] >= 0
    assert 'phase' not in outside and 'unit' not in outside
    assert outside['checkpoint_prepare_s'] is None
    assert outside['checkpoint_total_s'] is None


def test_save_logs_committed_file_size_and_complete_stage(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    with stage_timing.checkpoint_stage('pg_train', 4):
        metrics = training._save_checkpoint(str(path), {'weight': torch.tensor([3.])})
    row = _events(tmp_path)[0]
    assert row['checkpoint'] == path.name
    assert row['phase'] == 'pg_train' and row['unit'] == 4
    assert row['checkpoint_bytes'] == path.stat().st_size == metrics['checkpoint_bytes']
    durations = [row[key] for key in ('checkpoint_prepare_s', 'checkpoint_pack_s',
                 'checkpoint_serialize_s', 'checkpoint_replace_s')]
    assert all(value >= 0 for value in durations)
    assert row['checkpoint_total_s'] >= sum(durations)
    assert torch.equal(torch.load(path, weights_only=False)['weight'], torch.tensor([3.]))


@pytest.mark.parametrize('failure_point', ['serialize', 'replace'])
def test_failed_write_does_not_emit_a_committed_timing_event(tmp_path, monkeypatch, failure_point):
    path = tmp_path / 'checkpoint.pt'
    training._save_checkpoint(str(path), {'version': 1})
    old = path.read_bytes()
    original_events = (tmp_path / 'checkpoint_timing.jsonl').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('simulated checkpoint failure')
    with monkeypatch.context() as patch:
        if failure_point == 'serialize':
            patch.setattr(training, 'save_checkpoint_payload', fail)
        else:
            patch.setattr(os, 'replace', fail)
        with pytest.raises(OSError, match='simulated checkpoint failure'):
            with stage_timing.checkpoint_stage('dqn_train', 2):
                training._save_checkpoint(str(path), {'version': 2})
    assert path.read_bytes() == old
    assert (tmp_path / 'checkpoint_timing.jsonl').read_bytes() == original_events


def test_graceful_stop_logs_after_commit_before_exiting(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    with cooperative_worker_stop():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with pytest.raises(SystemExit) as stopped:
            with stage_timing.checkpoint_stage('dqn_train', 1):
                training._save_checkpoint(str(path), _stoppable_payload())
    assert stopped.value.code == 128 + signal.SIGTERM
    restored = torch.load(path, weights_only=False)
    assert restored['hpo_state']['budget']['attempts_started'] == 1
    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]['checkpoint_bytes'] == path.stat().st_size
    assert events[0]['phase'] == 'dqn_train' and events[0]['unit'] == 1
    assert events[0]['checkpoint_total_s'] >= events[0]['checkpoint_serialize_s']
    assert stage_timing._checkpoint_context.get() is None


def test_timing_io_failure_warns_but_preserves_committed_graceful_stop(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    # A concrete telemetry-only failure: this path cannot be opened as a file.
    (tmp_path / 'checkpoint_timing.jsonl').mkdir()
    with cooperative_worker_stop():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with pytest.warns(RuntimeWarning, match='Checkpoint saved'):
            with pytest.raises(SystemExit) as stopped:
                with stage_timing.checkpoint_stage('dqn_train', 1):
                    training._save_checkpoint(str(path), _stoppable_payload())
    assert stopped.value.code == 128 + signal.SIGTERM
    assert torch.load(path, weights_only=False)['hpo_state']['budget']['pending'] == 0
    assert not (tmp_path / 'checkpoint.pt.tmp').exists()
    assert stage_timing._checkpoint_context.get() is None


@pytest.mark.parametrize('method,target,phase', [
    ('dqn', 'mc', 'dqn_train'), ('dqn', 'bootstrap', 'dqn_train'),
    ('a2c', 'bootstrap', 'pg_train'),
])
def test_real_training_emits_one_timing_record_per_committed_checkpoint(tmp_path, method, target, phase):
    config = _config()
    config.update(method=method, dqn_target_mode=target, exploration_adaptive=True,
                  dqn_warmup_eps=2, dqn_batch_size=1, exploration_target=1., pg_warmup_eps=2,
                  pg_batch_eps=1, train_count_duplicates=False, train_budget=6,
                  hpo_recording=True, hpo_phase='train', hpo_target_unique=6,
                  hpo_ratio_limit=20., num_gen_eps=0)
    config.pop('pg_num_iters')
    config.pop('dqn_num_train_eps')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / 'run'
    _assert_success(_run_recorded(config_path, out))
    committed = [json.loads(line) for line in (out / 'checkpoint_events.jsonl').read_text().splitlines()]
    measured = _events(out)
    assert len(measured) == len(committed) > 1
    assert [row['unit'] for row in measured] == list(range(1, len(measured) + 1))
    assert all(row['phase'] == phase and row['checkpoint_prepare_s'] >= 0 for row in measured)
    assert all(row['checkpoint_total_s'] >= row['checkpoint_serialize_s'] for row in measured)
    assert measured[-1]['checkpoint_bytes'] == (out / 'checkpoint.pt').stat().st_size
