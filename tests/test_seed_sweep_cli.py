"""Independent public-CLI checks for paired seed sweeps.

Most tests stop at validation or guard rails. One tiny two-run CPU experiment
uses the built-in dummy predictor to exercise real process dispatch and output.
"""
from __future__ import annotations

from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from rl_matdesign import experiment
from rl_matdesign.config_schema import ConfigError


REPO = Path(__file__).resolve().parents[1]


def write_config(tmp_path, data):
    path = tmp_path / 'scenario.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def run_in_process(monkeypatch, path, *arguments):
    monkeypatch.setattr(sys, 'argv', ['rl-matdesign', '--config', str(path), *map(str, arguments)])
    with ExitStack() as resources:
        return experiment._run_experiment(resources)


def forbid_predictor(monkeypatch):
    monkeypatch.setattr(experiment, 'build_predictor',
                        lambda *a, **k: pytest.fail('validation/guards must precede predictor construction'))


def test_scalar_validation_preserves_original_shape_and_creates_no_output(tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {'eval_seed': 0, 'train_seed': 17, 'gen_seed': 23})
    output = tmp_path / 'not_created'
    run_in_process(monkeypatch, path, '--validate-config', '--out', output)
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {'config', 'sources'}
    assert [result['config'][key] for key in ('eval_seed', 'train_seed', 'gen_seed')] == [0, 17, 23]
    assert 'parallel_jobs' not in result['config']
    assert not output.exists()


def test_validation_pairs_lists_broadcasts_scalars_and_retains_duplicate_positions(
        tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {'eval_seed': [0, 0, 0], 'train_seed': [11, 11, 29],
                                   'gen_seed': 91, 'parallel_jobs': 2})
    output = tmp_path / 'not_created'
    run_in_process(monkeypatch, path, '--validate-config', '--out', output)
    result = json.loads(capsys.readouterr().out)
    assert result['run_count'] == 3 and result['parallel_jobs'] == 2
    assert [row['index'] for row in result['runs']] == [1, 2, 3]
    assert [[row['config'][key] for key in ('eval_seed', 'train_seed', 'gen_seed')]
            for row in result['runs']] == [[0, 11, 91], [0, 11, 91], [0, 29, 91]]
    assert all('parallel_jobs' not in row['config'] for row in result['runs'])
    assert not output.exists()


def test_scalar_seed_overrides_apply_before_expansion_and_parallel_override_wins(
        tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {'eval_seed': 0, 'train_seed': [1, 2],
                                   'gen_seed': [7, 8, 9], 'parallel_jobs': 3})
    run_in_process(monkeypatch, path, '--train-seed', '101', '--parallel-jobs', '1', '--validate-config')
    result = json.loads(capsys.readouterr().out)
    assert result['run_count'] == 3 and result['parallel_jobs'] == 1
    assert [row['config']['train_seed'] for row in result['runs']] == [101, 101, 101]
    assert [row['config']['gen_seed'] for row in result['runs']] == [7, 8, 9]
    assert all(row['sources']['train_seed'] == 'override' for row in result['runs'])


def test_overriding_only_seed_list_restores_scalar_validation_shape(tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {'eval_seed': 0, 'train_seed': [1, 2], 'gen_seed': 41})
    run_in_process(monkeypatch, path, '--train-seed', '101', '--validate-config')
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {'config', 'sources'}
    assert result['config']['train_seed'] == 101 and result['config']['gen_seed'] == 41


@pytest.mark.parametrize('arguments', [
    ['--resume-training'], ['--only-generate'],
    ['--load-qnet', 'qnet.pt'], ['--load-policy', 'policy.pt'],
    ['--load-scaler', 'std_scaler.bin'], ['--load-value-net', 'value_net.pt'],
    ['--load-reward-calibration', 'reward_calibration.json'],
])
def test_ambiguous_artifact_and_resume_modes_fail_before_any_output(
        tmp_path, monkeypatch, arguments):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {'train_seed': [11, 29], 'gen_seed': [11, 29]})
    output = tmp_path / 'not_created'
    with pytest.raises((ConfigError, ValueError, SystemExit)) as error:
        run_in_process(monkeypatch, path, '--out', output, *arguments)
    message = str(error.value).lower()
    assert 'seed' in message or 'scalar' in message or 'sweep' in message
    assert not output.exists()


@pytest.mark.parametrize('data', [
    {'train_seed': []}, {'train_seed': [True]}, {'train_seed': [-1]},
    {'train_seed': [2**32]}, {'train_seed': [None]},
    {'train_seed': [11, 29], 'gen_seed': [11]},
    {'train_seed': [11, 29], 'parallel_jobs': 0},
    {'train_seed': [11, 29], 'parallel_jobs': True},
])
def test_invalid_sweeps_fail_before_any_output(tmp_path, monkeypatch, data):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, data)
    output = tmp_path / 'not_created'
    with pytest.raises((ConfigError, ValueError, SystemExit)):
        run_in_process(monkeypatch, path, '--out', output)
    assert not output.exists()


def cli_process(path, *arguments, timeout=90):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    return subprocess.run(
        [sys.executable, '-c', 'from rl_matdesign.cli import main; main()',
         'run', '--config', str(path), *map(str, arguments)],
        cwd=REPO, env=environment, capture_output=True, text=True, timeout=timeout,
    )


def test_installed_entry_point_run_alias_validates_without_outputs(tmp_path):
    path = write_config(tmp_path, {'train_seed': [11, 29], 'gen_seed': [11, 29], 'eval_seed': 0})
    output = tmp_path / 'not_created'
    result = cli_process(path, '--validate-config', '--out', output, '--parallel-jobs', '2')
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved['run_count'] == 2 and resolved['parallel_jobs'] == 2
    assert not output.exists()


def tiny_mc_config():
    return {
        'groups': [{'name': 'metal', 'kind': 'composition', 'species_set': ['Fe', 'Co', 'Ni'],
                    'amount_set': [1.], 'n_steps': 1, 'total': 1., 'unique': True}],
        'properties': [{'name': 'score', 'predictor': 'dummy', 'direction': 'max', 'objective': 'mean'}],
        'method': 'dqn', 'dqn_target_mode': 'mc', 'dqn_hidden_dim': 8,
        'train_budget': 3, 'dqn_warmup_eps': 1, 'dqn_batch_size': 1,
        'dqn_buffer_size': 16, 'dqn_grad_steps_per_ep': 1,
        'train_count_duplicates': True, 'num_gen_eps': 0,
        'exploration_adaptive': False, 'dqn_eps_min': .2, 'dqn_eps_anneal_eps': 4,
        'eval_seed': [0, 0], 'train_seed': [11, 29], 'gen_seed': [11, 29], 'parallel_jobs': 2,
    }


def test_tiny_parallel_mc_sweep_writes_separate_scalar_training_runs(tmp_path):
    path = write_config(tmp_path, tiny_mc_config())
    output = tmp_path / 'sweep'
    result = cli_process(path, '--out', output, '--skip-generation', '--device', 'cpu',
                         '--save-checkpoint-freq', '1', timeout=120)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    manifest = json.loads((output / 'seed_runs.json').read_text())
    runs = manifest['runs']
    assert len(runs) == 2
    for index, (record, seed) in enumerate(zip(runs, [11, 29]), 1):
        expected = output / f'run_{index:03d}_eval_0_train_{seed}_gen_{seed}'
        assert record['index'] == index
        assert record['seeds'] == {'eval': 0, 'train': seed, 'gen': seed}
        assert Path(record['out']).resolve() == expected.resolve()
        assert record['returncode'] == 0 and record['status'] == 'completed'
        assert (expected / 'config.yaml').is_file() and (expected / 'run.log').is_file()
        assert (expected / 'qnet.pt').is_file() and (expected / 'std_scaler.bin').is_file()
        assert not (expected / 'generated.csv').exists()
        saved = json.loads((expected / 'run_config.json').read_text())
        assert [saved[key] for key in ('eval_seed', 'train_seed', 'gen_seed')] == [0, seed, seed]
        assert 'parallel_jobs' not in saved
        assert saved['train_budget'] == 3 and saved['exploration_adaptive'] is False
        with (expected / 'training_log.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert sum(row['phase'] == 'dqn_train' for row in rows) == 2
