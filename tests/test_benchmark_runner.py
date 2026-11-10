"""Planning, incomplete-budget and actual four-arm baseline contracts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def tiny_config(tmp_path):
    cfg = {"groups": [{"name": "alloy", "kind": "composition", "species_set": ["Fe", "Ni"],
                        "fraction_set": ["0", "0.25", "0.5", "0.75", "1"], "n_components": 2}],
           "predictor": "dummy", "dqn_warmup_eps": 1, "dqn_batch_size": 1,
           "dqn_grad_steps_per_ep": 1, "pg_warmup_eps": 2, "pg_batch_eps": 2,
           "train_max_duplicate_attempts": 50, "eval_seed": 19,
           "num_gen_eps": 0}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_dry_plan_has_unique_outputs_and_exact_shared_budget(tiny_config, tmp_path):
    runner = module('scripts/benchmark.py')
    out = tmp_path / 'plan'
    runner.main(['--config', str(tiny_config), '--out', str(out), '--budget', '3', '--seeds', '7'])
    rows = json.loads((out / 'benchmark_plan.json').read_text())
    assert len(rows) == len({row['output'] for row in rows}) == 4
    assert {row['domain_size'] for row in rows} == {5}
    for row in rows:
        authored = yaml.safe_load((Path(row['output']) / 'config.yaml').read_text())
        assert authored['train_budget'] == 3
        assert authored['train_count_duplicates'] is False
        assert authored['eval_seed'] == 19
        assert 'pg_num_iters' not in authored and 'dqn_num_train_eps' not in authored
    # Repeating an identical dry plan is safe until any arm actually starts.
    runner.main(['--config', str(tiny_config), '--out', str(out), '--budget', '3', '--seeds', '7'])


@pytest.mark.parametrize('collision', ['config_stem', 'seed', 'method'])
def test_plan_rejects_output_collisions_before_writing(tiny_config, tmp_path, collision):
    runner = module('scripts/benchmark.py')
    out = tmp_path / 'out'
    args = ['--config', str(tiny_config), '--out', str(out), '--budget', '2', '--seeds', '7']
    if collision == 'config_stem':
        second = tmp_path / 'other' / tiny_config.name
        second.parent.mkdir(); second.write_text(tiny_config.read_text())
        args += ['--config', str(second)]
    elif collision == 'seed':
        args += ['7']
    else:
        args += ['--methods', 'random', 'random']
    with pytest.raises(SystemExit):
        runner.main(args)
    assert not out.exists()


def test_plan_does_not_overwrite_existing_run(tiny_config, tmp_path):
    runner = module('scripts/benchmark.py')
    out = tmp_path / 'out'
    directory = out / 'tiny' / 'random-seed7'
    directory.mkdir(parents=True)
    artifact = directory / 'benchmark.json'
    artifact.write_text('original result')
    with pytest.raises(SystemExit):
        runner.main(['--config', str(tiny_config), '--out', str(out), '--budget', '2',
                     '--seeds', '7', '--methods', 'random'])
    assert artifact.read_text() == 'original result'
    assert not (directory / 'config.yaml').exists()


def test_zero_exit_status_does_not_hide_partial_budget(tmp_path):
    runner = module('scripts/benchmark.py')
    row = {'method': 'random', 'output': str(tmp_path), 'budget': 4}
    (tmp_path / 'benchmark.json').write_text(json.dumps({'budget_completed': 2, 'stop_reason': 'duplicate_guard'}))
    result = runner._completion(row, 0)
    assert result['status'] == 'incomplete'
    assert result['budget_completed'] == 2
    row['method'] = 'a2c'
    (tmp_path / 'timing.json').write_text(json.dumps({'predictor': {'n_unique': 2}}))
    assert runner._completion(row, 0)['status'] == 'incomplete'


@pytest.mark.parametrize('failure', ['duplicate', 'candidate', 'protocol'])
def test_random_partial_or_fatal_runs_report_and_fail(tiny_config, tmp_path, monkeypatch, failure):
    random_run = module('scripts/baselines/run_random.py')
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    cfg = yaml.safe_load(tiny_config.read_text()); cfg['train_max_duplicate_attempts'] = 2
    tiny_config.write_text(yaml.safe_dump(cfg))
    closed, calls = [], []
    class Oracle:
        def predict(self, candidate):
            calls.append(candidate)
            if failure == 'candidate': raise ValueError('candidate failed')
            if failure == 'protocol': raise EvaluationProtocolError('wrong output schema')
            return 1., None
        def oracle_identity(self): return {'model_sha256': 'mock frozen identity'}
        def close(self): closed.append(True)
    monkeypatch.setattr(random_run, 'build_predictor', lambda *a, **kw: Oracle())
    monkeypatch.setattr(random_run, 'sampler_from_env', lambda env: SimpleNamespace(
        size=2, total_units=1, sample=lambda rng: {'Fe': 1}))
    out = tmp_path / failure
    error = EvaluationProtocolError if failure == 'protocol' else SystemExit
    with pytest.raises(error):
        random_run.main(['--config', str(tiny_config), '--out', str(out), '--budget', '2'])
    report = json.loads((out / 'benchmark.json').read_text())
    assert report['budget_completed'] < 2
    assert report['stop_reason'] == {'duplicate': 'duplicate_guard', 'candidate': 'consecutive_evaluation_failures',
                                     'protocol': 'protocol_error'}[failure]
    assert report['total_s'] > 0
    assert closed == [True]
    manifest = Path(report['manifest'])
    identity = json.loads((manifest.parent / 'oracle_identity.json').read_text())
    assert identity == {'model_sha256': 'mock frozen identity'}
    assert len(calls) == {'duplicate': 3, 'candidate': 11, 'protocol': 1}[failure]


def test_real_tiny_four_arm_run_completes_identical_unique_budget(tiny_config, tmp_path):
    runner = module('scripts/benchmark.py')
    out = tmp_path / 'four_arms'
    runner.main(['--config', str(tiny_config), '--out', str(out), '--budget', '3', '--seeds', '7', '--run'])
    results = json.loads((out / 'benchmark_results.json').read_text())
    assert len(results) == 4
    assert {row['method'] for row in results} == {'dqn-mc', 'dqn-bootstrap', 'a2c', 'random'}
    assert all(row['status'] == 'complete' and row['budget_completed'] == 3 for row in results)
    assert all(row['returncode'] == 0 for row in results)
