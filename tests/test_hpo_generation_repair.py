"""Repair a real one-observation generation save failure without buying it twice."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
import yaml

from rl_matdesign.hpo.artifacts import atomic_json, read_json
from rl_matdesign.hpo.controller import Campaign
from rl_matdesign.hpo.job_runner import JobRunner
from rl_matdesign.hpo.restart import inspect_restart
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_checkpoint_retention import _assert_nested_equal


_ROOT = Path(__file__).resolve().parents[1]
_FAULT_WRAPPER = textwrap.dedent("""
    import errno
    import os
    from pathlib import Path
    import torch
    from rl_matdesign import experiment
    original_save = torch.save
    def fail_seventh_snapshot(data, destination, *args, **kwargs):
        if (str(destination).endswith('generation_checkpoint.pt.tmp')
                and isinstance(data, dict) and data.get('type') == 'hpo_generation'
                and data['hpo_state']['budget']['unique_evaluated'] == 7):
            assert data['hpo_state']['budget']['attempts_started'] == 7
            original_save(data, os.environ['REPAIR_TEST_EXPECTED_CHECKPOINT'])
            Path(destination).write_bytes(b'PK' + bytes(126))
            raise OSError(errno.ENOSPC, 'No space left on device: injected checkpoint write failure')
        return original_save(data, destination, *args, **kwargs)
    torch.save = fail_seventh_snapshot
    experiment.main()
""")


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _repair_module(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location(
        'hpo_generation_repair_under_test', _ROOT / 'scripts/repair_hpo_generation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _public_plan(tmp_path, method):
    # This space has thousands of candidates. The first seven sampled materials
    # must all be distinct so the fixture is the exact reported failure class.
    scenario = {
        'method': 'a2c' if method == 'a2c' else 'dqn',
        'properties': [{'name': 'energy', 'predictor': 'dummy',
                        'direction': 'min', 'objective': 'mean'}],
        'groups': [{'name': 'metal', 'kind': 'composition',
                    'species_set': ['Ni', 'Fe', 'Co', 'Cr', 'Mn', 'Cu', 'Zn', 'Ti'],
                    'fraction_set': [f'{value / 10:.1f}' for value in range(1, 9)],
                    'n_components': 3}],
        'eval_seed': 13, 'pg_warmup_eps': 2, 'pg_batch_eps': 1,
        'dqn_hidden_dim': 16, 'dqn_buffer_size': 100,
        'dqn_batch_size': 1, 'dqn_grad_steps_per_ep': 1,
        'gen_temperature': 1.0, 'num_gen_eps': 12,
        'exploration_gen_eps': 0,
    }
    (tmp_path / 'scenario.yaml').write_text(yaml.safe_dump(scenario, sort_keys=False))
    ranges = ({'pg_batch_eps': {'type': 'categorical', 'choices': [1]},
               'pg_lr_actor': {'type': 'categorical', 'choices': [.0001]},
               'pg_lr_critic': {'type': 'categorical', 'choices': [.0001]}}
              if method == 'a2c' else {
                  'dqn_batch_size': {'type': 'categorical', 'choices': [1]},
                  'warmup_budget_fraction': {'type': 'categorical', 'choices': [.2]},
                  'dqn_lr': {'type': 'categorical', 'choices': [.0001]}})
    budget = dict(mode='absolute_unique', min_unique=12,
                  initial_unique=12, max_unique=12)
    source = tmp_path / 'hpo.yaml'
    source.write_text(yaml.safe_dump({
        'base_config': 'scenario.yaml', 'method': method, 'random_seed': 42,
        'evaluation': {'threshold': .5, 'max_attempts_per_unique': 1.5},
        'training': {'repeats': 1, 'budget': budget, 'parameter_ranges': ranges},
        'generation': {'repeats': 1, 'budget': budget, 'initial_values': [1.]},
    }, sort_keys=False))
    return load_plan(source)


def _broken_campaign(tmp_path, monkeypatch, method):
    plan = _public_plan(tmp_path, method)
    campaign = Campaign(plan, tmp_path / 'campaign')
    atomic_json(campaign.out / 'state.json', {'requested_stage': 'generation'})
    runner = campaign.runner
    parameters = {'pg_batch_eps': 1} if method == 'a2c' else {'dqn_batch_size': 1}
    model = runner(parameters, 12, 17, 'train')
    assert model['status'] == 'COMPLETED', model
    generated_parameters = {**parameters, 'gen_temperature': 1.}
    expected_checkpoint = tmp_path / 'would_have_been_checkpoint7.pt'
    original_popen = subprocess.Popen

    def faulty_popen(args, **kwargs):
        assert args[1:3] == ['-m', 'rl_matdesign.hpo.worker']
        assert '--only-generate' in args
        env = dict(kwargs['env'])
        env['REPAIR_TEST_EXPECTED_CHECKPOINT'] = str(expected_checkpoint)
        return original_popen([args[0], '-c', _FAULT_WRAPPER, *args[3:]],
                              **{**kwargs, 'env': env})

    with monkeypatch.context() as fault:
        fault.setattr('rl_matdesign.hpo.job_runner.subprocess.Popen', faulty_popen)
        result = runner(generated_parameters, 12, 2366732292, 'generation', model)
    assert result['status'] == 'SYSTEM_ERROR', result
    assert result['returncode'] == 1
    directory = campaign.out / 'jobs' / result['job_id']
    assert expected_checkpoint.is_file(), (directory / 'stderr.log').read_text()
    old = torch.load(directory / 'generation_checkpoint.pt', map_location='cpu', weights_only=False)
    assert old['hpo_state']['budget']['attempts_started'] == 6
    assert old['hpo_state']['budget']['unique_evaluated'] == 6
    ledger = read_json(directory / 'candidate_budget.json')
    assert ledger['attempts_started'] == ledger['unique_evaluated'] == 7
    assert ledger['pending'] == ledger['failed_attempts'] == 0
    assert (directory / 'generation_checkpoint.pt.tmp').stat().st_size == 128
    assert len(_records(directory / 'generation_evaluations.jsonl')) == 7
    assert len(_records(directory / 'candidate_records.jsonl')) == 7
    audit = inspect_restart(campaign.out)
    assert audit['job_counts']['CACHED'] == 1
    assert audit['job_counts']['BLOCKED'] == 1
    return campaign, runner, model, generated_parameters, result, expected_checkpoint


def _forbid_dummy_evaluation(monkeypatch):
    from rl_matdesign import registry
    original_factory = registry.PREDICTORS['dummy']
    evaluated = []

    def factory(*args, **kwargs):
        predictor = original_factory(*args, **kwargs)

        def forbidden(*args, **kwargs):
            evaluated.append((args, kwargs))
            raise AssertionError('Repair must consume stored observations, never predict again')

        predictor.predict = forbidden
        predictor.batch_predict = forbidden
        return predictor

    monkeypatch.setitem(registry.PREDICTORS, 'dummy', factory)
    return evaluated


def _assert_checkpoint_equivalent(repaired, expected):
    # Recovery provenance may be added, but every scientific state written by
    # the original generation checkpoint must be reproduced exactly.
    for name in ('hpo_state', 'rng_state', 'generation_contract', 'predictor_ledger',
                 'predictor_caches', 'predictor_extra'):
        _assert_nested_equal(repaired[name], expected[name])
    for name, cache in expected['predictor_caches'].items():
        assert list(repaired['predictor_caches'][name]) == list(cache)


@pytest.mark.parametrize('method', ['a2c', 'dqn-mc', 'dqn-bootstrap'])
def test_real_seventh_generation_save_failure_repairs_without_resampling(
        tmp_path, monkeypatch, method):
    campaign, runner, model, parameters, failed, expected_path = _broken_campaign(
        tmp_path, monkeypatch, method)
    directory = campaign.out / 'jobs' / failed['job_id']
    module = _repair_module(monkeypatch)
    before = _snapshot(campaign.out)
    expected = torch.load(expected_path, map_location='cpu', weights_only=False)
    with monkeypatch.context() as forbid:
        evaluated = _forbid_dummy_evaluation(forbid)
        preview = module.recover(campaign.out, failed['job_id'], apply=False)
        assert preview['ready_to_apply'] and not preview['applied']
        assert preview['predictor_evaluations'] == 0
        assert _snapshot(campaign.out) == before
        applied = module.recover(campaign.out, failed['job_id'], apply=True)
        assert applied['applied'] and applied['predictor_evaluations'] == 0
        after_apply = _snapshot(campaign.out)
        repeated = module.recover(campaign.out, failed['job_id'], apply=True)
        assert repeated['applied'] and repeated['already_applied']
        assert _snapshot(campaign.out) == after_apply
        assert not evaluated
    repaired = torch.load(directory / 'generation_checkpoint.pt', map_location='cpu', weights_only=False)
    _assert_checkpoint_equivalent(repaired, expected)
    after = _snapshot(campaign.out)
    changed = [name for name in before if before[name] != after[name]]
    assert changed == [f"jobs/{failed['job_id']}/generation_checkpoint.pt"]
    # The latest checkpoint advances; original records, costs, failures, and
    # corrupt temporary bytes remain preserved as evidence.
    assert read_json(directory / 'candidate_budget.json')['attempts_started'] == 7
    assert (directory / 'generation_checkpoint.pt.tmp').read_bytes() == b'PK' + bytes(126)
    assert inspect_restart(campaign.out)['ready']

    # An uninterrupted generation from the same immutable network and seed is
    # the control. Its timing differs; the candidate stream and all RNGs do not.
    control_runner = JobRunner(deepcopy(campaign.plan), tmp_path / 'control')
    control = control_runner(parameters, 12, failed['seed'], 'generation', model)
    assert control['status'] == 'COMPLETED', control
    control_directory = tmp_path / 'control' / 'jobs' / control['job_id']
    resumed = runner(parameters, 12, failed['seed'], 'generation', model)
    assert resumed['status'] == 'COMPLETED', resumed
    assert (directory / 'candidate_records.jsonl').read_bytes() == (
        control_directory / 'candidate_records.jsonl').read_bytes()
    assert read_json(directory / 'candidate_budget.json') == read_json(
        control_directory / 'candidate_budget.json')
    resumed_checkpoint = torch.load(directory / 'generation_checkpoint.pt', map_location='cpu', weights_only=False)
    control_checkpoint = torch.load(control_directory / 'generation_checkpoint.pt', map_location='cpu', weights_only=False)
    for name in ('hpo_state', 'rng_state', 'predictor_caches', 'predictor_extra'):
        _assert_nested_equal(resumed_checkpoint[name], control_checkpoint[name])
    observed = _records(directory / 'generation_evaluations.jsonl')
    uninterrupted = _records(control_directory / 'generation_evaluations.jsonl')
    assert len(observed) == len(uninterrupted) == resumed['ledger']['attempts_started']
    assert [row['candidate'] for row in observed] == [row['candidate'] for row in uninterrupted]
    assert [row['request'] for row in observed] == list(range(1, len(observed) + 1))
    # A second invocation is a cached reuse, with no additional predictor work.
    before_reuse = _snapshot(campaign.out)
    assert runner(parameters, 12, failed['seed'], 'generation', model) == resumed
    assert _snapshot(campaign.out) == before_reuse


def test_nested_group_filter_is_refused_before_external_plugin_construction(monkeypatch):
    module = _repair_module(monkeypatch)
    config = {
        'method': 'a2c', 'hpo_phase': 'generation', 'hpo_recording': True,
        'gen_temperature': 1., 'gen_epsilon': 0., 'gen_top_frac': 0.,
        'groups': [{'name': 'metal', 'kind': 'composition',
                    'filters': [{'type': 'unreviewed_plugin:ExternalConstraint'}]}],
        'properties': [{'name': 'energy', 'predictor': 'dummy'}],
    }
    evidence = {
        'checkpoint': {'generation_contract': {'execution': {'source': 'same'}}},
        'provenance': [{'execution_contract': {'source': 'same'},
                        'manifest': {'phase': 'generate', 'config': config,
                                     'runtime_args': {'only_generate': True}}}],
        'run_config': deepcopy(config),
    }
    with pytest.raises(ValueError, match='group filters'):
        module._provenance(evidence)


@pytest.mark.parametrize('uncertainty', [None, 'model_spread'])
def test_recorded_tail_uncertainty_must_match_selected_scientific_output(monkeypatch, uncertainty):
    from rl_matdesign.configuration import build_predictor
    from rl_matdesign.predictors.outputs import OutputSchema, ScientificOutput

    module = _repair_module(monkeypatch)
    evaluated = _forbid_dummy_evaluation(monkeypatch)
    prop = {'name': 'energy', 'predictor': 'dummy', 'direction': 'min', 'objective': 'mean'}
    if uncertainty is not None:
        prop['uncertainty'] = uncertainty
    engine = build_predictor({'properties': [prop]}, seed=13)
    output = ScientificOutput(OutputSchema('energy', None), .5,
                              model_spread=.2, native_uncertainty=.1)
    record = {'prediction_record': {'obj_energy_mean': .5, 'obj_energy_std': .9}}
    observation = {'outputs': {'energy': output.to_dict()}}
    try:
        with pytest.raises(ValueError, match='uncertainty disagrees'):
            module._record_from_observation(engine, record, observation)
        assert not engine._stats_cache and not engine._batch_stats
        assert not evaluated
    finally:
        engine.close()
