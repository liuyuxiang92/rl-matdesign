#!/usr/bin/env python3
"""Repair one fully recorded scalar generation episode beyond a saved checkpoint.

Default: validate contracts and replay the frozen policy without evaluating any
property model or changing campaign files. --apply additionally backs up original
evidence and atomically replaces only generation_checkpoint.pt. Stop all workers
first. Only trusted local checkpoints, pure composition GlassNet/dummy objectives,
and temperature sampling are supported; every other case fails closed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

sys.dont_write_bytecode = True


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _provenance(evidence):
    saved = evidence['checkpoint']['generation_contract']['execution']
    matches = [row for row in evidence['provenance'] if row.get('execution_contract') == saved
               and row['manifest'].get('phase') == 'generate']
    _require(len(matches) == 1, 'Require one original generation invocation matching the checkpoint execution contract')
    manifest = matches[0]['manifest']
    config = manifest['config']
    run = {k: v for k, v in evidence['run_config'].items() if k not in {'config_file', 'config_schema_version'}}
    _require(_json_equal(config, run), 'run_config.json differs from the original resolved invocation')
    _require(manifest['runtime_args'].get('only_generate') is True, 'Original invocation was not generation-only')
    _require(not config.get('hpo_resume_generation'), 'Only a first, uninterrupted generation invocation is supported')
    _require(config.get('hpo_phase') == 'generation' and config.get('hpo_recording') is True,
             'Original invocation lacks HPO generation accounting')
    _require(config.get('method') in {'a2c', 'dqn'}, 'Unsupported policy method')
    _require(not config.get('exploration_gen_eps'), 'Separate exploration phase cannot be reconstructed')
    _require(config.get('gen_temperature', 0) > 0 and not config.get('gen_epsilon') and not config.get('gen_top_frac'),
             'Only temperature-only generation is supported')
    _require(not config.get('filters'), 'Recovery currently requires no additional postfilters')
    # Reject unknown factories before their construction can perform external work.
    def check_factories(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'filters':
                    _require(not item, 'Recovery currently requires no additional postfilters, including group filters')
                if key == 'predictor':
                    _require(item in {'glassnet', 'dummy'}, 'Only built-in GlassNet/dummy composition predictors are audited')
                check_factories(item)
        elif isinstance(value, list):
            for item in value:
                check_factories(item)
    check_factories(config)
    return manifest, deepcopy(config)


def _model_paths(evidence, manifest, directory):
    from rl_matdesign.hpo.artifacts import file_digest
    argv = evidence['execution']['argv']
    paths = {}
    flags = {'policy.pt': '--load-policy', 'qnet.pt': '--load-qnet',
             'value_net.pt': '--load-value-net', 'std_scaler.bin': '--load-scaler'}
    for name, expected in evidence['job']['input_models'].items():
        _require(name in flags, 'Unknown frozen input artifact')
        flag = flags[name]
        _require(argv.count(flag) == 1, 'Frozen artifact must have one explicit load flag: ' + name)
        index = argv.index(flag)
        _require(index + 1 < len(argv), 'Missing frozen artifact path')
        path = Path(argv[index + 1])
        if not path.is_absolute():
            path = Path(manifest['cwd']) / path
        path = path.resolve()
        _require(path.is_file() and file_digest(path) == expected, 'Frozen artifact changed: ' + str(path))
        paths[name] = path
    frozen = evidence['checkpoint']['generation_contract']['frozen_artifacts']
    _require(all(name in paths and file_digest(paths[name]) == value for name, value in frozen.items()),
             'Generation checkpoint frozen artifacts disagree with the job')
    weight = 'policy.pt' if manifest['config']['method'] == 'a2c' else 'qnet.pt'
    _require(set(frozen) == {weight, 'std_scaler.bin'}, 'Unexpected generation frozen artifact contract')
    return paths


def _supported_engine(engine, env, config, saved):
    from rl_matdesign.env import CompositionEnv
    from rl_matdesign.predictors.structure_score import StructureScorePredictor
    from rl_matdesign.predictors.glassnet import GlassNetPredictor
    _require(type(engine) is StructureScorePredictor and type(env) is CompositionEnv,
             'Only the built-in reward engine and flat CompositionEnv are audited')
    _require(env.phase_filter is None, 'Environment postfilter state cannot be reconstructed')
    _require(not engine.source_plan.sources and engine.sweep is None,
             'Structure sources and sweep/custom state are unsupported')
    _require(engine.evaluation_batch_size() == 1, 'Only scalar evaluation waves are supported')
    _require(saved.get('predictor_extra') is None, 'Custom predictor checkpoint state cannot be reconstructed')
    for prop in engine.properties:
        _require(prop['backend'] == 'composition', 'All properties must be composition-only')
        leaves = prop.get('instance') or []
        _require(len(leaves) == 1, 'Ensembles are outside this bounded recovery protocol')
        leaf = leaves[0]
        is_dummy = (type(leaf).__module__ == 'rl_matdesign.registry'
                    and type(leaf).__qualname__ == '_make_dummy.<locals>._DummyPredictor')
        _require(type(leaf) is GlassNetPredictor or is_dummy, 'Unknown predictor implementation')
        # Defense in depth: after identity/contract validation no real evaluation
        # may occur, even accidentally through a cache helper.
        def forbidden(*args, **kwargs):
            raise RuntimeError('Recovery must not call a property predictor')
        for name in ('predict', 'predict_raw', 'predict_outputs', 'batch_predict', 'batch_predict_outputs'):
            if callable(getattr(leaf, name, None)):
                setattr(leaf, name, forbidden)
    for name in ('predict', 'predict_raw', 'batch_predict', 'predict_outputs', 'batch_predict_outputs',
                 '_evaluate_inputs', '_materialize_structures'):
        setattr(engine, name, forbidden)


def _record_from_observation(engine, record, observation):
    from rl_matdesign.predictors.outputs import ScientificOutput
    from rl_matdesign.predictors.structure_score import _Statistics, _EvaluationRecord
    from rl_matdesign.utils.prediction_records import prediction_record
    from rl_matdesign.utils.composition import composition_key
    raw = record['prediction_record']
    names = [prop['name'] for prop in engine.properties]
    _require(set(observation['outputs']) == set(names), 'Scientific output names differ from the original properties')
    outputs = {name: ScientificOutput.from_dict(observation['outputs'][name]) for name in names}
    stats = _Statistics({name: (raw['obj_' + name + '_mean'], raw['obj_' + name + '_std']) for name in names},
                        outputs=outputs)
    for prop in engine.properties:
        name = prop['name']
        _require(stats[name][0] == outputs[name].mean, 'Scientific output mean disagrees with the candidate record')
        # Match the supported leaf protocols without evaluating the predictor.
        from rl_matdesign.predictors.glassnet import GlassNetPredictor
        default_uncertainty = 'model_spread' if type(prop['instance'][0]) is GlassNetPredictor else 'native_uncertainty'
        selected_uncertainty = prop['uncertainty'] or default_uncertainty
        _require(stats[name][1] == getattr(outputs[name], selected_uncertainty),
                 'Scientific output uncertainty disagrees with the candidate record')
        _require(stats[name][1] is None or (math.isfinite(stats[name][1]) and stats[name][1] >= 0),
                 'Invalid recorded uncertainty')
    metadata = json.loads(raw['metadata_json'])
    _require(metadata == {}, 'Only empty-metadata composition evaluations are audited')
    reconstructed = _EvaluationRecord(stats, metadata, {}, outputs)
    candidate = observation['candidate']
    key = composition_key(candidate)
    old_cache, old_batch = engine._stats_cache, engine._batch_stats
    try:
        engine._stats_cache, engine._batch_stats = OrderedDict([(key, reconstructed)]), {}
        reward = engine._combine(stats)
        _require(reward == observation['reward'] == raw['reward'], 'Recorded reward differs from the reconstructed objective/gates')
        exported = prediction_record(engine, candidate, (reward, None), k=float(engine.k))
        _require(_json_equal(exported, raw), 'Cache-only candidate export differs from its recorded evaluation')
    finally:
        engine._stats_cache, engine._batch_stats = old_cache, old_batch
    return key, reconstructed


def _cache_signature(cache):
    _require(isinstance(cache, (dict, OrderedDict)), 'Unsupported result cache type')
    signature = []
    for key, record in cache.items():
        signature.append((key, dict(record.stats), record.metadata, record.source_diagnostics,
                          {k: v.to_dict() for k, v in record.outputs.items()},
                          {k: v.to_dict() for k, v in getattr(record.stats, 'outputs', {}).items()}))
    return signature


def _rebuild_caches(engine, evidence):
    caches = evidence['checkpoint'].get('predictor_caches')
    _require(isinstance(caches, dict) and set(caches) == {'_stats_cache', '_batch_stats'},
             'Unexpected checkpoint result-cache schema')
    _require(not caches['_batch_stats'], 'Scalar-only recovery requires an empty batch cache')
    n = evidence['checkpoint']['hpo_state']['budget']['unique_evaluated']
    _require(n >= 1, 'Recovery requires a nonempty checkpoint prefix to verify reconstruction')
    reconstructed = OrderedDict()
    for index, (record, observation) in enumerate(zip(evidence['records'], evidence['observations'])):
        key, value = _record_from_observation(engine, record, observation)
        reconstructed[key] = value
        cap = engine._stats_cache_size
        if cap and cap > 0:
            while len(reconstructed) > cap:
                reconstructed.popitem(last=False)
        if index + 1 == n:
            _require(_cache_signature(reconstructed) == _cache_signature(caches['_stats_cache']),
                     'Recorded observations do not reproduce the saved prefix cache exactly')
    return {'_stats_cache': reconstructed, '_batch_stats': deepcopy(caches['_batch_stats'])}


def _rebuild_predictor_ledger(evidence):
    from rl_matdesign.utils.composition import composition_key
    saved = evidence['checkpoint']['predictor_ledger']
    n = evidence['checkpoint']['hpo_state']['budget']['unique_evaluated']
    _require(saved.get('schema_version') == 1 and saved.get('phase') == 'generate', 'Unsupported predictor ledger')
    _require(saved.get('diagnostic_requests', 0) == 0 and saved.get('diagnostic_failures', 0) == 0,
             'Diagnostic prediction work cannot be reconstructed')
    _require(set(saved['phase_costs']) == {'generate'}, 'Unexpected predictor phases')

    def expected(rows):
        keys = {composition_key(row['candidate']) for row in rows}
        _require(len(keys) == len(rows), 'Predictor observations include repeated candidates')
        _require(all(isinstance(row.get('timing', {}).get('cache_hit'), bool) for row in rows),
                 'Original cache-hit accounting is unavailable')
        best = max(row['reward'] for row in rows)
        return {'n_calls': len(rows), 'seen': sorted(keys, key=repr),
                't_predict_s': rows[-1]['prediction_elapsed_s'], 'best_mean': best,
                'phase_costs': {'generate': {'requests': len(rows), 'successes': len(rows), 'failures': 0,
                    'seen': sorted(keys, key=repr), 'best_reward': best,
                    'known_cache_hits': sum(row['timing']['cache_hit'] for row in rows),
                    'cache_status_known': len(rows)}}}
    prefix = expected(evidence['observations'][:n])
    for name, value in prefix.items():
        _require(saved.get(name) == value, 'Recorded observations do not reproduce checkpoint predictor ledger: ' + name)
    rebuilt = deepcopy(saved)
    rebuilt.update(expected(evidence['observations']))
    _require(math.isfinite(rebuilt['t_predict_s']) and rebuilt['t_predict_s'] >= saved['t_predict_s'],
             'Invalid cumulative prediction time')
    return rebuilt


def _replay(evidence, campaign, external):
    import joblib
    import torch
    from rl_matdesign.configuration import build_predictor, build_env
    from rl_matdesign.utils.seeding import set_global_seed, isolated_rng, restore_rng_state, capture_rng_state
    from rl_matdesign.utils.provenance import write_run_manifest
    from rl_matdesign.utils.execution_contract import execution_contract, require_execution_contract, generation_sampling_contract
    from rl_matdesign.utils.model_contract import load_artifact, validate_scaler
    from rl_matdesign.hpo.artifacts import file_digest
    from rl_matdesign.action_features import value_alphabet
    from rl_matdesign.model import PolicyNet, QRegressor
    from rl_matdesign.training import _precompute_elem_features, _pg_single_episode_generate, _rollout_policy_episode
    from rl_matdesign.utils.candidate_records import candidate_identity

    manifest, config = _provenance(evidence)
    directory = Path(evidence['directory'])
    paths = _model_paths(evidence, manifest, directory)
    external.update({str(path): file_digest(path) for path in paths.values()})
    for row in manifest['input_files']:
        if row.get('status') == 'present':
            _require(file_digest(row['path']) == row['sha256'], 'Original invocation input changed: ' + row['path'])
            external[row['path']] = row['sha256']
    saved = evidence['checkpoint']
    _require(generation_sampling_contract(config) == saved['generation_contract']['sampling'], 'Generation sampling settings changed')
    set_global_seed(config['train_seed'], deterministic=True)
    device = torch.device(manifest['runtime_args']['resolved_device'])
    engine = None
    with _working_directory(manifest['cwd']), tempfile.TemporaryDirectory(prefix='rlmat-generation-repair-') as scratch:
        try:
            with isolated_rng(config['eval_seed']):
                engine = build_predictor(config, seed=config['eval_seed'])
            env, _, _ = build_env(config, engine)
            # Fresh runtime and input fingerprints, never copied historical runtime.
            current_manifest = write_run_manifest(scratch, config, phase='generation-repair-check',
                config_path=directory / 'config.yaml', input_files=list(paths.values()),
                seeds=manifest['seeds'], runtime_args=manifest['runtime_args'])
            with isolated_rng(config['eval_seed']):
                contract = execution_contract(config, engine, current_manifest, device)
            require_execution_contract(saved['generation_contract']['execution'], contract, allow_source_upgrade=True)
            env.evaluation_contract = contract['evaluation']
            # Same reviewed model/source gate as an ordinary HPO generation job.
            env.hpo_recorder = object()
            _supported_engine(engine, env, config, saved)
            rebuilt_cache = _rebuild_caches(engine, evidence)
            rebuilt_ledger = _rebuild_predictor_ledger(evidence)
            scaler = joblib.load(paths['std_scaler.bin'])
            validate_scaler(scaler, env, str(paths['std_scaler.bin']))
            features, _ = _precompute_elem_features(env.species_set, env.state_featurizer, env=env)
            alphabet = value_alphabet(env)
            dimensions = dict(state_dim=int(scaler.n_features_in_), step_dim=env.n_components,
                              elem_dim=features.shape[1], frac_dim=alphabet.feature_dim)
            if config['method'] == 'a2c':
                model = PolicyNet(**dimensions).to(device)
                weight = load_artifact(paths['policy.pt'], env, device=device, scaler=scaler)
                model.load_state_dict(weight['policy_state'])
            else:
                model = QRegressor(**dimensions, hidden_dim=int(config['dqn_hidden_dim'])).to(device)
                weight = load_artifact(paths['qnet.pt'], env, device=device, scaler=scaler)
                model.load_state_dict(weight['qnet_state'])
            model.eval()
            del env.hpo_recorder
            env.constraints_enabled = True
            expected = evidence['observations'][-1]['candidate']
            raw = evidence['records'][-1]
            calls = []
            def recorded_reward(candidate):
                _require(not calls, 'Replay requested more than one terminal evaluation')
                _require(_json_equal(candidate, expected), 'Replayed predictor input differs from the recorded candidate')
                calls.append(candidate)
                return raw['prediction_record']['reward']
            recorded_reward.evaluate_candidate = recorded_reward
            env.reward_fn = recorded_reward
            # Every initialization and cache inspection precedes this restoration.
            restore_rng_state(saved['rng_state'])
            args = dict(env=env, scaler=scaler, device=device, elem_feats_scaled=features,
                        fraction_set=alphabet, gen_epsilon=0., gen_top_frac=0.,
                        gen_temperature=config['gen_temperature'])
            if config['method'] == 'a2c':
                _pg_single_episode_generate(policy=model, **args)
            else:
                _rollout_policy_episode(qnet=model, **args)
            rng = capture_rng_state()
            _require(len(calls) == 1, 'Replay did not produce exactly one terminal candidate')
            _require(candidate_identity(env.terminal_comp_key()) == raw['identity'],
                     'Replayed canonical composition identity differs from the durable record')
            _require(env.terminal_formula == raw['formula'], 'Replayed formula differs from the durable record')
            repaired = deepcopy(saved)
            repaired['rng_state'] = rng
            repaired['predictor_caches'] = rebuilt_cache
            repaired['predictor_ledger'] = rebuilt_ledger
            repaired['hpo_state'] = {'phase': 'generation', 'budget': {**deepcopy(evidence['ledger']), 'status': 'RUNNING'},
                                     'seen': sorted(row['identity'] for row in evidence['records']),
                                     'records_sha256': file_digest(directory / 'candidate_records.jsonl')}
            repaired['hpo_state']['budget'].pop('phase', None)
            return repaired, {'runtime_contract_verified': True, 'frozen_artifacts_verified': True,
                              'prefix_cache_and_costs_reconstructed': True, 'replayed_episodes': 1,
                              'predictor_evaluations': 0, 'replayed_candidate_matches': True,
                              'from_unique': saved['hpo_state']['budget']['unique_evaluated'],
                              'to_unique': evidence['ledger']['unique_evaluated'],
                              'budget_target': evidence['ledger']['target_unique'],
                              'retained_prediction_seconds': rebuilt_ledger['t_predict_s']}
        finally:
            if engine is not None:
                engine.close()


def recover(out, job_id, *, apply=False):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from hpo_generation_repair_io import validate_guard, apply_repair, applied_receipt
    from rl_matdesign.hpo.artifacts import read_json, file_digest, package_digest
    from rl_matdesign.hpo.spec import load_plan
    from rl_matdesign.hpo.controller import Campaign
    _require(isinstance(job_id, str) and re.fullmatch(r'[a-f0-9]{24}', job_id), 'Expected one HPO job ID')
    out = Path(out).expanduser().resolve()
    directory = out / 'jobs' / job_id
    _require(directory.is_dir() and not directory.is_symlink() and directory.resolve().is_relative_to(out),
             'Job directory missing or outside the campaign')
    manifest = read_json(out / 'manifest.json')
    plan = load_plan(manifest['plan']['source_path'], method=manifest['plan']['method'])
    campaign = Campaign(plan, out)  # Read-only for an existing verified campaign.
    previous = applied_receipt(directory)
    if previous:
        return {'ready_to_apply': False, 'applied': True, 'already_applied': True, 'receipt': previous,
                'next_step': 'Run the normal hpo restart --check-only before continuing'}
    evidence = validate_guard(directory)
    external = {str(out / 'manifest.json'): file_digest(out / 'manifest.json')}
    for row in manifest['configuration_sources'].values():
        external[row['path']] = file_digest(row['path'])
    initial_source = package_digest()
    started = time.monotonic()
    repaired, details = _replay(evidence, campaign, external)
    _require(package_digest() == initial_source, 'Source files changed during recovery validation')
    for path, expected in external.items():
        _require(file_digest(path) == expected, 'External artifact changed during validation: ' + path)
    for path, expected in external.items():
        _require(path not in evidence['external_hashes'] or evidence['external_hashes'][path] == expected,
                 'Conflicting external artifact fingerprints: ' + path)
        evidence['external_hashes'][path] = expected
    details.update(software_digest=initial_source, external_file_hashes=external,
                   replay_wall_seconds=time.monotonic() - started,
                   checkpoint_contract_preserved=True, original_paid_attempts_preserved=True)
    report = {'job_id': job_id, 'ready_to_apply': True, 'applied': False, **details}
    if apply:
        report['receipt'] = apply_repair(directory, repaired, evidence, details)
        report['applied'] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--apply', action='store_true', help='Back up originals and install the validated reconstructed checkpoint')
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            report = recover(args.out, args.job_id, apply=args.apply)
        code = 0
    except Exception as exc:
        report = {'job_id': args.job_id, 'applied': None if args.apply else False, 'status': 'ERROR' if args.apply else 'REFUSED',
                  'error': f'{type(exc).__name__}: {exc}',
                  'next_step': 'Preserve all original files and share this error; an interrupted apply can be checked by rerunning this tool; no budget rollback is permitted'}
        code = 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
