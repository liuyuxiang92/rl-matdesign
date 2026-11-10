#!/usr/bin/env python3
"""Uniform random search on unique bounded, gridded composition maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl_matdesign.configuration import load_config, build_env, build_predictor
from rl_matdesign.config_schema import resolve_config
from rl_matdesign.utils.design_count import sampler_from_env
from rl_matdesign.utils.provenance import write_run_manifest
from rl_matdesign.utils.seeding import isolated_rng
from rl_matdesign.utils.timing import PredictorTimer
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError


def main(argv=None):
    start = time.perf_counter()  # Includes parsing, configuration, domain counting and setup, as RL does.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.budget < 1 or args.seed < 0:
        parser.error('--budget must be positive and --seed must be nonnegative')
    authored = load_config(args.config)
    # This arm has no DQN warmup. Its explicit CLI budget is the canonical budget
    # recorded in provenance, even when the source YAML has larger RL defaults.
    authored.pop('dqn_num_train_eps', None)
    authored.pop('pg_num_iters', None)
    cfg = resolve_config(authored, {'train_seed': args.seed, 'train_count_duplicates': False,
                                    'train_budget': args.budget, 'dqn_warmup_eps': 0}).config
    env, _, _ = build_env(cfg, None)
    sampler = sampler_from_env(env)
    if args.budget > sampler.size:
        parser.error(f'Budget exceeds the {sampler.size} unique candidates')
    out = Path(args.out)
    if out.exists() and any(path.name != 'config.yaml' for path in out.iterdir()):
        parser.error(f'Refusing to overwrite an existing random-search run at {out}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = write_run_manifest(out, cfg, config_path=args.config, phase='uniform_random',
                                  seeds={'eval': cfg['eval_seed'], 'search': args.seed})
    inner, timer, error = None, None, None
    seen, duplicates, failures = set(), 0, 0
    stop_reason = 'setup_failed'
    try:
        with isolated_rng(cfg['eval_seed']):
            inner = build_predictor(cfg, seed=cfg['eval_seed'])
            identity_hook = getattr(inner, 'oracle_identity', None)
            identity = identity_hook() if callable(identity_hook) else {
                'coverage': 'adapter has no oracle_identity; external model assets are unverified'}
            (manifest.parent / 'oracle_identity.json').write_text(json.dumps(identity, indent=2, allow_nan=False))
            source_hook = getattr(inner, 'source_identity', None)
            if callable(source_hook):
                (manifest.parent / 'source_identity.json').write_text(json.dumps(source_hook(), indent=2, allow_nan=False))
        timer = PredictorTimer(inner, t0=start, episode_log_path=str(out / 'episode_timing.csv'), eval_seed=cfg['eval_seed'])
        timer.set_phase('train')
        rng = random.Random(args.seed)
        stop_reason = 'in_progress'
        while len(seen) < args.budget:
            units = sampler.sample(rng)
            key = tuple(sorted(units.items()))
            comp = {name: amount / sampler.total_units for name, amount in units.items()}
            # Sampling is uniform with replacement. Duplicate requests and any
            # recomputation after cache eviction remain in the cost ledger.
            try:
                timer.predict(comp)
            except EvaluationProtocolError:
                raise
            except Exception:
                failures += 1
                if failures >= 11:
                    stop_reason = 'consecutive_evaluation_failures'
                    break
                continue
            failures = 0
            duplicates = duplicates + 1 if key in seen else 0
            seen.add(key)
            if duplicates >= cfg['train_max_duplicate_attempts'] and len(seen) < args.budget:
                stop_reason = 'duplicate_guard'
                break
        if len(seen) == args.budget:
            stop_reason = 'budget_complete'
    except BaseException as exc:
        error = exc
        stop_reason = ('protocol_error' if isinstance(exc, EvaluationProtocolError) else
                       'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'execution_error')
    finally:
        close = getattr(inner, 'close', None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                if error is None:
                    error, stop_reason = exc, 'cleanup_error'
        report = {'method': 'uniform_random', 'domain_size': sampler.size,
                  'budget_requested': args.budget, 'budget_completed': len(seen),
                  'stop_reason': stop_reason, 'total_s': time.perf_counter() - start,
                  'predictor': timer.summary() if timer is not None else None,
                  'manifest': str(manifest),
                  'error': None if error is None else f'{type(error).__name__}: {error}'}
        (out / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    if error is not None:
        raise error
    if stop_reason != 'budget_complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
