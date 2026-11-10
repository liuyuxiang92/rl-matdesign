#!/usr/bin/env python3
"""Plan or run identical composition domains under four search methods.

Each arm starts in a fresh process with an empty oracle cache. The primary
comparison counts successful unique paid candidates; extra generation is skipped.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rl_matdesign.configuration import load_config, build_env
from rl_matdesign.config_schema import resolve_config
from rl_matdesign.utils.design_count import sampler_from_env


def _completion(row, returncode):
    """A zero process status alone does not prove that the paid budget finished."""
    name = 'benchmark.json' if row['method'] == 'random' else 'timing.json'
    path = Path(row['output']) / name
    completed, reason = None, 'missing_completion_evidence'
    if path.exists():
        try:
            report = json.loads(path.read_text())
            completed = (report['budget_completed'] if row['method'] == 'random'
                         else report['predictor']['n_unique'])
            reason = report.get('stop_reason', 'budget_complete')
        except (ValueError, KeyError, TypeError):
            reason = 'invalid_completion_evidence'
    success = returncode == 0 and completed == row['budget'] and reason == 'budget_complete'
    return {**row, 'returncode': returncode, 'budget_completed': completed,
            'status': 'complete' if success else 'incomplete', 'stop_reason': reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    parser.add_argument('--methods', nargs='+', choices=['dqn-mc', 'dqn-bootstrap', 'a2c', 'random'],
                        default=['dqn-mc', 'dqn-bootstrap', 'a2c', 'random'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run', action='store_true', help='Execute; otherwise only write the exact plan and counts')
    args = parser.parse_args(argv)
    if args.budget < 1:
        parser.error('--budget must be positive')
    if any(seed < 0 for seed in args.seeds):
        parser.error('--seeds must be nonnegative')
    out = Path(args.out).resolve()
    rows, outputs, configurations = [], set(), {}
    # Validate the whole plan before writing any arm; a later collision must not
    # overwrite an earlier arm's configuration or a completed experiment.
    for config in args.config:
        cfg = load_config(config)
        env, _, _ = build_env(cfg, None)
        domain = sampler_from_env(env)
        if args.budget > domain.size:
            parser.error('Requested budget exceeds the unique domain size')
        for seed in args.seeds:
            for method in args.methods:
                directory = out / Path(config).stem / f'{method}-seed{seed}'
                if directory in outputs:
                    parser.error(f'Duplicate benchmark output {directory}; use distinct config stems, seeds and methods')
                outputs.add(directory)
                existing = list(directory.iterdir()) if directory.exists() else []
                if any(path.name != 'config.yaml' for path in existing):
                    parser.error(f'Refusing to overwrite an existing run at {directory}; choose a new --out')
                authored = {**cfg, 'train_budget': args.budget, 'train_count_duplicates': False,
                            'train_seed': seed, 'gen_seed': seed,
                            'method': 'a2c' if method == 'a2c' else 'dqn'}
                authored.pop('dqn_num_train_eps', None)
                authored.pop('pg_num_iters', None)
                authored['dqn_warmup_eps'] = min(int(cfg.get('dqn_warmup_eps', 100)), args.budget)
                resolve_config(authored)  # Reject a non-runnable arm before executing any others.
                trial = directory / 'config.yaml'
                content = yaml.safe_dump(authored, sort_keys=False)
                if trial.exists() and trial.read_text() != content:
                    parser.error(f'Existing dry-plan configuration differs at {trial}; choose a new --out')
                configurations[trial] = content
                if method == 'random':
                    command = [sys.executable, str(ROOT / 'scripts/baselines/run_random.py'),
                               '--config', str(trial), '--out', str(directory), '--budget', str(args.budget), '--seed', str(seed)]
                else:
                    command = [sys.executable, str(ROOT / 'scripts/run_experiment.py'), '--config', str(trial),
                               '--out', str(directory), '--device', args.device, '--method', authored['method'],
                               '--skip-generation', '--save-checkpoint-freq', '100']
                    if method.startswith('dqn'):
                        command += ['--dqn-target-mode', method.removeprefix('dqn-')]
                rows.append({'config': str(Path(config).resolve()), 'domain_size': domain.size,
                             'horizon': env.n_components, 'initial_action_count': len(env.allowed_actions()),
                             'seed': seed, 'method': method, 'budget': args.budget,
                             'command': command, 'output': str(directory)})
    for trial, content in configurations.items():
        trial.parent.mkdir(parents=True, exist_ok=True)
        if not trial.exists():
            trial.write_text(content)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / 'benchmark_plan.json'
    plan.write_text(json.dumps(rows, indent=2))
    print(f'{len(rows)} runs; exact plan: {plan}')
    if args.run:
        results = []
        for row in rows:
            process = subprocess.run(row['command'], cwd=ROOT, check=False)
            results.append(_completion(row, process.returncode))
            (out / 'benchmark_results.json').write_text(json.dumps(results, indent=2))
        if any(row['status'] != 'complete' for row in results):
            raise SystemExit('Some arms failed or did not complete the exact budget; inspect benchmark_results.json')


if __name__ == '__main__':
    main()
