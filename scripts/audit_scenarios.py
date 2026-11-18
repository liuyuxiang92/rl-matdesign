#!/usr/bin/env python3
"""Audit authored scenario/HPO contracts without importing scientific models.

Optional domain checks construct the configured filters and initial legal-action
mask. They do not certify terminal feasibility, plugin-owned peptide constraints,
model capabilities, assets, scientific units or numerical predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rl_matdesign.config_schema import load_yaml_config, validate_config
from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.hpo.search_space import validate_search_space
from rl_matdesign.structure_sources import resolve_structure_plan

_STRUCTURE = {'dp_energy', 'dp_property', 'dp_classification', 'mgtransformer',
              'rl_matdesign.predictors.mgtransformer:MGTransformerPredictor', 'ooh_frames'}
_COMPOSITION = {'rf_magpie', 'lookup', 'glassnet', 'dummy'}


class _NoOracle:
    def predict(self, candidate):
        raise AssertionError('Schema/domain audit must never evaluate an oracle')


def audit_file(path: Path, root: Path, *, domains=False):
    row = {'path': str(path.relative_to(root)),
           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'model_inference': 'not_run', 'limitations': []}
    raw = load_yaml_config(path)
    if 'search_space' in raw:
        row['kind'] = 'hpo'
        ref = Path(raw['base_config'])
        candidates = [ref] if ref.is_absolute() else [path.parent / ref, root / ref]
        base = next((p.resolve() for p in candidates if p.is_file()), None)
        if base is None:
            raise ValueError(f'base_config does not exist: {ref}')
        cfg = load_config(str(base))
        overrides = raw.get('fixed_overrides', {})
        validate_config({**cfg, **overrides, 'method': raw['method']})
        validate_search_space(raw['search_space'], fixed_overrides=overrides, base_values=cfg)
        # A typo in a sampled parameter must fail before an expensive HPO run.
        from rl_matdesign.hpo.search_space import sample_from_search_space
        class LowerBoundTrial:
            def suggest_float(self, name, low, high, **kwargs): return low
            def suggest_int(self, name, low, high, **kwargs): return low
            def suggest_categorical(self, name, choices): return choices[0]
        sample = sample_from_search_space(LowerBoundTrial(), raw['search_space'], resolved_values={**cfg, **overrides})
        validate_config({**cfg, **overrides, **sample, 'method': raw['method']})
        try:
            base_label = str(base.relative_to(root))
        except ValueError:
            base_label = str(base)
        row.update(base_config=base_label, status='passed',
                   limitations=['Search-space schema and one representative trial config only; no HPO executed.'])
        return row
    cfg = load_config(str(path))
    row['kind'] = 'scenario'
    props = cfg['properties']
    external = [p for p in props if p['predictor'] not in _STRUCTURE | _COMPOSITION]
    needs_structure = [p['name'] for p in props if p['predictor'] in _STRUCTURE]
    plan = resolve_structure_plan(cfg, needs_structure=needs_structure)
    row['structure'] = None if plan.structure is None else plan.structure.builder
    row['structure_properties'] = list(plan.property_names)
    row['outputs'] = []
    for p in props:
        uncertainty = p.get('uncertainty')
        objective = p.get('objective', 'mean')
        paths = p.get('models', [])
        count = len(paths) if isinstance(paths, (list, tuple)) else 1
        if objective != 'mean' and uncertainty == 'model_spread' and count < 2:
            raise ValueError(f"{p['name']} requests unavailable singleton model_spread")
        row['outputs'].append({k: p.get(k) for k in ('name','predictor','output','objective','uncertainty','unit','representation','transform') if k in p})
        if p.get('unit') is None:
            row['limitations'].append(f"{p['name']}: physical unit unverified; YAML acceptance is not model provenance.")
    if external:
        row['limitations'].append('External plugin capabilities and predictor-supplied peptide constraints require installed-plugin validation.')
    if any(p['predictor'] == 'ooh_frames' for p in props):
        row['limitations'].append('OOH frame/output contract needs its actual template and models for inference validation.')
    if domains:
        env, _, _ = build_env(cfg, _NoOracle())
        env.initialize()
        legal = env.allowed_actions()
        row['domain'] = {'status': 'constructed_with_configured_filters',
                         'initial_legal_actions': len(legal),
                         'predictor_constraints': 'not_loaded', 'rollout': 'not_run'}
        if not len(legal):
            raise ValueError('No initial legal actions')
    row['status'] = 'passed'
    return row


def audit_inventory(root, paths, *, domains=False):
    root = Path(root).resolve()
    entries = []
    for rel in paths:
        try:
            entries.append(audit_file(root / rel, root, domains=domains))
        except Exception as exc:
            entries.append({'path': str(rel), 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
    return {'schema_version': 1, 'scope': 'Executable config/structure/HPO audit; not scientific model validation.',
            'configuration_count': len(entries), 'failed': sum(e['status']=='failed' for e in entries), 'entries': entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inventory', type=Path,
                        help='Local JSON inventory with entries containing relative config paths')
    inputs.add_argument('--config', type=Path, action='append', help='Scenario or HPO YAML; repeatable')
    parser.add_argument('--domains', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    paths = ([row['path'] for row in json.loads(args.inventory.read_text())['entries']]
             if args.inventory else [str(path.resolve().relative_to(args.root.resolve())) for path in args.config])
    report = audit_inventory(args.root, paths, domains=args.domains)
    text = json.dumps(report, indent=2) + '\n'
    if args.output: args.output.write_text(text)
    else: print(text, end='')
    print(f"Audited {report['configuration_count']} files; failures={report['failed']}", file=sys.stderr)
    return bool(report['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
