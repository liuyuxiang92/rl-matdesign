#!/usr/bin/env python3
"""Evaluate explicit LiPS precursor choices with the scenario's structure pipeline.

Example: --config configs/lips_sse.yaml --dopant Ni2O3:0.04 --dopant Mn:0.02
--cl 1.2 --save-poscar scored.vasp. Amounts are cation fractions. Compound labels
retain the chosen precursor form; no oxygen-form flag or formula inference is used.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _sub in ('src', 'scripts'):
    sys.path.insert(0, str(_ROOT / _sub))
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')


class SceneMeta:
    """Read the precursor and halide roles from the explicit pipeline operation."""
    def __init__(self, cfg):
        steps = cfg.get('structure', {}).get('steps', [])
        balances = [step for step in steps if step.get('type') == 'sse_charge_balance']
        if len(balances) != 1:
            raise ValueError('LiPS evaluation requires exactly one sse_charge_balance operation')
        balance = balances[0]
        self.p_group = balance['p_site_group']
        self.s_group = balance['halide_group']
        self.host_p = balance.get('host', {}).get('P', 'P')
        self.groups = {group['name']: group for group in cfg['groups']}
        if set(self.groups) != {self.p_group, self.s_group}:
            raise ValueError('LiPS shorthand requires the configured precursor and halide groups only')
        self.species_set = self.groups[self.p_group]['species_set']
        if self.host_p not in self.species_set:
            raise ValueError('The host must be an explicit precursor choice')
        from rl_matdesign.predictors.builders.pipeline import pipeline_group_constraints
        self.validators = pipeline_group_constraints(cfg['structure'], cfg['groups'])


def parse_dopant_flag(spec):
    parts = spec.split(':')
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError('--dopant expects CHOICE:AMOUNT, for example Ni2O3:0.04')
    amount = float(parts[1])
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError('Dopant amounts must be finite and positive')
    return parts[0].strip(), amount


def validate_candidate(meta, candidate):
    """Check menu/budget reachability and intrinsic chemistry without model I/O."""
    from rl_matdesign.amounts import ChemicalAmountsGroup
    if not isinstance(candidate, dict) or set(candidate) != set(meta.groups):
        raise ValueError('Candidate group names must match the scenario')
    result = {}
    for name, spec in meta.groups.items():
        values = candidate[name]
        if not isinstance(values, dict) or set(values) - set(spec['species_set']):
            raise ValueError(f'{name}: unknown choice or non-mapping amount recipe')
        exact = {}
        for label in spec['species_set']:
            raw = values.get(label, 0)
            if isinstance(raw, bool):
                raise ValueError(f'{name}.{label}: amount must be numerical')
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}.{label}: amount must be finite and nonnegative')
            lower, upper = spec.get('element_bounds', {}).get(label, (-math.inf, math.inf))
            if not float(lower) <= value <= float(upper):
                raise ValueError(f'{name}.{label}: amount violates final bounds')
            exact[label] = [value, value]
        # Fix every final amount; feasibility then proves the original action
        # menus and exact budget can construct this candidate, including zeros.
        ChemicalAmountsGroup({**spec, 'element_bounds': exact}, state_featurizer=lambda _: [0.0])
        validator = meta.validators.get(name)
        if validator is not None and not validator(values):
            raise ValueError(f'{name}: use at most one positive precursor form per metal')
        result[name] = {label: float(value) for label, value in values.items() if float(value)}
    return result


def build_candidate(meta, dopants, cl_count):
    p_site = {}
    for label, amount in dopants:
        if label == meta.host_p or label not in meta.species_set:
            raise ValueError(f'Unknown dopant choice {label!r}; use an explicit configured precursor')
        p_site[label] = p_site.get(label, Fraction(0)) + Fraction(str(amount))
    total = Fraction(str(meta.groups[meta.p_group].get('total', 1)))
    p_site[meta.host_p] = total - sum(p_site.values(), Fraction(0))
    return validate_candidate(meta, {
        meta.p_group: {label: float(value) for label, value in p_site.items()},
        meta.s_group: {'Cl': cl_count},
    })


def evaluate_one(predictor, candidate):
    """Use one scored ensemble for statistics and optional structure export."""
    breakdown = predictor.score_breakdown(candidate)
    stats = breakdown['stats']
    structure = next((samples[0] for samples in breakdown.get('structures', {}).values()
                      if isinstance(samples, (list, tuple)) and samples), None)
    formula = (structure.get_chemical_formula() if structure is not None
               else predictor.composition_formula(candidate))
    return {'formula': formula or '?', 'reward': float(breakdown['reward']),
            'dp_mean': float(predictor.raw_combine(stats)), 'stats': stats,
            'selected_alternative': breakdown.get('reward_details', {}).get('selected_alternative'),
            'structure': structure}


def save_poscar(path, structure):
    if structure is None:
        raise ValueError('No scored structure was returned; cannot export an unevaluated rebuild')
    from ase.io import write
    write(path, structure, format='vasp', sort=True, vasp5=True)


def write_csv(path, rows):
    names = list(rows[0]['stats'])
    fields = ['formula', 'reward', 'dp_mean', 'selected_alternative']
    fields += [f'{name}_{stat}' for name in names for stat in ('mean', 'std')]
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields[:4]}
            for name in names:
                flat[f'{name}_mean'], flat[f'{name}_std'] = row['stats'][name]
            writer.writerow(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dopant', action='append', default=[], metavar='CHOICE:AMOUNT')
    parser.add_argument('--cl', type=float)
    parser.add_argument('--candidates-file', type=Path, help='JSON list of complete named-group recipes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-poscar', type=Path, help='Export the scored structure; one candidate only')
    parser.add_argument('--out-csv', type=Path)
    args = parser.parse_args(argv)
    from rl_matdesign.configuration import load_config
    cfg = load_config(args.config)
    meta = SceneMeta(cfg)
    items = []
    try:
        if args.dopant:
            if args.cl is None:
                raise ValueError('--cl is required with --dopant')
            items.append(build_candidate(meta, [parse_dopant_flag(x) for x in args.dopant], args.cl))
        if args.candidates_file:
            values = json.loads(args.candidates_file.read_text())
            if not isinstance(values, list):
                raise ValueError('--candidates-file must contain a JSON list')
            items.extend(validate_candidate(meta, value) for value in values)
        if not items:
            raise ValueError('Supply --dopant and --cl, or --candidates-file')
        if args.save_poscar and len(items) != 1:
            raise ValueError('--save-poscar requires exactly one candidate')
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    from rl_matdesign.registry import build_reward
    predictor = build_reward(cfg, seed=args.seed)
    try:
        rows = []
        for candidate in items:
            row = evaluate_one(predictor, candidate)
            rows.append(row)
            print(f"{row['formula']}: reward={row['reward']:.8g}, selected_alternative={row['selected_alternative']}")
            for name, (mean, std) in row['stats'].items():
                print(f'  {name}: mean={mean}, std={std}')
            if args.save_poscar:
                save_poscar(args.save_poscar, row['structure'])
        if args.out_csv:
            write_csv(args.out_csv, rows)
    finally:
        close = getattr(predictor, 'close', None)
        if callable(close):
            close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
