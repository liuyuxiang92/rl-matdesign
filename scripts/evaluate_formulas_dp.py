#!/usr/bin/env python3
"""Evaluate elemental amount recipes using a complete current scenario YAML.

Example: --config configs/ooh.yaml --formula Ni0.70Fe0.15Ce0.05Er0.05Tm0.05. Formula amounts are
exact design-group amounts, without a scaffold suffix. Templates, adsorbates,
simulation, models and uncertainty are taken from the scenario configuration.
For compound precursor choices use evaluate_lips.py or the full named candidate
interface rather than trying to infer recipes from an elemental formula.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import sys

_ROOT=Path(__file__).resolve().parents[1]
for _sub in ('src','scripts'):
    sys.path.insert(0,str(_ROOT/_sub))
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')


def formula_candidate(formula, group):
    from pymatgen.core import Composition
    from rl_matdesign.amounts import ChemicalAmountsGroup
    if any(re.fullmatch(r'[A-Z][a-z]?', label) is None for label in group['species_set']):
        raise ValueError('Elemental formula input cannot preserve compound precursor identity')
    composition=Composition(formula).get_el_amt_dict()
    unknown=set(composition)-set(group['species_set'])
    if unknown:
        raise ValueError(f'Formula contains choices outside the design group: {sorted(unknown)}; omit scaffold atoms')
    exact={}
    for label in group['species_set']:
        value=float(composition.get(label,0))
        lower,upper=group.get('element_bounds',{}).get(label,(-float('inf'),float('inf')))
        if not lower <= value <= upper:
            raise ValueError(f'{label}: final amount violates element_bounds')
        exact[label]=[value,value]
    ChemicalAmountsGroup({**group,'element_bounds':exact},state_featurizer=lambda _: [0.])
    return {group['name']:{label:float(value) for label,value in composition.items() if value}}


def iter_formulas(args):
    yield from (formula.strip() for formula in args.formula if formula.strip())
    if args.formulas_file:
        for line in args.formulas_file.read_text().splitlines():
            line=line.strip()
            if line and not line.startswith('#'):
                yield line.split()[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--formula',action='append',default=[])
    parser.add_argument('--formulas-file',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--out-csv',type=Path)
    args=parser.parse_args(argv)
    from rl_matdesign.configuration import load_config
    from rl_matdesign.registry import build_reward
    cfg=load_config(args.config)
    groups=cfg['groups']
    if len(groups)!=1 or groups[0].get('kind')!='composition' or not cfg.get('structure'):
        parser.error('Formula evaluation requires one composition group and an explicit structure pipeline')
    try:
        items=[(formula,formula_candidate(formula,groups[0])) for formula in iter_formulas(args)]
        if not items: raise ValueError('Supply --formula or --formulas-file')
    except (ValueError,TypeError) as exc:
        parser.error(str(exc))
    predictor=build_reward(cfg,seed=args.seed)
    from evaluate_lips import evaluate_one
    rows=[]
    try:
        for formula,candidate in items:
            result=evaluate_one(predictor,candidate)
            row={'candidate_formula':formula,'structure_formula':result['formula'],'reward':result['reward'],'dp_mean':result['dp_mean']}
            for name,(mean,std) in result['stats'].items():
                row[f'{name}_mean']=mean; row[f'{name}_std']=std
            rows.append(row)
            print(row)
        if args.out_csv:
            with args.out_csv.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finally:
        close=getattr(predictor,'close',None)
        if callable(close): close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
