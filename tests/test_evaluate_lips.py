from pathlib import Path
from types import SimpleNamespace
import csv

import pytest
import yaml

from scripts.evaluate_lips import SceneMeta, build_candidate, evaluate_one, parse_dopant_flag, save_poscar, write_csv


def meta():
    root = Path(__file__).resolve().parents[1]
    return SceneMeta(yaml.safe_load((root / 'configs/lips_sse.yaml').read_text()))


def test_explicit_precursor_and_halide_recipe_retains_cation_basis():
    candidate = build_candidate(meta(), [('Ni2O3', .04), ('Mn', .02)], 1.2)
    assert candidate == {'P_site': {'Ni2O3': .04, 'Mn': .02, 'P': .94}, 'halide': {'Cl': 1.2}}


@pytest.mark.parametrize('dopants,cl,match', [
    ([('Ni', .04), ('Ni2O3', .02)], 1.2, 'precursor form'),
    ([('Ni', .01)], 1.2, 'bounds'),
    ([('Ni', .04)], .7, 'infeasible'),
    ([('Ni', .025)], 1.2, 'infeasible'),
    ([('Unknown', .04)], 1.2, 'Unknown dopant'),
])
def test_explicit_recipe_validates_bounds_menus_and_intrinsic_chemistry(dopants, cl, match):
    with pytest.raises(ValueError, match=match):
        build_candidate(meta(), dopants, cl)


def test_old_oxygen_form_flag_is_not_interpreted():
    with pytest.raises(ValueError, match='CHOICE:AMOUNT'):
        parse_dopant_flag('Ni:.04:1')
    assert parse_dopant_flag('Ni2O3:.04') == ('Ni2O3', .04)


def test_export_uses_the_scored_structure_without_another_evaluation(tmp_path):
    from ase import Atoms
    atoms = Atoms('Li2O', cell=[10, 10, 10], pbc=True)
    calls = []
    def breakdown(candidate):
        calls.append(candidate)
        return dict(stats={'T300': (1., None), 'T500': (2., .1)}, reward=2.,
                    reward_details={'selected_alternative': 'T500'}, structures={'T300': [atoms], 'T500': [atoms]})
    predictor = SimpleNamespace(score_breakdown=breakdown, raw_combine=lambda stats: 2.)
    row = evaluate_one(predictor, {'candidate': 1})
    assert row['structure'] is atoms and len(calls) == 1
    save_poscar(tmp_path / 'scored.vasp', row['structure'])
    assert len(calls) == 1
    write_csv(tmp_path / 'result.csv', [row])
    with (tmp_path / 'result.csv').open() as stream:
        result = next(csv.DictReader(stream))
    assert result['T300_std'] == '' and result['T500_mean'] == '2.0'
    assert result['selected_alternative'] == 'T500'
    with pytest.raises(ValueError, match='scored structure'):
        save_poscar(tmp_path / 'missing.vasp', None)



def test_removed_geometry_override_is_rejected_before_loading_config(capsys):
    from scripts.evaluate_lips import main
    with pytest.raises(SystemExit) as error:
        main(['--config', 'unused.yaml', '--no-geo-opt'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --no-geo-opt' in capsys.readouterr().err


@pytest.mark.parametrize('include_geometry', [False, True])
def test_evaluator_uses_exactly_the_declared_structure_operations(tmp_path, monkeypatch, include_geometry):
    from scripts.evaluate_lips import main
    import rl_matdesign.registry as registry

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'configs/lips_sse.yaml').read_text())
    steps = [step for step in cfg['structure']['steps'] if step['type'] != 'geo_opt']
    if include_geometry:
        steps.append({'type': 'geo_opt', 'model': 'unused.pt', 'relax_cell': False})
    cfg['structure']['steps'] = steps
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    seen, closed = [], []

    def build(config, *, seed):
        seen.append(config['structure']['steps'])
        return SimpleNamespace(
            score_breakdown=lambda candidate: {'reward': 1., 'stats': {}},
            raw_combine=lambda stats: 1.,
            composition_formula=lambda candidate: 'Li6PS5Cl',
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(registry, 'build_reward', build)
    assert main(['--config', str(path), '--dopant', 'Ni:0.04', '--cl', '1.2']) == 0
    assert seen == [steps] and closed == [True]
