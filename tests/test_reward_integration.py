"""Reward rules across structure, composition, sweep and generation consumers."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.registry import build_reward


class CompositionProperty:
    """Deterministic model-free leaf with property values supplied by candidate."""
    def __init__(self, cfg, *, seed=None):
        self.name = cfg['name']

    def score(self, candidate):
        return candidate[self.name]


def _cfg():
    return {
        'properties': [
            {'name': name, 'predictor': f'{__name__}:CompositionProperty',
             'direction': 'max', 'objective': 'mean', 'weight': 17.0}
            for name in ['delta', 'pure', 'cost']
        ],
        'reward': {
            'type': 'gated',
            'gates': [{'name': 'synthesis', 'score': {'type': 'weighted_sum',
                       'terms': {'pure': 1.0}}, 'min': 0.8, 'penalty_weight': 2.0}],
            'objective': {'type': 'weighted_sum', 'terms': {'delta': 1.0, 'cost': 0.5}},
            'failure': {'reward': -10.0},
        },
    }


def test_composition_only_gate_has_no_builder_and_preserves_raw_delta():
    cfg = _cfg()
    cfg['properties'][2]['direction'] = 'min'
    predictor = build_reward(cfg)
    assert predictor._shared_builder is None
    good = {'delta': 1200.0, 'pure': 0.8, 'cost': 20.0}
    bad = {'delta': 100000.0, 'pure': 0.6, 'cost': 0.0}
    # Explicit reward weights replace each property's 17, with no delta scaling.
    assert predictor.predict(good) == (1190.0, 0.0)
    assert predictor.predict(bad)[0] == pytest.approx(-10.4)
    assert predictor.predict_raw(bad)[0] == pytest.approx(-10.4)
    assert predictor.per_objective_stats(good)['delta'] == (1200.0, 0.0)
    assert predictor.reward_details(bad)['gates'][0]['violation'] == pytest.approx(0.2)


def test_gates_use_raw_means_while_objective_keeps_direction_scale_and_uncertainty():
    cfg = _cfg()
    cfg['properties'][0].update(objective='mean_minus_kstd')
    cfg['properties'][1].update(direction='min', scale=100.0)
    cfg['properties'][2].update(direction='min', scale=2.0)
    cfg['k'] = 2.0
    p = StructureScorePredictor(cfg)
    stats = {'delta': (100.0, 3.0), 'pure': (0.8, 0.4), 'cost': (20.0, 0.0)}
    assert p._combine(stats) == pytest.approx(89.0)
    assert p.raw_combine(stats) == pytest.approx(95.0)
    assert p._reward_details(stats)['gate_passed'] is True


def test_multiple_gates_and_weighted_gate_score():
    cfg = _cfg()
    cfg['reward']['gates'][0]['score']['terms'] = {'pure': 0.5, 'delta': 0.5}
    cfg['reward']['gates'].append({'name': 'cost', 'score': {'terms': {'cost': 1}},
                                   'max': 5.0, 'penalty_weight': 3.0})
    p = StructureScorePredictor(cfg)
    candidate = {'delta': 0.8, 'pure': 0.8, 'cost': 6.0}
    assert p.predict(candidate)[0] == -13.0
    details = p.reward_details(candidate)
    assert [gate['passed'] for gate in details['gates']] == [True, False]


def test_gated_sweep_and_breakdown_choose_feasible_value_and_reuse_build():
    cfg = _cfg()
    cfg.update(base_poscar='unused', sweep={'name': 'temperature', 'values': [300, 500]})
    for prop in cfg['properties']:
        prop.update(predictor='dp_property', models=['unused'], fparam=[None])
    p = StructureScorePredictor(cfg)
    builds = []
    p._shared_builder = SimpleNamespace(build=lambda *a, **k: builds.append(1) or [object()])
    def score(prop, structures, *, fparam=None):
        values = {'delta': 1000 if fparam[0] == 500 else 20,
                  'pure': 0.2 if fparam[0] == 500 else 0.9, 'cost': 0}
        return float(values[prop['name']]), 0.0
    p._score = score
    candidate = {'Fe': 1}
    assert p.predict(candidate)[0] == 20.0
    assert p.per_objective_stats(candidate)['temperature'] == (300.0, 0.0)
    assert p.reward_details(candidate)['gate_passed'] is True
    assert len(builds) == 1  # diagnostics used the cached stats
    breakdown = p.score_breakdown(candidate)
    assert breakdown['best']['value'] == 300.0
    assert breakdown['best']['reward'] == 20.0
    assert breakdown['rows'][1]['reward_details']['gate_passed'] is False
    assert p.raw_combine(breakdown['best']['stats']) == 20.0


def _class_cfg(index=2, **overrides):
    prop = {'name': 'pure', 'predictor': 'dp_classification', 'models': ['model.pt'],
            'head': 'synthesizability', 'output_index': index, 'direction': 'max',
            'objective': 'mean'}
    prop.update(overrides)
    return {'base_poscar': 'unused', 'n_random_configs': 3, 'properties': [prop]}


@pytest.mark.parametrize('index', [None, -1, 1.5, True])
def test_classification_requires_explicit_class_index(index):
    cfg = _class_cfg(index)
    with pytest.raises(ValueError, match='output_index'):
        StructureScorePredictor(cfg)


@pytest.mark.parametrize('override,match', [
    ({'head': 1}, 'head'), ({'transform': 'exp'}, 'transform'),
    ({'models': 'model.pt'}, 'models'), ({'output_kind': 'logits'}, 'predict_proba'),
])
def test_classification_rejects_ambiguous_or_non_probability_config(override, match):
    with pytest.raises(ValueError, match=match):
        StructureScorePredictor(_class_cfg(**override))


@pytest.mark.parametrize('nclasses', [2, 3])
@pytest.mark.parametrize('shared', [True, False])
def test_classification_probability_fold_for_each_structure_regime(monkeypatch, nclasses, shared):
    import rl_matdesign.utils.dp_eval as dp_eval
    cfg = _class_cfg(index=nclasses - 1)
    cfg['share_structure'] = shared
    p = StructureScorePredictor(cfg)
    seen = []
    def build(candidate, *, n_configs, rng):
        seen.append(n_configs)
        return [object()] * n_configs
    if shared:
        p._shared_builder = SimpleNamespace(build=build)
    else:
        p._builders['pure'] = SimpleNamespace(build=build)
    p._get_prop_models = lambda prop: ([object()], {})
    def evaluate(structures, models, elements, **kwargs):
        assert len(structures) == 3
        assert kwargs['output_index'] == nclasses - 1
        return np.array([[0.1, 0.9, 0.8]])
    monkeypatch.setattr(dp_eval, 'eval_classification_ensemble', evaluate)
    assert p.predict({'Fe': 1}) == pytest.approx((0.6, 0.0))
    assert p.per_objective_stats({'Fe': 1})['pure'] == pytest.approx((0.6, 0.0))
    assert seen == [3]


def test_classification_loads_public_factory_once_and_checks_type_maps(monkeypatch):
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_type_map=lambda: ['O', 'Fe'], predict_proba=lambda: None)
    deepmd = ModuleType('deepmd')
    infer = ModuleType('deepmd.infer')
    infer.DeepEval = factory
    deepmd.infer = infer
    monkeypatch.setitem(sys.modules, 'deepmd', deepmd)
    monkeypatch.setitem(sys.modules, 'deepmd.infer', infer)
    p = StructureScorePredictor(_class_cfg())
    first = p._get_prop_models(p.properties[0])
    assert first[1] == {'O': 0, 'Fe': 1}
    assert p._get_prop_models(p.properties[0]) is first
    assert calls == [{'model_file': 'model.pt', 'auto_batch_size': False, 'head': 'synthesizability'}]
    infer.DeepEval = lambda **kwargs: SimpleNamespace(get_type_map=lambda: ['O'])
    p = StructureScorePredictor(_class_cfg())
    with pytest.raises(ValueError, match='predict_proba'):
        p._get_prop_models(p.properties[0])
    maps = iter([['O', 'Fe'], ['Fe', 'O']])
    infer.DeepEval = lambda **kwargs: SimpleNamespace(get_type_map=lambda: next(maps),
                                                      predict_proba=lambda: None)
    p = StructureScorePredictor(_class_cfg(models=['a', 'b']))
    with pytest.raises(ValueError, match='type_map'):
        p._get_prop_models(p.properties[0])


def test_combination_rule_without_properties_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match='properties'):
        build_reward({'predictor': 'dummy', 'reward': {'type': 'gated'}})


@pytest.mark.parametrize('method', ['dqn', 'reinforce', 'a2c'])
def test_gated_reward_reaches_training_and_generated_csv(tmp_path, method):
    """All RL methods consume the same generic rule, with inspectable CSV output."""
    import csv
    import subprocess
    from pathlib import Path
    import yaml

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / 'configs/test_dummy.yaml').read_text())
    cfg.pop('predictor')
    cfg.update(
        properties=[{'name': name, 'predictor': 'dummy', 'direction': 'max',
                     'objective': 'mean'} for name in ['delta', 'pure']],
        reward=_cfg()['reward'],
        pg_warmup_eps=4, pg_num_iters=2, pg_batch_eps=2,
        pg_repeat_penalty_coef=0.0, dqn_warmup_eps=4, dqn_num_train_eps=6,
        dqn_grad_steps_per_ep=1, dqn_batch_size=4, num_gen_eps=4,
        exploration_gen_eps=2,
    )
    cfg['reward']['objective']['terms'] = {'delta': 1.0}
    cfg['reward']['gates'][0]['min'] = 0.5
    path = tmp_path / 'gated.yaml'
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / method
    proc = subprocess.run(
        [sys.executable, str(repo / 'scripts/run_experiment.py'), '--config', str(path),
         '--method', method, '--out', str(out), '--dp-seed', '0',
         '--train-seed', '0', '--gen-seed', '0'],
        cwd=repo, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with (out / 'generated.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        probability = float(row['obj_pure_mean'])
        objective = float(row['obj_delta_mean'])
        passed = probability >= 0.5
        violation = max(0.0, 0.5 - probability)
        expected = objective if passed else -10.0 - 2.0 * violation
        assert row['gate_passed'] == str(passed)
        assert row['gate_synthesis_passed'] == str(passed)
        assert float(row['gate_synthesis_value']) == pytest.approx(probability)
        assert float(row['gate_synthesis_violation']) == pytest.approx(violation)
        assert float(row['reward_objective']) == pytest.approx(objective)
        assert float(row['reward']) == pytest.approx(expected)
        assert float(row['dp_mean']) == pytest.approx(expected)
