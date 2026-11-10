"""Scientific failure/uncertainty semantics, independent of private RF assets."""
from types import SimpleNamespace

import numpy as np
import pytest

from rl_matdesign.predictors.rf_magpie import RFMagpiePredictor


def predictor(monkeypatch, *, tree_values=()):
    p = RFMagpiePredictor('unused.joblib')
    p._model = SimpleNamespace(predict=lambda x: [1200.0])
    if tree_values is not None:
        p._model.estimators_ = [SimpleNamespace(predict=lambda x, v=v: [v]) for v in tree_values]
    monkeypatch.setattr(p, '_get_featurizer', lambda: SimpleNamespace(featurize=lambda c: [1.0, 2.0]))
    return p


@pytest.mark.parametrize('trees', [None, (), (1200.,)])
def test_missing_spread_is_unavailable(monkeypatch, trees):
    p = predictor(monkeypatch, tree_values=trees)
    assert p.predict({'Ba': 1, 'Ti': 1, 'O': 3}) == (1200., None)


def test_tree_population_statistic_preserved(monkeypatch):
    p = predictor(monkeypatch, tree_values=(1100., 1300.))
    assert p.predict({'Ba': 1, 'Ti': 1, 'O': 3}) == (1200., 100.)


@pytest.mark.parametrize('composition', [{}, {'Ni': 0}, {'Ni': -1}, {'Ni': float('nan')}, {'Ni': float('inf')}])
def test_invalid_composition_never_becomes_scientific_value(monkeypatch, composition):
    p = predictor(monkeypatch)
    with pytest.raises(ValueError):
        p.predict(composition)
    assert not p._cache


def test_featurizer_failure_is_not_fake_temperature(monkeypatch):
    p = predictor(monkeypatch)
    def fail(c):
        raise RuntimeError('descriptor failed')
    monkeypatch.setattr(p, '_get_featurizer', lambda: SimpleNamespace(featurize=fail))
    with pytest.raises(RuntimeError, match='descriptor failed'):
        p.predict({'Ni': 1})
    assert not p._cache


@pytest.mark.parametrize('boundary', ['features', 'model', 'tree'])
def test_nonfinite_outputs_fail_without_cache(monkeypatch, boundary):
    p = predictor(monkeypatch, tree_values=(1., np.nan) if boundary == 'tree' else ())
    if boundary == 'features':
        monkeypatch.setattr(p, '_get_featurizer', lambda: SimpleNamespace(featurize=lambda c: [np.nan]))
    if boundary == 'model':
        p._model.predict = lambda x: [np.inf]
    with pytest.raises(ValueError, match='nonfinite'):
        p.predict({'Ni': 1})
    assert not p._cache


def test_large_search_does_not_grow_leaf_cache_without_limit(monkeypatch):
    p = predictor(monkeypatch)
    for amount in range(1, 4100):
        p.predict({'Ni': amount, 'Fe': 1})
    assert len(p._cache) == 4096
    assert (('Fe', 1.), ('Ni', 1.)) not in p._cache
