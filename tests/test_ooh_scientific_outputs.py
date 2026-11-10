"""OOH integration with real ASE structures and explicitly fake DeepMD models."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path

import numpy as np
import pytest

from abcde_ooh import dp_predictor as dpp
from rl_matdesign.predictors.ooh import OOHCatalystPredictor, _OutputLRU
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
from tests.test_ooh_adsorbates import slab_poscar


COMP = {'Ni': .5, 'Fe': .5}


@pytest.fixture
def oracle_factory(monkeypatch, slab_poscar, tmp_path):
    """Actual adapter + ASE path; deterministic fake model/config matrix."""
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import read, write
    counter = 0
    def make(matrix=((0., 10.), (10., 0.)), **kwargs):
        nonlocal counter
        root = tmp_path / f'oracle-{counter}'
        counter += 1
        root.mkdir()
        paths = []
        for index in range(len(matrix)):
            path = root / f'model-{index}.pt'
            path.write_bytes(f'fake unit-test weights {index}'.encode())
            paths.append(str(path))
        counters = {'loaded': 0, 'evals': 0}
        class FakeModel:
            def __init__(self, model_file, **_):
                self.index = paths.index(str(model_file))
                self.calls = 0
                counters['loaded'] += 1
            def eval(self, **_):
                value = matrix[self.index][self.calls % len(matrix[self.index])]
                self.calls += 1
                counters['evals'] += 1
                return np.asarray([[value, value + 100.]])
        monkeypatch.setattr(dpp, '_lazy_import_ase_deepmd',
                            lambda: (Atoms, read, write, chemical_symbols, FakeModel))
        options = {'adsorbates': [], 'n_random_configs': len(matrix[0]), **kwargs}
        predictor = OOHCatalystPredictor(slab_poscar, paths, **options)
        return predictor, counters, paths
    return make


def test_named_raw_and_legacy_views_use_one_matrix_without_sign_or_penalty_duplication(oracle_factory):
    predictor, counts, _ = oracle_factory()
    scientific = predictor.predict_outputs(COMP)['overpotential']
    assert scientific.mean == 5
    assert scientific.schema.unit is None and scientific.schema.representation == 'physical'
    assert scientific.model_spread == scientific.config_spread == 0
    assert scientific.pooled_spread == 5
    assert scientific.native_uncertainty is None
    assert [row.values for row in scientific.samples] == [(0, 10), (10, 0)]
    assert scientific.samples[0].configuration_ids == scientific.samples[1].configuration_ids
    assert len(set(scientific.samples[0].configuration_ids)) == 2
    assert predictor.predict_raw(COMP) == (5, 0)
    assert predictor.predict(COMP) == (-5, 0)
    predictor.uncertainty = 'total'
    predictor.objective = 'mean_minus_kstd'
    predictor.k = 2
    assert predictor.predict(COMP) == (-15, 5)
    assert predictor.predict_outputs(COMP)['overpotential'] is scientific
    predictor.batch_predict_outputs([COMP, {'Fe': .5, 'Ni': .5}, {'Co': 0, **COMP}])
    assert counts == {'loaded': 2, 'evals': 4}


@pytest.mark.parametrize('matrix,model_spread,config_spread,pooled', [
    (((2.,),), None, None, None),
    (((1., 3.),), None, 1., 1.),
    (((1.,), (3.,)), 1., None, 1.),
    (((1., 3.), (5., 7.)), 2., 1., math.sqrt(5)),
])
def test_population_axis_definitions_and_missing_singleton_statistics(oracle_factory, matrix, model_spread, config_spread, pooled):
    predictor, _, _ = oracle_factory(matrix)
    output = predictor.predict_outputs(COMP)['overpotential']
    assert output.model_spread == model_spread
    assert output.config_spread == config_spread
    assert output.pooled_spread == pytest.approx(pooled) if pooled is not None else output.pooled_spread is None
    raw_mean, raw_spread = predictor.predict_raw(COMP)
    assert raw_mean == output.mean and raw_spread == model_spread


def test_missing_penalty_statistic_is_configuration_error_not_zero(oracle_factory):
    predictor, counts, _ = oracle_factory(((2., 4.),), objective='mean_minus_kstd')
    assert predictor.predict_outputs(COMP)['overpotential'].model_spread is None
    with pytest.raises(EvaluationProtocolError, match='insufficient observations'):
        predictor.predict(COMP)
    predictor.objective = 'mean'
    assert predictor.predict(COMP) == (-3, None)
    predictor.objective = 'mean_minus_kstd'
    predictor.k = 0
    assert predictor.predict(COMP) == (-3, None)
    assert counts['evals'] == 2


@pytest.mark.parametrize('bad_values', [[1.], [1., 2., 3.], [1., float('nan')],
                                        [1., float('inf')], [1., True], [1., '2']])
def test_every_configuration_requires_all_models_and_finite_real_scalars(oracle_factory, bad_values):
    predictor, _, _ = oracle_factory()
    predictor._predictor._eval_models_on_prepared_inputs = lambda *_: bad_values
    with pytest.raises(EvaluationProtocolError, match='every model'):
        predictor.predict_outputs(COMP)
    assert len(predictor._cache) == 0
    assert predictor._predictor.last_scientific_output is None


def test_failed_later_configuration_does_not_leave_partial_or_cached_scientific_output(oracle_factory):
    predictor, _, _ = oracle_factory()
    responses = iter([[1., 2.], [3.]])
    predictor._predictor._eval_models_on_prepared_inputs = lambda *_: next(responses)
    with pytest.raises(EvaluationProtocolError):
        predictor.predict_outputs(COMP)
    assert len(predictor._predictor.last_sample_records) == 1
    assert predictor._predictor.last_scientific_output is None
    assert not predictor._cache


def test_direct_compatibility_details_are_derived_from_same_population_matrix(oracle_factory):
    predictor, counts, _ = oracle_factory(((1., 3.), (5., 7.)))
    mean, spread, details = predictor._predictor.predict_overpotential(
        COMP, uncertainty='configs', return_per_model=True)
    assert (mean, spread) == (4., 1.)
    assert details['per_model_means'] == [2., 6.]
    assert details['per_model_stds_over_configs'] == [1., 1.]
    assert details['pooled_spread'] == pytest.approx(math.sqrt(5))
    assert len(details['samples']) == 2 and counts['evals'] == 4


def test_missing_assets_fail_before_importing_model_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(dpp, '_lazy_import_ase_deepmd',
                        lambda: pytest.fail('Must validate assets before model import'))
    with pytest.raises(FileNotFoundError):
        OOHCatalystPredictor(str(tmp_path / 'missing'), [str(tmp_path / 'missing.pt')])


@pytest.mark.parametrize('asset_kind', ['model', 'template', 'deleted_model'])
def test_cached_results_reject_changed_or_deleted_assets(oracle_factory, asset_kind):
    predictor, counters, paths = oracle_factory()
    predictor.predict_outputs(COMP)
    expected_calls = counters['evals']
    path = Path(paths[0] if asset_kind != 'template' else predictor._predictor.cfg.base_poscar)
    stat = path.stat()
    if asset_kind == 'deleted_model':
        path.unlink()
    else:
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        # Even a same-size edit with restored mtime must be caught by ctime.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(EvaluationProtocolError, match='asset changed'):
        predictor.predict_outputs(COMP)
    assert counters['evals'] == expected_calls


@pytest.mark.parametrize('change', [
    {'seed': 222}, {'n_random_configs': 3}, {'output_index': 1}, {'adsorbates': ('O*',)},
])
def test_cached_results_reject_in_place_scientific_configuration_changes(oracle_factory, change):
    predictor, _, _ = oracle_factory()
    predictor.predict_outputs(COMP)
    predictor._predictor.cfg = replace(predictor._predictor.cfg, **change)
    with pytest.raises(EvaluationProtocolError, match='configuration changed'):
        predictor.predict_outputs(COMP)


def test_oracle_identity_contains_all_assets_seed_source_and_selected_output(oracle_factory):
    predictor, _, paths = oracle_factory(output_index=1, rng_seed=27)
    identity = predictor.oracle_identity()
    assert identity['config']['eval_seed'] == 27 and identity['config']['output_index'] == 1
    assert set(identity['assets']) == {*paths, predictor._predictor.cfg.base_poscar}
    assert all(len(record['sha256']) == 64 for record in identity['assets'].values())
    assert 'source' in identity and identity['statistic_ddof'] == 0
    original = deepcopy(identity)
    predictor._predictor.cfg.debug_dir = 'debug_only'
    predictor._predictor.cfg.debug_max_configs = 1
    predictor.k = 5
    predictor.uncertainty = 'total'
    assert predictor.oracle_identity() == original
    source = predictor.source_identity()
    assert source['eval_seed'] == 27 and source['property_sources'] == {'overpotential': 'ooh_slab'}


def test_lru_evicts_old_candidates_and_bounds_checkpoint_update(oracle_factory):
    predictor, counts, _ = oracle_factory(((2.,),))
    predictor._cache = _OutputLRU(capacity=2)
    a, b, c = COMP, {'Ni': .6, 'Fe': .4}, {'Ni': .7, 'Fe': .3}
    predictor.predict_outputs(a)
    predictor.predict_outputs(b)
    predictor.predict_outputs(a)  # Make b the least recently used candidate.
    predictor.predict_outputs(c)
    assert len(predictor._cache) == 2 and predictor._comp_key(b) not in predictor._cache
    predictor.predict_outputs(b)
    assert counts['evals'] == 4
    predictor._cache.update({index: ('legacy', 'entry') for index in range(10)})
    assert list(predictor._cache) == [8, 9]
    # Even a stale tuple installed under the current identity is not scientific data.
    predictor._cache[predictor._comp_key(a)] = (123, 0)
    assert predictor.predict_outputs(a)['overpotential'].mean == 2
    assert counts['evals'] == 5



def test_output_selection_incompatible_with_model_is_protocol_error(oracle_factory):
    predictor, _, _ = oracle_factory(output_index=99)
    with pytest.raises(EvaluationProtocolError, match='configured scalar output'):
        predictor.predict_outputs(COMP)
    assert not predictor._cache


def test_asset_drift_during_evaluation_is_rejected_before_caching(oracle_factory):
    predictor, _, paths = oracle_factory(((1.,),))
    model = predictor._predictor.dp_models[0]
    original = model.eval
    def changing_model(**kwargs):
        Path(paths[0]).write_bytes(b'changed during evaluation')
        return original(**kwargs)
    model.eval = changing_model
    with pytest.raises(EvaluationProtocolError, match='asset changed'):
        predictor.predict_outputs(COMP)
    assert predictor._predictor.last_scientific_output is None
    assert not predictor._cache
