"""OOH population statistics, cache and asset integrity through shared scoring."""
from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np
import pytest

from rl_matdesign.predictors.ooh_frames import OOHFramesPredictor
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
from tests.ooh_fixtures import COMP, leaf, oracle_factory, slab_poscar


def test_physical_outputs_and_reward_share_one_matrix(oracle_factory):
    predictor,counts,_=oracle_factory()
    scientific=predictor.predict_outputs(COMP)['overpotential']
    assert scientific.mean==5
    assert scientific.schema.unit is None and scientific.schema.representation=='physical'
    assert scientific.model_spread==scientific.config_spread==0
    assert scientific.pooled_spread==5 and scientific.native_uncertainty is None
    assert [row.values for row in scientific.samples]==[(0,10),(10,0)]
    assert scientific.samples[0].configuration_ids==scientific.samples[1].configuration_ids
    assert len(set(scientific.samples[0].configuration_ids))==2
    assert predictor.predict_raw(COMP)==(-5,None)
    assert predictor.predict(COMP)==(-5,None)
    predictor.batch_predict_outputs([COMP,{'alloy':{'Fe':.5,'Ni':.5}}])
    assert counts['loaded']==2 and counts['evals']==4


def test_selected_uncertainty_penalty_applies_once(oracle_factory):
    predictor,counts,_=oracle_factory(uncertainty='pooled_spread',objective='mean_minus_kstd',k=2)
    assert predictor.predict(COMP)==(-15,None)
    assert predictor.per_objective_stats(COMP)['overpotential']==(5,5)
    assert predictor.predict_raw(COMP)==(-5,None)
    assert counts['evals']==4


@pytest.mark.parametrize('matrix,model_spread,config_spread,pooled',[
    (((2.,),),None,None,None), (((1.,3.),),None,1.,1.),
    (((1.,),(3.,)),1.,None,1.), (((1.,3.),(5.,7.)),2.,1.,math.sqrt(5))])
def test_population_axis_definitions_and_missing_singletons(oracle_factory,matrix,model_spread,config_spread,pooled):
    predictor,_,_=oracle_factory(matrix)
    output=predictor.predict_outputs(COMP)['overpotential']
    assert output.model_spread==model_spread
    assert output.config_spread==config_spread
    assert output.pooled_spread==pytest.approx(pooled) if pooled is not None else output.pooled_spread is None
    assert predictor.per_objective_stats(COMP)['overpotential']==(output.mean,model_spread)


def test_missing_penalty_statistic_is_not_invented_zero(oracle_factory):
    predictor,_,_=oracle_factory(((2.,4.),),objective='mean_minus_kstd')
    with pytest.raises((ValueError,EvaluationProtocolError),match='unavailable|insufficient'):
        predictor.predict(COMP)
    mean_only,_,_=oracle_factory(((2.,4.),),objective='mean')
    assert mean_only.predict(COMP)==(-3,None)
    assert mean_only.per_objective_stats(COMP)['overpotential']==(3,None)


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_model_scalar_does_not_enter_cache(oracle_factory,bad):
    predictor,_,_=oracle_factory()
    leaf(predictor).models[1].eval=lambda **_: np.asarray([[bad]])
    with pytest.raises(EvaluationProtocolError,match='scalar') as error:
        predictor.predict_outputs(COMP)
    assert 'finite' in str(error.value.__cause__)
    assert not predictor._stats_cache


def test_later_configuration_failure_cannot_cache_partial_output(oracle_factory):
    predictor,_,_=oracle_factory()
    model=leaf(predictor).models[1]
    values=iter([np.asarray([[3.]]),np.empty((0,))])
    model.eval=lambda **_: next(values)
    with pytest.raises(EvaluationProtocolError,match='scalar'):
        predictor.predict_outputs(COMP)
    assert not predictor._stats_cache


def test_missing_assets_fail_before_model_import(tmp_path):
    with pytest.raises(FileNotFoundError):
        OOHFramesPredictor({'models':[str(tmp_path/'missing.pt')]})


@pytest.mark.parametrize('kind',['model','template','deleted_model'])
def test_cached_results_reject_changed_or_deleted_assets(oracle_factory,kind):
    predictor,counts,paths=oracle_factory()
    predictor.predict_outputs(COMP)
    calls=counts['evals']
    path=Path(paths[0] if kind!='template' else predictor.source_plan.structure.effective_config['base_poscar'])
    stat=path.stat()
    if kind=='deleted_model': path.unlink()
    else:
        content=path.read_bytes(); path.write_bytes(bytes([content[0]^1])+content[1:])
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(EvaluationProtocolError,match='changed|disappeared'):
        predictor.predict_outputs(COMP)
    assert counts['evals']==calls


def test_oracle_identity_records_assets_seed_and_selected_output_without_reward(oracle_factory):
    predictor,_,paths=oracle_factory(output_index=1,seed=27)
    identity=predictor.oracle_identity()
    assert identity['sources']['eval_seed']==27
    assert identity['outputs']['overpotential']['output_index']==1
    model_records=[item for records in identity['model_assets'].values() for item in records]
    assert {record['path'] for record in model_records}==set(paths)
    assert all(len(record['sha256'])==64 for record in model_records)
    assert identity['sources']['structure']['physical_protocol']['asset_sha256']
    original=deepcopy(identity); predictor.k=5
    assert predictor.oracle_identity()==original


def test_lru_bounds_results_and_recomputes_evicted_candidates(oracle_factory, monkeypatch):
    monkeypatch.setattr("rl_matdesign.predictors.structure_score._PREDICTION_CACHE_SIZE", 2)
    predictor,counts,_=oracle_factory(((2.,),))
    a,b,c=COMP,{'alloy':{'Ni':.6,'Fe':.4}},{'alloy':{'Ni':.7,'Fe':.3}}
    for candidate in (a,b,a,c): predictor.predict_outputs(candidate)
    assert len(predictor._stats_cache)==2 and predictor._key(b) not in predictor._stats_cache
    predictor.predict_outputs(b)
    assert counts['evals']==4


def test_incompatible_output_index_is_protocol_error(oracle_factory):
    predictor,_,_=oracle_factory(output_index=99)
    with pytest.raises(EvaluationProtocolError,match='scalar'):
        predictor.predict_outputs(COMP)
    assert not predictor._stats_cache


def test_asset_drift_during_inference_rejected_before_caching(oracle_factory):
    predictor,_,paths=oracle_factory(((1.,),))
    model=leaf(predictor).models[0]; original=model.eval
    def changing(**kwargs):
        Path(paths[0]).write_bytes(b'changed during evaluation')
        return original(**kwargs)
    model.eval=changing
    with pytest.raises(EvaluationProtocolError,match='changed'):
        predictor.predict_outputs(COMP)
    assert not predictor._stats_cache


@pytest.mark.parametrize('bad',[True,'2'])
def test_non_numeric_model_scalar_is_not_coerced_into_scientific_data(oracle_factory,bad):
    predictor,_,_=oracle_factory()
    leaf(predictor).models[1].eval=lambda **_: np.asarray([[bad]])
    with pytest.raises(EvaluationProtocolError,match='real|numeric|scalar'):
        predictor.predict_outputs(COMP)
    assert not predictor._stats_cache
