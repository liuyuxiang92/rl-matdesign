"""Adsorbate frame order, padding and masks through the current pipeline."""
import numpy as np
import pytest

from rl_matdesign.predictors.builders.pipeline import PipelineBuilder, pipeline_frames
from tests.ooh_fixtures import COMP, oracle_factory, slab_poscar


@pytest.mark.parametrize('modes,masked', [
    (['O*','OH*','OOH*'], [2,1,0]), ([],[0]), (['O*'],[2]),
    (['OH*','OOH*'],[1,0]), (['OOH*','O*'],[0,2]),
])
def test_configured_frame_order_and_padding_reach_model(oracle_factory,modes,masked):
    predictor,counts,_=oracle_factory(((2.,),),adsorbates=modes)
    predictor.predict(COMP)
    call=counts['inputs'][0]
    assert call['coords'].shape == (len(masked),9+(3 if modes else 0),3)
    assert [int((row == -1).sum()) for row in call['atom_types']] == masked
    assert call['mixed_type'] is True


def test_bare_selection_cuts_frame_work_threefold(oracle_factory):
    full,full_counts,_=oracle_factory(((2.,2.,2.,2.),))
    bare,bare_counts,_=oracle_factory(((2.,2.,2.,2.),),adsorbates=[])
    full.predict(COMP); bare.predict(COMP)
    assert sum(x['coords'].shape[0] for x in full_counts['inputs']) == 3*sum(x['coords'].shape[0] for x in bare_counts['inputs'])
    assert full_counts['evals'] == bare_counts['evals'] == 4


@pytest.mark.parametrize('modes',[['O'],['o*'],['O*','O*'],['OOOH*'],['O*','none']])
def test_pipeline_rejects_noncanonical_or_duplicate_modes(slab_poscar,modes):
    with pytest.raises(ValueError,match='adsorbates'):
        PipelineBuilder({'builder':'pipeline','base_poscar':slab_poscar,
                         'steps':[{'type':'ooh_adsorbates','adsorbates':modes}]})


def test_cache_prevents_a_second_model_call(oracle_factory):
    predictor,counts,_=oracle_factory(((2.,),),adsorbates=[])
    predictor.predict_outputs(COMP)
    predictor.predict_outputs({'alloy': {'Fe': .5,'Ni': .5}})
    assert counts['evals']==1


def test_oracle_identity_distinguishes_frame_settings_and_output_selection(oracle_factory):
    full,_,_=oracle_factory(((2.,),))
    bare,_,_=oracle_factory(((2.,),),adsorbates=[])
    selected,_,_=oracle_factory(((2.,),),output_index=1)
    assert full.source_identity()['structure']['config']['steps'] != bare.source_identity()['structure']['config']['steps']
    assert full.oracle_identity()['outputs']['overpotential']['output_index'] != selected.oracle_identity()['outputs']['overpotential']['output_index']


def test_exported_frame_is_the_scored_frame_without_a_rebuild(oracle_factory,tmp_path):
    from ase.io import read,write
    predictor,counts,_=oracle_factory(((2.,),),adsorbates=[])
    result=predictor.score_breakdown(COMP)
    parent=result['structures']['overpotential'][0]
    label,frame,masks=pipeline_frames(parent)[0]
    assert label=='bare' and masks==[]
    path=tmp_path/'scored.vasp'
    write(path,frame,format='vasp')
    exported=read(path,format='vasp')
    assert np.allclose(exported.positions,counts['inputs'][0]['coords'][0])
    assert counts['evals']==1
