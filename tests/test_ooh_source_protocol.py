"""OOH uses singular structure identities and candidate-local random samples."""
from copy import deepcopy
import random

import numpy as np

from rl_matdesign.predictors.builders.pipeline import PipelineBuilder
from rl_matdesign.structure_sources import StructureSpec, resolve_source_assets
from rl_matdesign.utils.structure import _fractions_to_counts
from tests.ooh_fixtures import COMP, oracle_factory, slab_poscar


def test_count_allocation_resolves_equal_remainders_by_label():
    first={'Ni':.5,'Fe':.5}
    assert _fractions_to_counts(first,3)==_fractions_to_counts(dict(reversed(list(first.items()))),3)=={'Fe':2,'Ni':1}


def test_source_seed_excludes_property_models_and_reward(oracle_factory):
    first,_,_=oracle_factory(((2.,2.),),output_index=0,k=1)
    second,_,_=oracle_factory(((3.,3.),),output_index=1,k=5)
    a,b=first.source_plan.structure,second.source_plan.structure
    assert a.to_schema()==b.to_schema()
    assert a.seed_for(COMP,eval_seed=1,sample_index=0)!=a.seed_for(COMP,eval_seed=1,sample_index=1)
    assert a.seed_for(COMP,eval_seed=1,sample_index=0)!=a.seed_for(COMP,eval_seed=2,sample_index=0)


def test_direct_sampling_is_stable_and_zero_components_have_no_physical_effect(oracle_factory):
    predictor,_,_=oracle_factory(((1.,1.,1.,1.),))
    builder=predictor._source_builders['structure']; spec=predictor.source_plan.structure
    first=builder.build_seeded(COMP,n_configs=4,source_spec=spec,eval_seed=7)
    state=random.getstate()
    second=builder.build_seeded({'alloy':{'Co':0,'Fe':.5,'Ni':.5}},n_configs=4,source_spec=spec,eval_seed=7)
    assert random.getstate()==state
    assert [x.get_chemical_symbols() for x in first]==[x.get_chemical_symbols() for x in second]
    records=[x.info['_rl_matdesign_sample'] for x in first]
    assert records==[x.info['_rl_matdesign_sample'] for x in second]
    assert len({record['seed'] for record in records})==4


def test_finer_than_historical_grid_remains_in_candidate_identity(oracle_factory):
    predictor,_,_=oracle_factory(((1.,),))
    first={'alloy':{'Ni':.501,'Fe':.499}}
    second={'alloy':{'Ni':.502,'Fe':.498}}
    assert predictor._key(first)!=predictor._key(second)


def test_relaxation_model_is_hashed_only_when_its_operation_is_present(slab_poscar,tmp_path):
    model=tmp_path/'relax.pt'; model.write_bytes(b'relax model A')
    cfg={'builder':'pipeline','base_poscar':slab_poscar,'steps':[
        {'type':'substitute','a_site_symbol':'Co','a_dopant_group':'alloy'}]}
    bare,_=resolve_source_assets(StructureSpec(cfg))
    relaxed=deepcopy(cfg); relaxed['steps'].append({'type':'geo_opt','model':str(model)})
    before,refs=resolve_source_assets(StructureSpec(relaxed))
    assert {item['path'] for item in refs}=={slab_poscar,str(model)}
    model.write_bytes(b'relax model B')
    after,_=resolve_source_assets(StructureSpec(relaxed))
    unchanged,_=resolve_source_assets(StructureSpec(cfg))
    assert before.protocol_fingerprint!=after.protocol_fingerprint
    assert bare.to_schema()==unchanged.to_schema()


def test_adsorbate_random_consumption_cannot_move_later_sample_decoration(oracle_factory):
    predictor,_,_=oracle_factory(((1.,1.,1.,1.),))
    spec=predictor.source_plan.structure
    ordinary=PipelineBuilder(spec.effective_config)
    extra=PipelineBuilder(spec.effective_config)
    prepare=extra._adsorbates
    def consume(atoms,plan,step,rng):
        rng.random(100)
        return prepare(atoms,plan,step,rng)
    extra._adsorbates=consume
    first=ordinary.build_seeded(COMP,n_configs=4,source_spec=spec,eval_seed=17)
    second=extra.build_seeded(COMP,n_configs=4,source_spec=spec,eval_seed=17)
    assert [x.get_chemical_symbols() for x in first]==[x.get_chemical_symbols() for x in second]
    assert all(np.array_equal(a.positions,b.positions) for a,b in zip(first,second))
    assert [x.info['_rl_matdesign_sample'] for x in first]==[x.info['_rl_matdesign_sample'] for x in second]
