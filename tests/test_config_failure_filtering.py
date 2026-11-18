"""Only usable configurations contribute to a formula's properties and reward."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_cross_formula_predictor import RecordingBuilder, _candidate, _config
from rl_matdesign.predictors.structure_score import StructureScorePredictor


def _predictor(n_random_configs):
    cfg = _config()
    cfg["structure"]["n_random_configs"] = n_random_configs
    return StructureScorePredictor(cfg, seed=17)


@pytest.mark.parametrize("mode", ["scalar", "batch", "batch_scalar_builder"])
def test_pipeline_relaxation_drops_failed_samples_once_and_averages_survivors(mode, tmp_path, monkeypatch):
    from collections import Counter
    from ase import Atoms
    path=tmp_path / "POSCAR"
    Atoms("Ni10",positions=np.arange(30).reshape(-1,3),cell=[50]*3,pbc=True).write(path,format="vasp")
    cfg=_config()
    cfg["structure"]={"builder":"pipeline","base_poscar":str(path),"n_random_configs":4,"steps":[
        {"type":"substitute","a_site_symbol":"Ni","a_dopant_group":"alloy"},
        {"type":"geo_opt","model":"unused.pt"}]}
    predictor=StructureScorePredictor(cfg,seed=17)
    builder=predictor._source_builders["structure"]
    if mode=="batch_scalar_builder":
        builder.build_many_seeded=None
    calls={1:0,2:0}; relaxed=[]; scored=[]
    def relax(atoms,*args):
        iron=Counter(atoms.get_chemical_symbols())["Fe"]
        index=calls[iron];calls[iron]+=1
        relaxed.append((iron,index))
        if iron==1 or index<2:
            raise RuntimeError("optimizer failed or did not converge")
        atoms.info['test_index']=index
        return atoms
    monkeypatch.setattr(builder,"_relax",relax)
    def score(prop,cells):
        indices=[cell.info['test_index'] for cell in cells]
        scored.append((prop['name'],indices))
        return float(np.mean([10.*index for index in indices])),0.
    predictor._score=score
    bad={'alloy':{'Fe':.1,'Ni':.9}}; good={'alloy':{'Fe':.2,'Ni':.8}}
    if mode=='scalar':
        with pytest.raises(RuntimeError,match='No structure samples succeeded'):
            predictor.predict(bad)
        result=predictor.predict(good)
    else:
        results=predictor.batch_predict([bad,good],return_exceptions=True)
        assert isinstance(results[0],RuntimeError)
        result=results[1]
    assert result==pytest.approx((2*25-25/5,None))
    assert predictor.per_objective_stats(good)=={'benefit':(25.,0.),'cost':(25.,0.)}
    assert all(indices==[2,3] for _,indices in scored)
    assert len(relaxed)==8 and predictor._key(bad) not in predictor._stats_cache
    diagnostics=predictor.source_diagnostics(good)['structure']
    assert [row['status'] for row in diagnostics['samples']]==['failed','failed','succeeded','succeeded']
    assert len(diagnostics['surviving_samples'])==2




@pytest.mark.parametrize("mode", ["scalar", "batch", "batch_scalar_builder"])
def test_serial_external_md_builder_observes_strict_nonconvergence(mode, monkeypatch):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.optimize import LBFGS
    from rl_matdesign.utils.structure import relax_structure

    monkeypatch.setattr(LBFGS, "run", lambda self, **kwargs: self.atoms.info["index"] >= 2)
    predictor = _predictor(n_random_configs=4)
    builder = RecordingBuilder()
    failures = []

    def build(candidate, *, n_configs, rng):
        usable = []
        for index in range(n_configs):
            try:
                if index == 0:
                    raise RuntimeError("MD process failed")
                atoms = Atoms("H", positions=[[0, 0, 0]], info={"index": index})
                atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((1, 3)))
                optimized = relax_structure(atoms, calc=atoms.calc, relax_cell=False)
                usable.append(SimpleNamespace(candidate=candidate, index=optimized.info["index"]))
            except Exception as exc:
                failures.append((index, str(exc)))
        return usable

    builder.build = build
    if mode == "batch_scalar_builder":
        builder.build_many_seeded = None
    predictor._source_builders["structure"] = builder
    seen = []
    predictor._score = lambda prop, cells: (seen.append([c.index for c in cells]) or 1.0, 0.0)
    candidate = _candidate(0.2)
    if mode == "scalar":
        predictor.predict(candidate)
    else:
        predictor.batch_predict([candidate])
    assert seen == [[2, 3], [2, 3]]
    assert [index for index, _ in failures] == [0, 1]
    assert "converg" in failures[1][1]
