"""Real ASE pipeline fixtures with only the DeepMD model call replaced."""
from copy import deepcopy
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from rl_matdesign.configuration import build_predictor

COMP = {'alloy': {'Ni': .5, 'Fe': .5}}


@pytest.fixture
def slab_poscar(tmp_path):
    from ase import Atoms
    from ase.io import write
    atoms = Atoms('Co6O2H', positions=[(0,0,0),(2,0,0),(0,2,0),(2,2,0),(1,1,2),(3,1,2),(1,3,4),(3,3,4),(1,1,5)],
                  cell=[4,4,12], pbc=True)
    path = tmp_path / 'POSCAR'
    write(path, atoms, format='vasp')
    return str(path)


@pytest.fixture
def oracle_factory(monkeypatch, slab_poscar, tmp_path):
    counter = 0
    def make(matrix=((0.,10.),(10.,0.)), *, adsorbates=None, output_index=0, seed=7,
             objective='mean', uncertainty='model_spread', k=1., predict_cache_size=200000):
        nonlocal counter
        root = tmp_path / f'oracle-{counter}'
        counter += 1
        root.mkdir()
        paths=[]
        for index in range(len(matrix)):
            path=root/f'model-{index}.pt'
            path.write_bytes(f'explicit fake model {index}'.encode())
            paths.append(str(path))
        counts={'loaded':0,'evals':0,'inputs':[]}
        class Model:
            def __init__(self, model_file, **kwargs):
                self.index=paths.index(str(model_file))
                self.calls=0
                counts['loaded']+=1
            def eval(self, **kwargs):
                value=matrix[self.index][self.calls % len(matrix[self.index])]
                self.calls+=1
                counts['evals']+=1
                counts['inputs'].append(deepcopy(kwargs))
                return np.asarray([[value,value+100.]])
        for name in ('deepmd','deepmd.pt','deepmd.pt.infer','deepmd.pt.infer.deep_eval'):
            monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
        sys.modules['deepmd.pt.infer.deep_eval'].DeepProperty=Model
        cfg={'properties':[{'name':'overpotential','predictor':'ooh_frames','models':paths,
             'output_index':output_index,'direction':'min','objective':objective,'uncertainty':uncertainty}],
             'k':k,'predict_cache_size':predict_cache_size,
             'structure':{'builder':'pipeline','base_poscar':slab_poscar,'n_random_configs':len(matrix[0]),'steps':[
                 {'type':'substitute','a_site_symbol':'Co','a_dopant_group':'alloy'},
                 {'type':'ooh_adsorbates','adsorbates':['O*','OH*','OOH*'] if adsorbates is None else list(adsorbates)}]}}
        return build_predictor(cfg,seed=seed), counts, paths
    return make


def leaf(predictor):
    return predictor.properties[0]['instance'][0]
