"""OOH adapter boundary regressions independent of real DeepMD weights."""
import sys
import types

import numpy as np
import pytest

from rl_matdesign.configuration import build_predictor
from rl_matdesign.predictors.ooh_frames import OOHFramesPredictor
from rl_matdesign.predictors.builders.pipeline import PipelineBuilder
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError


def install_model(monkeypatch):
    for name in ("deepmd", "deepmd.pt", "deepmd.pt.infer", "deepmd.pt.infer.deep_eval"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    class Model:
        def __init__(self, **kwargs):
            pass
        def eval(self, **kwargs):
            return np.asarray([[2.]])
    sys.modules["deepmd.pt.infer.deep_eval"].DeepProperty = Model


def config(tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text("fixture\n1\n20 0 0\n0 20 0\n0 0 20\nCo O H\n2 1 1\nDirect\n.1 .1 .1\n.3 .3 .3\n.5 .5 .5\n.7 .7 .7\n")
    model = tmp_path / "relative.pt"
    model.write_text("fake weights")
    return dict(properties=[dict(name="overpotential",predictor="ooh_frames",models=["relative.pt"],
        direction="min",objective="mean")], structure=dict(
        builder="pipeline",base_poscar=str(poscar),n_random_configs=1,steps=[
            dict(type="substitute",a_site_symbol="Co",a_dopant_group="alloy"),
            dict(type="ooh_adsorbates",adsorbates=["O*","OH*","OOH*"])]))


@pytest.mark.parametrize("predictor_name", [
    "ooh_frames", "rl_matdesign.predictors.ooh_frames:OOHFramesPredictor",
])
@pytest.mark.parametrize("change", ["modify", "delete"])
def test_relative_ooh_model_asset_change_is_rejected_before_cached_result(tmp_path, monkeypatch, change, predictor_name):
    monkeypatch.chdir(tmp_path)
    install_model(monkeypatch)
    cfg = config(tmp_path)
    cfg["properties"][0]["predictor"] = predictor_name
    predictor=build_predictor(cfg,seed=2)
    candidate={"alloy":{"Ni":1}}
    assert predictor.predict(candidate)==(-2.,None)
    if change == "modify":
        (tmp_path/"relative.pt").write_text("changed weights")
    else:
        (tmp_path/"relative.pt").unlink()
    with pytest.raises(EvaluationProtocolError,match="changed|disappeared"):
        predictor.predict(candidate)


def test_duplicate_resolved_ooh_checkpoint_aliases_are_rejected(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_model(monkeypatch)
    cfg=config(tmp_path)
    with pytest.raises((ValueError,EvaluationProtocolError),match="distinct|duplicate|repeat"):
        OOHFramesPredictor({"models":["relative.pt",str(tmp_path/"relative.pt")]})


def test_ooh_invalid_scalar_selection_is_a_protocol_error(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_model(monkeypatch)
    cfg=config(tmp_path)
    parent=PipelineBuilder(cfg["structure"]).build({"alloy":{"Ni":1}})[0]
    predictor=OOHFramesPredictor({"models":["relative.pt"],"output_index":2})
    with pytest.raises(EvaluationProtocolError,match="scalar|output"):
        predictor.predict_structure_outputs([parent])


def test_ooh_types_follow_each_coordinate_and_masks_only_disable_padding(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg=config(tmp_path)
    parent=PipelineBuilder(cfg["structure"]).build({"alloy":{"Ni":.5,"Fe":.5}},rng=np.random.default_rng(2))[0]
    coords,cells,types_=OOHFramesPredictor.prepared_inputs(parent)
    from ase.data import atomic_numbers
    slab_types=[atomic_numbers[s]-1 for s in parent.get_chemical_symbols()]
    assert coords.shape==(3,7,3) and cells.shape==(3,3,3)
    assert types_[:,:4].tolist()==[slab_types]*3
    assert types_[:,-3:].tolist()==[[7,-1,-1],[7,-1,0],[7,7,0]]
    assert np.array_equal(coords[:,:4],np.tile(parent.positions,(3,1,1)))


@pytest.mark.parametrize("predictor_name", [
    "ooh_frames", "rl_matdesign.predictors.ooh_frames:OOHFramesPredictor",
])
@pytest.mark.parametrize("close_fails", [False, True])
def test_partial_ooh_model_construction_closes_loaded_members_and_preserves_error(
    tmp_path, monkeypatch, predictor_name, close_fails,
):
    monkeypatch.chdir(tmp_path)
    install_model(monkeypatch)
    cfg = config(tmp_path)
    (tmp_path / "second.pt").write_text("second checkpoint")
    cfg["properties"][0].update(predictor=predictor_name, models=["relative.pt", "second.pt"])
    loaded = []
    class Model:
        def __init__(self, model_file, **kwargs):
            if loaded:
                raise RuntimeError("second checkpoint cannot load")
            self.closes = 0
            loaded.append(self)
        def close(self):
            self.closes += 1
            if close_fails:
                raise ValueError("cleanup must not mask the load failure")
    sys.modules["deepmd.pt.infer.deep_eval"].DeepProperty = Model
    with pytest.raises(RuntimeError, match="second checkpoint cannot load"):
        build_predictor(cfg, seed=2)
    assert len(loaded) == 1 and loaded[0].closes == 1


def test_ooh_direct_close_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_model(monkeypatch)
    config(tmp_path)
    predictor = OOHFramesPredictor({"models": ["relative.pt"]})
    closes = []
    predictor.models[0].close = lambda: closes.append(1)
    predictor.close()
    predictor.close()
    assert closes == [1]
