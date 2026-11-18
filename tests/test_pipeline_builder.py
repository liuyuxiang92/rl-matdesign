"""Physics-independent contracts for the composable structure builder."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from rl_matdesign.predictors.builders.pipeline import (
    FRAMES_INFO, ORIGINAL_IDS, PIPELINE_INFO, PipelineBuilder,
    pipeline_cation_amounts, pipeline_frames, pipeline_group_constraints,
    validate_pipeline_config,
)
from rl_matdesign.structure_sources import StructureSpec
from rl_matdesign.utils.structure_samples import SAMPLE_INFO_KEY


def poscar(tmp_path, blocks, name="POSCAR"):
    count = sum(n for _, n in blocks)
    # Explicit headers intentionally preserve duplicate species blocks.
    text = ["test", "1", "50 0 0", "0 50 0", "0 0 50",
            " ".join(s for s, _ in blocks), " ".join(str(n) for _, n in blocks), "Direct"]
    text += [f"{((i % 10)+.5)/10} {(((i//10)%10)+.5)/10} {((i//100)+.5)/10}" for i in range(count)]
    path = tmp_path / name
    path.write_text("\n".join(text) + "\n")
    return str(path)


def builder(path, *steps):
    return PipelineBuilder(dict(base_poscar=path, steps=list(steps)))


def test_repeated_symbols_consume_original_blocks_in_letter_order_and_cache(tmp_path, monkeypatch):
    path = poscar(tmp_path, [("Sr", 2), ("Fe", 3), ("Fe", 4)])
    stage = dict(type="substitute", c_site_symbol="Sr", c_dopant_group="C",
                 b_site_symbol="Fe", b_dopant_group="B", a_site_symbol="Fe", a_dopant_group="A")
    b = builder(path, stage)
    candidate = {"A": {"Mn": 1}, "B": {"Co": 1}, "C": {"Ca": 1}}
    actual = b.build(candidate)[0]
    assert actual.get_chemical_symbols() == ["Ca"]*2 + ["Mn"]*3 + ["Co"]*4
    assert actual.arrays[ORIGINAL_IDS].tolist() == list(range(9))
    plan = b.plan_candidate(candidate)
    assert [plan["bindings"][g]["index"] for g in ["A", "B", "C"]] == [1, 2, 0]
    assert plan["counts"] == {"Ca": 2, "Co": 4, "Mn": 3}
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("template reloaded"))
    assert b.build(candidate)[0].get_chemical_symbols() == actual.get_chemical_symbols()
    json.dumps(actual.info)  # bundle metadata is portable through worker boundaries


def test_exhausted_symbol_is_error_and_unselected_blocks_remain_fixed(tmp_path):
    path = poscar(tmp_path, [("Fe", 2), ("O", 3), ("Fe", 1)])
    stage = dict(type="substitute", a_site_symbol="Fe", a_dopant_group="A")
    b = builder(path, stage)
    assert b.build({"A": {"Ni": 1}})[0].get_chemical_symbols() == ["Ni"]*2 + ["O"]*3 + ["Fe"]
    stage.update(b_site_symbol="Fe", b_dopant_group="B", c_site_symbol="Fe", c_dopant_group="C")
    with pytest.raises(ValueError, match="No unused original"):
        builder(path, stage).plan_candidate({"A": {"Ni": 1}, "B": {"Co": 1}, "C": {"Mn": 1}})


def test_multiple_substitution_steps_use_original_identity(tmp_path):
    path = poscar(tmp_path, [("Sr", 2), ("Fe", 2)])
    b = builder(path, dict(type="substitute", a_site_symbol="Sr", a_dopant_group="A"),
                dict(type="substitute", a_site_symbol="Fe", a_dopant_group="B"))
    # The second step must not grab Fe atoms just produced by the first step.
    assert b.build({"A": {"Fe": 1}, "B": {"Co": 1}})[0].get_chemical_symbols() == ["Fe"]*2 + ["Co"]*2


def test_lazy_initialization_and_validation_do_not_read_template_or_load_models(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("eager asset read"))
    cfg = dict(base_poscar="missing.POSCAR", steps=[
        dict(type="substitute", a_site_symbol="Fe", a_dopant_group="A"),
        dict(type="geo_opt", model="missing.pt")])
    validate_pipeline_config(cfg)
    b = PipelineBuilder(cfg)
    assert b._templates == {} and b._calculators == {}
    source = StructureSpec({"builder": "pipeline", **cfg})
    assert len(b.sample_plan({"A": {"Fe": 1}}, n_configs=2, source_spec=source, eval_seed=5)) == 2


def test_multi_template_seeded_samples_keep_parent_ids_and_override_counts(tmp_path):
    paths = [poscar(tmp_path, [("Ti", 6)], name=f"P{i}") for i in range(2)]
    cfg = dict(builder="pipeline", base_poscar=[dict(path=paths[0], n_configs=2), dict(path=paths[1], n_configs=3)],
               steps=[dict(type="substitute", a_site_symbol="Ti", a_dopant_group="alloy")])
    b = PipelineBuilder(cfg)
    candidate = {"alloy": {"Ti": .5, "Zr": .5}}
    source = StructureSpec(cfg)
    kwargs = dict(n_configs=9, source_spec=source, eval_seed=17)
    plan = b.sample_plan(candidate, **kwargs)
    assert [(s.template_index,s.sample_index) for s in plan] == [(0,0),(0,1),(1,0),(1,1),(1,2)]
    a = b.build_seeded(candidate, **kwargs)
    z = [b.build_sample(candidate, sample=s) for s in reversed(plan)]
    assert [x.get_chemical_symbols() for x in a] == [x.get_chemical_symbols() for x in reversed(z)]
    assert [x.info[SAMPLE_INFO_KEY] for x in a] == [s.record() for s in plan]


def sse_cfg(path):
    return dict(base_poscar=path, steps=[
        dict(type="substitute", a_site_symbol="P", a_dopant_group="P_site", amount_basis="cation_fraction"),
        dict(type="sse_charge_balance", p_site_group="P_site", halide_group="halide",
             eligible_region=dict(symbol="S", take="last", count=5),
             eligible_region_fallback=dict(symbol="S", take="first", count=60),
             valences={"Li":1,"P":5,"S":-2,"O":-2,"Cl":-1,"Br":-1,
                       "Ni":dict(oxide=3,sulfide=2),"Mn":dict(oxide=4,sulfide=2),"Ru":dict(oxide=2,sulfide=4)})])


def test_cation_recipes_and_intrinsic_duplicate_form_constraint():
    assert pipeline_cation_amounts({"P": .92,"Ni2O3": .04,"MnO2": .04}) == {"P": .92,"Ni": .04,"O": .14,"Mn": .04}
    predicate = pipeline_group_constraints(sse_cfg("absent"))["P_site"]
    assert predicate({"Ni": .02, "Ni2O3": 0})
    assert not predicate({"P": .96, "Ni": .02, "Ni2O3": .02})


def test_sse_independent_rounding_charge_balance_and_original_region_fallback(tmp_path):
    path = poscar(tmp_path, [("Li",60),("P",10),("S",60)])
    b = PipelineBuilder(sse_cfg(path))
    candidate = {"P_site": {"P": .84,"Ni2O3": .08,"Mn": .08}, "halide": {"Cl": .8}}
    p = b.plan_candidate(candidate)
    c = p["sse"]["P_site"]
    # Independent .8 -> 1 rounding for each metal, residual P=8, oxide O=round(1.5)=2.
    assert c["metals"] == {"Mn":1,"Ni":1}
    assert {k:c[k] for k in ["P","O","Cl","Br","S","Li","Li_delete"]} == dict(P=8,O=2,Cl=8,Br=9,S=41,Li=58,Li_delete=2)
    assert len(c["anion_original_ids"]) == 60  # selected fallback only on capacity overflow
    actual = b.build(candidate, rng=np.random.default_rng(2))[0]
    assert dict(Counter(actual.get_chemical_symbols())) == p["counts"]
    original_ids = actual.arrays[ORIGINAL_IDS]
    assert len(set(original_ids)) == len(actual)
    changed_anions = [int(original_ids[i]) for i,a in enumerate(actual) if a.symbol in ("O","Cl","Br")]
    assert all(70 <= i < 130 for i in changed_anions)
    assert sum(i < 60 for i in original_ids) == 58
    assert "Li5.8" in b.composition_formula(candidate)
    bad = deepcopy(candidate)
    bad["P_site"] = {"P": .84, "Ni": .08, "Ni2O3": .08}
    with pytest.raises(ValueError, match="one positive form"):
        b.plan_candidate(bad)


def test_sse_rejects_mislabeled_oxide_and_never_reinterprets_metal_as_zero_valence(tmp_path):
    path = poscar(tmp_path, [("Li",60),("P",10),("S",60)])
    b = PipelineBuilder(sse_cfg(path))
    with pytest.raises(ValueError, match="disagrees"):
        b.plan_candidate({"P_site":{"P":.9,"NiO":.1},"halide":{"Cl":.8}})
    c = b.plan_candidate({"P_site":{"P":.9,"Ru":.1},"halide":{"Cl":.8}})["sse"]["P_site"]
    assert c["Li"] == 54  # Ru uses configured sulfide +4, never neutral Ru(0)


def test_signed_defect_removes_selected_species_and_keeps_other_block_ids(tmp_path):
    path = poscar(tmp_path, [("Sr",4),("Fe",2)])
    stage = dict(type="substitute", a_site_symbol="Sr", a_dopant_group="A", a_defect_group="defect",
                 b_site_symbol="Fe", b_dopant_group="B")
    b = builder(path, stage)
    candidate = {"A":{"Sr":.5,"Ca":.5},"B":{"Co":1},"defect":{"Ca":-.75}}
    plan = b.plan_candidate(candidate)
    assert plan["bindings"]["A"]["defects"] == {"Ca":-2}  # legacy clamp
    result = b.build(candidate)[0]
    assert Counter(result.get_chemical_symbols()) == {"Sr":2,"Co":2}
    assert set(result.arrays[ORIGINAL_IDS]) >= {4,5}
    with pytest.raises(ValueError, match="exceeds"):
        builder(path,{**stage,"vacancy_policy":"error"}).plan_candidate(candidate)


def test_positive_defect_keeps_inserted_identity_separate(tmp_path, monkeypatch):
    from ase import Atom
    from rl_matdesign.utils import structure
    path = poscar(tmp_path,[("Sr",2),("Fe",1)])
    b = builder(path,dict(type="substitute",a_site_symbol="Sr",a_dopant_group="A",a_defect_group="defect"))
    def insert(atoms,specs,rng,**kwargs):
        for element,n in specs:
            for _ in range(n): atoms.append(Atom(element,position=[1,1,1]))
        return atoms
    monkeypatch.setattr(structure,"_insert_interstitials",insert)
    out = b.build({"A":{"Sr":1},"defect":{"Sr":.5}})[0]
    assert out.arrays[ORIGINAL_IDS].tolist() == [0,1,2,-1]


def test_ooh_padded_frames_anchor_map_and_relaxation_consume_previous_stage(tmp_path, monkeypatch):
    from rl_matdesign.utils import structure
    path = poscar(tmp_path,[("Co",4),("O",2),("Co",2)])
    b = builder(path,dict(type="substitute",a_site_symbol="Co",a_dopant_group="ooh"),
        dict(type="ooh_adsorbates",adsorbates=["O*","OH*","OOH*"],ads_height=1.9,ads_dz=1),
        dict(type="geo_opt",model="fake.pt",head="test",relax_cell=False,steps=3,fmax=.1))
    class DP:
        def __init__(self,**kwargs): self.kwargs=kwargs
    monkeypatch.setitem(sys.modules,"deepmd.calculator",types.SimpleNamespace(DP=DP))
    calls=[]
    def relax(atoms,**kwargs):
        calls.append((atoms.get_chemical_symbols(),kwargs))
        result=atoms.copy()
        result.positions[0,0] += 1
        return result
    monkeypatch.setattr(structure,"relax_structure",relax)
    out=b.build({"ooh":{"Ni":1}},rng=np.random.default_rng(8))[0]
    decoded=pipeline_frames(out)
    assert [x[0] for x in decoded] == ["O*","OH*","OOH*"]
    assert [x[2] for x in decoded] == [[9,10],[9],[]]
    assert all(len(frame)==11 for _,frame,_ in decoded)
    assert all(frame.get_chemical_symbols()==["Ni"]*4+["O"]*2+["Co"]*2+["O","O","H"] for _,frame,_ in decoded)
    assert len(calls)==3 and len(b._calculators)==1
    for label,frame,masks in decoded:
        metadata=frame.info["_rl_matdesign_adsorbate"]
        assert metadata["anchor_original_id"] in [0,1,2,3]
        assert metadata["eligible_anchor_original_ids"] == [0,1,2,3]
        assert frame.arrays[ORIGINAL_IDS][-3:].tolist()==[-1,-1,-1]
        assert frame.positions[0,0] == out.positions[0,0]+1
        assert FRAMES_INFO not in frame.info
    json.dumps(out.info)


def test_md_training_checkpoint_is_rejected_before_loading_external_runtime(tmp_path):
    from rl_matdesign.predictors.builders.pipeline_md import md_average
    from ase import Atoms
    with pytest.raises(ValueError,match="frozen LAMMPS model"):
        md_average(Atoms("Fe"),dict(model="model.ckpt.pt",lmp_bin="/missing/lmp"),rng=np.random.default_rng(0))


@pytest.mark.parametrize("step",[
    dict(type="substitute",a_site_symbol="Fe"),
    dict(type="substitute",a_site_symbol="Fe",a_dopant_group="A",site_map={"A":"Fe"}),
    dict(type="unknown"),
    dict(type="ooh_adsorbates",adsorbates=["O*","O*"]),
])
def test_unknown_or_ambiguous_stage_schema_is_rejected(step):
    with pytest.raises(ValueError): validate_pipeline_config(dict(base_poscar="not_read",steps=[step]))


@pytest.mark.parametrize("exit_code", [0, 1])
def test_md_operation_preserves_input_identity_and_averages_previous_structure(tmp_path, monkeypatch, exit_code):
    from ase import Atoms
    from rl_matdesign.predictors.builders import pipeline_md
    model = tmp_path / "frozen.pt"
    model.write_bytes(b"test")
    original = Atoms("NiFe", positions=[[0,0,0],[5,5,5]], cell=[20,20,20], pbc=True)
    original.set_array(ORIGINAL_IDS,np.asarray([7,3]))
    original.info[PIPELINE_INFO] = {"template_index":2}
    original.info[SAMPLE_INFO_KEY] = {"seed":19}
    calls = {}
    def write(path,structure,type_map,source):
        calls["written"] = structure.copy()
        calls["types"] = type_map
        Path(path).write_text("fake conf")
    def render(**kwargs):
        calls["render"] = kwargs
        return "fake deck"
    def read(path,min_step):
        calls["min_step"] = min_step
        return ["fake frames"]
    def average(frames,rms_max):
        assert frames == ["fake frames"] and rms_max == .4
        return dict(n_atoms=2,types=np.array([2,1]),cell=np.eye(3)*21,coords=np.array([[1,2,3],[6,7,8]])),dict(n_frames=5)
    runtime = types.SimpleNamespace(write_conf_lmp=write,render_input_lammps=render,
                                  read_frames_since=read,average_frames=average)
    monkeypatch.setitem(sys.modules,"clc_workflow",types.ModuleType("clc_workflow"))
    monkeypatch.setitem(sys.modules,"clc_workflow.lammps_io",runtime)
    monkeypatch.setitem(sys.modules,"pymatgen.io.ase",types.SimpleNamespace(
        AseAtomsAdaptor=types.SimpleNamespace(get_structure=lambda atoms: atoms)))
    def run(command,**kwargs):
        calls["run"] = (command,kwargs)
        return types.SimpleNamespace(returncode=exit_code,stdout="done",stderr="")
    monkeypatch.setattr(pipeline_md.subprocess,"run",run)
    cfg=dict(model=str(model),lmp_bin="/mock/lmp",nsteps=100,timestep_ps=.1,average_last_ps=2,
             avg_rms_max=.4,temperature_k=470,velocity_seed=31,min_dist=1.7,scratch_dir=str(tmp_path),
             head="frozen_head",lmp_args=["-k","on"])
    from rl_matdesign.utils.stage_timing import capture_stage_times
    with capture_stage_times() as timing:
        if exit_code:
            with pytest.raises(RuntimeError, match="LAMMPS failed"):
                pipeline_md.md_average(original,cfg,rng=np.random.default_rng(1))
            assert timing.counts == {"md": 1}
            assert Path(calls["run"][1]["cwd"]).exists()
            return
        out=pipeline_md.md_average(original,cfg,rng=np.random.default_rng(1))
    assert timing.counts == {"md": 1} and timing.seconds["md"] >= 0
    assert calls["written"].get_chemical_symbols() == ["Ni","Fe"]
    assert calls["types"] == ["Fe","Ni"]
    assert calls["render"]["model_head"] == "frozen_head"
    assert calls["render"]["seed"] == 31 and calls["min_step"] == 80
    assert out.arrays[ORIGINAL_IDS].tolist() == [7,3]
    assert out.info[SAMPLE_INFO_KEY] == {"seed":19}
    assert out.info[PIPELINE_INFO] == {"template_index":2}
    assert np.array_equal(out.positions,np.array([[1,2,3],[6,7,8]]))
    assert np.array_equal(original.positions,np.array([[0,0,0],[5,5,5]]))
    assert not Path(calls["run"][1]["cwd"]).exists()
    assert out.info["_rl_matdesign_md"] == dict(n_frames=5,temperature_k=470,velocity_seed=31)


def test_defects_allow_new_interstitial_species_and_clamp_absent_vacancies(tmp_path, monkeypatch):
    from ase import Atom
    from rl_matdesign.utils import structure
    path = poscar(tmp_path,[("Sr",2),("Fe",1)])
    stage=dict(type="substitute",a_site_symbol="Sr",a_dopant_group="A",a_defect_group="defect")
    b = builder(path,stage)
    positive={"A":{"Sr":1},"defect":{"Ca":.5}}
    negative={"A":{"Sr":1},"defect":{"Ca":-.5}}
    assert b.plan_candidate(positive)["counts"] == {"Sr":2,"Fe":1,"Ca":1}
    assert b.plan_candidate(negative)["counts"] == {"Sr":2,"Fe":1}
    def insert(atoms,specs,rng,**kwargs):
        for element,n in specs:
            for _ in range(n): atoms.append(Atom(element,position=[1,1,1]))
        return atoms
    monkeypatch.setattr(structure,"_insert_interstitials",insert)
    out = b.build(positive)[0]
    assert Counter(out.get_chemical_symbols()) == {"Sr":2,"Fe":1,"Ca":1}
    assert out.arrays[ORIGINAL_IDS].tolist() == [0,1,2,-1]
    assert Counter(b.build(negative)[0].get_chemical_symbols()) == {"Sr":2,"Fe":1}
    with pytest.raises(ValueError,match="Vacancy request exceeds"):
        builder(path,{**stage,"vacancy_policy":"error"}).plan_candidate(negative)


def test_pipeline_unknown_top_level_settings_cannot_be_silently_ignored():
    cfg=dict(base_poscar="missing",steps=[dict(type="substitute",a_site_symbol="Fe",a_dopant_group="A")])
    with pytest.raises(ValueError,match="Unsupported pipeline structure fields"):
        validate_pipeline_config({**cfg,"site_symbol":"Fe"})
    with pytest.raises(ValueError,match="Unsupported pipeline structure fields"):
        validate_pipeline_config({**cfg,"geo_opt":{"enabled":True}})
