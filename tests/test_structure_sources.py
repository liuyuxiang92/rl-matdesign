"""Pure singular structure declarations and reproducible sampling identities."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rl_matdesign.structure_sources import StructureSpec, resolve_structure_plan, resolve_source_assets


def pipeline(**changes):
    cfg = {"builder": "pipeline", "base_poscar": "template.vasp", "n_random_configs": 4,
           "steps": [{"type": "substitute", "a_site_symbol": "Ti", "a_dopant_group": "alloy"}]}
    cfg.update(changes)
    return cfg


def config(**changes):
    cfg = {"structure": pipeline(),
           "properties": [{"name": "a", "predictor": "dp_energy", "models": ["energy.pt"], "direction": "min"},
                          {"name": "b", "predictor": "dp_property", "models": ["property.pt"], "direction": "max"}]}
    cfg.update(changes)
    return cfg


def source(**changes):
    return resolve_structure_plan(config(**changes), needs_structure=["a", "b"]).structure


def test_one_structure_and_defensive_frozen_declarations():
    cfg = config()
    plan = resolve_structure_plan(cfg, needs_structure=["a", "b"])
    original = plan.structure.to_schema()
    cfg["structure"]["base_poscar"] = "changed.vasp"
    plan.property_sources["a"] = "changed"
    plan.structure.effective_config["steps"][0]["a_site_symbol"] = "Fe"
    plan.structure.physical_protocol["config"]["base_poscar"] = "other.vasp"
    assert plan.structure.to_schema() == original
    assert plan.property_sources == {"a": "structure", "b": "structure"}
    assert plan.sources == (plan.structure,)
    with pytest.raises(FrozenInstanceError):
        plan.structure.source_id = "changed"
    assert plan.source("structure").builder == "pipeline"
    with pytest.raises(KeyError):
        plan.source("missing")


def test_pure_composition_and_peptide_design_need_no_ase_structure():
    cfg = {"properties": [{"predictor": "composition"}]}
    for structure in (None, {"builder": "peptide", "sequence_group": "sequence"}):
        if structure is not None:
            cfg["structure"] = structure
        plan = resolve_structure_plan(cfg, needs_structure=[])
        assert plan.structure is None and plan.property_sources == {}


@pytest.mark.parametrize("change,needed,message", [
    ({"structure": None}, ["a", "b"], "require a structure"),
    ({"structure": {}}, ["a", "b"], "builder"),
    ({"structure": {"builder": "peptide"}}, ["a"], "does not produce ASE"),
    ({"properties": [{"name": "a", "source": "structure"}]}, ["a"], "implicitly"),
    ({}, [], "no property consumes"),
    ({"base_poscar": "conflicting.vasp"}, ["a", "b"], "conflict"),
    ({"share_structure": True}, ["a", "b"], "Removed"),
    ({"structure_sources": {}}, ["a", "b"], "Removed"),
    ({"sweep": {"values": [460]}}, ["a", "b"], "Removed"),
    ({"geo_opt": {}}, ["a", "b"], "conflict"),
    ({"properties": [{"name": "a", "n_random_configs": 2}]}, ["a"], "structure-owned"),
    ({"properties": [{"name": "a"}, {"name": "a"}]}, ["a"], "Duplicate"),
    ({}, ["unknown"], "Unknown structure properties"),
    ({}, ["a", "a"], "duplicate"),
])
def test_invalid_structure_relationships_fail(change, needed, message):
    with pytest.raises(ValueError, match=message):
        resolve_structure_plan(config(**change), needs_structure=needed)


@pytest.mark.parametrize("changes", [{"weight": 2}, {"eval_seed": 8}, {"dqn_lr": .1},
                                    {"n_random_configs": True}, {"n_random_configs": 0},
                                    {"geo_opt": {}}, {"geo_opt": {"enabled": True}}])
def test_structure_rejects_unowned_parameters_and_invalid_counts(changes):
    with pytest.raises(ValueError):
        StructureSpec(pipeline(**changes))


def test_seed_is_candidate_local_typed_and_order_independent():
    spec = source()
    candidate = {"alloy": {"Ni": .2, "Fe": .4}, "peptide": ["01", "1"]}
    reordered = {"peptide": ["01", "1"], "alloy": {"Fe": .4, "Ni": .2}}
    expected = spec.seed_for(candidate, eval_seed=17, template_id="a", sample_index=2)
    spec.seed_for({"other": True}, eval_seed=17)
    assert spec.seed_for(reordered, eval_seed=17, template_id="a", sample_index=2) == expected
    assert spec.seed_for({"Ni": 1}, eval_seed=17) == spec.seed_for({"Ni": 1.0}, eval_seed=17)
    assert len({spec.seed_for({"field": value}, eval_seed=17) for value in [True, 1, "1", "01"]}) == 4
    assert spec.seed_for({"peptide": ["01", "1"]}, eval_seed=17) != spec.seed_for({"peptide": ["1", "01"]}, eval_seed=17)
    candidates = [{"Ni": .2}, {"Ni": .4}, {"Ni": .8}]
    joined = [spec.seed_for(c, eval_seed=17) for c in candidates]
    split = [spec.seed_for(c, eval_seed=17) for batch in [candidates[:1], candidates[1:]] for c in batch]
    assert joined == split and all(0 <= value < 2**64 for value in joined)


def test_reward_property_order_and_runtime_settings_cannot_change_seed():
    cfg = config()
    before = source()
    cfg["properties"].reverse()
    cfg["properties"][0].update(scale=100, direction="min")
    cfg["reward"] = {"objective": {"terms": {"a": 1, "b": 91}}}
    cfg.update(k=5, method="dqn", dqn_batch_size=32)
    cfg["structure"]["structure_parallel"] = {"devices": [0, 1], "candidate_batch_size": 8}
    after = resolve_structure_plan(cfg, needs_structure=["b", "a"]).structure
    assert before.physical_protocol == after.physical_protocol
    assert before.seed_for({"Ti": 1}, eval_seed=1) == after.seed_for({"Ti": 1}, eval_seed=1)
    assert after.effective_config["structure_parallel"]["devices"] == [0, 1]


def test_physics_asset_content_purpose_template_and_sample_change_seed():
    base = source()
    candidate = {"Ti": 1}
    changed = base.effective_config
    changed["steps"][0]["a_site_symbol"] = "Fe"
    alternate = StructureSpec(changed)
    content = base.with_asset_hashes({"template": "a" * 64})
    assert content.with_asset_hashes({"template": "b" * 64}).protocol_fingerprint != content.protocol_fingerprint
    assert len({base.seed_for(candidate, eval_seed=11), alternate.seed_for(candidate, eval_seed=11),
                content.seed_for(candidate, eval_seed=11), base.seed_for(candidate, eval_seed=12),
                base.seed_for(candidate, eval_seed=11, purpose="md"),
                base.seed_for(candidate, eval_seed=11, template_id="another"),
                base.seed_for(candidate, eval_seed=11, sample_index=0)}) == 7


def test_seed_is_stable_in_fresh_processes_with_different_python_hash_seeds():
    code = ("from rl_matdesign.structure_sources import StructureSpec; "
            "s=StructureSpec({'builder':'external:Builder','base_poscar':'a.vasp'}); "
            "print(s.seed_for({'Fe':.4,'Ni':.2},eval_seed=42,purpose='build',template_id='a',sample_index=0)); "
            "import sys; assert not any(name in sys.modules for name in ('torch','deepmd','ase'))")
    root = Path(__file__).resolve().parents[1]
    outputs = []
    for seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(root / "src")}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=True, timeout=30, env=env)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
    assert 0 <= int(outputs[0]) < 2**64


def test_generated_property_name_and_external_builder_fields_are_preserved():
    cfg = {"structure": {"builder": "external.builder:Builder", "special_input": {"condition": 3}},
           "properties": [{"predictor": "dp_energy"}]}
    plan = resolve_structure_plan(cfg, needs_structure=["prop0"])
    assert plan.property_sources == {"prop0": "structure"}
    assert plan.structure.effective_config["special_input"] == {"condition": 3}


def test_runtime_asset_resolution_preserves_adapter_hashes_and_nested_models(tmp_path):
    path = tmp_path / "template.vasp"
    path.write_text("template content")
    model = tmp_path / "relax.pt"
    model.write_text("model content")
    cfg = pipeline(base_poscar=str(path))
    cfg["steps"].append({"type": "geo_opt", "model": str(model)})
    spec = StructureSpec(cfg, asset_hashes={"adapter_owned": "a" * 64})
    frozen, references = resolve_source_assets(spec)
    hashes = frozen.physical_protocol["asset_sha256"]
    assert hashes["adapter_owned"] == "a" * 64
    assert hashes["base_poscar[0]"] == references[0]["sha256"]
    assert hashes["steps[1].model[0]"] == references[1]["sha256"]
