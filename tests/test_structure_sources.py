"""Pure source declarations and deterministic seed identities; no models needed."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rl_matdesign.structure_sources import (
    LegacySourceMigrationRequired, StructureSourceSpec, normalize_structure_sources,
)


def config(**changes):
    result = {
        "structure_sources": {"bulk": {"builder": "substitute", "base_poscar": "template.vasp",
                                         "site_symbol": "Ti", "n_random_configs": 4}},
        "properties": [{"name": "a", "predictor": "dp_energy", "models": ["energy.pt"],
                        "direction": "min"},
                       {"name": "b", "predictor": "dp_property", "models": ["property.pt"],
                        "direction": "max"}],
    }
    result.update(changes)
    return result


def source(**changes):
    return normalize_structure_sources(config(**changes), needs_structure=["a", "b"]).sources[0]


def test_one_source_auto_reference_and_defensive_frozen_declarations():
    cfg = config()
    plan = normalize_structure_sources(cfg, needs_structure=["a", "b"])
    original = plan.sources[0].to_schema()
    cfg["structure_sources"]["bulk"]["base_poscar"] = "changed.vasp"
    plan.property_sources["a"] = "changed"
    plan.sources[0].builder_config["site_symbol"] = "Fe"
    plan.sources[0].geo_opt["enabled"] = True
    plan.sources[0].physical_protocol["config"]["site_symbol"] = "Co"
    assert plan.sources[0].to_schema() == original
    assert plan.property_sources == {"a": "bulk", "b": "bulk"}
    with pytest.raises(FrozenInstanceError):
        plan.sources[0].source_id = "changed"
    assert plan.source("bulk").builder == "substitute"
    with pytest.raises(KeyError):
        plan.source("missing")


def test_two_equal_definitions_remain_independent_named_sources():
    cfg = config()
    cfg["structure_sources"]["independent"] = deepcopy(cfg["structure_sources"]["bulk"])
    cfg["properties"][0]["source"] = "bulk"
    cfg["properties"][1]["source"] = "independent"
    plan = normalize_structure_sources(cfg, needs_structure=["a", "b"])
    first, second = plan.sources
    assert first.physical_protocol == second.physical_protocol
    assert first.seed_for({"Ti": 1}, eval_seed=11) != second.seed_for({"Ti": 1}, eval_seed=11)
    assert plan.property_sources == {"a": "bulk", "b": "independent"}


def test_legacy_shared_normalizes_without_reward_model_or_run_parameters():
    cfg = {"builder": "sse", "base_poscar": ["a.vasp", {"path": "b.vasp", "n_configs": 2}],
           "groups": [{"name": "S_site", "choices": [{"name": "O_a", "element": "O", "values": [0, 1]}]}],
           "valences": {"Sn": 4}, "geo_opt": {"model": "relax.pt"}, "n_random_configs": 5,
           "k": 3, "method": "a2c", "eval_seed": 4, "pg_batch_eps": 10,
           "properties": [{"name": "a", "models": ["property.pt"], "weight": 9}]}
    plan = normalize_structure_sources(cfg, needs_structure=["a"])
    spec = plan.source("default")
    assert plan.property_sources == {"a": "default"}
    assert spec.n_random_configs == 5
    assert spec.geo_opt == {"model": "relax.pt"}
    assert spec.builder_config["groups"] == cfg["groups"]  # SSE O-form slot context.
    assert not ({"properties", "models", "weight", "k", "method", "eval_seed", "pg_batch_eps"}
                & set(spec.effective_config))
    assert plan.migration_notes


def test_legacy_false_requires_explicit_decision_and_reports_actual_old_inheritance():
    cfg = {"share_structure": False, "builder": "substitute", "base_poscar": "top.vasp",
           "site_map": {"Ti": "host"}, "n_random_configs": 3,
           "geo_opt": {"model": "global.pt"},
           "properties": [{"name": "a", "n_random_configs": 2},
                          {"name": "b", "base_poscar": "other.vasp", "geo_opt": {"model": "ignored.pt"}}]}
    with pytest.raises(LegacySourceMigrationRequired) as exc:
        normalize_structure_sources(cfg, needs_structure=["a", "b"])
    report = exc.value.migration_report
    sources = report["structure_sources"]
    assert report["property_sources"] == {"a": "property:a", "b": "property:b"}
    assert sources["property:a"]["n_random_configs"] == 2
    assert sources["property:b"]["n_random_configs"] == 3
    assert sources["property:b"]["base_poscar"] == "other.vasp"
    assert sources["property:b"]["geo_opt"] == {"model": "global.pt"}
    assert "site_map" not in sources["property:a"]
    assert "site_map" in report["ignored_top_level_builder_fields"]
    report.clear()
    assert exc.value.migration_report["reason"] == "legacy_independent_sampling"


def test_legacy_false_equal_settings_are_not_silently_merged():
    with pytest.raises(LegacySourceMigrationRequired) as exc:
        normalize_structure_sources({"share_structure": False, "base_poscar": "same.vasp",
                                     "properties": [{"name": "a"}, {"name": "b"}]},
                                    needs_structure=["a", "b"])
    assert len(exc.value.migration_report["structure_sources"]) == 2


def test_pure_composition_needs_no_sources_or_builder_imports():
    plan = normalize_structure_sources({"builder": "does_not_exist:Builder",
                                       "properties": [{"predictor": "composition"}]},
                                      needs_structure=[])
    assert plan.sources == () and plan.property_sources == {}


@pytest.mark.parametrize("change,needed,message", [
    ({"structure_sources": {"bulk": {}, "other": {}}}, ["a", "b"], "explicitly name"),
    ({"structure_sources": {}}, ["a", "b"], "explicitly name"),
    ({"properties": [{"name": "a", "source": "missing"}]}, ["a"], "unknown source"),
    ({"properties": [{"name": "a", "source": "bulk"}]}, [], "cannot reference"),
    ({}, [], "Unused"),
    ({"base_poscar": "conflicting.vasp"}, ["a", "b"], "conflict"),
    ({"share_structure": True}, ["a", "b"], "conflict"),
    ({"geo_opt": {"enabled": False}}, ["a", "b"], "conflict"),
    ({"properties": [{"name": "a", "n_random_configs": 2}]}, ["a"], "source-owned"),
    ({"properties": [{"name": "a"}, {"name": "a"}]}, ["a"], "Duplicate"),
    ({}, ["unknown"], "Unknown structure properties"),
    ({}, ["a", "a"], "duplicate"),
])
def test_invalid_or_ambiguous_source_relationships_fail(change, needed, message):
    with pytest.raises(ValueError, match=message):
        normalize_structure_sources(config(**change), needs_structure=needed)


@pytest.mark.parametrize("changes", [{"weight": 2}, {"eval_seed": 8}, {"dqn_lr": .1},
                                    {"n_random_configs": True}, {"n_random_configs": 0},
                                    {"geo_opt": True}, {"geo_opt": {"enabled": 1}}])
def test_source_rejects_unowned_parameters_and_invalid_counts(changes):
    with pytest.raises(ValueError):
        StructureSourceSpec("s", {"base_poscar": "a.vasp", **changes})


def test_seed_is_candidate_local_typed_and_order_independent():
    spec = source()
    candidate = {"alloy": {"Ni": .2, "Fe": .4}, "peptide": ["01", "1"]}
    reordered = {"peptide": ["01", "1"], "alloy": {"Fe": .4, "Ni": .2}}
    expected = spec.seed_for(candidate, eval_seed=17, template_id="a", sample_index=2)
    spec.seed_for({"other": True}, eval_seed=17)
    assert spec.seed_for(reordered, eval_seed=17, template_id="a", sample_index=2) == expected
    assert spec.seed_for({"Ni": 1}, eval_seed=17) == spec.seed_for({"Ni": 1.0}, eval_seed=17)
    seeds = {spec.seed_for({"field": value}, eval_seed=17) for value in [True, 1, "1", "01"]}
    assert len(seeds) == 4
    assert spec.seed_for({"peptide": ["01", "1"]}, eval_seed=17) != spec.seed_for({"peptide": ["1", "01"]}, eval_seed=17)
    candidates = [{"Ni": .2}, {"Ni": .4}, {"Ni": .8}]
    joined = [spec.seed_for(c, eval_seed=17) for c in candidates]
    split = [spec.seed_for(c, eval_seed=17) for batch in [candidates[:1], candidates[1:]] for c in batch]
    assert joined == split
    assert all(0 <= value < 2**64 for value in joined)


def test_reward_property_order_runtime_and_cache_settings_cannot_change_seed():
    cfg = config()
    before = source()
    cfg["properties"].reverse()
    cfg["properties"][0].update(weight=91, scale=100, direction="min")
    cfg.update(k=5, predict_cache_size=1, method="dqn", dqn_batch_size=32,
               structure_parallel={"devices": [0, 1], "batch_size": 8})
    after = normalize_structure_sources(cfg, needs_structure=["b", "a"]).source("bulk")
    assert before.physical_protocol == after.physical_protocol
    assert before.seed_for({"Ti": 1}, eval_seed=1) == after.seed_for({"Ti": 1}, eval_seed=1)
    assert after.effective_config["structure_parallel"]["devices"] == [0, 1]


def test_physics_asset_content_purpose_template_and_sample_change_seed():
    base = source()
    candidate = {"Ti": 1}
    original = base.seed_for(candidate, eval_seed=11)
    alternate = StructureSourceSpec("bulk", {**base.builder_config, "site_symbol": "Fe"})
    content = base.with_asset_hashes({"template": "a" * 64})
    assert content.with_asset_hashes({"template": "b" * 64}).protocol_fingerprint != content.protocol_fingerprint
    assert len({original, alternate.seed_for(candidate, eval_seed=11),
                content.seed_for(candidate, eval_seed=11),
                base.seed_for(candidate, eval_seed=12),
                base.seed_for(candidate, eval_seed=11, purpose="md"),
                base.seed_for(candidate, eval_seed=11, template_id="another"),
                base.seed_for(candidate, eval_seed=11, sample_index=0)}) == 7


def test_seed_is_stable_in_fresh_processes_with_different_python_hash_seeds():
    code = ("from rl_matdesign.structure_sources import StructureSourceSpec; "
            "s=StructureSourceSpec('bulk', {'builder':'substitute','base_poscar':'a.vasp'}); "
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
    # Version 1 golden vector: changing this derivation requires a version bump,
    # rather than silently changing every source's recorded sampling stream.
    assert outputs[0] == "9568421796554556625"


def test_source_names_generated_by_existing_predictor_and_fqn_fields_are_preserved():
    cfg = {"builder": "external.builder:Builder", "special_input": {"condition": 3},
           "properties": [{"predictor": "dp_energy"}]}
    plan = normalize_structure_sources(cfg, needs_structure=["prop0"])
    assert plan.property_sources == {"prop0": "default"}
    assert plan.sources[0].builder_config["special_input"] == {"condition": 3}
    cfg["share_structure"] = False
    cfg["properties"][0]["special_input"] = {"condition": 4}
    with pytest.raises(LegacySourceMigrationRequired) as exc:
        normalize_structure_sources(cfg, needs_structure=["prop0"])
    assert exc.value.migration_report["structure_sources"]["property:prop0"]["special_input"] == {"condition": 4}



def test_runtime_asset_resolution_preserves_explicit_adapter_asset_hashes(tmp_path):
    from rl_matdesign.structure_sources import resolve_source_assets
    path = tmp_path / "template.vasp"
    path.write_text("template content")
    spec = StructureSourceSpec("bulk", {"base_poscar": str(path)},
                               asset_hashes={"adapter_owned": "a" * 64})
    frozen, references = resolve_source_assets(spec)
    assert frozen.physical_protocol["asset_sha256"]["adapter_owned"] == "a" * 64
    assert frozen.physical_protocol["asset_sha256"]["base_poscar[0]"] == references[0]["sha256"]
