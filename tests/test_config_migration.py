"""Authored migrations preserve scientific intent or require an explicit choice."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_config.py"
_SPEC = importlib.util.spec_from_file_location("migrate_config", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
migrate_config = _MODULE.migrate_config


def structure_config(**changes):
    result = {
        "builder": "substitute", "base_poscar": [{"path": "first.vasp", "n_configs": 3},
                                                   {"path": "second.vasp", "n_configs": 2}],
        "n_random_configs": 7, "site_symbol": "Ni", "geo_opt": {"enabled": False},
        "properties": [{"name": "energy", "predictor": "dp_energy", "models": ["one.pt", "two.pt"],
                        "direction": "min", "objective": "mean_minus_kstd"}],
    }
    result.update(changes)
    return result


def test_shared_sources_preserve_templates_counts_and_do_not_mutate_input():
    raw = structure_config(dp_seed=11, share_structure=True)
    before = deepcopy(raw)
    result = migrate_config(raw, source_name="bulk")
    assert result.ready
    assert raw == before
    assert "dp_seed" not in result.config and result.config["eval_seed"] == 11
    source = result.config["structure_sources"]["bulk"]
    assert source["base_poscar"] == before["base_poscar"]
    assert source["n_random_configs"] == 7
    assert source["geo_opt"] == {"enabled": False}
    assert result.config["properties"][0]["source"] == "bulk"
    assert result.config["properties"][0]["uncertainty"] == "model_spread"
    assert all(key not in result.config for key in ("builder", "base_poscar", "n_random_configs", "share_structure"))
    assert migrate_config(result.config).config == result.config


def test_independent_sources_require_explicit_choice_even_when_equal():
    raw = structure_config(share_structure=False)
    raw["properties"].append({**raw["properties"][0], "name": "other"})
    ambiguous = migrate_config(raw)
    assert not ambiguous.ready
    assert ambiguous.config["share_structure"] is False
    independent = migrate_config(raw, source_policy="independent")
    assert independent.ready
    assert set(independent.config["structure_sources"]) == {"property:energy", "property:other"}
    shared = migrate_config(raw, source_policy="shared", source_name="bulk")
    assert shared.ready
    assert set(shared.config["structure_sources"]) == {"bulk"}
    assert all(prop["source"] == "bulk" for prop in shared.config["properties"])
    assert any("changes property sampling correlation" in item for item in shared.changes)


def test_merging_different_independent_sources_does_not_discard_overrides():
    raw = structure_config(share_structure=False)
    raw["properties"].append({**raw["properties"][0], "name": "other", "n_random_configs": 2})
    result = migrate_config(raw, source_policy="shared")
    assert not result.ready
    assert "structure_sources" not in result.config
    independent = migrate_config(raw, source_policy="independent")
    assert independent.config["structure_sources"]["property:other"]["n_random_configs"] == 2


def test_shared_ignored_property_override_is_reported_and_not_activated():
    raw = structure_config(share_structure=True)
    raw["properties"][0]["n_random_configs"] = 99
    result = migrate_config(raw)
    assert result.ready
    assert result.config["structure_sources"]["default"]["n_random_configs"] == 7
    assert "n_random_configs" not in result.config["properties"][0]
    assert any("ignored property" in item["message"] for item in result.reviews)


def test_unknown_external_structure_capability_requires_declaration_without_import():
    raw = structure_config()
    raw["properties"][0] = {"name": "custom", "predictor": "uninstalled.package:Adapter", "objective": "mean"}
    result = migrate_config(raw)
    assert not result.ready
    assert "uninstalled.package" not in sys.modules
    explicit = migrate_config(raw, structure_properties=["custom"])
    assert explicit.ready
    assert explicit.config["properties"][0]["source"] == "default"


def test_external_builder_fields_are_moved_without_guessing():
    raw = structure_config(builder="uninstalled.builder:Adapter", md={"temperature": 500},
                           special_protocol={"value": "custom"})
    result = migrate_config(raw)
    assert result.ready
    assert "special_protocol" not in result.config
    assert result.config["structure_sources"]["default"]["special_protocol"] == {"value": "custom"}
    assert result.config["structure_sources"]["default"]["md"] == {"temperature": 500}
    assert "uninstalled.builder" not in sys.modules


def test_implicit_old_objective_is_preserved_and_single_model_blocked():
    raw = structure_config()
    raw["properties"][0] = {"name": "formation", "predictor": "mgtransformer", "model": "one.pt", "direction": "min"}
    result = migrate_config(raw)
    assert not result.ready
    assert result.config["properties"][0]["objective"] == "mean_minus_kstd"
    assert any("One declared model" in item["message"] for item in result.reviews)
    explicit = migrate_config(raw, objective_overrides={"formation": "mean"})
    assert explicit.ready
    assert explicit.config["properties"][0]["objective"] == "mean"
    assert "uncertainty" not in explicit.config["properties"][0]


def test_zero_coefficient_proves_old_implicit_objective_equivalent_to_mean():
    raw = {"k": 0, "properties": [{"name": "temperature", "predictor": "rf_magpie", "model": "rf.pkl", "direction": "min"}]}
    result = migrate_config(raw)
    assert result.ready
    assert result.config["properties"][0]["objective"] == "mean"
    assert "unit" not in result.config["properties"][0]
    assert any(item["field"].endswith(".unit") and not item["blocking"] for item in result.reviews)


def test_native_uncertainty_is_not_inferred_from_arbitrary_external_class_name():
    raw = {"properties": [{"name": "pIC50", "predictor": "somewhere:GP", "objective": "mean_minus_kstd"}]}
    assert not migrate_config(raw).ready
    raw["properties"][0]["uncertainty"] = "native_uncertainty"
    assert migrate_config(raw).ready


@pytest.mark.parametrize("shared", [True, False])
def test_composition_only_models_need_no_structure_sharing_switch(shared):
    raw = {"share_structure": shared, "properties": [{"name": "q", "predictor": "lookup", "objective": "mean"}]}
    result = migrate_config(raw)
    assert result.ready
    assert "share_structure" not in result.config
    assert "structure_sources" not in result.config


@pytest.mark.parametrize("transform,representation", [("exp", "ln"), ("pow10", "log10")])
def test_transform_input_representation_is_explicit_without_inventing_units(transform, representation):
    raw = structure_config()
    raw["properties"][0]["transform"] = transform
    result = migrate_config(raw)
    assert result.config["properties"][0]["representation"] == representation
    assert "unit" not in result.config["properties"][0]


def test_fixed_full_list_and_bound_guaranteed_filter_become_free_assignments():
    raw = {"predictor": "dummy", "groups": [{"name": "alloy", "kind": "composition",
        "species_set": ["Ni", "Fe"], "n_components": 1, "episode_style": "fixed_order_amount",
        "fraction_set": [0, 0.5, 1], "element_bounds": {"Ni": [0.5, 1]},
        "filters": [{"type": "last_step_element", "required_elements": ["Ni"], "reserve_for_last": True},
                    {"type": "other:Filter", "option": 4}]}]}
    result = migrate_config(raw)
    group = result.config["groups"][0]
    assert group["n_components"] == 2
    assert "episode_style" not in group
    assert group["filters"] == [{"type": "other:Filter", "option": 4}]
    assert group["element_bounds"] == raw["groups"][0]["element_bounds"]


def test_integer_oxygen_requirement_preserves_other_chemical_filters():
    raw = {"predictor": "dummy", "groups": [{"name": "oxide", "kind": "integer",
        "species_set": ["Ni", "O"], "ratio_set": [0, 1, 2], "n_components": 3,
        "filters": [{"type": "last_step_element", "required_elements": ["O"],
                     "nonzero_ratio_at_last": True, "reserve_for_last": True}, {"type": "smact_charge"}]}]}
    result = migrate_config(raw)
    assert result.config["groups"][0]["token_requirements"] == {"O": {"count": 1, "positive": True}}
    assert result.config["groups"][0]["filters"] == [{"type": "smact_charge"}]


def test_hpo_reference_only_changes_when_explicitly_supplied():
    raw = {"base_config": "../../unknown.yaml", "search_space": {"dqn_lr": {"low": 0.001, "high": 0.1}}}
    assert migrate_config(raw).config == raw
    assert migrate_config(raw, base_config="../unknown.yaml").config["base_config"] == "../unknown.yaml"


@pytest.mark.parametrize("kwargs", [{"source_policy": "auto"}, {"objective_overrides": {"missing": "mean"}},
                                    {"objective_overrides": {"energy": "bad"}}, {"structure_properties": []},
                                    {"base_config": "unrelated.yaml"}])
def test_invalid_explicit_decisions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        migrate_config(structure_config(), **kwargs)


def test_cli_defaults_to_diff_and_never_overwrites_input_or_existing_output(tmp_path):
    path = tmp_path / "config.yaml"
    text = "# Retain the historical file\npredictor: dummy\ndp_seed: 11\n"
    path.write_text(text)
    command = [sys.executable, str(_SCRIPT), str(path)]
    dry = subprocess.run(command, capture_output=True, text=True)
    assert dry.returncode == 0 and "+eval_seed: 11" in dry.stdout
    assert path.read_text() == text
    same = subprocess.run(command + ["--output", str(path)], capture_output=True, text=True)
    assert same.returncode == 2 and "never overwritten" in same.stderr
    output = tmp_path / "new.yaml"
    written = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    assert written.returncode == 0
    assert yaml.safe_load(output.read_text()) == {"predictor": "dummy", "eval_seed": 11}
    repeated = subprocess.run(command + ["--output", str(output)], capture_output=True, text=True)
    assert repeated.returncode == 2 and "File exists" in repeated.stderr
    assert path.read_text() == text


def test_cli_blocked_science_produces_report_but_no_output(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(structure_config(share_structure=False)))
    output, report = tmp_path / "new.yaml", tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(_SCRIPT), str(config), "--output", str(output), "--report", str(report)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert report.exists() and not output.exists()
    assert "Blocking migration reviews remain" in result.stderr
