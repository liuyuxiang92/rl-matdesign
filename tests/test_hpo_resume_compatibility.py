"""Only reviewed HPO source changes may cross otherwise exact contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_matdesign.utils import hpo_resume_compatibility as compatibility
from rl_matdesign.utils.execution_contract import require_execution_contract


OLD_PACKAGE, NEW_PACKAGE = "a" * 64, "b" * 64
OLD_SOURCE, NEW_SOURCE = "c" * 64, "d" * 64


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "compatibility.json"
    document = {"schema_version": 1, "migrations": [{
        "id": "reviewed-storage-fix", "reason": "Storage only",
        "from": {"package": OLD_PACKAGE, "source": OLD_SOURCE},
        "to": {"package": NEW_PACKAGE, "source": NEW_SOURCE},
    }]}
    path.write_text(json.dumps(document))
    monkeypatch.setattr(compatibility, "_REGISTRY_PATH", path)
    return path, document


def evaluation(source):
    return {"schema_version": 1, "source_sha256": source,
            "configuration": {"property": "target", "threshold": 0},
            "assets": [{"sha256": "frozen-oracle"}],
            "runtime": {"device": "cpu"},
            "numerical_runtime": {"packages": {"torch": "fixed"}}}


def execution(source):
    return {"evaluation": evaluation(source),
            "learning": {"train_seed": 42, "checkpoint_frequency": 1}}


def generation(source):
    return {"execution": execution(source),
            "sampling": {"gen_seed": 101, "gen_temperature": .5},
            "frozen_artifacts": {"policy.pt": "immutable-model", "std_scaler.bin": "immutable-scaler"}}


def set_nested(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


def test_registry_matches_exact_forward_pair_and_returns_independent_evidence(registry):
    assert compatibility.approved_upgrade(OLD_PACKAGE, NEW_PACKAGE)["id"] == "reviewed-storage-fix"
    entry = compatibility.approved_upgrade(OLD_SOURCE, NEW_SOURCE, kind="source")
    entry["id"] = "modified"
    assert compatibility.approved_upgrade(OLD_SOURCE, NEW_SOURCE, kind="source")["id"] == "reviewed-storage-fix"
    assert compatibility.approved_upgrade(NEW_SOURCE, OLD_SOURCE, kind="source") is None
    assert compatibility.approved_upgrade(OLD_SOURCE, "e" * 64, kind="source") is None
    assert compatibility.approved_upgrade("e" * 64, NEW_SOURCE, kind="source") is None
    assert compatibility.approved_upgrade(OLD_SOURCE, OLD_SOURCE, kind="source") is None
    assert compatibility.approved_upgrade(None, NEW_SOURCE, kind="source") is None
    assert compatibility.approved_upgrade(OLD_PACKAGE, NEW_PACKAGE, kind="source") is None
    with pytest.raises(ValueError, match="fingerprint kind"):
        compatibility.approved_upgrade(OLD_PACKAGE, NEW_PACKAGE, kind="anything")


def test_unfinished_registry_does_not_partially_enable_a_migration(registry):
    path, document = registry
    document["migrations"][0]["to"]["package"] = None
    path.write_text(json.dumps(document))
    assert compatibility.approved_upgrade(OLD_SOURCE, NEW_SOURCE, kind="source") is None


def test_invalid_registry_fails_closed(registry):
    path, _ = registry
    path.write_text('{"schema_version": 100, "migrations": []}')
    with pytest.raises(ValueError, match="Invalid HPO"):
        compatibility.approved_upgrade(OLD_SOURCE, NEW_SOURCE, kind="source")


@pytest.mark.parametrize("factory,compare", [
    (evaluation, compatibility.evaluation_contracts_equal),
    (execution, compatibility.execution_contracts_equal),
    (generation, compatibility.generation_contracts_equal),
])
def test_upgrade_requires_explicit_opt_in_and_does_not_modify_saved_evidence(registry, factory, compare):
    actual, expected = factory(OLD_SOURCE), factory(NEW_SOURCE)
    before = copy.deepcopy((actual, expected))
    assert compare(actual, actual)
    assert not compare(actual, expected)
    assert compare(actual, expected, allow_source_upgrade=True)
    assert not compare(expected, actual, allow_source_upgrade=True)
    assert not compare(factory("e" * 64), expected, allow_source_upgrade=True)
    assert not compare(None, expected, allow_source_upgrade=True)
    assert (actual, expected) == before


@pytest.mark.parametrize("path,value", [
    (("configuration", "threshold"), 1),
    (("configuration", "threshold"), False),
    (("configuration", "property"), "changed-target"),
    (("assets", 0, "sha256"), "changed-oracle"),
    (("runtime", "device"), "cuda"),
    (("numerical_runtime", "packages", "torch"), "changed-version"),
])
def test_evaluation_fields_remain_strict_during_reviewed_upgrade(registry, path, value):
    actual, expected = evaluation(OLD_SOURCE), evaluation(NEW_SOURCE)
    set_nested(expected, path, value)
    assert not compatibility.evaluation_contracts_equal(actual, expected, allow_source_upgrade=True)


def test_source_exception_is_not_recursive_or_missing_field_tolerant(registry):
    actual, expected = evaluation(OLD_SOURCE), evaluation(NEW_SOURCE)
    actual["configuration"]["source_sha256"] = OLD_SOURCE
    expected["configuration"]["source_sha256"] = NEW_SOURCE
    assert not compatibility.evaluation_contracts_equal(actual, expected, allow_source_upgrade=True)
    expected = evaluation(NEW_SOURCE)
    expected.pop("runtime")
    assert not compatibility.evaluation_contracts_equal(evaluation(OLD_SOURCE), expected, allow_source_upgrade=True)


@pytest.mark.parametrize("path,value", [
    (("learning", "train_seed"), 43),
    (("learning", "checkpoint_frequency"), 10),
    (("evaluation", "assets", 0, "sha256"), "new-oracle"),
    (("evaluation", "configuration", "threshold"), 1),
])
def test_execution_protocol_changes_are_rejected(registry, path, value):
    actual, expected = execution(OLD_SOURCE), execution(NEW_SOURCE)
    set_nested(expected, path, value)
    assert not compatibility.execution_contracts_equal(actual, expected, allow_source_upgrade=True)
    with pytest.raises(ValueError, match="differs"):
        require_execution_contract(actual, expected, allow_source_upgrade=True)


def test_training_entry_remains_strict_without_hpo_opt_in(registry):
    actual, expected = execution(OLD_SOURCE), execution(NEW_SOURCE)
    with pytest.raises(ValueError, match="differs"):
        require_execution_contract(actual, expected)
    require_execution_contract(actual, expected, allow_source_upgrade=True)


@pytest.mark.parametrize("path,value", [
    (("sampling", "gen_seed"), 102),
    (("sampling", "gen_temperature"), .9),
    (("frozen_artifacts", "policy.pt"), "new-model"),
    (("frozen_artifacts", "std_scaler.bin"), "new-scaler"),
    (("execution", "learning", "train_seed"), 43),
])
def test_generation_remains_bound_to_weights_sampling_and_learning(registry, path, value):
    actual, expected = generation(OLD_SOURCE), generation(NEW_SOURCE)
    set_nested(expected, path, value)
    assert not compatibility.generation_contracts_equal(actual, expected, allow_source_upgrade=True)


def test_frozen_model_upgrade_is_available_only_for_hpo_env(registry, tmp_path):
    import numpy as np
    import torch
    from rl_matdesign.env import CompositionEnv
    from rl_matdesign.utils.model_contract import load_artifact, save_weights

    env = CompositionEnv(species_set=("Ni", "Fe", "Co"), fraction_set=["0.25", "0.50", "0.75"],
                         total_units=4, n_components=2, state_featurizer=lambda _: np.zeros(2))
    env.evaluation_contract = evaluation(OLD_SOURCE)
    path = tmp_path / "policy.pt"
    save_weights(path, torch.nn.Linear(2, 1), env, "policy")
    before = path.read_bytes()
    env.evaluation_contract = evaluation(NEW_SOURCE)
    with pytest.raises(ValueError, match="evaluation contract differs"):
        load_artifact(path, env)
    env.hpo_recorder = object()
    restored = load_artifact(path, env)
    assert restored["evaluation_contract"]["source_sha256"] == OLD_SOURCE
    assert path.read_bytes() == before
    env.evaluation_contract["configuration"]["threshold"] = 10
    with pytest.raises(ValueError, match="evaluation contract differs"):
        load_artifact(path, env)


def test_generation_restore_checks_upgrade_before_mutating_state(registry, monkeypatch):
    from rl_matdesign.utils import generation_resume

    loaded = []
    restored_rng = []
    predictor = SimpleNamespace(load_state_dict=loaded.append)
    monkeypatch.setattr(generation_resume, "restore_rng_state", restored_rng.append)
    saved = {"generation_contract": generation(OLD_SOURCE),
             "predictor_ledger": {"calls": 7}, "rng_state": {"state": "saved"}}
    changed = generation(NEW_SOURCE)
    changed["sampling"]["gen_seed"] += 1
    with pytest.raises(ValueError, match="Generation resume changed"):
        generation_resume.restore_generation(predictor, saved, changed)
    assert loaded == restored_rng == []
    generation_resume.restore_generation(predictor, saved, generation(NEW_SOURCE))
    assert loaded == [{"calls": 7}]
    assert restored_rng == [{"state": "saved"}]


def test_registry_resource_is_packaged():
    import importlib.resources
    from rl_matdesign import utils

    path = importlib.resources.files(utils).joinpath("hpo_resume_compatibility.json")
    document = json.loads(path.read_text())
    assert document["schema_version"] == 1
    assert document["migrations"][0]["reviewed_from_revision"] == "5d0d0dae"
    assert document["migrations"][0]["from"]["source"] == (
        "d8ce284b84fe7ca2b78d4e3a9f06749a84c437cb611745c22bc5b323592ad646")


def test_release_registry_matches_the_actual_frozen_source_tree():
    """Do not ship a null or stale upgrade target after editing production code."""
    import hashlib
    from rl_matdesign.hpo.artifacts import package_digest
    from rl_matdesign.utils import hpo_resume_compatibility as compatibility
    source_root = Path(compatibility.__file__).resolve().parents[2]
    worker = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        worker.update(str(path.relative_to(source_root)).encode())
        worker.update(hashlib.sha256(path.read_bytes()).digest())
    registry = json.loads(compatibility._REGISTRY_PATH.read_text())
    entry = registry["migrations"][0]
    assert entry["to"] == {"package": package_digest(), "source": worker.hexdigest()}
    assert compatibility.approved_upgrade(entry["from"]["package"], package_digest()) == entry
    assert compatibility.approved_upgrade(entry["from"]["source"], worker.hexdigest(), kind="source") == entry


def test_registered_legacy_predictor_variant_preserves_exact_file_set():
    import hashlib
    from pathlib import PurePosixPath
    from rl_matdesign.utils import hpo_resume_compatibility as compatibility
    source_root = Path(compatibility.__file__).resolve().parents[2]
    document = json.loads(compatibility._REGISTRY_PATH.read_text())
    inventory = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(source_root.rglob("*.py"))}
    extras = document["audited_extra_python_files"]
    assert set(extras) == {"predictors/composite.py", "predictors/dp_property.py",
                          "predictors/dp_structure.py", "predictors/hea.py",
                          "predictors/perovskite.py", "predictors/sinter_calcine.py"}
    combined = {**inventory, **{"rl_matdesign/" + path: sha for path, sha in extras.items()}}
    package = {path.removeprefix("rl_matdesign/"): sha for path, sha in combined.items()
               if path.startswith("rl_matdesign/")}
    package_digest = hashlib.sha256(json.dumps(package, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    worker = hashlib.sha256()
    for path in sorted(combined, key=PurePosixPath):
        worker.update(path.encode())
        worker.update(bytes.fromhex(combined[path]))
    expected = {"package": package_digest, "source": worker.hexdigest()}
    entries = document["migrations"]
    variants = [row for row in entries if row["source_variant"] == "plus_six_legacy_predictors"]
    assert len(variants) == 3
    assert all(row["to"] == expected for row in variants)
    # User's exact original campaign and current installed package are covered.
    assert {row["from"]["package"] for row in variants} == {
        "d495c572bb16672e6cfad8b2a0ba7d3bbee3b61fcd7a548df413f2185f5e27c2",
        "0a6c2c8cc5299bb8fb10e0c58218f98a64b3da57134e6f20041774a39b1cda25",
        "5621cee85788648dc17edcf384980309f5c45b419efebbc6c4fe428bc8b4bbe8"}
    clean = next(row for row in entries if row["source_variant"] == "clean")
    for row in variants:
        assert compatibility.approved_upgrade(row["from"]["package"], package_digest) == row
        assert compatibility.approved_upgrade(row["from"]["source"], worker.hexdigest(), kind="source") == row
        # Do not silently remove modules referenced by a custom predictor config.
        assert compatibility.approved_upgrade(row["from"]["package"], clean["to"]["package"]) is None
        assert compatibility.approved_upgrade(row["from"]["source"], clean["to"]["source"], kind="source") is None
