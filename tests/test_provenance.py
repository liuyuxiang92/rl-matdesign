"""Evidence must survive reuse of run directories and expose incomplete archives."""
import json
from pathlib import Path
import random
import subprocess

import pytest

from rl_matdesign.utils.provenance import (
    create_bundle, referenced_files, sha256_file, verify_bundle, write_run_manifest,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "test@example.invalid"],
                 ["config", "user.name", "Test"]):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    (root / "code.py").write_text("answer = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "code.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
    return root


def test_manifest_append_only_exact_command_and_seed_state(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    config = tmp_path / "input.yaml"
    config.write_text("models: [model.pt]\n")
    (repo / "code.py").write_text("answer = 2\n")
    state = random.getstate()
    first = write_run_manifest(tmp_path / "run", {"models": ["model.pt"]},
                               argv=["rl-matdesign", "--config", "input.yaml"],
                               config_path=config, seeds={"train": 8}, phase="train",
                               config_sources={"models": "yaml"}, runtime_args={"method": "a2c"},
                               repo_dir=repo)
    initial = first.read_bytes()
    second = write_run_manifest(tmp_path / "run", {}, phase="generate", repo_dir=repo)
    assert random.getstate() == state
    assert first != second
    assert first.read_bytes() == initial
    data = json.loads(initial)
    assert data["git"]["dirty_tracked"] is True
    assert "answer = 2" in (first.parent / "git.patch").read_text()
    assert data["argv"][-1] == "input.yaml"
    assert data["input_files"][0]["sha256"] == sha256_file(model)
    assert data["runtime_args"]["method"] == "a2c"
    assert not data["missing_inputs"]
    assert (first.parent / "input_config.yaml").read_text() == config.read_text()


def test_reference_paths_use_runtime_cwd_and_do_not_guess(tmp_path):
    refs = referenced_files({"properties": [{"models": ["a.pt", "b.pt"]}],
                             "out": "/some/output", "predictor": "module:Predictor"}, base_dir=tmp_path)
    assert [r["path"] for r in refs] == [str(tmp_path / "a.pt"), str(tmp_path / "b.pt")]


def test_bundle_preserves_outputs_and_detects_tamper(tmp_path, repo):
    run = tmp_path / "run"
    run.mkdir()
    (run / "generated.csv").write_text("formula,reward\nNi,1\n")
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    manifest = write_run_manifest(run, {"models": [str(model)]}, repo_dir=repo)
    (run / "pip-freeze.txt").write_text("example==1.0\n")
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    result = verify_bundle(bundle)
    assert result["integrity_ok"]
    assert result["complete"], result
    assert (bundle / "source/head.tar").is_file()
    assert (bundle / "source/working_tree/code.py").read_text() == "answer = 1\n"
    copy = bundle / "runs/000-run" / manifest.relative_to(run)
    assert copy.read_bytes() == manifest.read_bytes()
    (bundle / "runs/000-run/generated.csv").write_text("tampered")
    assert not verify_bundle(bundle)["integrity_ok"]


def test_legacy_run_does_not_inherit_packaging_commit(tmp_path, repo):
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"models": [str(tmp_path / "missing.pt")]}))
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    metadata = json.loads((bundle / "bundle.json").read_text())
    assert metadata["runs"][0]["historical_provenance"] == "unknown"
    assert metadata["packaging_source"]["commit"]
    assert not metadata["complete"]
    assert any("Missing" in p for p in metadata["problems"])
    assert any("NOT proof" in p for p in metadata["problems"])
    assert verify_bundle(bundle)["integrity_ok"]


def test_changed_asset_not_silently_presented_as_original(tmp_path, repo):
    model = tmp_path / "model.pt"
    model.write_bytes(b"original")
    run = tmp_path / "run"
    write_run_manifest(run, {"model": str(model)}, repo_dir=repo)
    model.write_bytes(b"changed")
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    result = verify_bundle(bundle)
    assert result["integrity_ok"]
    assert not result["complete"]
    assert any("changed since invocation" in p for p in result["problems"])


def test_no_overwrite_recursive_copy_or_symlink_escape(tmp_path, repo):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="inside"):
        create_bundle(run / "archive", [run], repo_dir=repo)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        create_bundle(existing, [run], repo_dir=repo)
    (run / "escape").symlink_to(repo, target_is_directory=True)
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    assert not (bundle / "runs/000-run/escape").exists()
    assert any("symlink" in p for p in verify_bundle(bundle)["problems"])


def test_checksum_path_traversal_is_reported(tmp_path, repo):
    run = tmp_path / "run"
    run.mkdir()
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    checksums = json.loads((bundle / "checksums.json").read_text())
    checksums["../outside"] = "invalid"
    (bundle / "checksums.json").write_text(json.dumps(checksums))
    result = verify_bundle(bundle)
    assert not result["integrity_ok"]
    assert any("Unsafe" in error for error in result["errors"])


def test_recorded_historical_revision_and_patch_are_archived(tmp_path, repo):
    run = tmp_path / "run"
    (repo / "code.py").write_text("answer = 5\n")
    invocation = write_run_manifest(run, {}, repo_dir=repo)
    original_commit = json.loads(invocation.read_text())["git"]["commit"]
    (run / "pip-freeze.txt").write_text("example==1.0\n")
    subprocess.run(["git", "-C", str(repo), "commit", "-am", "newer source"], check=True,
                   stdout=subprocess.PIPE)
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    metadata = json.loads((bundle / "bundle.json").read_text())
    assert original_commit != metadata["packaging_source"]["commit"]
    assert (bundle / metadata["historical_sources"][original_commit]).is_file()
    assert "answer = 5" in (bundle / "runs/000-run" / invocation.parent.relative_to(run) / "git.patch").read_text()
    assert verify_bundle(bundle)["complete"]


def test_invalid_bundle_metadata_is_a_verification_failure(tmp_path, repo):
    run = tmp_path / "run"
    run.mkdir()
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    (bundle / "bundle.json").write_text("broken JSON")
    report = verify_bundle(bundle)
    assert not report["integrity_ok"]
    assert any("Unreadable bundle metadata" in error for error in report["errors"])


def test_manifest_environment_whitelist_and_hash_dedup(tmp_path, repo, monkeypatch):
    import rl_matdesign.utils.provenance as provenance
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-record")
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    calls = []
    original_hash = provenance.sha256_file
    def count_hash(path):
        calls.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(provenance, "sha256_file", count_hash)
    manifest = write_run_manifest(tmp_path / "run", {"models": ["model.pt", "model.pt"]},
                                  input_files=[model], repo_dir=repo)
    data = json.loads(manifest.read_text())
    assert calls.count(model) == 1
    assert data["runtime"]["determinism_environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "do-not-record" not in manifest.read_text()


def test_missing_environment_export_is_not_complete(tmp_path, repo):
    run = tmp_path / "run"
    write_run_manifest(run, {}, repo_dir=repo)
    bundle = create_bundle(tmp_path / "archive", [run], repo_dir=repo)
    report = verify_bundle(bundle)
    assert report["integrity_ok"]
    assert not report["complete"]
    assert any("environment export/lock" in problem for problem in report["problems"])
    assert not json.loads((bundle / "bundle.json").read_text())["reproduction_validated"]


def test_mgtransformer_assets_resolve_under_repo_not_cwd(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mgt = tmp_path / "external_mgt"
    (mgt / "ckpt").mkdir(parents=True)
    actual_model = mgt / "ckpt/model.pt"
    actual_model.write_bytes(b"actual model used by subprocess")
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt/model.pt").write_bytes(b"wrong cwd model")
    (mgt / "serve-config.json").write_text("{}")
    absolute = tmp_path / "absolute.pt"
    absolute.write_bytes(b"absolute model")
    config = {"properties": [{"predictor": "mgtransformer", "mgt_repo": "external_mgt",
                              "models": ["ckpt/model.pt", str(absolute)],
                              "config": "serve-config.json"}]}
    refs = referenced_files(config)
    assert refs[0]["path"] == str(actual_model)
    assert refs[0]["path_resolution"] == "mgtransformer_repo"
    assert refs[1]["path"] == str(absolute)
    assert refs[2]["path"] == str(mgt / "serve-config.json")
    path = write_run_manifest(tmp_path / "run", config, repo_dir=repo)
    manifest = json.loads(path.read_text())
    assert manifest["input_files"][0]["sha256"] == sha256_file(actual_model)
    assert not manifest["missing_inputs"]
    bundle = create_bundle(tmp_path / "archive", [tmp_path / "run"], repo_dir=repo)
    archived = json.loads((bundle / "bundle.json").read_text())["assets"][0]
    assert (bundle / archived["bundle_path"]).read_bytes() == actual_model.read_bytes()


def test_rf_path_stays_cwd_and_unknown_plugin_is_not_guessed(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"cwd decoy")
    config = {"properties": [{"predictor": "rf_magpie", "model": "model.pt"},
                             {"predictor": "external.plugin:Predictor", "model": "model.pt"}]}
    refs = referenced_files(config)
    assert refs[0]["path"] == str(tmp_path / "model.pt")
    assert refs[1]["path"] is None
    assert refs[1]["status"] == "unresolved_path"
    path = write_run_manifest(tmp_path / "run", config, repo_dir=repo)
    manifest = json.loads(path.read_text())
    assert len(manifest["missing_inputs"]) == 1
    assert "sha256" not in manifest["missing_inputs"][0]
    bundle = create_bundle(tmp_path / "archive", [tmp_path / "run"], repo_dir=repo)
    assert any("Unresolved asset path" in p for p in verify_bundle(bundle)["problems"])


def test_named_source_templates_relaxation_and_property_models_are_archived(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = [tmp_path / name for name in ("phase_a.POSCAR", "phase_b.POSCAR", "relax.pt", "energy.pt")]
    for index, path in enumerate(paths):
        path.write_bytes(f"asset-{index}".encode())
    config = {
        "structure_sources": {"bulk": {
            "builder": "substitute",
            "base_poscar": [{"path": "phase_a.POSCAR", "n_configs": 3},
                            {"poscar": "phase_b.POSCAR", "n_configs": 2}],
            "geo_opt": {"model": "relax.pt"},
        }},
        "properties": [{"name": "energy", "predictor": "dp_energy", "source": "bulk", "model": "energy.pt"}],
    }
    refs = referenced_files(config)
    assert [row["path"] for row in refs] == [str(path) for path in paths]
    assert [row["field"] for row in refs[:3]] == [
        "structure_sources.bulk.base_poscar[0].path", "structure_sources.bulk.base_poscar[1].poscar",
        "structure_sources.bulk.geo_opt.model[0]",
    ]
    path = write_run_manifest(tmp_path / "run", config, repo_dir=repo)
    manifest = json.loads(path.read_text())
    assert not manifest["missing_inputs"]
    assert all(row["status"] == "present" for row in manifest["input_files"])
    assert [row["sha256"] for row in manifest["input_files"]] == [sha256_file(path) for path in paths]
    (tmp_path / "run" / "pip-freeze.txt").write_text("example==1.0\n")
    bundle = create_bundle(tmp_path / "archive", [tmp_path / "run"], repo_dir=repo)
    assert verify_bundle(bundle)["complete"]
    records = json.loads((bundle / "bundle.json").read_text())["assets"]
    assert {record["sha256"] for record in records} >= {sha256_file(path) for path in paths}


def test_external_source_builder_relative_paths_remain_unresolved_but_geo_opt_is_owned(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("template", "builder.pt", "relax.pt"):
        (tmp_path / name).write_bytes(name.encode())
    config = {"structure_sources": {"bulk": {"builder": "external:Builder", "base_poscar": "template",
                                              "md": {"model": "builder.pt"}, "geo_opt": {"model": "relax.pt"}}}}
    refs = {row["field"]: row for row in referenced_files(config)}
    for field in ["structure_sources.bulk.base_poscar[0]", "structure_sources.bulk.md.model[0]"]:
        assert refs[field]["path"] is None and refs[field]["status"] == "unresolved_path"
    assert refs["structure_sources.bulk.geo_opt.model[0]"]["path"] == str(tmp_path / "relax.pt")
    path = write_run_manifest(tmp_path / "run", config, repo_dir=repo)
    manifest = json.loads(path.read_text())
    assert len(manifest["missing_inputs"]) == 2
    assert all(row["path"] is None and "sha256" not in row for row in manifest["missing_inputs"])
    assert next(row for row in manifest["input_files"] if row["path"] == str(tmp_path / "relax.pt"))["status"] == "present"


def test_named_source_default_relaxation_model_and_missing_template_are_explicit(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    default_model = tmp_path / "models" / "DPA-3.1-3M.pt"
    default_model.write_bytes(b"default relaxation model")
    config = {"structure_sources": {"bulk": {"base_poscar": "missing.POSCAR", "geo_opt": {"enabled": True}}}}
    refs = referenced_files(config)
    assert refs[0]["status"] == "missing_or_not_file"
    assert refs[1]["field"] == "structure_sources.bulk.geo_opt.model[default]"
    assert refs[1]["path"] == str(default_model)
    path = write_run_manifest(tmp_path / "run", config, repo_dir=repo)
    manifest = json.loads(path.read_text())
    assert [row["field"] for row in manifest["missing_inputs"]] == ["structure_sources.bulk.base_poscar[0]"]


def test_named_sources_preserve_distinct_identity_for_shared_file_and_do_not_rehash_in_manifest(tmp_path, repo, monkeypatch):
    import rl_matdesign.utils.provenance as provenance
    monkeypatch.chdir(tmp_path)
    template = tmp_path / "POSCAR"
    template.write_bytes(b"template")
    calls = []
    original_hash = provenance.sha256_file
    def counted_hash(path):
        calls.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(provenance, "sha256_file", counted_hash)
    cfg = {"structure_sources": {name: {"base_poscar": "POSCAR"} for name in ["first", "second"]}}
    path = write_run_manifest(tmp_path / "run", cfg, repo_dir=repo)
    rows = json.loads(path.read_text())["input_files"]
    assert [row["source"] for row in rows] == ["first", "second"]
    assert calls.count(template) == 2  # Each source resolver once; no second manifest pass.
    assert rows[0]["sha256"] == rows[1]["sha256"]
