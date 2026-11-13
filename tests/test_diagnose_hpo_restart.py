"""Campaign-gate diagnosis must expose mismatches without mutating evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from rl_matdesign.hpo import artifacts, domain, spec
from rl_matdesign.utils import hpo_resume_compatibility as compatibility


_script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_hpo_restart.py"
_module = importlib.util.spec_from_file_location("diagnose_hpo_restart", _script)
diagnose = importlib.util.module_from_spec(_module)
_module.loader.exec_module(diagnose)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    hpo, base = tmp_path / "hpo.yaml", tmp_path / "base.yaml"
    hpo.write_text("training: {}\n")
    base.write_text("properties: []\n")
    plan = {"source_path": str(hpo), "base_config": str(base),
            "method": "a2c", "parameter_profile": "builtin:a2c", "parallel_jobs": 3}
    current = {"plan": plan, "domain": {"size": 100}, "assets": [{"sha256": "old-model"}]}
    monkeypatch.setattr(spec, "load_plan", lambda *a, **kw: copy.deepcopy(current["plan"]))
    monkeypatch.setattr(domain, "design_space_info", lambda p: copy.deepcopy(current["domain"]))
    monkeypatch.setattr(artifacts, "asset_fingerprints", lambda p: copy.deepcopy(current["assets"]))
    contract = {**copy.deepcopy(current), "software_digest": artifacts.package_digest(),
                "configuration_sources": {
                    name: {"path": str(path), "sha256": artifacts.file_digest(path)}
                    for name, path in (("hpo_yaml", hpo), ("base_yaml", base))}}
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "manifest.json"
    artifacts.atomic_json(manifest, {**contract, "contract_digest": artifacts.digest(contract)})
    return out, current, manifest, hpo


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_same_contract_is_read_only_and_not_a_checkpoint_readiness_claim(campaign, monkeypatch):
    out, _, _, _ = campaign
    from rl_matdesign.hpo import controller, job_runner
    def forbidden(*a, **kw):
        raise AssertionError("Diagnosis must not construct a campaign or runner")
    monkeypatch.setattr(controller, "Campaign", forbidden)
    monkeypatch.setattr(job_runner, "JobRunner", forbidden)
    before = snapshot(out.parent)
    report = diagnose.inspect_campaign(out)
    assert report["campaign_contract_accepted"]
    assert report["read_only"] and not report["checkpoint_state_checked"]
    assert report["differences"] == []
    assert snapshot(out.parent) == before


@pytest.mark.parametrize("approved", [False, True])
def test_reports_unsupported_vs_registered_old_software(campaign, monkeypatch, approved):
    out, _, manifest, _ = campaign
    saved = artifacts.read_json(manifest)
    saved["software_digest"] = "a" * 64
    saved["contract_digest"] = artifacts.digest({k: saved[k] for k in diagnose.CONTRACT_KEYS})
    artifacts.atomic_json(manifest, saved)
    monkeypatch.setattr(compatibility, "approved_upgrade",
                        lambda *a: {"id": "audited-test"} if approved else None)
    before = snapshot(out.parent)
    report = diagnose.inspect_campaign(out)
    assert report["software_upgrade_registered"] is approved
    assert report["campaign_contract_accepted"] is approved
    assert report["changed_contract_sections"] == ["software_digest"]
    assert snapshot(out.parent) == before


@pytest.mark.parametrize("section,key,value", [
    ("plan", "parallel_jobs", 10), ("domain", "size", 101),
    ("assets", None, [{"sha256": "changed-model"}]),
])
def test_pinpoints_changed_plan_domain_or_model(campaign, section, key, value):
    out, current, _, _ = campaign
    if key is None:
        current[section] = value
    else:
        current[section][key] = value
    report = diagnose.inspect_campaign(out)
    assert not report["campaign_contract_accepted"]
    assert report["changed_contract_sections"] == [section]
    assert any(row["path"].startswith(section + ("." if key else "["))
               for row in report["differences"])


def test_comment_only_yaml_change_is_visible(campaign):
    out, _, _, hpo = campaign
    hpo.write_text("# only a comment changed\ntraining: {}\n")
    report = diagnose.inspect_campaign(out)
    assert not report["campaign_contract_accepted"]
    assert report["changed_contract_sections"] == ["configuration_sources"]
    assert report["configuration_sources"]["hpo_yaml"]["unchanged"] is False


def test_invalid_yaml_keeps_software_and_source_evidence(campaign, monkeypatch):
    out, _, _, hpo = campaign
    hpo.write_text("invalid: [\n")
    def fail(*a, **kw):
        raise ValueError("invalid YAML")
    monkeypatch.setattr(spec, "load_plan", fail)
    report = diagnose.inspect_campaign(out)
    assert not report["campaign_contract_accepted"]
    assert report["package"]["digest"]
    assert report["configuration_sources"]["hpo_yaml"]["unchanged"] is False
    assert report["errors"] == [{"type": "ValueError", "message": "invalid YAML"}]


def test_damaged_manifest_never_accepted(campaign):
    out, _, manifest, _ = campaign
    saved = artifacts.read_json(manifest)
    saved["contract_digest"] = "corrupted"
    artifacts.atomic_json(manifest, saved)
    report = diagnose.inspect_campaign(out)
    assert not report["manifest_integrity"]
    assert not report["campaign_contract_accepted"]


def test_missing_package_has_actionable_json_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(artifacts, "package_digest", lambda: (_ for _ in ()).throw(
        ImportError("Wrong Python environment")))
    assert diagnose.main(["--out", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["errors"][0]["type"] == "ImportError"
    assert report["python"]


def test_difference_distinguishes_missing_from_null():
    diff = list(diagnose.differences({"items": [None]}, {"items": [], "new": None}))
    assert diff == [{"path": "items[0]", "saved": None, "current": {"missing": True}},
                    {"path": "new", "saved": {"missing": True}, "current": None}]


@pytest.fixture
def source_repo(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
    git("init")
    root = repo / "src" / "rl_matdesign"
    root.mkdir(parents=True)
    (root / "model.py").write_text("VERSION = 1\n")
    (root / "remove.py").write_text("OLD = True\n")
    (repo / ".gitignore").write_text(".ipynb_checkpoints/\n")
    # Git archives can omit source via export-ignore; diagnosis must not.
    (repo / ".gitattributes").write_text("src/rl_matdesign/remove.py export-ignore\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "old")
    old = git("rev-parse", "HEAD")
    old_tree = diagnose._tree_inventory(repo, old, "src/rl_matdesign")
    extra = root / ".ipynb_checkpoints" / "model-checkpoint.py"
    extra.parent.mkdir()
    extra.write_text("BACKUP = 1\n")
    return repo, root, extra, old, old_tree, git


def test_source_details_find_ignored_backup_and_old_tree_hypothesis(source_repo, tmp_path):
    import hashlib
    repo, root, extra, old, old_tree, git = source_repo
    saved = diagnose._inventory_digest({**old_tree,
        str(extra.relative_to(root)): hashlib.sha256(extra.read_bytes()).hexdigest()})
    (root / "model.py").write_text("VERSION = 2\n")
    git("add", "src")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "new")
    out = tmp_path / "campaign"
    provenance = out / "jobs" / "first" / "provenance" / "first"
    provenance.mkdir(parents=True)
    (provenance / "manifest.json").write_text(json.dumps({"git": {
        "commit": old, "dirty_tracked": False, "patch_sha256": "empty", "untracked_file_count": 1}}))
    before = snapshot(out)
    report = diagnose.source_evidence(root, out, saved)
    assert report["available"]
    assert report["modified_python_files"] == []
    assert report["missing_python_files"] == []
    assert report["extra_python_files"][0]["path"] == ".ipynb_checkpoints/model-checkpoint.py"
    historical = report["historical_sample"][0]
    assert not historical["committed_tree_matches_saved_digest"]
    assert historical["committed_tree_plus_current_extra_files_matches_saved_digest"]
    assert snapshot(out) == before


def test_source_details_distinguish_edits_and_deletions_from_untracked_files(source_repo, tmp_path):
    repo, root, extra, old, old_tree, git = source_repo
    (root / "model.py").write_text("LOCAL = 3\n")
    (root / "remove.py").unlink()
    report = diagnose.source_evidence(root, tmp_path / "absent", "unknown")
    assert report["available"]
    assert [row["path"] for row in report["modified_python_files"]] == ["model.py"]
    assert report["missing_python_files"] == ["remove.py"]
    assert report["extra_python_files"][0]["path"] == ".ipynb_checkpoints/model-checkpoint.py"


def test_source_details_support_installed_package_without_git(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "model.py").write_text("VERSION = 1\n")
    report = diagnose.source_evidence(root, tmp_path, "unknown")
    assert not report["available"]
    assert report["python_file_count"] == 1
    assert report["actual_package_digest"]
    assert report["inspection_error"]


@pytest.mark.parametrize("document", [[], {"git": None}])
def test_malformed_provenance_stays_a_source_diagnostic_error(source_repo, tmp_path, document):
    _, root, _, _, _, _ = source_repo
    out = tmp_path / "campaign"
    provenance = out / "jobs" / "first" / "provenance" / "first"
    provenance.mkdir(parents=True)
    (provenance / "manifest.json").write_text(json.dumps(document))
    report = diagnose.source_evidence(root, out, "unknown")
    assert report["available"]
    assert "Invalid provenance" in report["inspection_error"]
