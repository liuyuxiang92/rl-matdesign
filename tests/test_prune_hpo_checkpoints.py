"""Standalone cleanup leaves model/resume dependencies and evidence intact."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prune_hpo_checkpoints.py"
spec = importlib.util.spec_from_file_location("prune_hpo_checkpoints", SCRIPT)
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


def campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"protocol": "hpo_v2", "contract_digest": "unchanged"}))
    jobs = root / "jobs"
    jobs.mkdir()
    job = jobs / ("a" * 24)
    job.mkdir()
    (job / "job.json").write_text(json.dumps({"phase": "train"}))
    return root, job


def history(job, family="ep", count=4):
    paths = []
    for number in range(1, count + 1):
        path = job / f"checkpoint-{family}{number}.pt"
        path.write_bytes(bytes([number]) * number)
        paths.append(path)
    (job / "checkpoint.pt").symlink_to(paths[-1].name)
    return paths


@pytest.mark.parametrize("family", ["ep", "iter"])
def test_standalone_dry_run_then_apply_preserves_latest_models_and_metadata(tmp_path, family):
    root, job = campaign(tmp_path)
    paths = history(job, family)
    for name in ("qnet.pt", "policy.pt", "value_net.pt", "partial_checkpoint.pt",
                 "generation_checkpoint.pt", "std_scaler.bin", "candidate_records.jsonl"):
        (job / name).write_bytes(b"keep")
    before = {p: p.read_bytes() for p in job.iterdir() if p.is_file()}
    result = subprocess.run([sys.executable, str(SCRIPT), "--out", str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Superseded files: 3; bytes: 6" in result.stdout
    assert all(p.read_bytes() == data for p, data in before.items())
    result = subprocess.run([sys.executable, str(SCRIPT), "--out", str(root), "--apply"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Removed 3 files (6 bytes)" in result.stdout
    assert all(not p.exists() for p in paths[:-1])
    assert (job / "checkpoint.pt").read_bytes() == before[paths[-1]]
    assert all(p.read_bytes() == data for p, data in before.items() if p not in paths[:-1])


def test_metadata_references_and_newer_or_different_family_are_protected(tmp_path):
    root, job = campaign(tmp_path)
    paths = history(job)
    (job / "checkpoint.pt").unlink()
    (job / "checkpoint.pt").symlink_to(paths[2].name)
    (root / "frozen_models.json").write_text(json.dumps({str(paths[0]): "hash"}))
    (job / "config.yaml").write_text("load_checkpoint: checkpoint-ep2.pt\n")
    (job / "checkpoint-iter1.pt").write_bytes(b"different family")
    assert not prune.plan_pruning(root)["candidates"]


def test_regular_latest_allows_history_but_protects_explicit_reference(tmp_path):
    root, job = campaign(tmp_path)
    paths = history(job)
    (job / "checkpoint.pt").unlink()
    (job / "checkpoint.pt").write_bytes(b"new atomic latest")
    (job / "result.json").write_text(json.dumps({"checkpoint_paths": {"old": str(paths[1])}}))
    plan = prune.plan_pruning(root)
    assert {p["path"] for p in plan["candidates"]} == set(paths) - {paths[1]}
    prune.apply_pruning(plan)
    assert paths[1].exists()
    assert (job / "checkpoint.pt").read_bytes() == b"new atomic latest"


@pytest.mark.parametrize("target", [None, "checkpoint-ep99.pt", "../checkpoint-ep1.pt", "/tmp/checkpoint-ep1.pt", "policy.pt"])
def test_missing_dangling_or_unusual_pointer_fails_closed_per_job(tmp_path, target):
    root, job = campaign(tmp_path)
    paths = history(job)
    (job / "checkpoint.pt").unlink()
    if target is not None:
        (job / "checkpoint.pt").symlink_to(target)
    plan = prune.plan_pruning(root)
    assert not plan["candidates"]
    assert plan["skipped"]
    assert all(p.exists() for p in paths)


def test_never_deletes_numbered_symlinks_unrelated_or_generation_files(tmp_path):
    root, job = campaign(tmp_path)
    paths = history(job)
    paths[0].unlink()
    external = tmp_path / "external.pt"
    external.write_bytes(b"external")
    paths[0].symlink_to(external)
    (job / "not-checkpoint-ep1.pt").write_bytes(b"other")
    (job / "checkpoint-ep0.pt.backup").write_bytes(b"backup")
    generation = root / "jobs" / ("b" * 24)
    generation.mkdir()
    (generation / "job.json").write_text('{"phase":"generation"}')
    generation_paths = history(generation)
    prune.apply_pruning(prune.plan_pruning(root))
    assert paths[0].is_symlink() and external.read_bytes() == b"external"
    assert all(p.exists() for p in generation_paths)
    assert (job / "not-checkpoint-ep1.pt").exists()
    assert (job / "checkpoint-ep0.pt.backup").exists()


@pytest.mark.parametrize("change", ["alias", "target", "candidate", "metadata", "new_metadata"])
def test_apply_rechecks_before_deleting_any_file(tmp_path, change):
    root, job = campaign(tmp_path)
    paths = history(job)
    plan = prune.plan_pruning(root)
    if change == "alias":
        (job / "checkpoint.pt").unlink()
        (job / "checkpoint.pt").symlink_to(paths[2].name)
    elif change == "target":
        paths[-1].write_bytes(b"changed latest")
    elif change == "candidate":
        paths[1].write_bytes(b"changed history")
    elif change == "metadata":
        (job / "job.json").write_text('{"phase":"train","changed":true}')
    else:
        (job / "result.json").write_text(json.dumps({"protected": str(paths[0])}))
    with pytest.raises(prune.UnsafeCampaign):
        prune.apply_pruning(plan)
    assert all(p.exists() for p in paths)


def test_empty_campaign_and_invalid_manifest(tmp_path):
    root, job = campaign(tmp_path)
    assert prune.apply_pruning(prune.plan_pruning(root)) == {"removed_files": 0, "removed_bytes": 0}
    (root / "manifest.json").write_text('{"protocol":"ordinary_training"}')
    with pytest.raises(prune.UnsafeCampaign, match="HPO manifest"):
        prune.plan_pruning(root)


@pytest.mark.parametrize("kind", ["jobs", "job", "metadata"])
def test_symlink_scopes_are_refused(tmp_path, kind):
    root, job = campaign(tmp_path)
    history(job)
    if kind == "metadata":
        (job / "result.json").symlink_to(root / "manifest.json")
    else:
        path = root / "jobs" if kind == "jobs" else job
        moved = tmp_path / "moved"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(prune.UnsafeCampaign):
        prune.plan_pruning(root)


def test_cleanup_preserves_hpo_completed_result_artifact_validation(tmp_path):
    import hashlib
    from rl_matdesign.hpo.job_runner import JobRunner
    root, job = campaign(tmp_path)
    history(job)
    (job / "policy.pt").write_bytes(b"frozen model")
    (job / "std_scaler.bin").write_bytes(b"scaler")
    originals = {str(job / name): hashlib.sha256((job / name).read_bytes()).hexdigest()
                 for name in ("checkpoint.pt", "policy.pt", "std_scaler.bin")}
    result = {"status": "COMPLETED", "artifact_hashes": originals}
    (job / "result.json").write_text(json.dumps(result))
    result_bytes = (job / "result.json").read_bytes()
    JobRunner._validate_artifacts(result, "Original completed job")
    prune.apply_pruning(prune.plan_pruning(root))
    assert (job / "result.json").read_bytes() == result_bytes
    JobRunner._validate_artifacts(result, "Pruned completed job")
    assert JobRunner._completed_result(job / "result.json") == result
