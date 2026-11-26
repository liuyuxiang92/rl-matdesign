"""A single audited source update never relaxes scientific restart evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_matdesign.hpo.artifacts import atomic_json, read_json
from rl_matdesign.hpo.cli import main
from rl_matdesign.hpo.controller import Campaign, CampaignError
from rl_matdesign.hpo.spec import load_plan
from rl_matdesign.utils import source_compatibility as compat
from rl_matdesign.utils.execution_contract import contracts_equal, require_execution_contract
from tests.test_hpo_v2_job_runner import public_plan_path


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Test the real validator using an isolated inventory, not a mocked allow."""
    shipped = json.loads(Path(compat.__file__).with_name("dqn_performance_source.json").read_text())
    baseline = shipped["baseline_files"]
    overrides = {"rl_matdesign/utils/source_compatibility.py":
                 hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()}
    target = {**baseline, **overrides}
    path = tmp_path / "release.json"
    record = {"schema_version": 1, "id": compat.UPGRADE_ID,
              "baseline_revision": compat.BASELINE_REVISION,
              "baseline_files": baseline, "target_files": overrides}
    path.write_text(json.dumps(record))
    installed = deepcopy(target)
    monkeypatch.setattr(compat, "_manifest_path", lambda: path)
    monkeypatch.setattr(compat, "source_inventory", lambda _: deepcopy(installed))
    return SimpleNamespace(path=path, record=record, installed=installed,
                           baseline=compat.inventory_digests(baseline),
                           target=compat.inventory_digests(target))


def execution(source):
    return {"evaluation": {"source_sha256": source,
              "configuration": {"property": "T3", "direction": "min", "threshold": 0},
              "assets": [{"sha256": "frozen-oracle"}],
              "adapter_identity": {"device": "cpu", "schema": 2},
              "numerical_runtime": {"platform": "linux", "torch": "same"},
              "runtime": {"device": "cuda", "python": "same"}},
            "learning": {"method": "dqn", "train_seed": 42, "checkpoint_frequency": 1,
                         "dqn_grad_steps_per_ep": 1, "dqn_batch_size": 128}}


def generation(source):
    return {"execution": execution(source), "sampling": {"gen_seed": 101, "gen_exploration_target": 1},
            "frozen_artifacts": {"qnet.pt": "same-model", "std_scaler.bin": "same-scaler"}}


def replace(value, path, item):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = item


@pytest.mark.parametrize("kind", ["package", "execution"])
def test_exact_audited_forward_upgrade_only(release, kind):
    evidence = compat.approved_performance_upgrade(release.baseline[kind], release.target[kind], kind=kind)
    assert evidence["id"] == compat.UPGRADE_ID
    assert evidence["saved_source_sha256"] == release.baseline[kind]
    assert evidence["current_source_sha256"] == release.target[kind]
    assert compat.approved_performance_upgrade(release.target[kind], release.baseline[kind], kind=kind) is None
    assert compat.approved_performance_upgrade("unknown", release.target[kind], kind=kind) is None
    assert compat.approved_performance_upgrade(release.baseline[kind], "unknown", kind=kind) is None


@pytest.mark.parametrize("kind", ["package", "execution"])
@pytest.mark.parametrize("saved_legacy", [False, True])
@pytest.mark.parametrize("installed_legacy", [False, True])
def test_exact_server_legacy_source_variants(release, kind, saved_legacy, installed_legacy):
    baseline = release.record["baseline_files"]
    if saved_legacy:
        baseline = {**baseline, **compat.LEGACY_PREDICTOR_FILES}
    if installed_legacy:
        release.installed.update(compat.LEGACY_PREDICTOR_FILES)
    old = compat.inventory_digests(baseline)[kind]
    new = compat.inventory_digests(release.installed)[kind]
    evidence = compat.approved_performance_upgrade(old, new, kind=kind)
    assert evidence is not None
    assert evidence["saved_inventory_variant"] == ("legacy_predictors" if saved_legacy else "clean")
    assert evidence["current_inventory_variant"] == ("legacy_predictors" if installed_legacy else "clean")


@pytest.mark.parametrize("mutation", ["changed", "missing_one", "unexpected_extra"])
def test_legacy_sources_must_be_exact_all_or_none(release, mutation):
    release.installed.update(compat.LEGACY_PREDICTOR_FILES)
    name = next(iter(compat.LEGACY_PREDICTOR_FILES))
    if mutation == "changed":
        release.installed[name] = "f" * 64
    elif mutation == "missing_one":
        release.installed.pop(name)
    else:
        release.installed["rl_matdesign/predictors/unreviewed.py"] = "f" * 64
    new = compat.inventory_digests(release.installed)["execution"]
    assert compat.approved_performance_upgrade(compat.LEGACY_EXECUTION_SHA256,
                                               new, kind="execution") is None


def test_server_inventory_hashes_are_independently_reproducible(release):
    hashes = compat.inventory_digests({**release.record["baseline_files"], **compat.LEGACY_PREDICTOR_FILES})
    assert hashes == {"package": compat.LEGACY_PACKAGE_SHA256, "execution": compat.LEGACY_EXECUTION_SHA256}


@pytest.mark.parametrize("mutation", ["changed_scorer", "changed_target", "extra_file", "missing_file"])
def test_installed_inventory_drift_is_rejected(release, mutation):
    if mutation == "changed_scorer":
        release.installed["rl_matdesign/hpo/scoring.py"] = "f" * 64
    elif mutation == "changed_target":
        release.installed["rl_matdesign/utils/source_compatibility.py"] = "f" * 64
    elif mutation == "extra_file":
        release.installed["rl_matdesign/unreviewed.py"] = "f" * 64
    else:
        release.installed.pop("rl_matdesign/training.py")
    assert compat.approved_performance_upgrade(release.baseline["execution"],
        release.target["execution"], kind="execution") is None


@pytest.mark.parametrize("mutation", ["bad_baseline", "scoring_override", "unsealed", "missing", "invalid_json", "invalid_type"])
def test_untrusted_or_incomplete_inventory_is_rejected(release, mutation):
    if mutation == "missing":
        release.path.unlink()
    elif mutation == "invalid_json":
        release.path.write_text("not JSON")
    elif mutation == "invalid_type":
        release.path.write_text("[]")
    else:
        if mutation == "bad_baseline":
            release.record["baseline_files"]["rl_matdesign/training.py"] = "f" * 64
        elif mutation == "scoring_override":
            release.record["target_files"]["rl_matdesign/hpo/scoring.py"] = "f" * 64
        else:
            release.record["target_files"] = {}
        release.path.write_text(json.dumps(release.record))
    assert compat.approved_performance_upgrade(release.baseline["package"],
        release.target["package"], kind="package") is None


def test_source_upgrade_does_not_redefine_generic_exact_equality(release):
    old, new = execution(release.baseline["execution"]), execution(release.target["execution"])
    before = deepcopy(old)
    assert not contracts_equal(old, new)
    require_execution_contract(old, new)
    assert compat.evaluation_contracts_equal(old["evaluation"], new["evaluation"])
    assert old == before


@pytest.mark.parametrize("path,value", [
    (("evaluation", "configuration", "threshold"), False),
    (("evaluation", "configuration", "direction"), "max"),
    (("evaluation", "assets", 0, "sha256"), "new-oracle"),
    (("evaluation", "adapter_identity", "device"), "cuda"),
    (("evaluation", "numerical_runtime", "platform"), "another-platform"),
    (("evaluation", "runtime", "python"), "new-python"),
    (("learning", "train_seed"), 43),
    (("learning", "checkpoint_frequency"), 2),
    (("learning", "dqn_batch_size"), 256),
    (("learning", "dqn_grad_steps_per_ep"), 7),
])
def test_source_upgrade_never_hides_other_execution_changes(release, path, value):
    old, new = execution(release.baseline["execution"]), execution(release.target["execution"])
    replace(new, path, value)
    with pytest.raises(ValueError, match="first differing fields"):
        require_execution_contract(old, new)


@pytest.mark.parametrize("path,value", [
    (("sampling", "gen_seed"), 102),
    (("sampling", "gen_exploration_target"), .5),
    (("frozen_artifacts", "qnet.pt"), "new-model"),
    (("frozen_artifacts", "std_scaler.bin"), "new-scaler"),
    (("execution", "learning", "train_seed"), 43),
])
def test_generation_upgrade_still_freezes_models_seeds_and_sampling(release, path, value):
    old, new = generation(release.baseline["execution"]), generation(release.target["execution"])
    assert compat.generation_contracts_equal(old, new)
    replace(new, path, value)
    assert not compat.generation_contracts_equal(old, new)


def test_generation_restore_uses_source_only_comparator(release, monkeypatch):
    from rl_matdesign.exploration import ExplorationController
    from rl_matdesign.utils import generation_resume
    ledgers, rng = [], []
    monkeypatch.setattr(generation_resume, "restore_rng_state", rng.append)
    contract = generation(release.baseline["execution"])
    controller = ExplorationController(contract["sampling"]["gen_exploration_target"], True)
    saved = {"generation_contract": contract, "exploration_state": controller.state_dict(),
             "predictor_ledger": {"n_calls": 7}, "rng_state": {"saved": True}}
    generation_resume.restore_generation(SimpleNamespace(load_state_dict=ledgers.append),
        saved, generation(release.target["execution"]))
    assert ledgers == [{"n_calls": 7}] and rng == [{"saved": True}]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_frozen_model_load_accepts_upgrade_without_rewriting_weights(release, tmp_path, checkpoint):
    import numpy as np
    import torch
    from rl_matdesign.env_multigroup import MultiGroupEnv
    from rl_matdesign.utils.model_contract import load_artifact, save_weights
    env = MultiGroupEnv(groups=[{'species_set': ("Ni", "Fe", "Co"), 'amount_set': ["0.25", "0.50", "0.75"], 'n_steps': 2, 'name': 'composition', 'kind': 'composition', 'unique': True, 'total': 1}], state_featurizer=lambda _: np.zeros(2))
    env.evaluation_contract = execution(release.baseline["execution"])["evaluation"]
    path = tmp_path / "qnet.pt"
    save_weights(path, torch.nn.Linear(2, 1), env, "qnet")
    if checkpoint:
        raw = torch.load(path, weights_only=True)
        raw["execution_contract"] = execution(release.baseline["execution"])
        torch.save(raw, path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    env.evaluation_contract = execution(release.target["execution"])["evaluation"]
    load_artifact(path, env, checkpoint=checkpoint)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    env.evaluation_contract["assets"][0]["sha256"] = "changed-oracle"
    with pytest.raises(ValueError, match="evaluation contract differs"):
        load_artifact(path, env, checkpoint=checkpoint)


@pytest.fixture
def campaign(release, tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    plan = load_plan(source)
    monkeypatch.setattr("rl_matdesign.hpo.controller.design_space_info",
                        lambda _: {"status": "unknown", "size": None, "episode_steps": 1})
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: release.baseline["package"])
    run = Campaign(plan, tmp_path / "campaign")
    atomic_json(run.out / "state.json", {"status": "INTERRUPTED", "requested_stage": "ranges"})
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: release.target["package"])
    return SimpleNamespace(plan=plan, out=run.out)


def test_upgrade_reuses_cached_training_generation_and_job_ids(campaign, release, monkeypatch):
    from tests.test_hpo_v2_job_runner import FakeWorker
    worker = FakeWorker(monkeypatch)
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: release.baseline["package"])
    original = Campaign(campaign.plan, campaign.out)
    trained = original.runner({"pg_batch_eps": 2}, 8, 11, "train")
    generated = original.runner({"pg_batch_eps": 2, "gen_exploration_target": 1}, 4, 401, "generation", trained)
    before = {p: p.read_bytes() for p in (campaign.out / "jobs").rglob("*") if p.is_file()}
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: release.target["package"])
    upgraded = Campaign(campaign.plan, campaign.out)
    assert upgraded.source_upgrade
    assert upgraded.runner({"pg_batch_eps": 2}, 8, 11, "train") == trained
    assert upgraded.runner({"pg_batch_eps": 2, "gen_exploration_target": 1}, 4, 401, "generation", trained) == generated
    assert len(worker.calls) == 2
    assert {p: p.read_bytes() for p in (campaign.out / "jobs").rglob("*") if p.is_file()} == before


def test_check_only_accepts_upgrade_without_any_filesystem_writes(campaign, monkeypatch, capsys):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.out.rglob("*") if p.is_file()}
    monkeypatch.setattr(Campaign, "run", lambda *a: pytest.fail("check-only ran workers"))
    assert main(["restart", "--out", str(campaign.out), "--check-only", "--parallel-jobs", "10"]) == 0
    report = json.loads(capsys.readouterr().out)["restart_check"]
    assert report["source_upgrade"]["id"] == compat.UPGRADE_ID
    assert report["execution_options"]["effective_parallel_jobs"] == 10
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.out.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("command", ["restart", "resume"])
def test_actual_restart_records_idempotent_receipt_preserving_manifest(campaign, monkeypatch, capsys, command):
    manifest = (campaign.out / "manifest.json").read_bytes()
    calls = []
    monkeypatch.setattr(Campaign, "run", lambda self, scope: calls.append(scope) or {"status": "COMPLETED"})
    args = [command, "--out", str(campaign.out)]
    assert main(args) == 0
    path = campaign.out / "source_upgrades" / (compat.UPGRADE_ID + ".json")
    first = path.read_bytes(), path.stat().st_mtime_ns
    assert read_json(path)["recorded_at"]
    assert main(args) == 0
    assert calls == ["ranges", "ranges"]
    assert (path.read_bytes(), path.stat().st_mtime_ns) == first
    assert (campaign.out / "manifest.json").read_bytes() == manifest
    capsys.readouterr()


def test_failed_preflight_never_records_upgrade(campaign, monkeypatch, capsys):
    monkeypatch.setattr("rl_matdesign.hpo.restart.inspect_restart",
                        lambda _: {"ready": False, "jobs": [{"classification": "BLOCKED"}]})
    before = {p: p.read_bytes() for p in campaign.out.rglob("*") if p.is_file()}
    assert main(["restart", "--out", str(campaign.out)]) == 2
    assert {p: p.read_bytes() for p in campaign.out.rglob("*") if p.is_file()} == before
    capsys.readouterr()


@pytest.mark.parametrize("change", ["plan", "domain", "yaml", "manifest", "inventory"])
def test_campaign_upgrade_keeps_every_other_guard(campaign, release, change):
    plan = deepcopy(campaign.plan)
    kwargs = {}
    if change == "plan":
        plan["objective"]["threshold"] += 1
    elif change == "domain":
        kwargs["domain"] = {"status": "exact", "size": 2, "episode_steps": 1}
    elif change == "yaml":
        path = Path(plan["source_path"])
        path.write_text(path.read_text() + "\n# changed\n")
    elif change == "manifest":
        original = read_json(campaign.out / "manifest.json")
        original["software_digest"] = "tampered"
        atomic_json(campaign.out / "manifest.json", original)
    else:
        release.installed["rl_matdesign/hpo/scoring.py"] = "f" * 64
    with pytest.raises(CampaignError):
        Campaign(plan, campaign.out, **kwargs)


def test_tampered_upgrade_receipt_is_not_overwritten(campaign):
    run = Campaign(campaign.plan, campaign.out)
    run.record_source_upgrade()
    path = campaign.out / "source_upgrades" / (compat.UPGRADE_ID + ".json")
    receipt = read_json(path)
    receipt["saved_source_sha256"] = "changed"
    atomic_json(path, receipt)
    before = path.read_bytes()
    with pytest.raises(CampaignError, match="receipt changed"):
        run.record_source_upgrade()
    assert path.read_bytes() == before


@pytest.mark.parametrize("installed_legacy", [False, True])
def test_shipped_sealed_manifest_authenticates_only_its_recorded_target(monkeypatch, installed_legacy):
    """The historical release remains valid without sealing today's runtime."""
    path = Path(compat.__file__).with_name("dqn_performance_source.json")
    original = path.read_bytes()
    record = json.loads(original)
    assert compat.inventory_digests(record["baseline_files"]) == {
        "package": compat.BASELINE_PACKAGE_SHA256,
        "execution": compat.BASELINE_EXECUTION_SHA256,
    }
    target = {**record["baseline_files"], **record["target_files"]}
    if installed_legacy:
        target.update(compat.LEGACY_PREDICTOR_FILES)
    # Only the observed inventory is isolated. The shipped seal and the real
    # validator are unchanged, unlike the synthetic mutation fixtures above.
    monkeypatch.setattr(compat, "source_inventory", lambda _: dict(target))
    current = compat.inventory_digests(target)
    for kind, baseline in (("package", compat.BASELINE_PACKAGE_SHA256),
                           ("execution", compat.BASELINE_EXECUTION_SHA256)):
        evidence = compat.approved_performance_upgrade(baseline, current[kind], kind=kind)
        assert evidence is not None
        assert evidence["inventory_sha256"] == hashlib.sha256(original).hexdigest()
        assert evidence["changed_source_files"] == sorted(record["target_files"])
        assert evidence["current_inventory_variant"] == ("legacy_predictors" if installed_legacy else "clean")
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["package", "execution"])
@pytest.mark.parametrize("saved_legacy", [False, True])
def test_semantic_material_group_runtime_is_not_the_sealed_performance_release(kind, saved_legacy):
    installed = compat.source_inventory(compat._source_root())
    record = json.loads(Path(compat.__file__).with_name("dqn_performance_source.json").read_text())
    sealed = {**record["baseline_files"], **record["target_files"]}
    # This refactor adds a new scientific action domain. It must never inherit
    # the historical storage-only migration's exact-learning guarantee.
    assert "rl_matdesign/amounts.py" in installed
    assert "rl_matdesign/amounts.py" not in sealed
    baseline = (compat.LEGACY_PACKAGE_SHA256 if kind == "package" else compat.LEGACY_EXECUTION_SHA256) \
        if saved_legacy else (compat.BASELINE_PACKAGE_SHA256 if kind == "package" else compat.BASELINE_EXECUTION_SHA256)
    current = compat.inventory_digests(installed)[kind]
    assert compat.approved_performance_upgrade(baseline, current, kind=kind) is None


def test_evidence_script_is_standalone_and_does_not_import_or_modify_package(tmp_path):
    import os
    import subprocess
    import sys
    root = tmp_path / "installed"
    package = root / "rl_matdesign"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('evidence collection imported the package')\n")
    campaign = tmp_path / "runs" / "dqn_mc_medium"
    campaign.mkdir(parents=True)
    contract = {"plan": {"method": "dqn_mc"}, "domain": {}, "assets": [],
                "software_digest": "old-source", "configuration_sources": {}}
    from rl_matdesign.hpo.artifacts import digest
    atomic_json(campaign / "manifest.json", {**contract, "contract_digest": digest(contract)})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    script = Path(__file__).parents[1] / "scripts" / "hpo_source_evidence.py"
    result = subprocess.run([sys.executable, str(script), "--runs-dir", str(campaign.parent)],
        env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["read_only"]
    assert report["installed_package"] == str(package)
    assert report["campaigns"][0]["manifest_integrity"]
    assert report["campaigns"][0]["software_digest"] == "old-source"
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()} == before
