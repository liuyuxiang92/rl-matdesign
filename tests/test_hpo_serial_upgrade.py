"""A reviewed old serial campaign may adopt runtime concurrency, not new science."""
from copy import deepcopy

import pytest

from rl_matdesign.hpo.artifacts import atomic_json, read_json
from rl_matdesign.hpo.controller import Campaign, CampaignError
from rl_matdesign.hpo.spec import load_plan
from rl_matdesign.utils.hpo_resume_compatibility import normalize_legacy_serial_contract
from tests.test_hpo_v2_job_runner import public_plan_path


@pytest.fixture
def old_serial(tmp_path, monkeypatch):
    plan = load_plan(public_plan_path(tmp_path))
    original = deepcopy(plan)
    original.pop("parallel_jobs")
    original["sources"].pop("parallel_jobs")
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "old-source")
    monkeypatch.setattr("rl_matdesign.hpo.controller.design_space_info", lambda _: {
        "status": "unknown", "size": None, "episode_steps": 1})
    run = Campaign(original, tmp_path / "campaign")
    before = (run.out / "manifest.json").read_bytes()
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "new-source")
    monkeypatch.setattr("rl_matdesign.hpo.controller.approved_upgrade", lambda *a, **kw: {
        "id": "reviewed-serial-upgrade"})
    return plan, run.out, before


def test_old_serial_defaults_normalize_without_rewriting_original(old_serial):
    plan, out, before = old_serial
    untouched = deepcopy(plan)
    run = Campaign(plan, out)
    assert run.parallel_jobs == 1
    assert run.software_migration["normalized_plan_defaults"] == [
        "plan.parallel_jobs", "plan.sources.parallel_jobs"]
    run.record_execution_options(10)
    run.record_software_migration()
    assert run.parallel_jobs == 10
    assert plan == untouched
    assert run.plan["parallel_jobs"] == 1
    assert run.runner.plan["parallel_jobs"] == 1
    assert (out / "manifest.json").read_bytes() == before
    assert "parallel_jobs" not in read_json(out / "manifest.json")["plan"]
    assert read_json(out / "execution_options.json")["history"][-1]["effective_parallel_jobs"] == 10


def test_default_migration_does_not_authorize_unknown_software(old_serial, monkeypatch):
    plan, out, before = old_serial
    monkeypatch.setattr("rl_matdesign.hpo.controller.approved_upgrade", lambda *a, **kw: None)
    with pytest.raises(CampaignError, match="registered"):
        Campaign(plan, out)
    assert (out / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("change", ["parallel", "authored_parallel", "threshold", "seed", "budget"])
def test_default_migration_never_allows_other_plan_changes(old_serial, change):
    plan, out, before = old_serial
    changed = deepcopy(plan)
    if change == "parallel":
        changed["parallel_jobs"] = 10
    elif change == "authored_parallel":
        changed["sources"]["parallel_jobs"] = "yaml"
        changed["authored_config"]["parallel_jobs"] = 1
    elif change == "threshold":
        changed["objective"]["threshold"] += 1
    elif change == "seed":
        changed["training_search"]["seeds"][0] += 1
    else:
        changed["budget"]["train"]["initial_unique"] += 1
    with pytest.raises(CampaignError, match="configuration"):
        Campaign(changed, out)
    assert (out / "manifest.json").read_bytes() == before
    assert not (out / "software_migrations.json").exists()
    assert not (out / "execution_options.json").exists()


@pytest.mark.parametrize("old_parallel,old_source,new_parallel,new_source", [
    (None, None, True, "default"), (None, None, 2, "default"),
    (None, None, 1, "yaml"), (1, None, 1, "default"),
    (None, "default", 1, "default"), (2, "yaml", 1, "default"),
])
def test_normalization_is_only_the_paired_absent_to_serial_default(
        old_parallel, old_source, new_parallel, new_source):
    old = {"plan": {"sources": {}, "authored_config": {}}}
    if old_parallel is not None:
        old["plan"]["parallel_jobs"] = old_parallel
    if old_source is not None:
        old["plan"]["sources"]["parallel_jobs"] = old_source
    new = {"plan": {"parallel_jobs": new_parallel, "sources": {"parallel_jobs": new_source}}}
    before = deepcopy(old)
    comparable, changed = normalize_legacy_serial_contract(old, new)
    assert not changed
    assert comparable == old == before


def test_same_software_keeps_same_strict_manifest_guard(old_serial, monkeypatch):
    plan, out, before = old_serial
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "old-source")
    monkeypatch.setattr("rl_matdesign.hpo.controller.approved_upgrade", lambda *a, **kw: None)
    with pytest.raises(CampaignError, match="registered"):
        Campaign(plan, out)
