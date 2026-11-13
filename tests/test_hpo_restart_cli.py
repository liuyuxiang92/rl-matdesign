"""Restart permits only the audited storage repair, without rewriting evidence."""
from copy import deepcopy
import json

import pytest

from rl_matdesign.hpo.artifacts import atomic_json, read_json
from rl_matdesign.hpo.cli import main
from rl_matdesign.hpo.controller import Campaign, CampaignError
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_job_runner import public_plan_path


@pytest.fixture
def existing(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    plan = load_plan(source)
    domain = {"status": "unknown", "size": None, "episode_steps": 1}
    monkeypatch.setattr("rl_matdesign.hpo.controller.design_space_info", lambda _: domain)
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "old-software")
    run = Campaign(plan, tmp_path / "campaign")
    atomic_json(run.out / "state.json", {"status": "INTERRUPTED", "requested_stage": "ranges"})
    return plan, run.out


def approve(monkeypatch):
    upgrade = {"id": "test-storage-repair", "reason": "latest checkpoint only"}
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "new-software")
    monkeypatch.setattr("rl_matdesign.hpo.controller.approved_upgrade",
                        lambda old, new, **kw: deepcopy(upgrade)
                        if (old, new, kw["kind"]) == ("old-software", "new-software", "package") else None)
    return upgrade


def test_migration_keeps_manifest_bytes_and_records_once_on_execution(existing, monkeypatch):
    plan, out = existing
    before = (out / "manifest.json").read_bytes()
    approved = approve(monkeypatch)
    run = Campaign(plan, out)
    assert run.software_migration["migration"] == approved
    assert not (out / "software_migrations.json").exists()
    run.record_software_migration()
    first = (out / "software_migrations.json").read_bytes()
    Campaign(plan, out).record_software_migration()
    assert (out / "software_migrations.json").read_bytes() == first
    assert len(read_json(out / "software_migrations.json")["migrations"]) == 1
    assert (out / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("field", ["threshold", "budget", "seed", "parallel"])
def test_approved_source_never_allows_other_changes(existing, monkeypatch, field):
    plan, out = existing
    approve(monkeypatch)
    changed = deepcopy(plan)
    if field == "threshold":
        changed["objective"]["threshold"] += 1
    elif field == "budget":
        changed["budget"]["train"]["initial_unique"] += 1
    elif field == "seed":
        changed["training_search"]["seeds"][0] += 1
    else:
        changed["parallel_jobs"] += 1
    with pytest.raises(CampaignError, match="configuration"):
        Campaign(changed, out)
    assert not (out / "software_migrations.json").exists()


def test_unknown_source_and_modified_original_manifest_are_rejected(existing, monkeypatch):
    plan, out = existing
    monkeypatch.setattr("rl_matdesign.hpo.controller.package_digest", lambda: "unknown-software")
    with pytest.raises(CampaignError, match="registered"):
        Campaign(plan, out)
    approve(monkeypatch)
    damaged = read_json(out / "manifest.json")
    damaged["domain"]["size"] = 999
    atomic_json(out / "manifest.json", damaged)
    with pytest.raises(CampaignError, match="manifest contract"):
        Campaign(plan, out)


def test_restart_check_only_is_read_only_even_for_approved_upgrade(existing, monkeypatch, capsys):
    _, out = existing
    approve(monkeypatch)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    monkeypatch.setattr(Campaign, "run", lambda *a, **kw: pytest.fail("check-only started execution"))
    assert main(["restart", "--out", str(out), "--check-only"]) == 0
    result = json.loads(capsys.readouterr().out)["restart_check"]
    assert result["ready"] and result["requested_stage"] == "ranges"
    assert result["software_migration"]["migration"]["id"] == "test-storage-repair"
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("command", ["resume", "restart"])
def test_resume_and_restart_validate_then_continue_saved_scope(existing, monkeypatch, capsys, command):
    _, out = existing
    calls = []
    monkeypatch.setattr(Campaign, "run", lambda self, scope: calls.append(scope) or {"status": "COMPLETED"})
    assert main([command, "--out", str(out)]) == 0
    assert calls == ["ranges"]
    assert '"restart_check"' in capsys.readouterr().out


def test_blocked_restart_does_not_launch_any_job(existing, monkeypatch, capsys):
    _, out = existing
    monkeypatch.setattr("rl_matdesign.hpo.restart.inspect_restart", lambda _: {
        "ready": False, "jobs": [{"classification": "BLOCKED", "reason": "accounting advanced beyond checkpoint"}]})
    monkeypatch.setattr(Campaign, "run", lambda *a: pytest.fail("blocked restart sampled"))
    assert main(["restart", "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "advanced beyond checkpoint" in captured.out
    assert "no new sampling" in captured.err
