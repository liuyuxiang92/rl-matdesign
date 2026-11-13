"""Restart permits only the audited storage repair, without rewriting evidence."""
from copy import deepcopy
from pathlib import Path
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


@pytest.mark.parametrize("parallel_jobs", [None, 10])
def test_restart_check_only_is_read_only_even_for_approved_upgrade(existing, monkeypatch, capsys, parallel_jobs):
    _, out = existing
    approve(monkeypatch)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    monkeypatch.setattr(Campaign, "run", lambda *a, **kw: pytest.fail("check-only started execution"))
    argv = ["restart", "--out", str(out), "--check-only"]
    if parallel_jobs is not None:
        argv += ["--parallel-jobs", str(parallel_jobs)]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)["restart_check"]
    assert result["ready"] and result["requested_stage"] == "ranges"
    assert result["software_migration"]["migration"]["id"] == "test-storage-repair"
    assert result["execution_options"] == {
        "original_parallel_jobs": 1, "requested_parallel_jobs": parallel_jobs,
        "effective_parallel_jobs": parallel_jobs or 1,
    }
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("command", ["resume", "restart"])
def test_resume_and_restart_validate_then_continue_saved_scope(existing, monkeypatch, capsys, command):
    _, out = existing
    calls = []
    monkeypatch.setattr(Campaign, "run", lambda self, scope: calls.append(scope) or {"status": "COMPLETED"})
    assert main([command, "--out", str(out)]) == 0
    assert calls == ["ranges"]
    assert '"restart_check"' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["resume", "restart"])
def test_blocked_restart_does_not_launch_any_job(existing, monkeypatch, capsys, command):
    _, out = existing
    monkeypatch.setattr("rl_matdesign.hpo.restart.inspect_restart", lambda _: {
        "ready": False, "jobs": [{"classification": "BLOCKED", "reason": "accounting advanced beyond checkpoint"}]})
    monkeypatch.setattr(Campaign, "run", lambda *a: pytest.fail("blocked restart sampled"))
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    monkeypatch.setattr(Campaign, "record_execution_options",
                        lambda *a: pytest.fail("blocked restart changed execution history"))
    assert main([command, "--out", str(out), "--parallel-jobs", "10"]) == 2
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    captured = capsys.readouterr()
    assert "advanced beyond checkpoint" in captured.out
    assert "no new sampling" in captured.err


@pytest.mark.parametrize("command", ["resume", "restart"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "ten"])
def test_parallel_jobs_rejects_nonpositive_or_noninteger_before_loading(command, value, monkeypatch, capsys):
    monkeypatch.setattr(Campaign, "__init__", lambda *a, **kw: pytest.fail("invalid CLI loaded a campaign"))
    with pytest.raises(SystemExit) as exc:
        main([command, "--out", "unused", "--parallel-jobs", value])
    assert exc.value.code == 2
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["resume", "restart"])
@pytest.mark.parametrize("parallel_jobs", [None, 10])
def test_restart_parallel_override_is_runtime_only(existing, monkeypatch, capsys, command, parallel_jobs):
    plan, out = existing
    original = (out / "manifest.json").read_bytes()
    yaml_before = Path(plan["source_path"]).read_bytes()
    calls = []
    def run(self, scope):
        calls.append((scope, self.parallel_jobs, self.plan["parallel_jobs"],
                      self.runner.plan["parallel_jobs"]))
        return {"status": "COMPLETED"}
    monkeypatch.setattr(Campaign, "run", run)
    argv = [command, "--out", str(out)]
    if parallel_jobs is not None:
        argv += ["--parallel-jobs", str(parallel_jobs)]
    assert main(argv) == 0
    assert calls == [("ranges", parallel_jobs or 1, 1, 1)]
    history = read_json(out / "execution_options.json")["history"]
    assert len(history) == 1
    assert history[0]["original_parallel_jobs"] == 1
    assert history[0]["requested_parallel_jobs"] == parallel_jobs
    assert history[0]["effective_parallel_jobs"] == (parallel_jobs or 1)
    assert history[0]["recorded_at"]
    assert (out / "manifest.json").read_bytes() == original
    assert Path(plan["source_path"]).read_bytes() == yaml_before
    report, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert report["restart_check"]["execution_options"]["effective_parallel_jobs"] == (parallel_jobs or 1)


def test_actual_restart_override_history_is_idempotent_and_tracks_switchback(existing, monkeypatch, capsys):
    _, out = existing
    monkeypatch.setattr(Campaign, "run", lambda *a: {"status": "COMPLETED"})
    argv = ["restart", "--out", str(out)]
    assert main(argv + ["--parallel-jobs", "10"]) == 0
    first = (out / "execution_options.json").read_bytes()
    assert main(argv + ["--parallel-jobs", "10"]) == 0
    assert (out / "execution_options.json").read_bytes() == first
    assert main(argv) == 0
    assert main(argv) == 0
    history = read_json(out / "execution_options.json")["history"]
    assert [(row["requested_parallel_jobs"], row["effective_parallel_jobs"]) for row in history] == [
        (10, 10), (None, 1),
    ]
    capsys.readouterr()


def test_override_cannot_bypass_original_configuration_validation(existing, monkeypatch, capsys):
    plan, out = existing
    changed = deepcopy(plan)
    changed["parallel_jobs"] = 7
    monkeypatch.setattr("rl_matdesign.hpo.cli.load_plan", lambda *a, **kw: changed)
    monkeypatch.setattr(Campaign, "run", lambda *a: pytest.fail("changed plan ran"))
    monkeypatch.setattr(Campaign, "record_execution_options",
                        lambda *a: pytest.fail("changed plan wrote runtime options"))
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert main(["restart", "--out", str(out), "--parallel-jobs", "10"]) == 2
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert "configuration" in capsys.readouterr().err
