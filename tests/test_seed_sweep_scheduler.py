"""Exercise process bounds, failure accounting, and cancellation without ML."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rl_matdesign import seed_sweep
from rl_matdesign.config_schema import ConfigError, resolve_run_configs


def request(tmp_path, count=4, jobs=2, **extra):
    plan = resolve_run_configs({"train_seed": list(range(count)), "parallel_jobs": jobs, **extra})
    args = SimpleNamespace(config=str(tmp_path / "original.yaml"), out=str(tmp_path / "runs"),
                           device="cpu", save_checkpoint_freq=1, skip_generation=True)
    return args, plan


def test_real_children_overlap_but_obey_limit_and_keep_failed_seeds(tmp_path, monkeypatch):
    args, plan = request(tmp_path)
    script = tmp_path / "child.py"
    script.write_text('''import json, pathlib, sys, time
out = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
start = time.monotonic()
time.sleep(0.25)
(out / 'interval.json').write_text(json.dumps([start, time.monotonic()]))
raise SystemExit(9 if seed == 1 else 0)
''')
    monkeypatch.setattr(seed_sweep, "_child_command", lambda args, config, out:
                        [sys.executable, str(script), str(out), str(int(out.name.split('_train_')[1].split('_')[0]))])
    assert seed_sweep.run_seed_sweep(args, plan) == 1
    manifest = json.loads((Path(args.out) / "seed_runs.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_runs"] == 3 and manifest["unsuccessful_runs"] == 1
    assert [row["status"] for row in manifest["runs"]] == ["completed", "failed", "completed", "completed"]
    assert [row["returncode"] for row in manifest["runs"]] == [0, 9, 0, 0]
    events = []
    for row in manifest["runs"]:
        start, end = json.loads((Path(row["out"]) / "interval.json").read_text())
        events.extend([(start, 1), (end, -1)])
    active = maximum = 0
    for _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    assert maximum == 2
    assert not (Path(args.out) / ".seed_runs.lock").exists()


@pytest.mark.parametrize("budget_status,code,expected", [
    ("COMPLETED", 0, "completed"), ("COMPLETED", 1, "failed"),
    ("INFEASIBLE_EFFICIENCY", 0, "infeasible_efficiency"),
    ("INFEASIBLE_NUMERICAL", 1, "failed"),
    ("INCOMPLETE", 0, "incomplete"), ("RUNNING", 0, "failed"),
])
def test_budget_outcome_not_just_exit_code(tmp_path, budget_status, code, expected):
    (tmp_path / "candidate_budget.json").write_text(json.dumps({
        "status": budget_status, "phase": "train", "target_unique": 4,
        "ratio_limit": 1.5, "unique_evaluated": 4, "attempts_started": 4, "pending": 0}))
    row = {"out": str(tmp_path), "returncode": code}
    config = {"hpo_recording": True, "hpo_phase": "train", "hpo_target_unique": 4, "hpo_ratio_limit": 1.5}
    assert seed_sweep._completion_status(row, config) == expected
    assert row["budget_status"] == budget_status


@pytest.mark.parametrize("contents", [None, "broken json", "{}", '[]'])
def test_missing_budget_never_counts_as_success(tmp_path, contents):
    if contents is not None:
        (tmp_path / "candidate_budget.json").write_text(contents)
    row = {"out": str(tmp_path), "returncode": 0}
    assert seed_sweep._completion_status(row, {"hpo_recording": True}) == "failed"


def test_existing_results_are_preserved(tmp_path):
    args, plan = request(tmp_path)
    directory = Path(args.out)
    directory.mkdir()
    saved = directory / "precious.txt"
    saved.write_text("original")
    with pytest.raises(ConfigError, match="empty --out"):
        seed_sweep.run_seed_sweep(args, plan)
    assert saved.read_text() == "original"
    assert list(directory.iterdir()) == [saved]


def test_reserved_directory_is_not_stolen(tmp_path):
    args, plan = request(tmp_path)
    directory = Path(args.out)
    directory.mkdir()
    lock = directory / ".seed_runs.lock"
    lock.write_text("another process")
    with pytest.raises(ConfigError, match="reserved"):
        seed_sweep.run_seed_sweep(args, plan)
    assert lock.read_text() == "another process"


def test_spawn_failure_records_every_seed_and_continues(tmp_path, monkeypatch):
    args, plan = request(tmp_path)
    monkeypatch.setattr(seed_sweep, "_child_command", lambda args, config, out:
                        [str(tmp_path / "missing")] if "run_002" in out.name
                        else [sys.executable, "-c", "pass"])
    assert seed_sweep.run_seed_sweep(args, plan) == 1
    manifest = json.loads((Path(args.out) / "seed_runs.json").read_text())
    assert len(manifest["runs"]) == 4
    assert [row["status"] for row in manifest["runs"]] == ["completed", "failed", "completed", "completed"]
    assert "Could not start" in manifest["runs"][1]["error"]


def test_interrupt_reaps_active_child_and_cancels_pending(tmp_path, monkeypatch):
    args, plan = request(tmp_path, jobs=1)
    children = []
    original_popen = subprocess.Popen
    original_sleep = seed_sweep.time.sleep
    interrupted = False

    def popen(*positional, **keywords):
        child = original_popen(*positional, **keywords)
        children.append(child)
        return child

    def interrupt(duration):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_sleep(duration)

    monkeypatch.setattr(seed_sweep, "_child_command", lambda *unused:
                        [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(seed_sweep.subprocess, "Popen", popen)
    monkeypatch.setattr(seed_sweep.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        seed_sweep.run_seed_sweep(args, plan)
    assert len(children) == 1 and children[0].poll() is not None
    manifest = json.loads((Path(args.out) / "seed_runs.json").read_text())
    assert manifest["status"] == "interrupted"
    assert [row["status"] for row in manifest["runs"]] == ["interrupted", "cancelled", "cancelled", "cancelled"]
    assert not (Path(args.out) / ".seed_runs.lock").exists()


@pytest.mark.parametrize("change", [
    {"phase": "generation"}, {"target_unique": 3}, {"unique_evaluated": 3},
    {"ratio_limit": 2}, {"attempts_started": 6}, {"attempts_started": 3},
    {"pending": 1}, {"attempts_started": True},
])
def test_completed_hpo_budget_must_match_request(tmp_path, change):
    budget = {"status": "COMPLETED", "phase": "train", "target_unique": 4,
              "ratio_limit": 1.5, "unique_evaluated": 4, "attempts_started": 4, "pending": 0}
    budget.update(change)
    (tmp_path / "candidate_budget.json").write_text(json.dumps(budget))
    row = {"out": str(tmp_path), "returncode": 0}
    config = {"hpo_recording": True, "hpo_phase": "train", "hpo_target_unique": 4, "hpo_ratio_limit": 1.5}
    assert seed_sweep._completion_status(row, config) == "failed"


def test_identical_seed_tuples_get_independent_directories(tmp_path, monkeypatch):
    args, _ = request(tmp_path)
    plan = resolve_run_configs({"train_seed": [7, 7], "parallel_jobs": 2})
    monkeypatch.setattr(seed_sweep, "_child_command", lambda *unused: [sys.executable, "-c", "pass"])
    assert seed_sweep.run_seed_sweep(args, plan) == 0
    rows = json.loads((Path(args.out) / "seed_runs.json").read_text())["runs"]
    assert rows[0]["seeds"] == rows[1]["seeds"]
    assert rows[0]["out"] != rows[1]["out"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_popen_registers_and_reaps_child(tmp_path, monkeypatch, signum):
    args, plan = request(tmp_path, jobs=1)
    children = []
    original_popen = subprocess.Popen
    original_handler = signal.getsignal(signum)

    def popen(*positional, **keywords):
        child = original_popen(*positional, **keywords)
        children.append(child)
        signal.raise_signal(signum)
        return child

    monkeypatch.setattr(seed_sweep, "_child_command", lambda *unused:
                        [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(seed_sweep.subprocess, "Popen", popen)
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        seed_sweep.run_seed_sweep(args, plan)
    assert len(children) == 1 and children[0].poll() is not None
    assert signal.getsignal(signum) == original_handler
    rows = json.loads((Path(args.out) / "seed_runs.json").read_text())["runs"]
    assert [row["status"] for row in rows] == ["interrupted", "cancelled", "cancelled", "cancelled"]
