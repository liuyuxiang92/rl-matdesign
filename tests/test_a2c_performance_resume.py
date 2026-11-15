"""Native old A2C checkpoints retain paid state through the v2 storage upgrade."""
from __future__ import annotations

import json
import os
import shutil

import pytest
import yaml

from rl_matdesign.hpo.algorithm_profiles import initial_parameters
from rl_matdesign.utils import source_compatibility as compatibility
from rl_matdesign.utils.checkpoint_store import load_checkpoint
from tests.test_dqn_performance_resume import (
    ROOT, _cli, _snapshot, _success, legacy_source,
)
from tests.test_hpo_checkpoint_retention import _assert_nested_equal
from tests.test_hpo_cooperative_restart import _plan


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX cooperative signals")


def test_old_a2c_native_restart_then_second_restart_is_exact(tmp_path, legacy_source):
    plan, _ = _plan(tmp_path, "a2c")
    parameters = initial_parameters(plan)
    parameters["pg_batch_eps"] = 2
    hpo = tmp_path / "hpo.yaml"
    config = yaml.safe_load(hpo.read_text())
    config["training"]["parameter_ranges"] = {
        name: {"type": "categorical", "choices": [value]}
        for name, value in parameters.items()
    }
    hpo.write_text(yaml.safe_dump(config, sort_keys=False))

    interrupted = tmp_path / "old-interrupted"
    paused = _cli(legacy_source, tmp_path, "run", "--hpo-config", str(hpo),
                  "--out", str(interrupted), "--stage", "ranges", stop=True)
    assert paused.returncode == 130, paused.stdout[-6000:] + paused.stderr[-6000:]
    checkpoints = list((interrupted / "jobs").glob("*/checkpoint.pt"))
    assert len(checkpoints) == 1
    job = checkpoints[0].parent
    stopped = load_checkpoint(job / "checkpoint.pt")
    assert stopped["type"] == "pg" and stopped["rl_method"] == "a2c"
    assert 5 <= stopped["hpo_state"]["budget"]["unique_evaluated"] < 12
    assert stopped["iteration"] >= 3  # Stop completes the current on-policy batch.
    paid = json.loads((job / "candidate_budget.json").read_text())
    assert paid["pending"] == 0
    assert paid["attempts_started"] == stopped["hpo_state"]["budget"]["attempts_started"]
    prefix = (job / "candidate_records.jsonl").read_bytes()

    # Both controls begin at exactly the same real, paid, old A2C checkpoint.
    # The comparison differs only in an extra native stop/restart under v2.
    control = tmp_path / "new-control"
    shutil.copytree(interrupted, control)
    _success(_cli(ROOT / "src", tmp_path, "restart", "--out", str(control)))

    before = _snapshot(interrupted)
    checked = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted), "--check-only")
    _success(checked)
    assert "SYNCHRONIZED" in checked.stdout
    assert compatibility.UPGRADE_ID in checked.stdout
    assert _snapshot(interrupted) == before

    paused_again = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted), stop=True)
    assert paused_again.returncode == 130, paused_again.stdout[-6000:] + paused_again.stderr[-6000:]
    second = load_checkpoint(job / "checkpoint.pt")
    assert second["iteration"] > stopped["iteration"]
    assert second["hpo_state"]["budget"]["attempts_started"] > paid["attempts_started"]
    assert (job / "candidate_records.jsonl").read_bytes().startswith(prefix)
    _success(_cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted)))

    assert json.loads((interrupted / "state.json").read_text())["status"] == "COMPLETED"
    control_checkpoints = list((control / "jobs").glob("*/checkpoint.pt"))
    assert len(control_checkpoints) == 1
    control_job = control_checkpoints[0].parent
    expected = load_checkpoint(control_job / "checkpoint.pt")
    actual = load_checkpoint(job / "checkpoint.pt")
    for key in (
        "type", "rl_method", "scaler_fingerprint", "environment_contract",
        "policy_state", "value_net_state", "opt_actor_state", "opt_critic_state",
        "visit_counts", "entropy_coef_eff", "budget_target", "episodes_completed",
        "iteration", "rng_state", "training_budget_state", "training_sample_session_id",
        "training_budget_history_complete", "hpo_state",
    ):
        _assert_nested_equal(actual[key], expected[key])
    for key in ("n_calls", "seen", "best_mean", "diagnostic_requests",
                "diagnostic_failures", "phase", "phase_costs"):
        _assert_nested_equal(actual["predictor_ledger"][key], expected["predictor_ledger"][key])
    assert (job / "candidate_records.jsonl").read_bytes().startswith(prefix)
    assert (job / "candidate_records.jsonl").read_bytes() == (control_job / "candidate_records.jsonl").read_bytes()
    assert json.loads((job / "candidate_budget.json").read_text()) == json.loads((control_job / "candidate_budget.json").read_text())
    assert (job / "checkpoint.pt").is_file() and not (job / "checkpoint.pt").is_symlink()
    assert not list(job.glob("checkpoint-iter*.pt"))
    assert len(list((interrupted / "source_upgrades").glob("*.json"))) == 1
