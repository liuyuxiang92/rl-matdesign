"""Native HPO workers stop after committed learning units, not paid samples."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import torch
import yaml

from rl_matdesign import training
from rl_matdesign.utils.hpo_stop import checkpoint_committed, cooperative_worker_stop
from tests.test_hpo_checkpoint_retention import _assert_nested_equal
from tests.test_experiment_reproducibility import _config


ROOT = Path(__file__).resolve().parents[1]


def checkpoint(**changes):
    budget = dict(pending=0, attempts_started=3, unique_evaluated=3,
                  target_unique=12, max_attempts=17, status="RUNNING")
    budget.update(changes)
    return {"hpo_state": {"budget": budget}}


def test_first_signal_waits_for_committed_checkpoint_and_restores_handlers():
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(SystemExit) as stopped:
        with cooperative_worker_stop():
            signal.raise_signal(signal.SIGTERM)
            # No exception before a complete synchronized HPO boundary.
            checkpoint_committed({"hpo_state": None})
            checkpoint_committed(checkpoint(pending=1))
            checkpoint_committed(checkpoint())
    assert stopped.value.code == 128 + signal.SIGTERM
    assert {signum: signal.getsignal(signum) for signum in previous} == previous


@pytest.mark.parametrize("changes", [
    {"attempts_started": 0, "unique_evaluated": 0},
    {"pending": 1},
    {"unique_evaluated": 12, "attempts_started": 12},
    {"status": "COMPLETED"},
    {"status": "INFEASIBLE_EFFICIENCY"},
    {"status": "INFEASIBLE_NUMERICAL"},
    {"attempts_started": 10, "unique_evaluated": 4},
])
def test_zero_pending_or_terminal_budget_does_not_interrupt_finalization(changes):
    with cooperative_worker_stop():
        signal.raise_signal(signal.SIGINT)
        checkpoint_committed(checkpoint(**changes))


@pytest.mark.parametrize("first_signal", [signal.SIGINT, signal.SIGTERM])
def test_later_worker_sigint_forces_exit_without_checkpoint(first_signal):
    with pytest.raises(SystemExit) as stopped:
        with cooperative_worker_stop():
            signal.raise_signal(first_signal)
            signal.raise_signal(signal.SIGINT)
    assert stopped.value.code == 128 + signal.SIGINT


@pytest.mark.parametrize("first_signal", [signal.SIGINT, signal.SIGTERM])
def test_repeated_worker_sigterm_remains_deferred_until_checkpoint(first_signal):
    with cooperative_worker_stop():
        signal.raise_signal(first_signal)
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGTERM)
        checkpoint_committed(checkpoint(pending=1))
        with pytest.raises(SystemExit) as stopped:
            checkpoint_committed(checkpoint())
        assert stopped.value.code == 128 + first_signal


def test_normal_experiment_checkpoint_ignores_hpo_stop_without_worker_context():
    checkpoint_committed(checkpoint())


def test_committed_stop_runs_after_atomic_replace_and_failed_save_preserves_request(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.pt"
    training._save_checkpoint(str(target), {"weight": 1})
    before = target.read_bytes()
    with cooperative_worker_stop():
        signal.raise_signal(signal.SIGTERM)
        with monkeypatch.context() as patch:
            def fail_save(*args, **kwargs):
                raise OSError("injected serialization failure")
            patch.setattr(training.torch, "save", fail_save)
            with pytest.raises(OSError, match="serialization"):
                training._save_checkpoint(str(target), checkpoint())
        assert target.read_bytes() == before
        with pytest.raises(SystemExit) as stopped:
            training._save_checkpoint(str(target), checkpoint())
        assert stopped.value.code == 128 + signal.SIGTERM
        saved = torch.load(target, map_location="cpu", weights_only=False)
        assert saved["hpo_state"] == checkpoint()["hpo_state"]
        assert saved["rng_state"]["schema_version"] == 1


_SIGNALLED_WORKER = textwrap.dedent("""
    import os, runpy, signal, sys
    sys.path.insert(0, sys.argv.pop(1))
    stop_at = int(sys.argv.pop(1))
    from rl_matdesign.utils.candidate_records import CandidateRecorder
    original = CandidateRecorder.start
    def start(self):
        original(self)
        if self.budget.attempts_started == stop_at:
            os.kill(os.getpid(), signal.SIGTERM)
    CandidateRecorder.start = start
    runpy.run_module('rl_matdesign.hpo.worker', run_name='__main__')
""")


def run_worker(config, directory, stop_at=0, resume=False):
    args = [sys.executable, "-B", "-c", _SIGNALLED_WORKER, str(ROOT / "src"), str(stop_at),
            "--config", str(config), "--out", str(directory), "--device", "cpu",
            "--skip-generation", "--save-checkpoint-freq", "1"]
    if resume:
        args.append("--resume-training")
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("method,target,stop_at,batch", [
    ("dqn", "mc", 1, 2),         # request during paid warmup reservation
    ("dqn", "bootstrap", 3, 2),  # request after warmup, before predictor/update
    ("a2c", "bootstrap", 2, 4),  # one collected episode, partial batch of four
])
def test_real_worker_signal_finishes_unit_then_resume_matches_control(tmp_path, method, target, stop_at, batch):
    config = _config()
    config["groups"][0].update(
        species_set=["Ni", "Fe", "Co", "Cr", "Mn", "Cu", "Zn", "Ti"],
        fraction_set=[f"{value / 10:.1f}" for value in range(1, 9)], n_components=3)
    config.update(method=method, dqn_target_mode=target, dqn_warmup_eps=2,
                  dqn_batch_size=2, dqn_eps_min=1., pg_warmup_eps=2, pg_batch_eps=batch,
                  train_count_duplicates=False, train_budget=12,
                  hpo_recording=True, hpo_phase="train", hpo_target_unique=12,
                  hpo_ratio_limit=1.5, num_gen_eps=0, exploration_gen_eps=0)
    config.pop("pg_num_iters")
    config.pop("dqn_num_train_eps")
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config))
    resumed, control = tmp_path / "resumed", tmp_path / "control"

    first = run_worker(source, resumed, stop_at=stop_at)
    assert first.returncode == 128 + signal.SIGTERM, (first.stdout, first.stderr)
    saved = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    ledger = json.loads((resumed / "candidate_budget.json").read_text())
    assert ledger["status"] == "INTERRUPTED"
    assert ledger["pending"] == 0
    assert ledger["attempts_started"] == saved["hpo_state"]["budget"]["attempts_started"]
    assert ledger["unique_evaluated"] == saved["hpo_state"]["budget"]["unique_evaluated"]
    assert ledger["attempts_started"] >= (batch if method == "a2c" else 3)
    if method == "a2c":
        assert saved["iteration"] == 1
        assert saved["episodes_completed"] == batch
    else:
        assert saved["training_episodes_completed"] == 1
    prefix = (resumed / "candidate_records.jsonl").read_bytes()

    continued = run_worker(source, resumed, resume=True)
    assert continued.returncode == 0, (continued.stdout, continued.stderr)
    baseline = run_worker(source, control)
    assert baseline.returncode == 0, (baseline.stdout, baseline.stderr)
    assert (resumed / "candidate_records.jsonl").read_bytes().startswith(prefix)
    assert (resumed / "candidate_records.jsonl").read_bytes() == (control / "candidate_records.jsonl").read_bytes()
    actual = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    expected = torch.load(control / "checkpoint.pt", map_location="cpu", weights_only=False)
    state_keys = (["policy_state", "value_net_state", "opt_actor_state", "opt_critic_state", "iteration"]
                  if method == "a2c" else ["qnet_state", "target_net_state", "opt_state", "buffer"])
    for name in (*state_keys, "rng_state", "training_budget_state", "hpo_state"):
        _assert_nested_equal(actual[name], expected[name])
    assert not list(resumed.glob("checkpoint-ep*.pt"))
    assert not list(resumed.glob("checkpoint-iter*.pt"))


def test_terminal_target_signal_finishes_final_model_export(tmp_path):
    config = _config()
    config.update(method="a2c", pg_warmup_eps=2, pg_batch_eps=2, dqn_warmup_eps=1,
                  train_count_duplicates=False, train_budget=2,
                  hpo_recording=True, hpo_phase="train", hpo_target_unique=2,
                  hpo_ratio_limit=20., num_gen_eps=0, exploration_gen_eps=0)
    config.pop("pg_num_iters")
    config.pop("dqn_num_train_eps")
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config))
    out = tmp_path / "terminal"
    result = run_worker(source, out, stop_at=1)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads((out / "candidate_budget.json").read_text())["status"] == "COMPLETED"
    assert all((out / name).is_file() for name in ("policy.pt", "value_net.pt", "std_scaler.bin"))


_TERMINAL_COORDINATOR_INTERRUPT = textwrap.dedent("""
    import os, runpy, sys, time
    from pathlib import Path
    from rl_matdesign.utils.candidate_records import CandidateRecorder
    from rl_matdesign.utils import hpo_stop
    original = CandidateRecorder.success
    def success(self, episode):
        result = original(self, episode)
        if self.budget.unique_evaluated == self.budget.target_unique:
            Path(os.environ['HPO_TEST_TERMINAL_READY']).touch()
            deadline = time.monotonic() + 20
            while hpo_stop._active_stop.signum is None:
                if time.monotonic() > deadline:
                    raise RuntimeError('coordinator did not forward its stop request')
                time.sleep(.01)
        return result
    CandidateRecorder.success = success
    runpy.run_module('rl_matdesign.hpo.worker', run_name='__main__')
""")


def test_serial_coordinator_interrupt_after_last_sample_can_finalize_without_rebuying(tmp_path, monkeypatch):
    from rl_matdesign.hpo.controller import Campaign
    from rl_matdesign.hpo.spec import load_plan
    from rl_matdesign.hpo.restart import inspect_restart

    scenario = _config()
    scenario.pop("predictor")
    scenario.update(properties=[{"name": "energy", "predictor": "dummy", "direction": "min"}],
                    pg_warmup_eps=2, dqn_warmup_eps=1, pg_batch_eps=4)
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    budget = dict(mode="absolute_unique", min_unique=4, initial_unique=4, max_unique=4)
    hpo = {
        "base_config": "scenario.yaml", "method": "a2c", "random_seed": 42,
        "evaluation": {"threshold": .5, "max_attempts_per_unique": 20},
        "training": {"repeats": 1, "budget": budget, "parameter_ranges": {
            "pg_batch_eps": {"type": "categorical", "choices": [4]}}},
        "generation": {"repeats": 1, "budget": budget, "initial_values": [1.]},
    }
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump(hpo))
    plan = load_plan(source)
    campaign = Campaign(plan, tmp_path / "campaign")
    marker = tmp_path / "terminal-ready"
    original_popen = subprocess.Popen
    calls = []

    class InterruptingWait:
        def __init__(self, process):
            self.process = process
            self.interrupted = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            if not self.interrupted:
                deadline = time.monotonic() + 30
                while not marker.exists() and self.process.poll() is None:
                    assert time.monotonic() < deadline, "terminal worker readiness timed out"
                    time.sleep(.01)
                if marker.exists():
                    self.interrupted = True
                    raise KeyboardInterrupt("test coordinator stop during terminal batch")
            return self.process.wait(timeout=timeout)

    def launch(argv, **kwargs):
        calls.append(argv)
        environment = dict(kwargs["env"], HPO_TEST_TERMINAL_READY=str(marker))
        process = original_popen([argv[0], "-B", "-c", _TERMINAL_COORDINATOR_INTERRUPT, *argv[3:]],
                                 **{**kwargs, "env": environment})
        return InterruptingWait(process)

    with monkeypatch.context() as patch:
        patch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", launch)
        with pytest.raises(KeyboardInterrupt, match="terminal batch"):
            campaign.runner({"pg_batch_eps": 4}, 4, 17, "train")
    jobdir = next((campaign.out / "jobs").iterdir())
    preserved = json.loads((jobdir / "result.json").read_text())
    assert preserved["status"] == "INTERRUPTED"
    assert preserved["returncode"] == 0
    assert preserved["ledger"]["status"] == "COMPLETED"
    assert preserved["coordinator_interruption"]["type"] == "KeyboardInterrupt"
    assert "execution_failure" not in preserved
    assert all((jobdir / name).is_file() for name in ("policy.pt", "value_net.pt", "std_scaler.bin"))
    paid = (jobdir / "candidate_budget.json").read_bytes()
    archive = (jobdir / "candidate_records.jsonl").read_bytes()
    assert inspect_restart(campaign.out)["ready"] is True
    with monkeypatch.context() as patch:
        patch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen",
                      lambda *args, **kwargs: pytest.fail("terminal recovery launched another worker"))
        completed = Campaign(plan, campaign.out).runner({"pg_batch_eps": 4}, 4, 17, "train")
    assert completed["status"] == "COMPLETED"
    assert completed["recovered_from_terminal_artifacts"] is True
    assert (jobdir / "candidate_budget.json").read_bytes() == paid
    assert (jobdir / "candidate_records.jsonl").read_bytes() == archive
    assert len(calls) == 1
