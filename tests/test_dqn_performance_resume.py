"""Real old-source HPO jobs obey the sealed performance-release boundary.

An actually approved runtime must resume exactly. A later semantic runtime must
reject both check-only and restart without touching the historical campaign.
Current-runtime restart equivalence is covered by test_hpo_cooperative_restart.

Set RLMAT_PERFORMANCE_LEGACY_SRC to an immutable baseline ``src`` directory when
testing a source export without Git history. Normal checkouts materialize the
recorded release from Git; no checkpoint or compatibility evidence is mocked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap

import pytest
import torch
import yaml

from rl_matdesign.utils import source_compatibility as compatibility
from tests.test_hpo_checkpoint_retention import _assert_nested_equal


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX cooperative signals")


@pytest.fixture(scope="module")
def legacy_source(tmp_path_factory):
    supplied = os.environ.get("RLMAT_PERFORMANCE_LEGACY_SRC")
    if supplied:
        source = Path(supplied).resolve()
        assert source.is_dir(), f"Missing legacy source: {source}"
    else:
        archived = subprocess.run(
            ["git", "archive", compatibility.BASELINE_REVISION, "src"],
            cwd=ROOT, capture_output=True, timeout=30,
        )
        if archived.returncode:
            pytest.skip("Baseline Git revision unavailable; set RLMAT_PERFORMANCE_LEGACY_SRC")
        directory = tmp_path_factory.mktemp("dqn-legacy-source")
        with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as archive:
            for member in archive.getmembers():
                assert member.name.startswith("src/") or member.name == "src"
                assert member.isfile() or member.isdir()
                assert ".." not in Path(member.name).parts
            archive.extractall(directory, filter="data")
        source = directory / "src"
    assert compatibility.inventory_digests(compatibility.source_inventory(source)) == {
        "package": compatibility.BASELINE_PACKAGE_SHA256,
        "execution": compatibility.BASELINE_EXECUTION_SHA256,
    }
    return source


COORDINATOR = textwrap.dedent("""
    import os, subprocess, textwrap
    from rl_matdesign.hpo import job_runner
    if os.environ.get('STOP_LEGACY_JOB') == '1':
        worker = textwrap.dedent('''
            import os, runpy, signal, time
            from rl_matdesign.utils import hpo_stop
            from rl_matdesign.utils.candidate_records import CandidateRecorder
            original = CandidateRecorder.success
            sent = False
            def success(self, episode):
                global sent
                fresh = original(self, episode)
                if not sent and self.budget.unique_evaluated >= 5:
                    sent = True
                    os.kill(os.getppid(), signal.SIGTERM)
                    deadline = time.monotonic() + 20
                    while hpo_stop._active_stop.signum is None:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Coordinator did not forward the stop signal')
                        time.sleep(.01)
                return fresh
            CandidateRecorder.success = success
            runpy.run_module('rl_matdesign.hpo.worker', run_name='__main__')
        ''')
        class WorkerHook:
            def __getattr__(self, name):
                return getattr(subprocess, name)
            def Popen(self, argv, **kwargs):
                assert argv[1:3] == ['-m', 'rl_matdesign.hpo.worker']
                return subprocess.Popen([argv[0], '-B', '-c', worker, *argv[3:]], **kwargs)
        job_runner.subprocess = WorkerHook()
    from rl_matdesign.cli import main
    main()
""")


def _cli(source, directory, *arguments, stop=False):
    environment = dict(
        os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="", STOP_LEGACY_JOB="1" if stop else "0",
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", COORDINATOR, "hpo", *arguments],
        cwd=directory, env=environment, capture_output=True, text=True, timeout=180,
    )


def _success(result):
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]


def _snapshot(directory):
    return {str(path.relative_to(directory)): (
        path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in directory.rglob("*") if path.is_file()}


def _write_historical_inputs(path, method):
    """Frozen input syntax for the archived baseline, never today's loader."""
    scenario = {
        "method": "dqn",
        "properties": [{"name": "energy", "predictor": "dummy", "direction": "min", "objective": "mean"}],
        "groups": [{"name": "metal", "kind": "composition",
                    "species_set": ["Ni", "Fe", "Co", "Cr", "Mn", "Cu", "Zn", "Ti"],
                    "fraction_set": [f"{i / 10:.1f}" for i in range(1, 9)], "n_components": 3}],
        "eval_seed": 13, "pg_warmup_eps": 2, "pg_batch_eps": 2,
        "dqn_hidden_dim": 16, "dqn_buffer_size": 100, "dqn_grad_steps_per_ep": 2,
        "gen_temperature": 1., "num_gen_eps": 0, "exploration_gen_eps": 0,
    }
    parameters = {"dqn_batch_size": 8, "dqn_lr": .001, "dqn_eps_min": .2,
                  "warmup_budget_fraction": .2, "epsilon_anneal_budget_ratio": 1.}
    if method == "dqn-bootstrap":
        parameters["dqn_target_update_freq"] = 2
    budget = {"mode": "absolute_unique", "min_unique": 12, "initial_unique": 12, "max_unique": 12}
    (path / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False))
    config = {
        "base_config": "scenario.yaml", "method": method, "random_seed": 42, "parallel_jobs": 1,
        "evaluation": {"threshold": .5, "max_attempts_per_unique": 1.5},
        "training": {"repeats": 1, "budget": budget, "parameter_ranges": {
            name: {"type": "categorical", "choices": [value]} for name, value in parameters.items()}},
        "generation": {"repeats": 1, "budget": budget, "initial_values": [1.]},
    }
    hpo = path / "hpo.yaml"
    hpo.write_text(yaml.safe_dump(config, sort_keys=False))
    return hpo


@pytest.mark.parametrize("method", ["dqn-mc", "dqn-bootstrap"])
def test_legacy_hpo_restart_obeys_sealed_release_boundary(tmp_path, legacy_source, method):
    installed = compatibility.inventory_digests(compatibility.source_inventory(ROOT / "src"))
    approvals = [compatibility.approved_performance_upgrade(baseline, installed[kind], kind=kind)
                 for kind, baseline in (("package", compatibility.BASELINE_PACKAGE_SHA256),
                                        ("execution", compatibility.BASELINE_EXECUTION_SHA256))]
    approved_release = all(evidence is not None for evidence in approvals)
    hpo = _write_historical_inputs(tmp_path, method)
    control, interrupted = tmp_path / "old-control", tmp_path / "old-interrupted"
    # The reference full run is needed only for an authenticated target. An
    # unsealed runtime is tested against a real stopped legacy checkpoint.
    runs = ((control, False), (interrupted, True)) if approved_release else ((interrupted, True),)
    for out, stop in runs:
        result = _cli(legacy_source, tmp_path, "run", "--hpo-config", str(hpo),
                      "--out", str(out), "--stage", "ranges", stop=stop)
        if stop:
            assert result.returncode == 130, result.stdout[-6000:] + result.stderr[-6000:]
        else:
            _success(result)
    jobs = list((interrupted / "jobs").glob("*/checkpoint.pt"))
    assert len(jobs) == 1
    job = jobs[0].parent
    # These are locally generated trusted fixtures, never user-provided pickle.
    stopped = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert stopped["training_episodes_completed"] >= 3
    prefix = (job / "candidate_records.jsonl").read_bytes()
    assert stopped["hpo_state"]["budget"]["unique_evaluated"] == 5

    before = _snapshot(interrupted)
    checked = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted), "--check-only")
    if not approved_release:
        assert all(evidence is None for evidence in approvals)
        assert checked.returncode == 2, checked.stdout + checked.stderr
        # Retired configuration can reject before the source-upgrade guard.
        # Keep the archived fixture unchanged; every rejection must preserve it.
        expected_rejections = (
            "exact audited DQN performance release",
            "Group 'metal' has unused or removed parameters: ['fraction_set', 'n_components']",
            "Removed top-level parameters: ['gen_temperature']",
            "Exploration parameters ['gen_temperature'] require exploration_adaptive=false",
        )
        assert any(reason in checked.stderr for reason in expected_rejections)
        assert _snapshot(interrupted) == before
        rejected = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted))
        assert rejected.returncode == 2, rejected.stdout + rejected.stderr
        assert any(reason in rejected.stderr for reason in expected_rejections)
        assert _snapshot(interrupted) == before
        assert not (interrupted / "source_upgrades").exists()
        return
    _success(checked)
    assert "SYNCHRONIZED" in checked.stdout
    assert compatibility.UPGRADE_ID in checked.stdout
    assert _snapshot(interrupted) == before

    _success(_cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted)))
    assert json.loads((interrupted / "state.json").read_text())["status"] == "COMPLETED"
    assert (job / "candidate_records.jsonl").read_bytes().startswith(prefix)
    control_jobs = list((control / "jobs").glob("*/checkpoint.pt"))
    assert len(control_jobs) == 1
    control_job = control_jobs[0].parent
    expected = torch.load(control_job / "checkpoint.pt", map_location="cpu", weights_only=False)
    actual = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key in ("type", "scaler_fingerprint", "environment_contract", "dqn_buffer_size",
                "qnet_state", "target_net_state", "opt_state", "buffer", "eps", "dp_cache",
                "training_episodes_completed", "episodes_completed", "rng_state",
                "training_budget_state", "training_budget_history_complete", "hpo_state"):
        _assert_nested_equal(actual[key], expected[key])
    for key in ("n_calls", "seen", "best_mean", "diagnostic_requests",
                "diagnostic_failures", "phase", "phase_costs"):
        _assert_nested_equal(actual["predictor_ledger"][key], expected["predictor_ledger"][key])
    assert (job / "candidate_records.jsonl").read_bytes() == (control_job / "candidate_records.jsonl").read_bytes()
    assert json.loads((job / "candidate_budget.json").read_text()) == json.loads((control_job / "candidate_budget.json").read_text())
    assert (job / "checkpoint.pt").is_file() and not (job / "checkpoint.pt").is_symlink()
    assert not list(job.glob("checkpoint-ep*.pt"))
    assert len(list((interrupted / "source_upgrades").glob("*.json"))) == 1
