"""Real old-source HPO jobs continue through the audited protocol upgrade.

Set RLMAT_PERFORMANCE_LEGACY_SRC to an immutable baseline ``src`` directory when
testing a source export without Git history. Normal checkouts materialize the
recorded release from Git; no checkpoint or compatibility evidence is mocked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap

import pytest
import torch
import yaml

from rl_matdesign.hpo.algorithm_profiles import initial_parameters
from rl_matdesign.utils import source_compatibility as compatibility
from rl_matdesign.utils.checkpoint_store import load_checkpoint
from tests.test_hpo_checkpoint_retention import _assert_nested_equal
from tests.test_hpo_cooperative_restart import _plan


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX cooperative signals")


@pytest.fixture(scope="module", params=["baseline", "v1"])
def legacy_source(tmp_path_factory, request):
    version = request.param
    revision = compatibility.BASELINE_REVISION if version == "baseline" else compatibility.V1_REVISION
    supplied = os.environ.get("RLMAT_PERFORMANCE_LEGACY_SRC" if version == "baseline" else "RLMAT_PERFORMANCE_V1_SRC")
    if supplied:
        source = Path(supplied).resolve()
        assert source.is_dir(), f"Missing legacy source: {source}"
    else:
        archived = subprocess.run(
            ["git", "archive", revision, "src"],
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
    expected = {
        "package": compatibility.BASELINE_PACKAGE_SHA256 if version == "baseline" else compatibility.V1_PACKAGE_SHA256,
        "execution": compatibility.BASELINE_EXECUTION_SHA256 if version == "baseline" else compatibility.V1_EXECUTION_SHA256,
    }
    assert compatibility.inventory_digests(compatibility.source_inventory(source)) == expected
    extras = os.environ.get("RLMAT_PERFORMANCE_LEGACY_PREDICTORS_DIR")
    if extras:
        copied = tmp_path_factory.mktemp("source-with-legacy-predictors") / "src"
        shutil.copytree(source, copied)
        for name, sha in compatibility.LEGACY_PREDICTOR_FILES.items():
            origin = Path(extras) / Path(name).name
            assert hashlib.sha256(origin.read_bytes()).hexdigest() == sha
            shutil.copyfile(origin, copied / name)
        source = copied
        expected = {
            "package": compatibility.LEGACY_PACKAGE_SHA256 if version == "baseline" else compatibility.V1_LEGACY_PACKAGE_SHA256,
            "execution": compatibility.LEGACY_EXECUTION_SHA256 if version == "baseline" else compatibility.V1_LEGACY_EXECUTION_SHA256,
        }
        assert compatibility.inventory_digests(compatibility.source_inventory(source)) == expected
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


@pytest.mark.parametrize("method", ["dqn-mc", "dqn-bootstrap"])
def test_legacy_hpo_upgrade_preserves_paid_prefix_and_native_continuation(tmp_path, legacy_source, method):
    plan, _ = _plan(tmp_path, method)
    parameters = initial_parameters(plan)
    parameters["dqn_batch_size"] = 8
    if method == "dqn-bootstrap":
        parameters["dqn_target_update_freq"] = 2
    hpo = tmp_path / "hpo.yaml"
    config = yaml.safe_load(hpo.read_text())
    config["training"]["parameter_ranges"] = {
        name: {"type": "categorical", "choices": [value]}
        for name, value in parameters.items()
    }
    hpo.write_text(yaml.safe_dump(config, sort_keys=False))
    control, interrupted = tmp_path / "old-control", tmp_path / "old-interrupted"
    for out, stop in ((control, False), (interrupted, True)):
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
    stopped = load_checkpoint(job / "checkpoint.pt", map_location="cpu")
    assert stopped["training_episodes_completed"] >= 3
    prefix = (job / "candidate_records.jsonl").read_bytes()
    assert stopped["hpo_state"]["budget"]["unique_evaluated"] == 5

    if method == "dqn-bootstrap":
        # Batched target matrices can round differently from old scalar calls.
        # Compare two upgraded continuations of the SAME paid old checkpoint;
        # one is interrupted once more under v2. Do not claim old/new bit parity.
        shutil.rmtree(control)
        shutil.copytree(interrupted, control)
        _success(_cli(ROOT / "src", tmp_path, "restart", "--out", str(control)))

    before = _snapshot(interrupted)
    checked = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted), "--check-only")
    _success(checked)
    assert "SYNCHRONIZED" in checked.stdout
    assert compatibility.UPGRADE_ID in checked.stdout
    assert _snapshot(interrupted) == before

    if method == "dqn-bootstrap":
        paused_again = _cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted), stop=True)
        assert paused_again.returncode == 130, paused_again.stdout + paused_again.stderr
    _success(_cli(ROOT / "src", tmp_path, "restart", "--out", str(interrupted)))
    assert json.loads((interrupted / "state.json").read_text())["status"] == "COMPLETED"
    assert (job / "candidate_records.jsonl").read_bytes().startswith(prefix)
    control_jobs = list((control / "jobs").glob("*/checkpoint.pt"))
    assert len(control_jobs) == 1
    control_job = control_jobs[0].parent
    expected = load_checkpoint(control_job / "checkpoint.pt", map_location="cpu")
    actual = load_checkpoint(job / "checkpoint.pt", map_location="cpu")
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
