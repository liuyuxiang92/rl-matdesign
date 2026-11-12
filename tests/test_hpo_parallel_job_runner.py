"""Concurrent workers retain ordered calibration, durable costs and safe shutdown."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from rl_matdesign.hpo.artifacts import atomic_json, file_digest, read_json
from rl_matdesign.hpo.job_runner import JobError, JobRunner
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_job_runner import FakeWorker, public_plan_path


class OrderedWorker:
    """Seed 11 waits for release while other seeds can finish first."""
    def __init__(self, monkeypatch):
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.calls = []
        owner = self

        class Process:
            returncode = 0

            def __init__(self, args, **kwargs):
                self.seed = int(args[args.index("--train-seed") + 1])
                self.directory = Path(args[args.index("--out") + 1])
                cfg = yaml.safe_load(Path(args[args.index("--config") + 1]).read_text())
                self.budget = cfg["hpo_target_unique"]
                owner.calls.append((self.seed, kwargs))

            def wait(self, timeout=None):
                if self.seed == 11:
                    owner.first_started.set()
                    assert owner.release_first.wait(5), "test must release the first worker"
                values = [-4, -3, -2, -1, 0, 1, 2, 3]
                multiplier = 1 if self.seed == 11 else 100
                rows = [{"identity": str(i), "properties": {"energy": values[i] * multiplier},
                         "constraint_passed": True, "units": {"energy": "eV"}}
                        for i in range(self.budget)]
                (self.directory / "candidate_records.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n")
                atomic_json(self.directory / "candidate_budget.json", {
                    "status": "COMPLETED", "attempts_started": self.budget,
                    "unique_evaluated": self.budget, "target_unique": self.budget})
                (self.directory / "policy.pt").write_text(f"policy-{self.seed}")
                (self.directory / "std_scaler.bin").write_text("scaler")
                return self.returncode

        monkeypatch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", Process)


def test_raw_jobs_finish_out_of_order_without_calibrating_or_loading_archives(tmp_path, monkeypatch):
    plan = load_plan(public_plan_path(tmp_path))
    original = copy.deepcopy(plan)
    worker = OrderedWorker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "campaign")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner.execute, {"pg_batch_eps": 2}, 8, 11, "train")
        assert worker.first_started.wait(5)
        second = pool.submit(runner.execute, {"pg_batch_eps": 2}, 8, 29, "train")
        try:
            second_raw = second.result(timeout=5)
            assert not first.done()
            assert plan == original
            assert not (runner.out / "quality_calibration.json").exists()
            assert not list(runner.out.glob("jobs/*/result.json"))
            assert "records" not in second_raw and "ledger" not in second_raw
            assert len(json.dumps(second_raw)) < 5000
        finally:
            worker.release_first.set()
        first_raw = first.result(timeout=5)
    first_result = runner.finalize(first_raw)
    second_result = runner.finalize(second_raw)
    assert first_result["quality_normalization"]["scale"] == 3.5
    assert second_result["quality_normalization"] == first_result["quality_normalization"]
    calibration = read_json(runner.out / "quality_calibration.json")
    assert calibration["source_job_id"] == first_result["job_id"]
    assert all(kwargs["start_new_session"] is True for _, kwargs in worker.calls)
    assert first_result["wall_seconds"] >= first_result["current_execution_seconds"] >= 0
    assert not runner._processes


def test_duplicate_concurrent_requests_spend_one_job_and_keep_measured_timing(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    worker = OrderedWorker(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner.execute, {"pg_batch_eps": 2}, 8, 11, "train")
        assert worker.first_started.wait(5)
        second = pool.submit(runner.execute, {"pg_batch_eps": 2}, 8, 11, "train")
        worker.release_first.set()
        payloads = [first.result(timeout=5), second.result(timeout=5)]
    results = [runner.finalize(payload) for payload in payloads]
    assert len(worker.calls) == 1
    assert results[0] == results[1]
    assert results[1]["wall_seconds"] is not None
    assert results[1]["timing_status"] == "measured"


def test_finalization_rechecks_cached_artifacts_after_waiting(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    result = runner({"pg_batch_eps": 2}, 8, 11, "train")
    payload = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    Path(result["checkpoint_paths"]["policy.pt"]).write_text("changed while queued")
    with pytest.raises(JobError, match="Completed job artifact changed"):
        runner.finalize(payload)
    assert len(worker.calls) == 1


def test_parallel_generation_preserves_shared_frozen_model(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    FakeWorker(monkeypatch)
    trained = runner({"pg_batch_eps": 2}, 8, 11, "train")
    before = {name: file_digest(path) for name, path in trained["checkpoint_paths"].items()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(runner.execute, {"pg_batch_eps": 2, "gen_temperature": 1},
                               4, seed, "generation", trained) for seed in (401, 402, 403)]
        raw = [future.result(timeout=5) for future in futures]
    results = [runner.finalize(payload) for payload in raw]
    assert all(result["eligible_for_selection"] for result in results)
    assert len({result["job_id"] for result in results}) == 3
    assert before == {name: file_digest(path) for name, path in trained["checkpoint_paths"].items()}


def test_malformed_raw_archive_raises_during_ordered_finalization(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    FakeWorker(monkeypatch)
    raw = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    archive = Path(raw["directory"]) / "candidate_records.jsonl"
    archive.write_text(archive.read_text() + '{"truncated":')
    with pytest.raises(JobError, match="Incomplete record archive preserved"):
        runner.finalize(raw)
    assert read_json(Path(raw["directory"]) / "candidate_budget.json")["attempts_started"] == 8
    assert not (runner.out / "quality_calibration.json").exists()


# A real, lightweight worker exercises process groups and the same ledger cleanup
# used by experiments, without loading tensor libraries or spending model calls.
_CANCELLABLE_WORKER = r'''
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from rl_matdesign.utils.candidate_records import CandidateRecorder

def terminate(signum, frame):
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
directory = Path(sys.argv[1])
with ExitStack() as resources:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    resources.callback(child.wait)
    recorder = CandidateRecorder(directory, 8, 1.5, "train")
    resources.callback(recorder.close)
    recorder.start()
    recorder.success(SimpleNamespace(terminal_key="one", formula="one",
                                    prediction_record={"obj_energy_mean": -2}))
    (directory / "ready.json").write_text(json.dumps({"pid": os.getpid(),
        "pgid": os.getpgrp(), "child": child.pid}))
    while True:
        time.sleep(.1)
'''


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cancellation")
def test_cancel_all_reaps_workers_and_children_and_preserves_attempts(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    real_popen = subprocess.Popen
    processes = []

    def launch(args, **kwargs):
        directory = args[args.index("--out") + 1]
        process = real_popen([sys.executable, "-c", _CANCELLABLE_WORKER, directory], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", launch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runner.execute, {"pg_batch_eps": 2}, 8, seed, "train")
                   for seed in (11, 29)]
        try:
            deadline = time.monotonic() + 10
            while len(list(runner.out.glob("jobs/*/ready.json"))) < 2:
                assert time.monotonic() < deadline, "worker readiness timed out"
                time.sleep(.01)
            ready = [read_json(path) for path in runner.out.glob("jobs/*/ready.json")]
            assert all(row["pid"] == row["pgid"] != os.getpgrp() for row in ready)
        finally:
            runner.cancel_all()
        payloads = [future.result(timeout=5) for future in futures]
    assert all(process.poll() is not None for process in processes)
    for row in ready:
        with pytest.raises(ProcessLookupError):
            os.kill(row["child"], 0)
    assert not runner._processes
    results = [runner.finalize(payload) for payload in payloads]
    assert all(result["status"] == "INTERRUPTED" for result in results)
    assert all(result["ledger"]["status"] == "INTERRUPTED" for result in results)
    assert all(result["ledger"]["attempts_started"] == 1 for result in results)
    assert all(result["ledger"]["unique_evaluated"] == 1 for result in results)
    assert not (runner.out / "quality_calibration.json").exists()
    with pytest.raises(CancelledError):
        runner.execute({"pg_batch_eps": 2}, 8, 53, "train")
    assert len(processes) == 2
    restored = JobRunner(load_plan(source), runner.out)
    with pytest.raises(JobError, match="without a resumable checkpoint"):
        restored.execute({"pg_batch_eps": 2}, 8, 11, "train")
    assert len(processes) == 2


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(7), RuntimeError("wait failed")])
def test_direct_call_persists_interrupt_and_reraises_original_failure(tmp_path, monkeypatch, failure):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    original_process = subprocess.Popen

    class InterruptingProcess(original_process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_wait = True

        def wait(self, timeout=None):
            if self.first_wait:
                self.first_wait = False
                raise failure
            self.returncode = -signal.SIGTERM
            return self.returncode

    monkeypatch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", InterruptingProcess)
    monkeypatch.setattr(runner, "_signal_process", lambda process, sig: None)
    with pytest.raises(type(failure)) as raised:
        runner({"pg_batch_eps": 2}, 8, 11, "train")
    assert raised.value is failure
    result = read_json(worker.calls[0]["directory"] / "result.json")
    expected_status = "SYSTEM_ERROR" if isinstance(failure, RuntimeError) else "INTERRUPTED"
    assert result["status"] == expected_status
    assert result["ledger"]["attempts_started"] == 8
    assert not runner._processes
    assert not (runner.out / "quality_calibration.json").exists()


def test_failed_worker_does_not_calibrate_or_become_selectable(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    FakeWorker(monkeypatch)
    monkeypatch.setattr(subprocess.Popen, "returncode", 17)
    failed = runner.finalize(runner.execute({"pg_batch_eps": 2}, 8, 11, "train"))
    assert failed["status"] == "SYSTEM_ERROR"
    assert failed["returncode"] == 17
    assert not failed["eligible_for_selection"]
    assert failed["ledger"]["attempts_started"] == 8
    assert not (runner.out / "quality_calibration.json").exists()
    monkeypatch.setattr(subprocess.Popen, "returncode", 0)
    success = runner.finalize(runner.execute({"pg_batch_eps": 2}, 8, 29, "train"))
    assert success["eligible_for_selection"]
    assert read_json(runner.out / "quality_calibration.json")["source_job_id"] == success["job_id"]


def test_execution_uses_current_ranges_without_mutating_shared_plan(tmp_path, monkeypatch):
    plan = load_plan(public_plan_path(tmp_path))
    runner = JobRunner(plan, tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    plan["search_space"]["pg_batch_eps"] = {"type": "categorical", "choices": [4]}
    expected = copy.deepcopy(plan)
    raw = runner.execute({}, 8, 11, "train")
    assert plan == expected
    assert worker.calls[0]["config"]["pg_batch_eps"] == 4
    assert runner.finalize(raw)["eligible_for_selection"]


def test_preserve_after_degenerate_reference_never_calibrates_or_scores(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    worker.values = [0]
    first = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    worker.values = [-4, -3, -2, -1, 0, 1, 2, 3]
    later = runner.execute({"pg_batch_eps": 2}, 8, 29, "train")
    assert runner.finalize(first)["status"] == "QUALITY_CALIBRATION_REQUIRED"
    original_plan = copy.deepcopy(runner.plan)

    def no_scoring(*args, **kwargs):
        pytest.fail("cleanup must not calibrate or score later jobs")

    monkeypatch.setattr("rl_matdesign.hpo.job_runner.calibrate_quality_scale", no_scoring)
    monkeypatch.setattr("rl_matdesign.hpo.job_runner.score_records", no_scoring)
    preserved = runner.preserve(later)
    assert preserved["status"] == "INCOMPLETE"
    assert preserved["finalization_pending"] and not preserved["eligible_for_selection"]
    assert preserved["raw_metrics"] == preserved["normalized_metrics"] == {}
    assert "quality_normalization" not in preserved
    assert preserved["ledger"]["attempts_started"] == 8
    assert preserved["wall_seconds"] == later["elapsed"]
    assert runner.plan == original_plan
    assert not (runner.out / "quality_calibration.json").exists()
    restored = JobRunner(load_plan(source), runner.out)
    assert restored.plan["objective"]["normalization"]["scale"] is None
    assert len(worker.calls) == 2


def test_preserve_then_ordered_resume_uses_first_reference_and_retains_later_time(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    first = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    first_dir = Path(first["directory"])
    # Recreate an interrupted but synchronized training checkpoint after 4 of 8
    # unique observations; the fake runtime completes its remaining budget.
    archive = first_dir / "candidate_records.jsonl"
    archive.write_text("\n".join(archive.read_text().splitlines()[:4]) + "\n")
    atomic_json(first_dir / "candidate_budget.json", {
        "status": "INTERRUPTED", "attempts_started": 4, "unique_evaluated": 4,
        "target_unique": 8})
    (first_dir / "checkpoint.pt").write_text("synchronized checkpoint verified by runtime")
    first.update(interrupted=True, returncode=-signal.SIGTERM, elapsed=1.25)
    worker.values = [-400, -300, -200, -100, 0, 100, 200, 300]
    later = runner.execute({"pg_batch_eps": 2}, 8, 29, "train")
    later["elapsed"] = 2.5
    first_saved, later_saved = runner.preserve(first), runner.preserve(later)
    assert first_saved["status"] == "INTERRUPTED"
    assert later_saved["status"] == "INCOMPLETE"
    assert not (runner.out / "quality_calibration.json").exists()

    restored = JobRunner(load_plan(source), runner.out)
    worker.values = [-4, -3, -2, -1, 0, 1, 2, 3]
    resumed_first = restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert "--resume-training" in worker.calls[-1]["args"]
    assert resumed_first["wall_seconds"] == 1.25 + resumed_first["current_execution_seconds"]
    assert resumed_first["quality_normalization"]["scale"] == 3.5
    resumed_later = restored({"pg_batch_eps": 2}, 8, 29, "train")
    assert len(worker.calls) == 3  # Only the interrupted first worker reruns.
    assert resumed_later["eligible_for_selection"]
    assert resumed_later["quality_normalization"]["scale"] == 3.5
    assert resumed_later["wall_seconds"] == later_saved["wall_seconds"] == 2.5
    assert resumed_later["current_execution_seconds"] == 0
    assert resumed_later["timing_status"] == "recovered_from_preserved_execution"
    assert resumed_later["ledger"] == later_saved["ledger"]
    assert "finalization_pending" not in resumed_later
    assert read_json(runner.out / "quality_calibration.json")["source_job_id"] == resumed_first["job_id"]


@pytest.mark.parametrize("persist", ["finalize", "preserve"])
def test_known_nonzero_worker_cannot_be_blessed_by_terminal_recovery(tmp_path, monkeypatch, persist):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    monkeypatch.setattr(subprocess.Popen, "returncode", 17)
    raw = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    result = getattr(runner, persist)(raw)
    assert result["returncode"] == 17
    assert result["ledger"]["status"] == "COMPLETED"
    monkeypatch.setattr(subprocess.Popen, "returncode", 0)
    restored = JobRunner(load_plan(source), runner.out)
    with pytest.raises(JobError, match="terminal ledger but its execution failed"):
        restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert len(worker.calls) == 1
    assert read_json(Path(raw["directory"]) / "result.json") == result
    assert not (runner.out / "quality_calibration.json").exists()


def test_preserve_records_execution_failure_without_rethrowing_or_clearing_it(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    raw = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    raw["execution_error"] = RuntimeError("cannot safely finish execution")
    result = runner.preserve(raw)
    assert result["execution_failure"] == {
        "type": "RuntimeError", "message": "cannot safely finish execution"}
    restored = JobRunner(load_plan(source), runner.out)
    with pytest.raises(JobError, match="terminal ledger but its execution failed"):
        restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert len(worker.calls) == 1
    assert not (runner.out / "quality_calibration.json").exists()


def test_preserved_completed_archive_hash_is_checked_before_recovery(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    result = runner.preserve(runner.execute({"pg_batch_eps": 2}, 8, 11, "train"))
    Path(result["records_path"]).write_text("changed after cleanup")
    restored = JobRunner(load_plan(source), runner.out)
    with pytest.raises(JobError, match="Preserved job artifact changed"):
        restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert len(worker.calls) == 1
    assert not (runner.out / "quality_calibration.json").exists()


def test_preserve_accepts_malformed_archive_without_scoring_it(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    FakeWorker(monkeypatch)
    raw = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    archive = Path(raw["directory"]) / "candidate_records.jsonl"
    archive.write_text('{"truncated":')
    result = runner.preserve(raw)
    assert result["ledger"]["attempts_started"] == 8
    assert result["artifact_hashes"][str(archive)] == file_digest(archive)
    assert result["status"] == "INCOMPLETE"
    assert not (runner.out / "quality_calibration.json").exists()


def test_preserve_cached_result_is_a_verified_noop(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    result = runner({"pg_batch_eps": 2}, 8, 11, "train")
    raw = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    before = file_digest(Path(result["directory"]) / "result.json")
    assert runner.preserve(raw) == result
    assert file_digest(Path(result["directory"]) / "result.json") == before
    assert len(worker.calls) == 1


def test_preserved_numerical_stop_remains_infeasible_on_terminal_recovery(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker = FakeWorker(monkeypatch)
    worker.ledger_updates = {"status": "INFEASIBLE_NUMERICAL"}
    monkeypatch.setattr(subprocess.Popen, "returncode", 1)
    result = runner.preserve(runner.execute({"pg_batch_eps": 2}, 8, 11, "train"))
    restored = JobRunner(load_plan(source), runner.out)
    recovered = restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert recovered["status"] == "INFEASIBLE_NUMERICAL"
    assert recovered["returncode"] == 1
    assert recovered["wall_seconds"] == result["wall_seconds"]
    assert len(worker.calls) == 1
