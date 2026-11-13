"""Isolated train/generate jobs with durable cost and complete raw records."""
from __future__ import annotations

import copy
from concurrent.futures import CancelledError
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import yaml

from .artifacts import atomic_json, checkpoint_files, digest, file_digest, read_json
from .scoring import score_records
from .spec import phase_efficiency
from .algorithm_profiles import resolve_parameters
from .calibration import calibrate_quality_scale, restore_quality_scale, QualityCalibrationError
from ..config_schema import with_generation_selection


class JobError(RuntimeError):
    """A job cannot safely be executed or resumed."""


class JobRunner:
    def __init__(self, plan: dict, out: str | Path):
        self.plan, self.out = plan, Path(out)
        restore_quality_scale(self.plan, self.out)
        self._process_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._processes = {}
        self._job_locks = {}

    def __call__(self, parameters: dict, budget: int, seed: int, phase: str,
                 model: dict | None = None) -> dict:
        return self.finalize(self.execute(parameters, budget, seed, phase, model))

    @staticmethod
    def _completed_result(path: Path) -> dict | None:
        previous = read_json(path) if path.exists() else None
        if previous and previous["status"] not in ("INTERRUPTED", "INCOMPLETE", "SYSTEM_ERROR"):
            JobRunner._validate_artifacts(previous, "Completed job")
            return previous
        return None

    @staticmethod
    def _validate_artifacts(result, description):
        for artifact, expected in result.get("artifact_hashes", {}).items():
            if not Path(artifact).is_file() or file_digest(artifact) != expected:
                raise JobError(f"{description} artifact changed: {artifact}")

    @staticmethod
    def _signal_process(process, sig):
        try:
            if os.name == "posix":
                # start_new_session makes this group independent of terminal
                # SIGINT, including children created by the experiment.
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def cancel_all(self) -> None:
        """Stop/reap active workers and prevent queued jobs from starting.

        Cancellation is sticky. Resume uses a fresh runner, which validates the
        preserved ledgers/checkpoints before spending further attempts.
        """
        self._cancelled.set()
        with self._cancel_lock:
            with self._process_lock:
                active = list(self._processes.items())
                for _, cancelled in active:
                    cancelled.set()
            # Signal every worker promptly before waiting for any one of them.
            for process, _ in active:
                self._signal_process(process, signal.SIGTERM)
            deadline = time.monotonic() + 10
            for process, _ in active:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    self._signal_process(process, signal.SIGKILL)
                    process.wait()

    def stop_all(self) -> None:
        """Ask native workers to stop at their next complete checkpoint.

        Signal only the Python worker: its predictor children must finish the
        current evaluation. A second interrupt explicitly forces group cleanup.
        """
        self._cancelled.set()
        with self._cancel_lock:
            with self._process_lock:
                active = list(self._processes.items())
                for _, cancelled in active:
                    cancelled.set()
            if active:
                print("[HPO] Stopping after complete checkpoints; waiting for active "
                      "evaluations/training batches. Press Ctrl+C again to force exit.",
                      file=sys.stderr, flush=True)
            try:
                for process, _ in active:
                    try:
                        process.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for process, _ in active:
                    process.wait()
            except BaseException as forced:
                # Finish cleanup here instead of re-entering cancel_all while
                # holding its lock. The caller retains the original interrupt.
                for process, _ in active:
                    self._signal_process(process, signal.SIGKILL)
                for process, _ in active:
                    process.wait()
                if not isinstance(forced, KeyboardInterrupt):
                    raise

    def execute(self, parameters: dict, budget: int, seed: int, phase: str,
                model: dict | None = None) -> dict:
        """Run isolated work, returning a small payload without global scoring.

        The coordinator must call finalize on these payloads in request order.
        Archives stay on disk rather than accumulating in completed futures.
        """
        if self._cancelled.is_set():
            raise CancelledError("Job runner has been cancelled")
        if phase not in ("train", "generation"):
            raise ValueError(f"Unknown phase {phase}")
        if phase == "generation" and model is None:
            raise JobError("Generation requires a frozen training model")
        input_models = {} if model is None else {
            name: file_digest(path) for name, path in model["checkpoint_paths"].items()
            if name in ("policy.pt", "qnet.pt", "std_scaler.bin", "value_net.pt")}
        if model is not None:
            required = {"std_scaler.bin", "policy.pt" if self.plan["method"] == "a2c" else "qnet.pt"}
            if not required.issubset(input_models):
                raise JobError("Frozen model is missing its policy/Q network or scaler")
            expected_hashes = model.get("artifact_hashes", {})
            for name, observed in input_models.items():
                source = model["checkpoint_paths"][name]
                if expected_hashes.get(source) != observed:
                    raise JobError(f"Frozen model artifact changed or has no recorded fingerprint: {source}")
        key = {"parameters": copy.deepcopy(parameters), "budget": budget, "seed": seed,
               "phase": phase, "input_models": input_models}
        job_id = digest(key)[:24]
        with self._process_lock:
            lock = self._job_locks.setdefault(job_id, threading.Lock())
        # Duplicate requests must never race on the same ledger/archive.
        with lock:
            if self._cancelled.is_set():
                raise CancelledError("Job runner has been cancelled")
            return self._execute_job(key, job_id, model)

    def _execute_job(self, key, job_id, model):
        parameters, budget, seed, phase = (key[name] for name in ("parameters", "budget", "seed", "phase"))
        # Range search may replace the search space between batches. Snapshot
        # the current plan for this job without mutating shared campaign state.
        plan = copy.deepcopy(self.plan)
        directory = self.out / "jobs" / job_id
        result_path = directory / "result.json"
        completed = self._completed_result(result_path)
        if completed is not None:
            return {"cached_result_path": str(result_path)}
        previous = read_json(result_path) if result_path.exists() else None
        if previous and previous.get("finalization_pending"):
            self._validate_artifacts(previous, "Preserved job")
        directory.mkdir(parents=True, exist_ok=True)
        ledger_path = directory / "candidate_budget.json"
        resume = False
        recover_terminal = False
        if ledger_path.exists():
            old_ledger = read_json(ledger_path)
            recover_terminal = old_ledger.get("status") in {
                "COMPLETED", "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}
            if recover_terminal and previous and (
                    previous.get("execution_failure")
                    or (old_ledger.get("status") == "COMPLETED" and (
                        previous.get("status") == "SYSTEM_ERROR"
                        or previous.get("returncode") not in (None, 0)))):
                raise JobError(f"Job {job_id} has a terminal ledger but its execution failed; "
                               "the recorded failure and consumed cost cannot be cleared by recovery. "
                               "Inspect result.json and the preserved worker logs.")
            if not recover_terminal and not old_ledger.get("attempts_started", 0):
                raise JobError(f"Job {job_id} has initialized accounting artifacts but no consumed attempts; "
                               "inspect the preserved initialization failure before starting a new campaign. "
                               "Existing ledgers are never silently erased.")
            if not recover_terminal and old_ledger.get("attempts_started", 0):
                resume_checkpoint = directory / ("generation_checkpoint.pt" if phase == "generation" else "checkpoint.pt")
                if not resume_checkpoint.exists():
                    raise JobError(f"Job {job_id} consumed attempts without a resumable checkpoint; "
                                   "cost is preserved and will not be reset. See candidate_budget.json.")
                resume = True

        train_budget = budget if model is None else model["budget"]
        cfg = copy.deepcopy(plan["base_config_data"])
        generation_parameter = plan["generation_search"]["parameter"]
        training_parameters = {name: value for name, value in parameters.items()
                               if name not in ("gen_temperature", "gen_epsilon", "gen_top_frac")}
        cfg.update(resolve_parameters(plan, training_parameters, train_budget))
        efficiency = phase_efficiency(plan, phase)
        cfg.update(hpo_recording=True, hpo_phase=phase, hpo_target_unique=budget,
                   hpo_ratio_limit=efficiency["max_attempts_per_unique"],
                   train_count_duplicates=False,
                   train_max_duplicate_attempts=max(1000, 10 * budget),
                   exploration_gen_eps=0)
        if phase == "generation":
            cfg.update(num_gen_eps=budget, gen_epsilon=0.0, gen_top_frac=0.0, gen_temperature=0.0,
                       max_gen_attempts=max(1000, 10 * budget))
            cfg[generation_parameter] = parameters[generation_parameter]
            if resume:
                cfg["hpo_resume_generation"] = True
        # Full runtime configs contain defaults for all three strategies. Carry
        # the actual selection explicitly so YAML key sorting cannot change it.
        cfg = with_generation_selection(cfg, generation_parameter)
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        atomic_json(directory / "job.json", key)
        args = [sys.executable, "-m", "rl_matdesign.hpo.worker", "--config", str(config_path),
                "--out", str(directory), "--train-seed",
                str(seed if model is None else model["seed"]), "--gen-seed", str(seed)]
        if phase == "train":
            args += ["--skip-generation", "--save-checkpoint-freq", "1"]
            if resume:
                args.append("--resume-training")
        else:
            args.append("--only-generate")
            for name, flag in (("qnet.pt", "--load-qnet"), ("policy.pt", "--load-policy"),
                               ("std_scaler.bin", "--load-scaler"), ("value_net.pt", "--load-value-net")):
                if name in model["checkpoint_paths"]:
                    args += [flag, model["checkpoint_paths"][name]]
        environment = os.environ.copy()
        package_parent = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = package_parent + os.pathsep + environment.get("PYTHONPATH", "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment.setdefault(name, "1")
        started = time.monotonic()
        interrupted = False
        execution_error = None
        if recover_terminal:
            # A process may finish and persist its ledger before the parent
            # persists result.json. Reconstruct evidence, never buy it twice.
            returncode = (previous or {}).get("returncode")
            # Cleanup can preserve a measured worker duration before scoring.
            # Reusing terminal artifacts spends no additional worker time.
            elapsed = 0.0 if previous is not None and previous.get("wall_seconds") is not None else None
        else:
            atomic_json(directory / "execution.json", {"status": "RUNNING", "argv": args,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "resume": resume, "previous": previous})
            with (directory / "stdout.log").open("a") as stdout, (directory / "stderr.log").open("a") as stderr:
                cancelled = threading.Event()
                # Lock launch+registration against cancel_all's snapshot. Once
                # cancellation is requested, no late worker can escape it.
                with self._process_lock:
                    if self._cancelled.is_set():
                        raise CancelledError("Job runner has been cancelled")
                    process = subprocess.Popen(args, cwd=plan["base_config_dir"], env=environment,
                                               stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    self._processes[process] = cancelled
                try:
                    returncode = process.wait()
                except BaseException as exc:
                    execution_error = exc
                    if isinstance(exc, KeyboardInterrupt):
                        self.stop_all()
                    else:
                        self.cancel_all()
                    returncode = process.returncode
                finally:
                    with self._process_lock:
                        interrupted = cancelled.is_set()
                        self._processes.pop(process, None)
            elapsed = time.monotonic() - started
        if model:
            after = {name: file_digest(path) for name, path in model["checkpoint_paths"].items()
                     if name in key["input_models"]}
            if after != key["input_models"]:
                raise JobError("Frozen generation modified its input model or scaler")
        return {"key": key, "job_id": job_id, "directory": str(directory),
                "argv": args, "previous": previous, "resume": resume,
                "recover_terminal": recover_terminal, "elapsed": elapsed,
                "interrupted": interrupted, "returncode": returncode,
                "execution_error": execution_error}

    def preserve(self, payload: dict) -> dict:
        """Persist cleanup evidence without making any scientific decision.

        A later completed worker cannot become the quality reference while an
        earlier requested job is unresolved. Resume finalizes terminal ledgers
        in the original request order without rerunning their workers.
        """
        if "cached_result_path" in payload:
            completed = self._completed_result(Path(payload["cached_result_path"]))
            if completed is None:
                raise JobError("Completed job result disappeared before preservation")
            return completed
        directory = Path(payload["directory"])
        completed = self._completed_result(directory / "result.json")
        if completed is not None:
            return completed
        ledger_path = directory / "candidate_budget.json"
        interrupted = payload["interrupted"]
        ledger = read_json(ledger_path) if ledger_path.exists() else {
            "status": "INTERRUPTED" if interrupted else "SYSTEM_ERROR",
            "attempts_started": 0, "unique_evaluated": 0,
            "target_unique": payload["key"]["budget"]}
        pending = {"status": "INTERRUPTED" if interrupted else "INCOMPLETE",
                   "eligible_for_selection": False, "score": -1.0,
                   "raw_metrics": {}, "normalized_metrics": {},
                   "finalization_pending": True,
                   "reason": "Awaiting ordered finalization after batch interruption or failure"}
        paths = checkpoint_files(directory) if payload["key"]["phase"] == "train" else {}
        return self._save_result(payload, pending, ledger, paths, include_quality=False)

    def finalize(self, payload: dict) -> dict:
        """Calibrate, score and persist one payload on the coordinator thread."""
        if "cached_result_path" in payload:
            completed = self._completed_result(Path(payload["cached_result_path"]))
            if completed is None:
                raise JobError("Completed job result disappeared before finalization")
            return completed
        key, job_id = payload["key"], payload["job_id"]
        budget, phase = key["budget"], key["phase"]
        directory = Path(payload["directory"])
        result_path = directory / "result.json"
        # An identical request may have completed while this payload waited for
        # its turn; reuse its measured result rather than replacing its timing.
        completed = self._completed_result(result_path)
        if completed is not None:
            return completed
        ledger_path = directory / "candidate_budget.json"
        interrupted, returncode = payload["interrupted"], payload["returncode"]
        execution_error = payload["execution_error"]
        ledger = read_json(ledger_path) if ledger_path.exists() else {
            "status": "INTERRUPTED" if interrupted else "SYSTEM_ERROR",
            "attempts_started": 0, "unique_evaluated": 0, "target_unique": budget}
        record_path = directory / "candidate_records.jsonl"
        records = []
        if record_path.exists():
            try:
                records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise JobError(f"Incomplete record archive preserved at {record_path}: {exc}") from exc
        calibration_error = None
        try:
            if not interrupted and not returncode and execution_error is None:
                calibrate_quality_scale(self.plan, self.out, records, ledger, record_path, job_id, phase)
        except QualityCalibrationError as exc:
            calibration_error = str(exc)
        scored = score_records(records, ledger, self.plan, phase)
        if calibration_error:
            scored.update(status="QUALITY_CALIBRATION_REQUIRED", eligible_for_selection=False,
                          score=-1.0, error=calibration_error)
        if execution_error is not None and not isinstance(execution_error, (KeyboardInterrupt, SystemExit)):
            scored.update(status="SYSTEM_ERROR", eligible_for_selection=False, score=-1.0,
                          error=str(execution_error))
        elif interrupted:
            scored.update(status="INTERRUPTED", eligible_for_selection=False, score=-1.0)
        elif returncode and scored.get("status") not in ("INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"):
            scored.update(status="SYSTEM_ERROR", eligible_for_selection=False, score=-1.0)
        paths = checkpoint_files(directory) if phase == "train" else {}
        if phase == "train" and scored.get("eligible_for_selection"):
            required = {"std_scaler.bin", "policy.pt" if self.plan["method"] == "a2c" else "qnet.pt"}
            if not required.issubset(paths):
                scored.update(status="SYSTEM_ERROR", eligible_for_selection=False, score=-1.0,
                              reason="Completed training did not save all required model artifacts")
        result = self._save_result(payload, scored, ledger, paths, include_quality=True)
        if execution_error is not None:
            raise execution_error
        return result

    def _save_result(self, payload, evaluation, ledger, paths, *, include_quality):
        key, job_id = payload["key"], payload["job_id"]
        directory = Path(payload["directory"])
        result_path = directory / "result.json"
        record_path, ledger_path = directory / "candidate_records.jsonl", directory / "candidate_budget.json"
        resume, previous = payload["resume"], payload["previous"]
        elapsed, recover_terminal = payload["elapsed"], payload["recover_terminal"]
        returncode = payload["returncode"]
        prior_time_unknown = resume and (previous is None or previous.get("wall_seconds") is None)
        calibration_path = self.out / "quality_calibration.json"
        calibration_artifacts = [str(calibration_path)] if include_quality and calibration_path.is_file() else []
        result = {**evaluation, "job_id": job_id, **key,
                  "checkpoint_paths": paths, "ledger": ledger, "directory": str(directory),
                  "records_path": str(record_path), "returncode": returncode,
                  "wall_seconds": None if elapsed is None or prior_time_unknown else elapsed + ((previous or {}).get("wall_seconds") or 0),
                  "current_execution_seconds": elapsed,
                  "recovered_from_terminal_artifacts": recover_terminal,
                  "timing_status": ("recovered_from_preserved_execution" if recover_terminal and elapsed is not None else
                                    "unknown_after_parent_interruption" if recover_terminal else
                                    "partial_after_resume" if prior_time_unknown else "measured"),
                  "artifact_hashes": {p: file_digest(p) for p in
                      [*paths.values(), str(record_path), str(ledger_path), *calibration_artifacts] if Path(p).is_file()}}
        if include_quality:
            result["quality_normalization"] = copy.deepcopy(self.plan["objective"]["normalization"])
        if payload.get("execution_error") is not None:
            error = payload["execution_error"]
            settled = returncode == 0 or ledger.get("status") in {
                "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}
            field = ("coordinator_interruption" if isinstance(error, KeyboardInterrupt)
                     and payload["interrupted"] and settled else "execution_failure")
            result[field] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(result_path, result)
        atomic_json(directory / "execution.json", {"status": result["status"], "argv": payload["argv"],
                                                   "returncode": returncode, "wall_seconds": result["wall_seconds"]})
        return result
