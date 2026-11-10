"""GPU-isolated execution of independent configurations, without MD-specific code.

Workers are fresh interpreters, not forks of a process that may already own CUDA
models. Each keeps its builder (and optional relaxation calculator) for the run.
Only candidates and calculator-free ASE structures cross the private IPC pipe.
"""
from __future__ import annotations

import atexit
import copy
import math
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import warnings
from multiprocessing import Pipe
from multiprocessing.connection import wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .structure_samples import SamplingProtocolError, sample_plan_failure_record

if TYPE_CHECKING:
    from ase import Atoms


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"structure_parallel.{name} must be a positive integer")
    return value


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"structure_parallel.{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"structure_parallel.{name} must be finite and positive")
    return value


def _python_environment() -> dict[str, str]:
    env = dict(os.environ)
    # Runtime sys.path may include uninstalled source checkouts and FQN plugins.
    paths = [os.path.abspath(p or os.getcwd()) for p in sys.path]
    paths.extend(p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _cuda_device_count(timeout: float) -> int:
    """Discover allocated CUDA devices in a clean, model-free interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", "import json, torch; "
         "print(json.dumps({'count': torch.cuda.device_count() if torch.cuda.is_available() else 0}))"],
        env=_python_environment(), capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode:
        raise ValueError(f"Cannot discover CUDA GPUs: {result.stderr[-2000:]}")
    try:
        return int(json.loads(result.stdout.strip().splitlines()[-1])["count"])
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError("CUDA device discovery returned no valid device count") from exc


def _device_tokens(devices: Any, reserve: Any, timeout: float) -> list[str]:
    if not isinstance(reserve, list) or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in reserve
    ) or len(set(reserve)) != len(reserve):
        raise ValueError("structure_parallel.reserve_devices must list distinct nonnegative indices")
    if devices == "auto":
        count = _cuda_device_count(timeout)
        if any(d >= count for d in reserve):
            raise ValueError("structure_parallel.reserve_devices is outside the visible GPU allocation")
        devices = [d for d in range(count) if d not in reserve]
        if not devices:
            raise ValueError("No CUDA GPU is available for structure_parallel after reservations")
    elif reserve:
        raise ValueError("structure_parallel.reserve_devices requires devices: auto")
    if not isinstance(devices, list) or not devices:
        raise ValueError("structure_parallel.devices needs 'auto' or a non-empty list of GPU indices")
    if any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in devices):
        raise ValueError("structure_parallel.devices must contain nonnegative integer GPU indices")
    if len(set(devices)) != len(devices):
        raise ValueError("structure_parallel.devices must not repeat a GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(d) for d in devices]
    allocated = [s.strip() for s in visible.split(",") if s.strip()]
    if not allocated or "-1" in allocated or max(devices) >= len(allocated):
        raise ValueError(
            f"structure_parallel.devices {devices} is outside the inherited "
            f"CUDA_VISIBLE_DEVICES={visible!r}; indices refer to visible GPUs"
        )
    tokens = [allocated[d] for d in devices]
    if len(set(tokens)) != len(tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES maps workers to duplicate GPUs")
    return tokens


class _Worker:
    def __init__(self, token: str, cfg: dict, kind: str, seed: int | None,
                 threads: int) -> None:
        self.token = token
        self.connection, child = Pipe(duplex=True)
        env = _python_environment()
        # DeepMD chooses its device at import, and LAMMPS inherits this environment.
        env.update(CUDA_VISIBLE_DEVICES=token, LOCAL_RANK="0", DEVICE="cuda",
                   PYTHONUNBUFFERED="1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "TF_INTRA_OP_PARALLELISM_THREADS",
                    "TF_INTER_OP_PARALLELISM_THREADS"):
            env[key] = str(threads)
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "rl_matdesign.utils.structure_worker", str(child.fileno())],
                env=env, pass_fds=(child.fileno(),), start_new_session=True,
            )
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        try:
            self.connection.send({"kind": kind, "cfg": cfg, "seed": seed, "threads": threads})
        except BaseException:
            self.close()
            raise

    def receive(self) -> dict:
        try:
            return self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Structure worker on GPU {self.token} exited unexpectedly") from exc

    def close(self) -> None:
        # A worker may be blocked inside subprocess.run(LAMMPS). Kill its entire
        # private process group, including children that outlive the Python process.
        try:
            if self.process.poll() is None:
                self.connection.send({"op": "close"})
                self.process.wait(timeout=0.2)
        except (OSError, EOFError, subprocess.TimeoutExpired):
            pass
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Some desktop sandboxes report EPERM for an already-reaped
                    # process group. Never suppress a real failure on a live worker.
                    if self.process.poll() is None:
                        raise
                if sig == signal.SIGTERM:
                    try:
                        self.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            self.process.wait(timeout=5)
        finally:
            self.connection.close()



class ParallelStructureBuilder:
    """Run independent configurations with one structure worker per GPU.

    Seeded execution requires a plugin's explicit build_sample capability;
    custom sample plans are resolved by the worker-owned builder. Legacy direct
    calls require one ASE Atoms for n_configs=1 and honor the supplied RNG.
    Configurations from multiple formulas share the same queue. Each GPU starts
    the next queued configuration when its current build and optimization finish.
    """

    parallel_execution = True

    def __init__(self, kind: str, cfg: dict, *, seed: int | None = None) -> None:
        if os.name != "posix":
            raise ValueError("structure_parallel currently requires a POSIX host (Linux/macOS)")
        options = cfg.get("structure_parallel")
        if not isinstance(options, dict):
            raise ValueError("structure_parallel must be a mapping")
        known = {"devices", "reserve_devices", "workers_per_device", "min_speedup",
                 "candidate_batch_size",
                 "threads_per_worker", "timeout_s", "startup_timeout_s", "min_successful_configs"}
        extra = set(options) - known
        if extra:
            raise ValueError(f"Unknown structure_parallel options: {sorted(extra)}")
        self.threads = _positive_int(options.get("threads_per_worker", 1), "threads_per_worker")
        self.timeout = _positive_seconds(options.get("timeout_s", 3600), "timeout_s")
        self.startup_timeout = _positive_seconds(options.get("startup_timeout_s", 300), "startup_timeout_s")
        min_successful = _positive_int(options.get("min_successful_configs", 1), "min_successful_configs")
        if min_successful > 1:
            warnings.warn(
                "structure_parallel.min_successful_configs is deprecated and values "
                "above 1 are ignored; every nonempty successful ensemble is accepted",
                DeprecationWarning, stacklevel=2,
            )
        self.min_successful = 1
        self.candidate_batch_size = options.get("candidate_batch_size", "auto")
        if self.candidate_batch_size != "auto":
            self.candidate_batch_size = _positive_int(self.candidate_batch_size, "candidate_batch_size")
        self.devices = _device_tokens(options.get("devices", "auto"), options.get("reserve_devices", []), self.startup_timeout)
        workers_per_device = options.get("workers_per_device", 1)
        if workers_per_device == "auto":
            warnings.warn(
                "structure_parallel.workers_per_device: auto is deprecated and now "
                "uses one worker per GPU; remove this option or set it to 1",
                DeprecationWarning, stacklevel=2,
            )
        elif type(workers_per_device) is not int or workers_per_device != 1:
            raise ValueError(
                "structure_parallel.workers_per_device must be 1; each GPU runs "
                "one structure at a time"
            )
        self.workers_per_device = 1
        if "min_speedup" in options:
            warnings.warn(
                "structure_parallel.min_speedup is deprecated and ignored; "
                "GPU concurrency is fixed at one worker per GPU",
                DeprecationWarning, stacklevel=2,
            )
        overrides = ((cfg.get("md") or {}).get("env") or {})
        reserved = {"CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "DEVICE"}.intersection(overrides)
        if reserved:
            raise ValueError(
                f"md.env overrides {sorted(reserved)} would break structure_parallel GPU isolation; "
                "remove them and use structure_parallel.devices"
            )
        self.kind = kind
        self.cfg = copy.deepcopy(cfg)
        self.cfg.pop("structure_parallel", None)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._workers: dict[tuple[int, int], _Worker] = {}
        self._closed = False
        self._registered = False
        self._run_id = uuid.uuid4().hex[:12]
        self.last_build_timings: list[dict] = []
        self.last_build_job_timings: list[dict] = []
        self.last_build_wall_s = 0.0
        self.last_sample_records: list[list[dict]] = []
        go = cfg.get("geo_opt") or {}
        self.handles_geo_opt = bool(go) and bool(go.get("enabled", True))

    def _ensure_targets(self, targets: list[tuple[int, int]]) -> None:
        if self._closed:
            raise RuntimeError("ParallelStructureBuilder is closed")
        if not self._registered:
            atexit.register(self.close)
            self._registered = True
        new_workers = []
        try:
            for device, slot in targets:
                if (device, slot) in self._workers:
                    continue
                cfg = copy.deepcopy(self.cfg)
                if cfg.get("md_debug_dir"):
                    # Restarted workers reset the plugin's file counter.
                    cfg["md_debug_dir"] = str(
                        Path(cfg["md_debug_dir"]) / self._run_id /
                        f"gpu_{device}_slot_{slot}_{uuid.uuid4().hex[:8]}"
                    )
                worker = _Worker(self.devices[device], cfg, self.kind, self.seed, self.threads)
                self._workers[device, slot] = worker
                new_workers.append(worker)
            deadline = time.monotonic() + self.startup_timeout
            for worker in new_workers:
                if not worker.connection.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError(f"Structure worker startup timed out on GPU {worker.token}")
                reply = worker.receive()
                if not reply.get("ok"):
                    raise RuntimeError(f"Structure worker startup on GPU {worker.token}: {reply.get('error')}")
                worker.sample_protocol = reply.get("sample_protocol") is True
        except BaseException:
            self._stop_workers()
            raise

    def _run_jobs(self, candidates: list[Any], jobs: list[tuple],
                  targets: list[tuple[int, int]],
                  *, label: str) -> list[dict]:
        """Queue (candidate index, config index, seed, sample) across workers.

        Replies retain job order, independently of worker completion order.
        """
        if not jobs:
            return []
        self._ensure_targets(targets)
        if any(job[3] is not None for job in jobs):
            self._require_sample_capabilities(targets)
        replies: dict[int, dict] = {}
        active = {}
        next_index = 0

        def context(index: int) -> str:
            candidate_index, config_index, _, _sample = jobs[index]
            return (f"{label} candidate {candidate_index + 1}/{len(candidates)} "
                    f"config {config_index + 1} (job {index + 1}/{len(jobs)})")

        def dispatch(worker: _Worker, index: int) -> None:
            candidate_index, config_index, seed, sample = jobs[index]
            worker.connection.send({"op": "build", "candidate": candidates[candidate_index],
                                    "seed": seed, "index": index,
                                    "candidate_index": candidate_index,
                                    "config_index": config_index, "sample": sample})
            self.last_build_job_timings[index]["dispatched"] = True
            active[worker.connection] = (worker, index, time.monotonic())
            print(f"[structure_parallel] {context(index)} started on GPU {worker.token}", flush=True)

        try:
            for target in targets[:len(jobs)]:
                dispatch(self._workers[target], next_index)
                next_index += 1
            while active:
                remaining = min(self.timeout - (time.monotonic() - t) for _, _, t in active.values())
                if remaining <= 0:
                    worker, index, _ = min(active.values(), key=lambda item: item[2])
                    raise TimeoutError(
                        f"{context(index)} on GPU {worker.token} "
                        f"exceeded structure_parallel.timeout_s={self.timeout:g}"
                    )
                for connection in wait(list(active), timeout=min(remaining, 1.0)):
                    worker, index, t = active.pop(connection)
                    reply = worker.receive()
                    if reply.get("sampling_protocol_error"):
                        raise SamplingProtocolError(reply.get("error", "Invalid builder sample protocol"))
                    reply["elapsed"] = time.monotonic() - t
                    replies[index] = reply
                    self.last_build_job_timings[index].update(
                        completed=True, succeeded=bool(reply.get("ok")),
                        timing=reply.get("timing"), elapsed_s=reply["elapsed"],
                        error=reply.get("error"),
                    )
                    status = "done" if reply.get("ok") else "FAILED"
                    print(f"[structure_parallel] {context(index)} {status} on GPU {worker.token} ({reply['elapsed']:.1f}s)", flush=True)
                    if not reply.get("ok"):
                        print(reply.get("error", "unknown worker failure"), flush=True)
                    if next_index < len(jobs):
                        dispatch(worker, next_index)
                        next_index += 1
        except BaseException:
            self._stop_workers()
            raise
        return [replies[i] for i in range(len(jobs))]

    def _retire(self, device: int, slot: int) -> None:
        worker = self._workers.pop((device, slot), None)
        if worker is not None:
            worker.close()

    def _validate_n_configs(self, n_configs: int) -> int:
        return _positive_int(n_configs, "n_configs")

    def recommended_candidate_batch_size(self, n_configs: int) -> int:
        """Recommend enough formulas to fill the GPUs, without starting workers."""
        n_configs = self._validate_n_configs(n_configs)
        if self.candidate_batch_size != "auto":
            return self.candidate_batch_size
        slots = len(self.devices)
        return max(1, (slots + n_configs - 1) // n_configs)

    def build(self, candidate: Any, *, n_configs: int = 1,
              rng: np.random.Generator | None = None) -> list[Atoms]:
        result = self.build_many([candidate], n_configs=n_configs, rng=rng)[0]
        if isinstance(result, Exception):
            raise result
        return result

    def build_many(self, candidates: Any, *, n_configs: int = 1,
                   rng: np.random.Generator | None = None) -> list[list[Atoms] | Exception]:
        """Legacy caller-owned RNG path, preserving its sequential seed stream."""
        n_configs = self._validate_n_configs(n_configs)
        candidates = list(candidates)
        rng = self._rng if rng is None else rng
        seeds = rng.integers(0, 2**32, size=len(candidates) * n_configs, dtype=np.uint64)
        jobs = [(index // n_configs, index % n_configs, int(seed), None)
                for index, seed in enumerate(seeds)]
        return self._build_jobs(candidates, jobs)

    def build_seeded(self, candidate: Any, *, n_configs: int, source_spec, eval_seed: int):
        result = self.build_many_seeded([candidate], n_configs=n_configs,
                                        source_spec=source_spec, eval_seed=eval_seed)[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _require_sample_capabilities(self, targets: list[tuple[int, int]]) -> None:
        if any(not self._workers[target].sample_protocol for target in targets):
            self._stop_workers()
            raise SamplingProtocolError(
                "Seeded parallel execution requires explicit "
                "build_sample(candidate, *, sample) capability; native build ensembles "
                "cannot be split reproducibly."
            )

    def build_many_seeded(self, candidates: Any, *, n_configs: int, source_spec, eval_seed: int):
        """Resolve worker-owned plans, then execute their explicit sample protocol.

        No builder is imported or initialized in the parent. A missing sample
        capability fails before any scientific build task. Candidate-specific
        plan errors remain aligned with successful neighboring candidates.
        """
        n_configs = self._validate_n_configs(n_configs)
        candidates = list(candidates)
        self.last_build_timings = []
        self.last_build_job_timings = []
        self.last_sample_records = []
        self.last_build_wall_s = 0.0
        if not candidates:
            return []
        jobs, plan_failures = [], {}
        self._ensure_targets([(0, 0)])
        self._require_sample_capabilities([(0, 0)])
        worker = self._workers[0, 0]
        try:
            for candidate_index, candidate in enumerate(candidates):
                worker.connection.send({"op": "plan", "candidate": candidate,
                                        "n_configs": n_configs, "source_spec": source_spec,
                                        "eval_seed": eval_seed})
                if not worker.connection.poll(self.timeout):
                    raise TimeoutError("Structure worker sample_plan timed out")
                reply = worker.receive()
                if reply.get("sampling_protocol_error"):
                    raise SamplingProtocolError(reply.get("error", "Invalid builder sample protocol"))
                if not reply.get("ok"):
                    error = RuntimeError(reply.get("error", "Structure sample plan failed"))
                    plan_failures[candidate_index] = (error, sample_plan_failure_record(source_spec, error))
                    continue
                jobs.extend((candidate_index, index, sample.seed, sample)
                            for index, sample in enumerate(reply["samples"]))
        except BaseException:
            self._stop_workers()
            raise
        return self._build_jobs(candidates, jobs, plan_failures=plan_failures)

    def _build_jobs(self, candidates: list[Any], jobs: list[tuple], *,
                    plan_failures: dict[int, tuple[Exception, dict]] | None = None
                    ) -> list[list[Atoms] | Exception]:
        """Return ordered successful ensembles, retaining every sample's status."""
        self.last_build_timings = []
        self.last_build_job_timings = []
        self.last_sample_records = []
        self.last_build_wall_s = 0.0
        plan_failures = plan_failures or {}
        if not candidates:
            return []
        started = time.monotonic()
        self.last_build_job_timings = [
            {"candidate_index": candidate_index, "config_index": config_index,
             "dispatched": False, "completed": False, "succeeded": False,
             "timing": None}
            for candidate_index, config_index, _seed, _sample in jobs
        ]
        groups = [[] for _ in candidates]
        for index, (candidate_index, _config_index, _seed, _sample) in enumerate(jobs):
            groups[candidate_index].append(index)
        targets = [(device, 0) for device in range(min(len(self.devices), len(jobs)))]
        for device, slot in set(self._workers) - set(targets):
            self._retire(device, slot)
        print(f"[structure_parallel] scheduling {len(candidates)} candidates / "
              f"{len(jobs)} configurations on {len(targets)} GPUs (one structure per GPU)", flush=True)
        try:
            replies = self._run_jobs(candidates, jobs, targets, label="config")
        finally:
            self.last_build_wall_s = time.monotonic() - started
            for candidate_index, indices in enumerate(groups):
                records = [self.last_build_job_timings[index] for index in indices]
                incomplete = any(not record["completed"] for record in records)
                timing = {
                    "config_attempted": sum(int(record["dispatched"]) for record in records),
                    "config_succeeded": sum(int(record["succeeded"]) for record in records),
                    "timing_incomplete": incomplete,
                }
                for stage in ("md_task_s", "opt_task_s", "structure_other_task_s"):
                    values = [(record["timing"] or {}).get(stage) for record in records]
                    timing[stage] = (None if incomplete or any(value is None for value in values)
                                     else sum(values))
                self.last_build_timings.append(timing)
                sample_records = ([plan_failures[candidate_index][1]]
                                  if candidate_index in plan_failures else [])
                for index, record in zip(indices, records):
                    _candidate_index, config_index, seed, sample = jobs[index]
                    identity = sample.record() if sample else {"config_index": config_index, "seed": seed}
                    status = ("succeeded" if record["succeeded"] else "failed"
                              if record["completed"] else "interrupted"
                              if record["dispatched"] else "not_attempted")
                    identity["status"] = status
                    if record.get("error"):
                        identity["error"] = record["error"]
                    sample_records.append(identity)
                self.last_sample_records.append(sample_records)
        results: list[list[Atoms] | Exception] = []
        for index, indices in enumerate(groups):
            if index in plan_failures:
                results.append(plan_failures[index][0])
                continue
            candidate_replies = [replies[job_index] for job_index in indices]
            atoms = [reply["atoms"] for reply in candidate_replies if reply.get("ok")]
            print(f"[structure_parallel] candidate {index + 1}/{len(candidates)}: "
                  f"{len(atoms)}/{len(indices)} configurations succeeded", flush=True)
            if not atoms:
                failure = next((r.get("error") for r in candidate_replies if not r.get("ok")), "no structures")
                results.append(RuntimeError(
                    f"Only {len(atoms)}/{len(indices)} configurations succeeded for "
                    f"candidate {index + 1}/{len(candidates)}; "
                    f"at least one successful configuration is required. First failure: {failure}"
                ))
            else:
                results.append(atoms)
        print(f"[structure_parallel] batch completed in {time.monotonic() - started:.1f}s", flush=True)
        return results

    def composition_formula(self, candidate: Any) -> str | None:
        self._ensure_targets([(0, 0)])
        worker = self._workers[0, 0]
        try:
            worker.connection.send({"op": "formula", "candidate": candidate})
            if not worker.connection.poll(self.timeout):
                raise TimeoutError("Structure worker composition_formula timed out")
            reply = worker.receive()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "composition_formula failed"))
            return reply["formula"]
        except BaseException:
            self._stop_workers()
            raise

    def _stop_workers(self) -> None:
        workers, self._workers = list(self._workers.values()), {}
        for worker in workers:
            try:
                worker.close()
            except Exception as exc:
                # Do not strand other GPU workers or mask the original failure.
                print(f"[structure_parallel] cleanup failed for PID {worker.process.pid}: {exc}", flush=True)

    def close(self) -> None:
        self._closed = True
        self._stop_workers()
        if self._registered:
            atexit.unregister(self.close)
            self._registered = False
