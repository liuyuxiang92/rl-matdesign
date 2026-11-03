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
from multiprocessing import Pipe
from multiprocessing.connection import wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


class _CalibrationCandidateFailure(RuntimeError):
    """A normal candidate error in a calibration baseline, not lost workers."""


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
    """Run independent configurations on detected GPUs with optional calibration.

    A plugin must honor the supplied RNG and return one ASE Atoms for n_configs=1.
    RL episode order stays serial. Calibration repeats full-physics tasks once;
    these are excluded from the returned ensemble and from RL sample counts.
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
        self.min_successful = _positive_int(options.get("min_successful_configs", 1), "min_successful_configs")
        self.candidate_batch_size = options.get("candidate_batch_size", 1)
        if self.candidate_batch_size != "auto":
            self.candidate_batch_size = _positive_int(self.candidate_batch_size, "candidate_batch_size")
        self.devices = _device_tokens(options.get("devices", "auto"), options.get("reserve_devices", []), self.startup_timeout)
        self.workers_per_device = options.get("workers_per_device", "auto")
        if (self.workers_per_device != "auto" and
                (type(self.workers_per_device) is not int or self.workers_per_device not in (1, 2))):
            raise ValueError("structure_parallel.workers_per_device must be auto, 1 or 2")
        self.min_speedup = _positive_seconds(options.get("min_speedup", 1.15), "min_speedup")
        if self.min_speedup < 1:
            raise ValueError("structure_parallel.min_speedup must be at least 1")
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
        count = 1 if self.workers_per_device == "auto" else self.workers_per_device
        self._counts = [count] * len(self.devices)
        self._tuned = False
        self.autotune_report: dict | None = None
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
        except BaseException:
            self._stop_workers()
            raise

    def _run_jobs(self, candidates: list[Any], jobs: list[tuple[int, int, int]],
                  targets: list[tuple[int, int]],
                  *, label: str) -> list[dict]:
        """Queue (candidate index, config index, seed) jobs across shared workers.

        Replies retain job order. Calibration supplies exactly one job per target.
        """
        self._ensure_targets(targets)
        replies: dict[int, dict] = {}
        active = {}
        next_index = 0

        def context(index: int) -> str:
            candidate_index, config_index, _ = jobs[index]
            return (f"{label} candidate {candidate_index + 1}/{len(candidates)} "
                    f"config {config_index + 1} (job {index + 1}/{len(jobs)})")

        def dispatch(worker: _Worker, index: int) -> None:
            candidate_index, config_index, seed = jobs[index]
            worker.connection.send({"op": "build", "candidate": candidates[candidate_index],
                                    "seed": seed, "index": index,
                                    "candidate_index": candidate_index,
                                    "config_index": config_index})
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
                    reply["elapsed"] = time.monotonic() - t
                    replies[index] = reply
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

    def _calibrate(self, candidate: Any, seed: int, *,
                   tolerate_candidate_failure: bool = False) -> None:
        from .structure_autotune import autotune_concurrency

        print(
            "[structure_parallel] calibrating 1 vs 2 jobs per GPU using the same "
            "candidate/seed. Up to 5 EXTRA full build/opt tasks per GPU; "
            "calibration structures are excluded from reward.", flush=True,
        )

        phase = 0

        def run_batch(probes):
            nonlocal phase
            phase += 1
            targets = [(job["device"], job["slot"]) for job in probes]
            jobs = [(0, index, job["seed"]) for index, job in enumerate(probes)]
            replies = self._run_jobs([candidate], jobs, targets, label="calibration")
            # Distinguish worker-reported baseline failures from infrastructure
            # exceptions raised by _run_jobs; only the former may fall back.
            if phase <= 2:
                failure = next((reply for reply in replies if not reply.get("ok")), None)
                if failure is not None:
                    raise _CalibrationCandidateFailure(
                        f"Autotune baseline failed: {failure.get('error', 'unknown worker error')}"
                    )
            return replies

        try:
            report = autotune_concurrency(
                len(self.devices), run_batch, self._retire, seed=seed, min_speedup=self.min_speedup,
            )
        except _CalibrationCandidateFailure as exc:
            if not tolerate_candidate_failure:
                self._stop_workers()
                raise
            self._counts = [1] * len(self.devices)
            for device, slot in list(self._workers):
                if slot:
                    self._retire(device, slot)
            print(
                "[structure_parallel] first candidate failed calibration; using "
                "one slot per GPU for this batch and retrying calibration on a "
                f"later batch. {exc}", flush=True,
            )
            return
        except BaseException:
            self._stop_workers()
            raise
        self.autotune_report = report
        self._counts = report["counts"]
        self._tuned = True
        for token, measurement in zip(self.devices, report["measurements"]):
            print(f"[structure_parallel] GPU {token}: {measurement}", flush=True)

    def _validate_n_configs(self, n_configs: int) -> int:
        n_configs = _positive_int(n_configs, "n_configs")
        if self.min_successful > n_configs:
            raise ValueError("structure_parallel.min_successful_configs exceeds n_configs")
        return n_configs

    def recommended_candidate_batch_size(self, n_configs: int) -> int:
        """Recommend enough formulas to fill worker slots, without starting workers.

        Before automatic calibration, allow two potential slots per GPU so the
        first multi-formula batch can measure and use the available concurrency.
        """
        n_configs = self._validate_n_configs(n_configs)
        if self.candidate_batch_size != "auto":
            return self.candidate_batch_size
        slots = (2 * len(self.devices)
                 if self.workers_per_device == "auto" and not self._tuned
                 else sum(self._counts))
        return max(1, (slots + n_configs - 1) // n_configs)

    def build(self, candidate: Any, *, n_configs: int = 1,
              rng: np.random.Generator | None = None) -> list[Atoms]:
        result = self.build_many([candidate], n_configs=n_configs, rng=rng)[0]
        if isinstance(result, Exception):
            raise result
        return result

    def build_many(self, candidates: Any, *, n_configs: int = 1,
                   rng: np.random.Generator | None = None) -> list[list[Atoms] | Exception]:
        """Build formulas together, returning ordered ensembles or per-formula errors.

        Ordinary worker-reported errors only reduce that formula's ensemble;
        timeout, crash, and startup failures stop the pool and propagate. Seeds
        are assigned in candidate/config order before any dispatch or probes.
        """
        n_configs = self._validate_n_configs(n_configs)
        candidates = list(candidates)
        if not candidates:
            return []
        rng = self._rng if rng is None else rng
        total_configs = len(candidates) * n_configs
        seeds = [int(s) for s in rng.integers(0, 2**32, size=total_configs, dtype=np.uint64)]
        if (self.workers_per_device == "auto" and not self._tuned
                and total_configs > len(self.devices)):
            self._calibrate(candidates[0], seeds[0], tolerate_candidate_failure=len(candidates) > 1)
        # Fill distinct GPUs first, then second slots; don't initialize idle slots.
        targets = [(d, slot) for slot in range(max(self._counts))
                   for d, count in enumerate(self._counts) if slot < count][:total_configs]
        # Calibration explores both slots on every GPU. Do not leave unused
        # optimizer models resident when the actual ensemble needs fewer slots.
        for device, slot in set(self._workers) - set(targets):
            self._retire(device, slot)
        print(f"[structure_parallel] scheduling {len(candidates)} candidates x "
              f"{n_configs} configurations on "
              f"{len(targets)} slots across {len({d for d, _ in targets})} GPUs", flush=True)
        started = time.monotonic()
        jobs = [(index // n_configs, index % n_configs, seed)
                for index, seed in enumerate(seeds)]
        replies = self._run_jobs(candidates, jobs, targets, label="config")
        results: list[list[Atoms] | Exception] = []
        for index in range(len(candidates)):
            candidate_replies = replies[index * n_configs:(index + 1) * n_configs]
            atoms = [reply["atoms"] for reply in candidate_replies if reply.get("ok")]
            print(f"[structure_parallel] candidate {index + 1}/{len(candidates)}: "
                  f"{len(atoms)}/{n_configs} configurations succeeded", flush=True)
            if len(atoms) < self.min_successful:
                failure = next((r.get("error") for r in candidate_replies if not r.get("ok")), "no structures")
                results.append(RuntimeError(
                    f"Only {len(atoms)}/{n_configs} configurations succeeded for "
                    f"candidate {index + 1}/{len(candidates)}; "
                    f"min_successful_configs={self.min_successful}. First failure: {failure}"
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
