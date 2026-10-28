"""Measure whether each GPU benefits from two configuration workers.

This module only schedules calibration calls. The caller owns processes, GPU
binding, full-physics execution, timing, and cleanup on infrastructure failures.
Calibration structures never enter the returned candidate ensemble or RL data.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def autotune_concurrency(
    device_count: int,
    run_batch: Callable[[list[dict[str, int]]], list[dict[str, Any]]],
    retire: Callable[[int, int], None],
    *,
    seed: int,
    min_speedup: float = 1.15,
) -> dict[str, Any]:
    """Select one or two persistent configuration workers for each GPU.

    ``run_batch`` executes every supplied job concurrently and returns replies
    in the same order. Jobs contain ``device`` (index into the caller's GPU
    allocation), ``slot`` (0 or 1), and ``seed``. Replies contain ``ok``, elapsed
    wall seconds, and optionally ``error``. Infrastructure exceptions propagate.

    Both slots execute the same candidate (owned by the caller) and seed. Slot
    0 is warmed before its measured solo run. Slot 1 is warmed while slot 0
    remains resident, then both run together. Two workers are retained only if
    ``2 * solo_seconds / max(pair_seconds) >= min_speedup``. Failed second-slot
    warmups or pair runs fall back to one worker and retire slot 1 immediately.
    A failed first-slot warmup/measurement aborts calibration.

    Results include ordered ``counts`` and a ``measurements`` record per GPU.
    Unavailable pair measurements/speedups are ``None``. The caller must discard
    the calibration results; they are timing trials, not reward samples.
    """
    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 1:
        raise ValueError("device_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(min_speedup, bool):
        raise ValueError("min_speedup must be finite and at least 1")
    try:
        threshold = float(min_speedup)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_speedup must be finite and at least 1") from exc
    if not math.isfinite(threshold) or threshold < 1:
        raise ValueError("min_speedup must be finite and at least 1")

    def job(device: int, slot: int) -> dict[str, int]:
        return {"device": device, "slot": slot, "seed": seed}

    def run(jobs: list[dict[str, int]], phase: str) -> list[dict[str, Any]]:
        replies = run_batch(jobs)
        if (
            not isinstance(replies, Sequence)
            or isinstance(replies, (str, bytes))
            or len(replies) != len(jobs)
        ):
            raise RuntimeError(f"Autotune {phase}: expected {len(jobs)} ordered replies")
        checked = []
        for task, reply in zip(jobs, replies):
            context = f"Autotune {phase} on GPU {task['device']} slot {task['slot']}"
            if not isinstance(reply, Mapping) or not isinstance(reply.get("ok"), bool):
                raise RuntimeError(f"{context}: reply must contain a boolean 'ok'")
            elapsed = reply.get("elapsed")
            if isinstance(elapsed, bool):
                raise RuntimeError(f"{context}: elapsed must be finite and positive")
            try:
                elapsed = float(elapsed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{context}: elapsed must be finite and positive") from exc
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"{context}: elapsed must be finite and positive")
            checked.append({**reply, "elapsed": elapsed})
        return checked

    primary_jobs = [job(device, 0) for device in range(device_count)]
    for phase in ("primary warmup", "single-worker measurement"):
        baseline = run(primary_jobs, phase)
        for device, reply in enumerate(baseline):
            if not reply["ok"]:
                raise RuntimeError(
                    f"Autotune {phase} failed on GPU {device}: "
                    f"{reply.get('error', 'unknown worker error')}"
                )

    measurements = [
        {"device": device, "concurrency": 1, "reason": "insufficient_speedup",
         "t_single": reply["elapsed"], "t_pair": None, "speedup": None}
        for device, reply in enumerate(baseline)
    ]
    secondary = run([job(device, 1) for device in range(device_count)], "secondary warmup")
    pair_devices = []
    for device, reply in enumerate(secondary):
        if reply["ok"]:
            pair_devices.append(device)
        else:
            measurements[device].update(
                reason="secondary_warmup_failed", error=reply.get("error", "unknown worker error")
            )
            retire(device, 1)

    if pair_devices:
        pair_jobs = [job(device, slot) for device in pair_devices for slot in (0, 1)]
        paired = run(pair_jobs, "concurrent-pair measurement")
        for offset, device in enumerate(pair_devices):
            pair = paired[2 * offset:2 * offset + 2]
            report = measurements[device]
            failed = [reply for reply in pair if not reply["ok"]]
            if failed:
                report.update(
                    reason="concurrent_pair_failed",
                    error="; ".join(str(reply.get("error", "unknown worker error")) for reply in failed),
                )
                retire(device, 1)
                continue
            pair_seconds = max(reply["elapsed"] for reply in pair)
            speedup = 2 * report["t_single"] / pair_seconds
            if not math.isfinite(speedup) or speedup <= 0:
                raise RuntimeError(f"Autotune concurrent-pair measurement on GPU {device}: invalid speedup")
            report.update(t_pair=pair_seconds, speedup=speedup)
            if speedup >= threshold:
                report.update(concurrency=2, reason="throughput_gain")
            else:
                retire(device, 1)

    return {"counts": [report["concurrency"] for report in measurements],
            "measurements": measurements}
