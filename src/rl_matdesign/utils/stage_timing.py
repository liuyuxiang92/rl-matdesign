"""Optional, process-local timing hooks for expensive structure stages.

Builders may declare ``timing_stages = ("md", "opt")`` for stages they always
instrument, and surround their work with ``timed_stage("md")``. The hooks add
no configuration and are no-ops outside an active capture. Timings survive
exceptions; they measure elapsed task seconds, not GPU utilization or parallel
batch wall time.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Iterator


@dataclass
class StageTimes:
    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _active: set[str] = field(default_factory=set, repr=False)


_CAPTURE: ContextVar[StageTimes | None] = ContextVar("structure_stage_timing", default=None)


@contextmanager
def capture_stage_times() -> Iterator[StageTimes]:
    """Collect an independent set of stages, restoring any outer capture."""
    capture = StageTimes()
    token = _CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CAPTURE.reset(token)


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """Record a stage including unsuccessful attempts, without nested duplicates."""
    capture = _CAPTURE.get()
    if capture is None or name in capture._active:
        yield
        return
    capture._active.add(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        capture.seconds[name] = capture.seconds.get(name, 0.0) + time.perf_counter() - started
        capture.counts[name] = capture.counts.get(name, 0) + 1
        capture._active.remove(name)


def summarize_stage_times(
    capture: StageTimes,
    elapsed_s: float,
    *,
    md_expected: bool = False,
    opt_expected: bool = False,
    supported_stages=(),
    config_attempted: int | None = 1,
    config_succeeded: int = 0,
) -> dict:
    """Distinguish a known zero from an uninstrumented, unknown stage.

    A declared hook makes zero meaningful even when a precheck fails before
    reaching it. Without a declaration or an observed hook, an expected stage
    is unknown. Never infer MD time from the whole builder's elapsed time.
    ``config_attempted=None`` means a serial/plugin builder did not expose its
    actual attempts; requested ensemble size or stage-call counts cannot reveal
    early aborts, retries, or multiple optimization passes per configuration.
    """
    supported = {supported_stages} if isinstance(supported_stages, str) else set(supported_stages or ())

    def seconds(name: str, expected: bool) -> float | None:
        if name in supported or capture.counts.get(name, 0) or not expected:
            return float(capture.seconds.get(name, 0.0))
        return None

    md_s = seconds("md", md_expected)
    opt_s = seconds("opt", opt_expected)
    return {
        "md_task_s": md_s,
        "opt_task_s": opt_s,
        "structure_other_task_s": (
            max(0.0, float(elapsed_s) - md_s - opt_s)
            if md_s is not None and opt_s is not None else None
        ),
        "config_attempted": None if config_attempted is None else int(config_attempted),
        "config_succeeded": int(config_succeeded),
    }
