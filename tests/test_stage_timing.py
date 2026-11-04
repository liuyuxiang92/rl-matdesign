"""Measured stage durations, including failures and unknown external hooks."""
from __future__ import annotations

import pytest

from rl_matdesign.utils import stage_timing
from rl_matdesign.utils.stage_timing import (
    capture_stage_times, summarize_stage_times, timed_stage,
)


def test_stage_capture_records_failure_and_does_not_count_nested_hook_twice(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(stage_timing.time, "perf_counter", lambda: now[0])
    with capture_stage_times() as capture:
        with pytest.raises(ValueError, match="MD failed"):
            with timed_stage("md"):
                now[0] = 2
                with timed_stage("md"):
                    now[0] = 5
                    raise ValueError("MD failed")
        with timed_stage("md"):
            now[0] = 8
    assert capture.seconds == {"md": 8}
    assert capture.counts == {"md": 2}
    assert summarize_stage_times(capture, 10, md_expected=True) == {
        "md_task_s": 8, "opt_task_s": 0, "structure_other_task_s": 2,
        "config_attempted": 1, "config_succeeded": 0,
    }


def test_independent_nested_capture_restores_outer_and_uncaptured_hooks_are_noops(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(stage_timing.time, "perf_counter", lambda: now[0])
    with capture_stage_times() as outer:
        with capture_stage_times() as inner:
            with timed_stage("md"):
                now[0] += 3
        with timed_stage("opt"):
            now[0] += 2
    with timed_stage("md"):
        now[0] += 100
    assert inner.seconds == {"md": 3}
    assert outer.seconds == {"opt": 2}


def test_expected_uninstrumented_stage_is_unknown_and_declared_zero_is_known():
    with capture_stage_times() as capture:
        pass
    unknown = summarize_stage_times(capture, 20, md_expected=True, opt_expected=True)
    assert unknown["md_task_s"] is None
    assert unknown["opt_task_s"] is None
    assert unknown["structure_other_task_s"] is None
    declared = summarize_stage_times(
        capture, 20, md_expected=True, opt_expected=True, supported_stages=("md", "opt"),
    )
    assert declared["md_task_s"] == declared["opt_task_s"] == 0
    assert declared["structure_other_task_s"] == 20
    unknown_attempts = summarize_stage_times(capture, 0, config_attempted=None)
    assert unknown_attempts["config_attempted"] is None


def test_shared_optimizer_records_nonconverged_and_raised_runs(monkeypatch):
    ase = pytest.importorskip("ase")
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.optimize import LBFGS
    from rl_matdesign.utils.structure import (
        RelaxationNotConverged, relax_structure, strict_relaxation_errors,
    )

    now = [0.0]
    monkeypatch.setattr(stage_timing.time, "perf_counter", lambda: now[0])
    def failed_run(self, **kwargs):
        now[0] += 4
        return False

    monkeypatch.setattr(LBFGS, "run", failed_run)
    with capture_stage_times() as capture, strict_relaxation_errors():
        with pytest.raises(RelaxationNotConverged):
            relax_structure(ase.Atoms("H"), calc=SinglePointCalculator(ase.Atoms("H")), relax_cell=False)
    assert capture.seconds == {"opt": 4}
    def raised_run(self, **kwargs):
        now[0] += 3
        raise RuntimeError("optimization failed")
    monkeypatch.setattr(LBFGS, "run", raised_run)
    with capture_stage_times() as capture, strict_relaxation_errors():
        with pytest.raises(RuntimeError, match="optimization failed"):
            relax_structure(ase.Atoms("H"), calc=SinglePointCalculator(ase.Atoms("H")), relax_cell=False)
    assert capture.seconds == {"opt": 3}
