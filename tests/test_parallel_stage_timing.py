"""Timing crosses real worker IPC for successes, skipped configs and crashes."""
from __future__ import annotations

import pytest

from tests.test_parallel_structures import pool_factory


_TIMED_BUILDER = '''
import time
from parallel_test_builder import Builder as BaseBuilder
from rl_matdesign.utils.stage_timing import timed_stage

class Builder(BaseBuilder):
    timing_stages = ("md", "opt")
    def build(self, candidate, *, n_configs=1, rng=None):
        if candidate.get("timing_fail") == "precheck":
            raise ValueError("invalid initial structure")
        with timed_stage("md"):
            time.sleep(.015)
            if candidate.get("timing_fail") == "md":
                raise RuntimeError("MD failed after spending time")
        with timed_stage("opt"):
            time.sleep(.01)
            if candidate.get("timing_fail") == "opt":
                raise RuntimeError("optimization did not converge")
        return super().build(candidate, n_configs=n_configs, rng=rng)
'''


def _timed_pool(pool_factory, tmp_path, **kwargs):
    pool, root = pool_factory(cfg_extra={"md": {"model": "fake"}, "opt": {"model": "fake"}}, **kwargs)
    (tmp_path / "fake_modules" / "timed_builder.py").write_text(_TIMED_BUILDER)
    pool.kind = "timed_builder:Builder"
    return pool, root


def test_measured_stage_times_and_failures_are_returned_in_formula_order(pool_factory, tmp_path):
    pool, _ = _timed_pool(pool_factory, tmp_path)
    result = pool.build_many([
        {"id": "ok"}, {"timing_fail": "md"}, {"timing_fail": "opt"}, {"timing_fail": "precheck"},
    ], n_configs=2)
    assert len(result[0]) == 2
    assert all(isinstance(item, RuntimeError) for item in result[1:])
    success, md_failure, opt_failure, precheck_failure = pool.last_build_timings
    assert [item["config_attempted"] for item in pool.last_build_timings] == [2] * 4
    assert [item["config_succeeded"] for item in pool.last_build_timings] == [2, 0, 0, 0]
    assert all(not item["timing_incomplete"] for item in pool.last_build_timings)
    for item in (success, md_failure, opt_failure):
        assert item["md_task_s"] >= .03
    assert success["opt_task_s"] >= .02
    assert opt_failure["opt_task_s"] >= .02
    assert md_failure["opt_task_s"] == 0
    assert precheck_failure["md_task_s"] == precheck_failure["opt_task_s"] == 0
    assert all(item["structure_other_task_s"] >= 0 for item in pool.last_build_timings)
    assert pool.last_build_wall_s > 0
    assert len(pool.last_build_job_timings) == 8
    assert all(item["timing"] is not None for item in pool.last_build_job_timings)


def test_uninstrumented_external_md_is_unknown_not_builder_time(pool_factory):
    pool, _ = pool_factory(cfg_extra={"md": {"model": "old_plugin"}, "opt": {"model": "old_plugin"}})
    pool.build({}, n_configs=1)
    timing = pool.last_build_timings[0]
    assert timing["md_task_s"] is None
    assert timing["opt_task_s"] is None
    assert timing["structure_other_task_s"] is None
    assert timing["config_succeeded"] == 1


def test_shared_optimizer_failure_keeps_observed_duration_in_worker_reply(pool_factory):
    pool, _ = pool_factory(cfg_extra={"opt": {"model": "fake"}})
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        pool.build({"mode": "internal_relax_oom"}, n_configs=1)
    timing = pool.last_build_timings[0]
    assert timing["md_task_s"] == 0
    assert timing["opt_task_s"] > 0
    assert timing["config_attempted"] == 1
    assert timing["config_succeeded"] == 0


def test_timeout_preserves_finished_formula_and_marks_inflight_unknown(pool_factory, tmp_path):
    pool, _ = _timed_pool(pool_factory, tmp_path, parallel_extra={"timeout_s": 1})
    with pytest.raises(TimeoutError):
        pool.build_many([{"mode": "hang"}, {}], n_configs=1)
    hanging, completed = pool.last_build_timings
    assert hanging["config_attempted"] == 1
    assert hanging["config_succeeded"] == 0
    assert hanging["timing_incomplete"] is True
    assert hanging["md_task_s"] is None
    assert completed["config_succeeded"] == 1
    assert completed["timing_incomplete"] is False
    assert completed["md_task_s"] > 0
    assert pool.last_build_wall_s >= 1
    assert pool.last_build_job_timings[1]["timing"]["md_task_s"] > 0
    assert not pool._workers


def test_empty_batch_clears_previous_timings(pool_factory):
    pool, _ = pool_factory()
    pool.build({}, n_configs=1)
    assert pool.last_build_timings
    assert pool.build_many([]) == []
    assert pool.last_build_timings == pool.last_build_job_timings == []
    assert pool.last_build_wall_s == 0
