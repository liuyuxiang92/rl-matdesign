"""Exact fake-clock accounting through structure prediction and its caches."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_matdesign.utils import stage_timing
from rl_matdesign.utils.stage_timing import (
    capture_stage_times, summarize_stage_times, timed_stage,
)
from tests.test_cross_formula_predictor import _candidate, _predictor


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def advance(self, seconds):
        self.seconds += seconds

    def now(self):
        return self.seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(stage_timing.time, "perf_counter", clock.now)
    return clock


class TimedSerialBuilder:
    timing_stages = ("md", "opt")
    md = {"model": "fake-md"}
    opt = {"model": "fake-opt"}

    def __init__(self, clock, *, failures=None):
        self.clock = clock
        self.failures = failures or {}
        self.calls = []

    def build(self, candidate, *, n_configs, rng):
        self.calls.append(candidate)
        structures = []
        for index in range(n_configs):
            with timed_stage("md"):
                self.clock.advance(2)
                if self.failures.get(candidate["Fe"]) == "md":
                    raise RuntimeError("MD failed")
            with timed_stage("opt"):
                self.clock.advance(3)
                if self.failures.get(candidate["Fe"]) == "opt":
                    raise RuntimeError("OPT failed")
            self.clock.advance(1)  # Explicit setup/postprocessing overhead.
            structures.append(SimpleNamespace(candidate=candidate, index=index))
        return structures


class TimedBatchBuilder(TimedSerialBuilder):
    parallel_execution = True
    handles_geo_opt = True

    def build_many(self, candidates, *, n_configs, rng):
        self.last_build_timings = []
        results = []
        for candidate in candidates:
            started = self.clock.now()
            succeeded = 0
            with capture_stage_times() as capture:
                try:
                    structures = self.build(candidate, n_configs=n_configs, rng=rng)
                    succeeded = len(structures)
                    results.append(structures)
                except Exception as exc:
                    results.append(exc)
            self.last_build_timings.append(summarize_stage_times(
                capture, self.clock.now() - started, md_expected=True, opt_expected=True,
                supported_stages=self.timing_stages, config_attempted=n_configs,
                config_succeeded=succeeded,
            ))
        return results


def _timed_predictor(clock, *, shared=True, batch=False, failures=None):
    predictor = _predictor(share_structure=shared)
    factory = TimedBatchBuilder if batch else TimedSerialBuilder
    if shared:
        predictor._shared_builder = factory(clock, failures=failures)
    else:
        predictor._builders = {
            prop["name"]: factory(clock, failures=failures if index == 0 else None)
            for index, prop in enumerate(predictor.properties)
        }
    predictor._parallel_models_ready = True  # No DeepMD imports in fake scoring.

    def score(prop, structures):
        clock.advance(7 if prop["name"] == "benefit" else 5)
        return 1.0, 0.0

    predictor._score = score
    return predictor


@pytest.mark.parametrize("shared", [True, False])
def test_scalar_build_stages_and_properties_have_exact_nonoverlapping_times(clock, shared):
    predictor = _timed_predictor(clock, shared=shared)
    predictor.predict(_candidate(.2))
    row = predictor.last_evaluation_timings[0]
    multiplier = 1 if shared else 2
    assert row["md_task_s"] == 4 * multiplier
    assert row["opt_task_s"] == 6 * multiplier
    assert row["structure_other_task_s"] == 2 * multiplier
    assert row["property_s"] == 12
    assert row["config_succeeded"] == 2 * multiplier
    assert row["config_attempted"] is None
    assert not row["cache_hit"]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("batch", [True, False])
def test_batch_records_align_with_candidates_and_coalesced_duplicates_cost_zero(clock, shared, batch):
    predictor = _timed_predictor(clock, shared=shared, batch=batch)
    first, second = _candidate(.1), _candidate(.2)
    predictor.batch_predict([first, second, first.copy()])
    first_row, second_row, duplicate = predictor.last_evaluation_timings
    for row in (first_row, second_row):
        multiplier = 1 if shared else 2
        assert row["md_task_s"] == 4 * multiplier
        assert row["opt_task_s"] == 6 * multiplier
        assert row["structure_other_task_s"] == 2 * multiplier
        assert row["property_s"] == 12
        assert row["config_succeeded"] == 2 * multiplier
        assert row["config_attempted"] == (2 * multiplier if batch else None)
    assert duplicate["cache_hit"] is True
    assert duplicate["md_task_s"] == duplicate["opt_task_s"] == duplicate["property_s"] == 0
    assert duplicate["config_attempted"] == duplicate["config_succeeded"] == 0
    before = clock.now()
    predictor.batch_predict([second, first])
    assert clock.now() == before
    assert all(row["cache_hit"] for row in predictor.last_evaluation_timings)
    assert all(row["md_task_s"] == row["property_s"] == 0 for row in predictor.last_evaluation_timings)


def test_scalar_cache_and_diagnostic_reads_do_not_reuse_original_stage_cost(clock):
    predictor = _timed_predictor(clock)
    candidate = _candidate(.2)
    predictor.predict(candidate)
    assert predictor.last_evaluation_timings[0]["md_task_s"] == 4
    before = clock.now()
    predictor.predict(candidate)
    row = predictor.last_evaluation_timings[0]
    assert row["cache_hit"] is True
    assert row["md_task_s"] == row["opt_task_s"] == row["property_s"] == 0
    predictor.per_objective_stats(candidate)
    predictor.reward_details(candidate)
    assert clock.now() == before


@pytest.mark.parametrize("stage,md_seconds,opt_seconds", [("md", 2, 0), ("opt", 2, 3)])
def test_failed_serial_stages_keep_time_and_do_not_enter_cache(clock, stage, md_seconds, opt_seconds):
    predictor = _timed_predictor(clock, failures={.1: stage})
    bad = _candidate(.1)
    with pytest.raises(RuntimeError, match="failed"):
        predictor.predict(bad)
    row = predictor.last_evaluation_timings[0]
    assert row["md_task_s"] == md_seconds
    assert row["opt_task_s"] == opt_seconds
    assert row["property_s"] == 0
    assert row["config_succeeded"] == 0
    assert predictor._key(bad) not in predictor._stats_cache


def test_property_failure_retains_build_and_elapsed_inference_time(clock):
    predictor = _timed_predictor(clock)
    def failing_score(prop, structures):
        clock.advance(9)
        raise ValueError("model inference failed")
    predictor._score = failing_score
    with pytest.raises(ValueError, match="inference failed"):
        predictor.predict(_candidate(.1))
    row = predictor.last_evaluation_timings[0]
    assert row["md_task_s"] == 4
    assert row["opt_task_s"] == 6
    assert row["property_s"] == 9
    assert not predictor._stats_cache


def test_multiple_optimization_passes_do_not_invent_extra_configuration_attempts(clock):
    predictor = _timed_predictor(clock)
    predictor.geo_opt_enabled = True
    def relax(atoms):
        with timed_stage("opt"):
            clock.advance(11)
        return atoms
    predictor._relax = relax
    predictor.predict(_candidate(.1))
    row = predictor.last_evaluation_timings[0]
    assert row["opt_task_s"] == 6 + 22
    assert row["config_succeeded"] == 2
    assert row["config_attempted"] is None


@pytest.mark.parametrize("survivors", [0, 1])
def test_parent_relaxation_after_batch_counts_failed_opt_time_and_actual_survivors(clock, survivors):
    predictor = _timed_predictor(clock, batch=True)
    predictor.geo_opt_enabled = True
    predictor._shared_builder.handles_geo_opt = False
    def relax(atoms):
        with timed_stage("opt"):
            clock.advance(11)
            if atoms.index >= survivors:
                raise RuntimeError("unconverged")
        return atoms
    predictor._relax = relax
    result = predictor.batch_predict([_candidate(.1)], return_exceptions=True)
    row = predictor.last_evaluation_timings[0]
    assert isinstance(result[0], Exception) == (survivors == 0)
    assert row["md_task_s"] == 4
    assert row["opt_task_s"] == 6 + 22
    assert row["structure_other_task_s"] == 2
    assert row["config_attempted"] == 2
    assert row["config_succeeded"] == survivors
    assert row["property_s"] == (12 if survivors else 0)


def test_pool_timeout_sidechannel_stays_aligned_around_cache_hits_and_duplicates(clock):
    predictor = _timed_predictor(clock)
    cached, first, second = [_candidate(value) for value in (.1, .2, .3)]
    predictor.predict(cached)
    class TimeoutBuilder:
        parallel_execution = True
        handles_geo_opt = True
        def build_many(self, candidates, **kwargs):
            assert candidates == [first, second]
            self.last_build_timings = [
                {"md_task_s": 11, "opt_task_s": 22, "structure_other_task_s": 33,
                 "config_attempted": 2, "config_succeeded": 2, "timing_incomplete": False},
                {"md_task_s": None, "opt_task_s": None, "structure_other_task_s": None,
                 "config_attempted": 1, "config_succeeded": 0, "timing_incomplete": True},
            ]
            clock.advance(100)
            raise TimeoutError("worker timed out")
    predictor._shared_builder = TimeoutBuilder()
    result = predictor.batch_predict([cached, first, second, first.copy()], return_exceptions=True)
    assert all(isinstance(item, TimeoutError) for item in result[1:])
    cache, finished, unfinished, duplicate = predictor.last_evaluation_timings
    assert cache["cache_hit"] is True
    assert cache["md_task_s"] == duplicate["md_task_s"] == 0
    assert finished["md_task_s"] == 11
    assert finished["opt_task_s"] == 22
    assert finished["config_succeeded"] == 2
    assert unfinished["md_task_s"] is None
    assert unfinished["timing_incomplete"] is True
    assert unfinished["config_attempted"] == 1
    assert duplicate["cache_hit"] is False
    assert all(row["property_s"] == 0 for row in predictor.last_evaluation_timings)


def test_separate_property_groups_keep_alignment_when_first_group_skips_a_formula(clock):
    predictor = _timed_predictor(clock, shared=False, batch=True, failures={.1: "md"})
    bad, good = _candidate(.1), _candidate(.2)
    result = predictor.batch_predict([bad, good], return_exceptions=True)
    assert isinstance(result[0], RuntimeError)
    bad_row, good_row = predictor.last_evaluation_timings
    assert bad_row["md_task_s"] == 2
    assert bad_row["opt_task_s"] == bad_row["property_s"] == 0
    assert good_row["md_task_s"] == 8
    assert good_row["opt_task_s"] == 12
    assert good_row["property_s"] == 12
    assert predictor._builders["cost"].calls == [good]


def test_uninstrumented_serial_physics_reports_unknown_stage_costs(clock):
    predictor = _timed_predictor(clock)
    class OldBuilder:
        md = {"model": "old"}
        opt = {"model": "old"}
        def build(self, candidate, *, n_configs, rng):
            clock.advance(27)
            return [SimpleNamespace(candidate=candidate) for _ in range(n_configs)]
    predictor._shared_builder = OldBuilder()
    predictor.predict(_candidate(.1))
    row = predictor.last_evaluation_timings[0]
    assert row["md_task_s"] is None
    assert row["opt_task_s"] is None
    assert row["structure_other_task_s"] is None
    assert row["property_s"] == 12
