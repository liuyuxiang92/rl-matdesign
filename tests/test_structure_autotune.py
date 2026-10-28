"""Pure calibration scheduling tests; no CUDA or physics dependencies."""
from __future__ import annotations

import math

import pytest

from rl_matdesign.utils.structure_autotune import autotune_concurrency


def ok(elapsed=10):
    return {"ok": True, "elapsed": elapsed}


def fail(error="CUDA out of memory", elapsed=1):
    return {"ok": False, "elapsed": elapsed, "error": error}


class Batches:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.retired = []
        self.events = []

    def run(self, jobs):
        self.calls.append(jobs)
        self.events.append(("run", jobs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def retire(self, device, slot):
        self.retired.append((device, slot))
        self.events.append(("retire", device, slot))

    def tune(self, devices=2, **kwargs):
        return autotune_concurrency(devices, self.run, self.retire, seed=123, **kwargs)


def test_full_physics_sequence_uses_identical_seed_and_ordered_device_reports():
    batches = Batches(
        [ok(30), ok(30)],  # Primary model-load/physics warmup is not measured.
        [ok(10), ok(20)],
        [ok(30), ok(30)],  # Primary remains resident during secondary warmup.
        [ok(13), ok(14), ok(39), ok(40)],
    )
    report = batches.tune()

    assert report["counts"] == [2, 1]
    assert [m["device"] for m in report["measurements"]] == [0, 1]
    assert report["measurements"][0] == {
        "device": 0, "concurrency": 2, "reason": "throughput_gain",
        "t_single": 10.0, "t_pair": 14.0, "speedup": 20 / 14,
    }
    assert report["measurements"][1] == {
        "device": 1, "concurrency": 1, "reason": "insufficient_speedup",
        "t_single": 20.0, "t_pair": 40.0, "speedup": 1.0,
    }
    assert batches.calls == [
        [{"device": 0, "slot": 0, "seed": 123}, {"device": 1, "slot": 0, "seed": 123}],
        [{"device": 0, "slot": 0, "seed": 123}, {"device": 1, "slot": 0, "seed": 123}],
        [{"device": 0, "slot": 1, "seed": 123}, {"device": 1, "slot": 1, "seed": 123}],
        [{"device": 0, "slot": 0, "seed": 123}, {"device": 0, "slot": 1, "seed": 123},
         {"device": 1, "slot": 0, "seed": 123}, {"device": 1, "slot": 1, "seed": 123}],
    ]
    assert batches.retired == [(1, 1)]
    assert batches.replies == []


def test_secondary_oom_retires_worker_before_testing_remaining_gpu():
    batches = Batches([ok(), ok()], [ok(), ok()], [fail(), ok()], [ok(12), ok(14)])
    report = batches.tune()

    assert report["counts"] == [1, 2]
    assert report["measurements"][0] == {
        "device": 0, "concurrency": 1, "reason": "secondary_warmup_failed",
        "t_single": 10.0, "t_pair": None, "speedup": None, "error": "CUDA out of memory",
    }
    assert batches.calls[-1] == [
        {"device": 1, "slot": 0, "seed": 123}, {"device": 1, "slot": 1, "seed": 123},
    ]
    assert batches.events[3] == ("retire", 0, 1)
    assert batches.retired == [(0, 1)]


@pytest.mark.parametrize("failed_slot", [0, 1])
def test_pair_failure_falls_back_to_one_even_if_failure_is_primary_slot(failed_slot):
    pair = [ok(12), ok(12)]
    pair[failed_slot] = fail("pair model OOM")
    batches = Batches([ok()], [ok()], [ok()], pair)
    report = batches.tune(devices=1)

    assert report["counts"] == [1]
    assert report["measurements"][0]["reason"] == "concurrent_pair_failed"
    assert report["measurements"][0]["error"] == "pair model OOM"
    assert report["measurements"][0]["t_pair"] is None
    assert report["measurements"][0]["speedup"] is None
    assert batches.retired == [(0, 1)]


def test_all_secondary_warmups_fail_without_running_empty_batch():
    batches = Batches([ok(), ok()], [ok(), ok()], [fail(), fail("another OOM")])
    report = batches.tune()

    assert report["counts"] == [1, 1]
    assert batches.retired == [(0, 1), (1, 1)]
    assert len(batches.calls) == 3
    assert all(m["reason"] == "secondary_warmup_failed" for m in report["measurements"])


def test_all_concurrent_pairs_fail_but_valid_baseline_remains_usable():
    batches = Batches([ok(), ok()], [ok(), ok()], [ok(), ok()], [fail(), fail(), fail(), fail()])
    report = batches.tune()

    assert report["counts"] == [1, 1]
    assert batches.retired == [(0, 1), (1, 1)]
    assert all(m["reason"] == "concurrent_pair_failed" for m in report["measurements"])


@pytest.mark.parametrize("phase", [0, 1])
def test_baseline_failure_aborts_with_device_phase_and_error(phase):
    replies = [[ok(), ok()], [ok(), ok()]]
    replies[phase][1] = fail("bad input structure")
    batches = Batches(*replies)

    with pytest.raises(RuntimeError, match="GPU 1: bad input structure") as exc:
        batches.tune()
    assert ("primary warmup" if phase == 0 else "single-worker measurement") in str(exc.value)
    assert len(batches.calls) == phase + 1
    assert batches.retired == []  # Caller owns cleanup of fatal calibration failures.


def test_speedup_threshold_is_inclusive_and_uses_slowest_pair_member():
    batches = Batches([ok()], [ok(23)], [ok()], [ok(1), ok(40)])
    report = batches.tune(devices=1, min_speedup=1.15)

    assert report["counts"] == [2]
    assert report["measurements"][0]["speedup"] == pytest.approx(1.15)
    assert batches.retired == []


@pytest.mark.parametrize("phase", range(4))
def test_infrastructure_exceptions_propagate_for_caller_pool_cleanup(phase):
    failure = TimeoutError("GPU worker hung")
    replies = [[ok()], [ok()], [ok()], [ok(), ok()]]
    replies[phase] = failure
    batches = Batches(*replies)

    with pytest.raises(TimeoutError) as exc:
        batches.tune(devices=1)
    assert exc.value is failure


@pytest.mark.parametrize("elapsed", [None, 0, -1, float("nan"), float("inf"), True, "invalid"])
@pytest.mark.parametrize("phase", range(4))
def test_invalid_measurements_cannot_select_concurrency(elapsed, phase):
    replies = [[ok()], [ok()], [ok()], [ok(), ok()]]
    replies[phase][0] = ok(elapsed)
    batches = Batches(*replies)

    with pytest.raises(RuntimeError, match="elapsed must be finite and positive"):
        batches.tune(devices=1)


@pytest.mark.parametrize("reply", [[], [ok(), ok()], None, "not replies", [{"ok": 1, "elapsed": 10}]])
def test_invalid_reply_shape_is_reported(reply):
    batches = Batches(reply)
    with pytest.raises(RuntimeError, match="Autotune primary warmup"):
        batches.tune(devices=1)


@pytest.mark.parametrize("device_count", [0, -1, True, 1.5])
def test_device_count_validation(device_count):
    batches = Batches()
    with pytest.raises(ValueError, match="device_count"):
        batches.tune(devices=device_count)
    assert batches.calls == []


@pytest.mark.parametrize("threshold", [0, .9, -1, math.inf, math.nan, True, None])
def test_minimum_speedup_validation(threshold):
    batches = Batches()
    with pytest.raises(ValueError, match="min_speedup"):
        batches.tune(min_speedup=threshold)
    assert batches.calls == []


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_seed_validation(seed):
    batches = Batches()
    with pytest.raises(ValueError, match="seed"):
        autotune_concurrency(1, batches.run, batches.retire, seed=seed)
    assert batches.calls == []
