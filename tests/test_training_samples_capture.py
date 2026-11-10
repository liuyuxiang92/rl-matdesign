"""Training diagnostics respect timed scalar fallbacks and predictor contracts."""
import random

import numpy as np
import pytest
import torch

from rl_matdesign.configuration import _TerminalReward
from rl_matdesign.utils.episode_batch import CandidateEvaluationFailed
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
from rl_matdesign.utils.seeding import capture_rng_state, restore_rng_state
from rl_matdesign.utils.timing import PredictorTimer


class LatestPrediction:
    """A plugin retains just one raw realization, like many external adapters."""

    def __init__(self):
        self.calls = []
        self.hooks = []

    def predict(self, candidate):
        self.calls.append(candidate)
        self.last_evaluation_timings = [{"cache_hit": False, "property_s": candidate["x"] / 10}]
        value = random.random() + np.random.random() + torch.rand(()).item()
        if candidate["x"] < 0:
            raise RuntimeError("one failed realization")
        self.latest = (candidate, value)
        return value, None

    def predict_raw(self, candidate):
        raise AssertionError("export must never re-evaluate a candidate")

    def prediction_record(self, candidate, prediction, *, k=1.):
        self.hooks.append(candidate)
        assert self.latest[0] == candidate, "later candidate overwrote the raw record"
        return {"dp_mean": self.latest[1] * 10, "dp_std": None,
                "dp_mean_minus_std": None, "meta_position": candidate["x"]}


def _run_capture(enabled):
    inner = LatestPrediction()
    timer = PredictorTimer(inner, eval_seed=17)
    adapter = _TerminalReward(timer, "multi_group")
    adapter.capture_prediction_records = enabled
    candidates = [{"x": 1}, {"x": -1}, {"x": 2}, {"x": 1}]
    results = adapter.evaluate_many(candidates)
    return inner, timer, adapter, results


def test_timer_scalar_batch_snapshots_before_cache_is_overwritten_and_keeps_accounting():
    state = capture_rng_state()
    expected_random = (random.random(), np.random.random(), torch.rand(()).item())
    restore_rng_state(state)
    off = _run_capture(False)
    on = _run_capture(True)
    actual_random = (random.random(), np.random.random(), torch.rand(()).item())
    assert actual_random == expected_random

    inner, timer, adapter, results = on
    assert inner.calls == off[0].calls == [{"x": 1}, {"x": -1}, {"x": 2}, {"x": 1}]
    assert inner.hooks == [{"x": 1}, {"x": 2}, {"x": 1}]
    assert off[0].hooks == []
    assert isinstance(results[1], CandidateEvaluationFailed)
    assert [results[i] for i in (0, 2, 3)] == [off[3][i] for i in (0, 2, 3)]
    for i in (0, 2, 3):
        row = adapter.last_prediction_records[i]
        assert row["reward"] == results[i]
        assert row["dp_mean"] == results[i] * 10
        assert row["dp_std"] is None
        assert row["dp_mean_minus_std"] is None
        assert "prediction_record_error" not in row
    assert adapter.last_prediction_records[1] == {}
    assert len(adapter.last_evaluation_timings) == 4
    assert adapter.last_evaluation_timings == off[2].last_evaluation_timings
    assert timer.summary()["phases"] == off[1].summary()["phases"]
    assert timer.n_calls == off[1].n_calls == 3
    assert timer.summary()["diagnostic_requests"] == 0


def test_scalar_fallback_keeps_fatal_protocol_error_and_does_not_try_later_candidates():
    class Broken(LatestPrediction):
        def predict(self, candidate):
            if candidate["x"] == 2:
                self.calls.append(candidate)
                raise EvaluationProtocolError("invalid scientific output")
            return super().predict(candidate)

    inner = Broken()
    timer = PredictorTimer(inner)
    adapter = _TerminalReward(timer, "multi_group")
    adapter.capture_prediction_records = True
    with pytest.raises(EvaluationProtocolError, match="invalid scientific output"):
        adapter.evaluate_many([{"x": 1}, {"x": 2}, {"x": 3}])
    assert inner.calls == [{"x": 1}, {"x": 2}]
    assert inner.hooks == [{"x": 1}]


def test_native_candidate_seeded_batch_stays_batched_when_capturing():
    class SharedBatch:
        candidate_seeded_batch = True
        calls = 0

        def batch_predict(self, candidates, *, return_exceptions=False):
            self.calls += 1
            self.records = {item["x"]: item["x"] * 10 for item in candidates}
            return [(item["x"], None) for item in candidates]

        def predict(self, candidate):
            raise AssertionError("native shared batch must remain shared")

        def prediction_record(self, candidate, prediction, *, k=1.):
            return {"obj_delta_mean": self.records[candidate["x"]], "obj_delta_std": None}

    inner = SharedBatch()
    adapter = _TerminalReward(PredictorTimer(inner), "multi_group")
    adapter.capture_prediction_records = True
    assert adapter.evaluate_many([{"x": 1}, {"x": 2}]) == [1., 2.]
    assert inner.calls == 1
    assert [row["obj_delta_mean"] for row in adapter.last_prediction_records] == [10, 20]
    assert adapter.last_predictions == [(1, None), (2, None)]
