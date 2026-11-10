"""Native property snapshots are cache-only and retain gate/units semantics."""
from copy import deepcopy

import pytest

from rl_matdesign.configuration import _TerminalReward
from rl_matdesign.predictors.ooh import OOHCatalystPredictor
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.prediction_records import prediction_record
from rl_matdesign.utils.timing import PredictorTimer
from tests.test_reward_integration import _cfg


def test_structure_scalar_record_preserves_raw_gate_values_and_real_timing(monkeypatch):
    predictor = StructureScorePredictor(_cfg())
    timer = PredictorTimer(predictor)
    adapter = _TerminalReward(timer, "multi_group")
    adapter.capture_prediction_records = True
    candidate = {"delta": .07, "pure": .4, "cost": 0.0}
    reward = adapter(candidate)
    assert reward == pytest.approx(-10.8)
    record = adapter.last_prediction_records[0]
    assert record["reward"] == reward
    assert record["obj_delta_mean"] == .07
    assert record["dp_mean"] == pytest.approx(-10.8)
    assert record["reward_objective"] == .07
    assert record["gate_passed"] is False
    assert record["gate_synthesis_violation"] == .4
    assert timer.n_calls == 1
    before = deepcopy(predictor.last_evaluation_timings)
    assert before[0]["cache_hit"] is False
    monkeypatch.setattr(predictor, "_raw_stats", lambda _: pytest.fail("snapshot triggered inference"))
    assert predictor.prediction_record(candidate, (reward, 0))["obj_delta_mean"] == .07
    assert predictor.last_evaluation_timings == before
    assert adapter.last_evaluation_timings == before


def test_batch_records_survive_tiny_lru_without_another_build(monkeypatch):
    cfg = _cfg()
    cfg["predict_cache_size"] = 1
    predictor = StructureScorePredictor(cfg)
    adapter = _TerminalReward(predictor, "multi_group")
    adapter.capture_prediction_records = True
    candidates = [{"delta": value, "pure": .9, "cost": 0.0} for value in (.01, .02, .03)]
    assert adapter.evaluate_many(candidates) == pytest.approx([.01, .02, .03])
    assert len(predictor._stats_cache) == 1
    assert [r["obj_delta_mean"] for r in adapter.last_prediction_records] == [.01, .02, .03]
    before = deepcopy(predictor.last_evaluation_timings)
    monkeypatch.setattr(predictor, "_raw_stats", lambda _: pytest.fail("evicted candidate rebuilt"))
    for candidate, expected in zip(candidates, (.01, .02, .03)):
        assert predictor.prediction_record(candidate, (expected, 0))["obj_delta_mean"] == expected
    assert predictor.last_evaluation_timings == before
    assert all(not row["cache_hit"] for row in before)


def test_ooh_record_keeps_positive_overpotential_when_reward_is_negative(monkeypatch):
    # No DeepMD load is needed to validate the cache/serialization boundary.
    predictor = OOHCatalystPredictor.__new__(OOHCatalystPredictor)
    predictor._settings_key = (("O", "OH", "OOH"), 0)
    candidate = {"Ni": .7, "Fe": .3}
    predictor._cache = {predictor._comp_key(candidate): (310.0, 12.0)}
    monkeypatch.setattr(predictor, "check_phase", lambda _: (True, "NiFe"))
    monkeypatch.setattr(predictor, "predict_raw", lambda _: pytest.fail("raw re-evaluation"))
    record = prediction_record(predictor, candidate, (-322.0, 12.0), k=2)
    assert record["reward"] == -322.0
    assert record["dp_mean"] == 310.0
    assert record["dp_std"] == 12.0
    assert record["dp_mean_minus_std"] == 286.0
    assert record["primary_label"] == "NiFe"


def test_bad_optional_snapshot_does_not_turn_scored_candidate_into_retry():
    class Predictor:
        calls = 0

        def predict(self, _):
            self.calls += 1
            return .5, .1

        def prediction_record(self, *args, **kwargs):
            raise RuntimeError("diagnostics unavailable")

    predictor = Predictor()
    adapter = _TerminalReward(predictor, "multi_group")
    adapter.capture_prediction_records = True
    assert adapter({"Fe": 1}) == .5
    assert predictor.calls == 1
    assert adapter.last_prediction_records[0]["reward"] == .5
    assert "diagnostics unavailable" in adapter.last_prediction_records[0]["prediction_record_error"]
