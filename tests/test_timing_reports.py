"""Live timing reports remain readable across failures and resumed invocations."""
import csv
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.timing import PredictorTimer


def test_malformed_optional_timing_does_not_mask_prediction_failure():
    def fail(_candidate):
        raise RuntimeError("original prediction error")

    predictor = PredictorTimer(SimpleNamespace(
        predict=fail, last_evaluation_timings=[None],
    ))
    with pytest.raises(RuntimeError, match="original prediction error"):
        predictor.predict({"Ni": 1})
    assert predictor.last_evaluation_timings == [{}]
    assert predictor.t_predict_s >= 0
    assert predictor.n_calls == 0


def test_tuple_telemetry_is_copied_without_changing_prediction():
    inner = SimpleNamespace(predict=lambda _: (2, 0),
                            last_evaluation_timings=({"md_task_s": 3},))
    predictor = PredictorTimer(inner)
    assert predictor.predict({"Ni": 1}) == (2, 0)
    inner.last_evaluation_timings[0]["md_task_s"] = 100
    assert predictor.last_evaluation_timings == [{"md_task_s": 3}]


def test_resumed_csv_distinguishes_restarted_wave_and_elapsed_counters(tmp_path):
    path = tmp_path / "episode_timing.csv"
    row = {"phase": "dqn_train", "episode": 1, "formula": "Ni1", "status": "ok",
           "evaluation_wave_id": 1, "evaluation_wave_size": 1,
           "rollout_s": 1.0, "eval_wall_share_s": 3.0,
           "rl_update_s": 2.0, "episode_wall_share_s": 6.0,
           "md_task_s": None, "opt_task_s": 1.0}
    first = PredictorTimer(object(), episode_log_path=str(path))
    first.record_episode(row)
    # Each write closes the stream, so completed attempts survive an interrupted run.
    with path.open() as stream:
        first_rows = list(csv.DictReader(stream))
    assert len(first_rows) == 1
    assert first_rows[0]["md_task_s"] == ""
    resumed = PredictorTimer(object(), episode_log_path=str(path), append=True)
    resumed.record_episode({**row, "episode": 2, "status": "failed", "rl_update_s": 0})
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["timing_session"] != rows[1]["timing_session"]
    assert rows[0]["evaluation_wave_id"] == rows[1]["evaluation_wave_id"] == "1"
    summary = resumed.summary()["episode_timings"]
    assert summary["n_records"] == summary["n_failed"] == 1
    assert summary["timing_session"] == rows[1]["timing_session"]
    assert summary["md_task_s_unavailable_records"] == 1
