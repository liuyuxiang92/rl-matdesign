"""Timing attribution follows waves, failed attempts and actual learning batches."""
from types import SimpleNamespace

import pytest

from rl_matdesign import configuration, training
from rl_matdesign.utils import episode_batch
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator
from .test_cross_formula_training import Predictor, ScriptedEnv, pg_setup, run_dqn


class Clock:
    def __init__(self):
        self.value = 0.0

    def perf_counter(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class Recorder:
    def __init__(self):
        self.rows = []

    def record_episode(self, row):
        self.rows.append(dict(row))

    def snapshot(self):
        return {}

    def mark(self, _phase):
        pass


class TimedPredictor(Predictor):
    def __init__(self, clock, size):
        super().__init__(size)
        self.clock = clock

    @staticmethod
    def details(candidate):
        cached = next(iter(candidate)) == "C"
        return {
            "md_task_s": 0.0 if cached else 30.0,
            "opt_task_s": 0.0 if cached else 10.0,
            "structure_other_task_s": 0.0,
            "config_attempted": 0 if cached else 2,
            "config_succeeded": 0 if cached else 2,
            "cache_hit": cached,
            "property_s": 0.0 if cached else 1.0,
        }

    def predict(self, candidate):
        self.clock.advance(20.0)
        self.last_evaluation_timings = [self.details(candidate)]
        return super().predict(candidate)

    def batch_predict(self, candidates, *, return_exceptions=False):
        self.clock.advance(20.0)
        self.last_evaluation_timings = [self.details(c) for c in candidates]
        return super().batch_predict(candidates, return_exceptions=return_exceptions)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    for module in (episode_batch, configuration, training):
        monkeypatch.setattr(module, "time", clock)
    return clock


def rollout(env, clock):
    clock.advance(3.0)
    training._rollout_random_episode(env)


def test_wave_shares_include_failure_and_cache_without_repeating_task_cost(clock):
    recorder = Recorder()
    predictor = TimedPredictor(clock, 3)
    env = ScriptedEnv(["A", "X", "C"], predictor)
    evaluator = EpisodeBatchEvaluator(env, timer=recorder)
    first = evaluator.next(lambda: rollout(env, clock), limit=3,
                           retry_scalar=training._rollout_with_retry, phase="dqn_train")
    assert clock.value == 29
    second = evaluator.next(lambda: rollout(env, clock), limit=3,
                            retry_scalar=training._rollout_with_retry, phase="dqn_train")
    assert clock.value == 29  # Popping snapshots does not remeasure the wave.
    failed = recorder.rows[0]
    assert failed["formula"] == "X1" and failed["status"] == "failed"
    assert failed["phase"] == "dqn_train"
    timings = [first.timing, failed, second.timing]
    assert {r["evaluation_wave_id"] for r in timings} == {1}
    assert {r["evaluation_wave_size"] for r in timings} == {3}
    assert sum(r["episode_wall_share_s"] for r in timings) == pytest.approx(29)
    assert [r["eval_wall_share_s"] for r in timings] == pytest.approx([20 / 3] * 3)
    assert second.timing["md_task_s"] == second.timing["opt_task_s"] == 0
    assert second.timing["cache_hit"] is True
    assert first.timing["md_task_s"] > first.timing["eval_wall_share_s"]


def test_scalar_retry_logs_failed_work_separately_and_does_not_reuse_stale_timing(clock):
    recorder = Recorder()
    predictor = TimedPredictor(clock, 1)
    env = ScriptedEnv(["X", "A"], predictor)
    evaluator = EpisodeBatchEvaluator(env, timer=recorder)
    episode = evaluator.next(lambda: rollout(env, clock), limit=1,
                             retry_scalar=training._rollout_with_retry, phase="dqn_warmup")
    assert clock.value == 46
    assert len(recorder.rows) == 1
    assert recorder.rows[0]["episode_wall_share_s"] == 23
    assert episode.timing["episode_wall_share_s"] == 23
    assert episode.timing["evaluation_wave_id"] == 2
    assert episode.timing["rollout_s"] == 3
    assert recorder.rows[0]["rl_update_s"] == 0


def test_adapter_aligns_invalid_payloads_and_copies_timing(clock):
    predictor = TimedPredictor(clock, 2)
    reward = configuration._TerminalReward(predictor, "multi_group")
    reward._candidate = lambda payload: None if payload == "invalid" else payload
    result = reward.evaluate_many([{"A": 1}, "invalid", {"C": 1}])
    assert result == [1, configuration._FAILED_CANDIDATE_REWARD, 3]
    assert reward.last_evaluation_timings[1] == {}
    assert reward.last_evaluation_timings[0]["md_task_s"] == 30
    assert reward.last_evaluation_timings[2]["md_task_s"] == 0
    predictor.last_evaluation_timings[0]["md_task_s"] = 999
    assert reward.last_evaluation_timings[0]["md_task_s"] == 30
    reward("invalid")
    assert reward.last_evaluation_timings == [{}]
    assert reward.last_evaluation_wall_s == 0


def test_legacy_predictor_does_not_fabricate_md_opt_times(clock):
    reward = configuration._TerminalReward(SimpleNamespace(predict=lambda _: (1, 0)), "multi_group")
    assert reward({"A": 1}) == 1
    assert reward.last_evaluation_timings == [{}]


def test_dqn_rows_and_live_records_include_each_episode_update(clock, monkeypatch, tmp_path):
    recorder = Recorder()
    predictor = TimedPredictor(clock, 2)
    env = ScriptedEnv(["A", "B", "C", "D"], predictor)
    gradient = training._dqn_gradient_step

    def timed_gradient(*args, **kwargs):
        clock.advance(5.0)
        return gradient(*args, **kwargs)

    monkeypatch.setattr(training, "_dqn_gradient_step", timed_gradient)
    _, _, rows = run_dqn(env, tmp_path, n_warmup_eps=2, dqn_num_train_eps=2, timer=recorder)
    assert [r["rl_update_s"] for r in rows] == [0, 0, 5, 5]
    assert len(recorder.rows) == 4
    assert [r["formula"] for r in recorder.rows] == ["A1", "B1", "C1", "D1"]
    assert sum(r["episode_wall_share_s"] for r in recorder.rows) == pytest.approx(clock.value)
    assert all(r["status"] == "ok" for r in recorder.rows)


@pytest.mark.parametrize("method", ["reinforce", "a2c"])
def test_pg_update_split_uses_actual_batch_size(clock, monkeypatch, method):
    recorder = Recorder()
    _, _, kwargs = pg_setup(monkeypatch, [0, 0, 1, 1, 2, 2, 3], method=method)
    terms = training._episode_pg_terms

    def timed_terms(*args, **kwargs):
        clock.advance(2.0)
        return terms(*args, **kwargs)

    monkeypatch.setattr(training, "_episode_pg_terms", timed_terms)
    rows = training.train_pg(**kwargs, train_count_duplicates=False, timer=recorder)
    updates = [r for r in rows if r["phase"] == "pg_train"]
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    assert [r["batch_eps_actual"] for r in updates] == [4, 3]
    assert [r["rl_update_s"] for r in updates] == [8, 6]
    assert [r["rl_update_s"] for r in episodes] == [2] * 7
    assert [r["rl_update_s"] for r in recorder.rows] == [2] * 7
    assert sum(r["episode_wall_share_s"] for r in recorder.rows) == pytest.approx(clock.value)
    assert all(r["phase"] == "pg_episode" for r in recorder.rows)


@pytest.mark.parametrize("size", [1, 20])
def test_all_failed_run_still_keeps_eleven_live_timing_rows(clock, tmp_path, size):
    recorder = Recorder()
    predictor = TimedPredictor(clock, size)
    env = ScriptedEnv(["X"] * 20, predictor)
    with pytest.raises(RuntimeError, match="11 consecutive"):
        run_dqn(env, tmp_path, n_warmup_eps=12, timer=recorder)
    assert len(recorder.rows) == 11
    assert all(r["status"] == "failed" for r in recorder.rows)
    assert sum(r["episode_wall_share_s"] for r in recorder.rows) == pytest.approx(clock.value)


def test_generation_keeps_timing_out_of_candidate_schema(clock, monkeypatch):
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    recorder = Recorder()
    predictor = TimedPredictor(clock, 3)
    predictor.record_episode = recorder.record_episode
    env = ScriptedEnv(["X", "A", "A", "B"], predictor)
    monkeypatch.setattr(training, "_rollout_policy_episode",
                        lambda **kw: training._rollout_random_episode(kw["env"]))
    rows = training.generate_candidates(
        env=env, predictor=predictor, qnet=torch.nn.Linear(1, 1),
        scaler=StandardScaler().fit([[0.0, 1.0], [1.0, 1.0]]), device=torch.device("cpu"),
        elem_feats_scaled=np.array([[0.1, 0.2]]), fraction_set=["0.5"],
        n_exploit=2, n_explore=0, max_attempts=4,
    )
    assert [r["formula"] for r in rows] == ["A1", "B1"]
    assert [r["formula"] for r in recorder.rows] == ["X1", "A1", "A1", "B1"]
    assert [r["status"] for r in recorder.rows] == ["failed", "ok", "ok", "ok"]
    assert all(r["rl_update_s"] == 0 for r in recorder.rows)
    assert all("episode_wall_share_s" not in r for r in rows)


def test_pg_failure_guard_flushes_successful_untrained_batch(clock, monkeypatch):
    recorder = Recorder()
    predictor, _, kwargs = pg_setup(monkeypatch, [0] + [1] * 20)
    original_score = predictor.score
    original_batch = predictor.batch_predict

    def score(candidate):
        if next(iter(candidate)) != "A":
            raise RuntimeError("MD failed")
        return original_score(candidate)

    def timed_batch(candidates, *, return_exceptions=False):
        clock.advance(20.0)
        return original_batch(candidates, return_exceptions=return_exceptions)

    predictor.score = score
    predictor.batch_predict = timed_batch
    monkeypatch.setattr(training, "_episode_pg_terms",
                        lambda **_: pytest.fail("Incomplete aborted batch must not train"))
    with pytest.raises(RuntimeError, match="11 consecutive"):
        training.train_pg(**kwargs, timer=recorder)
    failed = [row for row in recorder.rows if row["status"] == "failed"]
    successful = [row for row in recorder.rows if row["status"] == "ok"]
    assert len(failed) == 11
    assert len(successful) == 1
    assert successful[0]["formula"] == "Na1.00"
    assert successful[0]["episode"] == 1
    assert successful[0]["rl_update_s"] == 0
    assert successful[0]["eval_wall_share_s"] > 0
    assert sum(row["episode_wall_share_s"] for row in recorder.rows) == pytest.approx(clock.value)
