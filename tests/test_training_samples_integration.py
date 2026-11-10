"""Budgeted material exports retain the exact evaluations that drove training."""
from __future__ import annotations

import csv
import subprocess
import sys

import numpy as np
import pytest
import torch

from rl_matdesign import training
from rl_matdesign.configuration import _TerminalReward
from rl_matdesign.utils.episode_batch import CandidateEvaluationFailed, EpisodeBatchEvaluator
from rl_matdesign.utils.training_samples import TrainingSampleWriter
from tests.test_cross_formula_training import Predictor, ScriptedEnv, pg_setup, run_dqn, saved
from tests.test_duplicate_budget_cli import _REPO, _RUNNER, _config, _run, _successful_rows


def _samples(out):
    with (out / "training_samples.csv").open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
@pytest.mark.parametrize("count_duplicates", [False, True])
def test_cli_exports_exactly_the_budgeted_training_episodes(tmp_path, method, count_duplicates):
    cfg = _config(method=method, count_duplicates=count_duplicates)
    budget = 6 if count_duplicates else 3
    if count_duplicates:
        # Force repeats: the search space contains only three materials.
        cfg.update(dqn_num_train_eps=5, pg_batch_eps=6)
    proc, out = _run(tmp_path, cfg)
    logs = _successful_rows(proc, out)
    samples = _samples(out)
    expected = [row for row in logs if row.get("budget_counted", "").lower() == "true"
                and row["phase"] in {"dqn_warmup", "dqn_train", "pg_episode"}]

    assert len(samples) == len(expected) == budget
    assert [row["phase"] for row in samples] == [row["phase"] for row in expected]
    assert [int(row["episode"]) for row in samples] == [int(row["episode"]) for row in expected]
    assert [int(row["training_budget_used"]) for row in samples] == list(range(1, budget + 1))
    assert all(row["formula"] for row in samples)
    assert all(row["budget_counted"].lower() == "true" for row in samples)
    assert all(None not in row for row in samples), "CSV rows must align with their header"
    for row, log in zip(samples, expected):
        assert float(row["reward"]) == pytest.approx(float(log["terminal_reward"]))
        assert float(row["dp_mean"]) == pytest.approx(float(row["reward"]))
        assert float(row["dp_std"]) >= 0.0
    if method == "dqn":
        assert [row["phase"] for row in samples] == ["dqn_warmup"] + ["dqn_train"] * (budget - 1)
    else:
        # The scaler's two random warmup episodes and per-iteration summaries
        # are outside the existing PG training budget.
        assert {row["phase"] for row in samples} == {"pg_episode"}
    if count_duplicates:
        assert len({row["formula"] for row in samples}) < len(samples)


@pytest.mark.parametrize("multiple_groups", [False, True])
def test_cli_material_export_supports_categorical_and_multigroup_spaces(tmp_path, multiple_groups):
    cfg = _config(count_duplicates=True, categorical=True)
    if multiple_groups:
        cfg["groups"].insert(0, {
            "name": "metal", "kind": "composition", "species_set": ["Fe", "Ni"],
            "fraction_set": ["1.0"], "n_components": 1,
        })
    proc, out = _run(tmp_path, cfg)
    _successful_rows(proc, out)
    samples = _samples(out)
    assert len(samples) == 3
    assert [int(row["training_budget_used"]) for row in samples] == [1, 2, 3]
    assert all(row["formula"] for row in samples)
    assert all(float(row["dp_mean"]) == float(row["reward"]) for row in samples)


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
def test_only_generate_preserves_samples_and_uses_the_same_property_columns(tmp_path, method):
    cfg = _config(method=method, count_duplicates=True)
    cfg.update(num_gen_eps=1, gen_temperature=0.0)
    proc, out = _run(tmp_path, cfg)
    _successful_rows(proc, out)
    sample_path = out / "training_samples.csv"
    original = sample_path.read_bytes()

    generated = subprocess.run(
        [sys.executable, str(_RUNNER), "--config", str(tmp_path / "config.yaml"),
         "--out", str(out), "--device", "cpu", "--only-generate", "--gen-seed", "0"],
        capture_output=True, text=True, cwd=_REPO, timeout=120,
    )
    assert generated.returncode == 0, generated.stdout[-3000:] + generated.stderr[-5000:]
    assert sample_path.read_bytes() == original
    with (out / "generated.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames)
        generated_rows = list(reader)
    assert len(generated_rows) == 1
    assert {"formula", "reward", "dp_mean", "dp_std", "dp_mean_minus_std"} <= fields
    assert fields <= set(_samples(out)[0])


def test_dqn_resume_appends_budgeted_samples_without_repeating_warmup(tmp_path):
    cfg = _config(count_duplicates=False)
    cfg["dqn_num_train_eps"] = 1
    proc, out = _run(tmp_path, cfg)
    _successful_rows(proc, out)
    original = _samples(out)
    assert len(original) == 2
    session_id = original[0]["training_session_id"]
    assert saved(out)["training_sample_session_id"] == session_id

    cfg["dqn_num_train_eps"] = 2
    proc, out = _run(tmp_path, cfg, "--resume-training", out=out)
    _successful_rows(proc, out)
    resumed = _samples(out)
    assert len(resumed) == 3
    assert resumed[:2] == original
    assert resumed[-1]["training_session_id"] == session_id
    assert saved(out)["training_sample_session_id"] == session_id
    assert [int(row["training_budget_used"]) for row in resumed] == [1, 2, 3]
    assert [row["phase"] for row in resumed] == ["dqn_warmup", "dqn_train", "dqn_train"]


class ChangingPredictor:
    """Uncached realizations change even when the terminal material repeats."""

    def __init__(self):
        self.calls = []

    def evaluation_batch_size(self):
        return 1

    def predict(self, candidate):
        self.calls.append(dict(candidate))
        return len(self.calls) + 0.25, 0.125


def test_scalar_snapshots_reuse_each_actual_stochastic_prediction():
    predictor = ChangingPredictor()
    env = ScriptedEnv(["A", "A", "B"], predictor)
    evaluator = EpisodeBatchEvaluator(env, capture_predictions=True)
    episodes = [evaluator.next(lambda: training._rollout_random_episode(env), limit=1)
                for _ in range(3)]

    assert predictor.calls == [{"A": 1.0}, {"A": 1.0}, {"B": 1.0}]
    assert [episode.path[-1].reward for episode in episodes] == [1.25, 2.25, 3.25]
    assert [episode.prediction["dp_mean"] for episode in episodes] == [1.25, 2.25, 3.25]
    assert [episode.prediction["dp_std"] for episode in episodes] == [0.125] * 3
    env.reward_fn.last_prediction_records[0]["dp_mean"] = -999
    assert episodes[-1].prediction["dp_mean"] == 3.25


def test_raw_predictor_without_a_record_hook_is_not_evaluated_again():
    predictor = ChangingPredictor()
    predictor.predict_raw = lambda _: pytest.fail("sample export must not predict again")
    adapter = _TerminalReward(predictor, "multi_group")
    adapter.capture_prediction_records = True
    assert adapter({"A": 1.0}) == 1.25
    assert len(predictor.calls) == 1
    record = adapter.last_prediction_records[0]
    assert record["predict_mean"] == 1.25
    assert record["predict_std"] == 0.125
    assert record.get("prediction_record_error")
    assert "dp_mean" not in record or record["dp_mean"] is None


class PartialGatedPredictor:
    def __init__(self):
        self.batches = []
        self.record_calls = []

    def evaluation_batch_size(self):
        return 3

    def predict(self, candidate):
        pytest.fail("batched snapshot export must not call scalar predict")

    def predict_raw(self, candidate):
        pytest.fail("batched snapshot export must not call predict_raw")

    def batch_predict(self, candidates, *, return_exceptions=False):
        assert return_exceptions
        self.batches.append(candidates)
        return [(-10.0, 0.0), RuntimeError("failed structure"), (0.12, 0.0)]

    def prediction_record(self, candidate, prediction, *, k=1.0):
        self.record_calls.append((dict(candidate), prediction))
        passed = prediction[0] > 0
        return {"dp_mean": prediction[0], "obj_delta_mean": 0.12 if passed else 0.07,
                "obj_delta_std": 0.002, "reward_objective": 0.12 if passed else 0.07,
                "gate_passed": passed, "gate_synthesis_value": 0.9 if passed else 0.4,
                "gate_synthesis_passed": passed}


def test_partial_batch_keeps_gate_properties_aligned_with_successful_episodes():
    predictor = PartialGatedPredictor()
    env = ScriptedEnv(["A", "X", "B"], predictor)
    evaluator = EpisodeBatchEvaluator(env, capture_predictions=True)
    first = evaluator.next(lambda: training._rollout_random_episode(env), limit=3)
    with pytest.raises(CandidateEvaluationFailed, match="failed structure"):
        evaluator.next(lambda: pytest.fail("must drain the queued wave"), limit=2)
    last = evaluator.next(lambda: pytest.fail("must drain the queued wave"), limit=1)

    assert len(predictor.batches) == 1
    assert predictor.record_calls == [({"A": 1.0}, (-10.0, 0.0)), ({"B": 1.0}, (0.12, 0.0))]
    assert first.formula == "A1" and last.formula == "B1"
    assert first.path[-1].reward == -10.0
    assert first.prediction["obj_delta_mean"] == 0.07
    assert first.prediction["gate_passed"] is False
    assert last.path[-1].reward == 0.12
    assert last.prediction["obj_delta_mean"] == 0.12
    assert last.prediction["gate_passed"] is True
    assert len(env.reward_fn.last_prediction_records) == 3
    assert not env.reward_fn.last_prediction_records[1]
    assert first.timing["evaluation_wave_id"] == last.timing["evaluation_wave_id"]
    assert first.timing["evaluation_wave_size"] == last.timing["evaluation_wave_size"] == 3


def _record_completed_prediction(candidate, prediction, *, k=1.0):
    """Expose separate physical units while preserving the existing reward."""
    return {"dp_mean": 10 * prediction[0], "dp_std": prediction[1],
            "obj_delta_mean": 0.01 * prediction[0]}


@pytest.mark.parametrize("target_mode", ["bootstrap", "mc"])
def test_dqn_sample_file_survives_duplicate_guard_with_partial_batch_failure(tmp_path, target_mode):
    predictor = Predictor(4)
    predictor.prediction_record = _record_completed_prediction
    env = ScriptedEnv(["A", "B", "A", "X", "C", "C", "C", "D"], predictor)
    writer = TrainingSampleWriter(tmp_path / "training_samples.csv")
    with pytest.raises(RuntimeError, match="2 consecutive duplicate"):
        run_dqn(env, tmp_path, n_warmup_eps=2, dqn_num_train_eps=3,
                dqn_target_mode=target_mode, train_count_duplicates=False,
                train_max_duplicate_attempts=2, training_samples=writer)

    samples = _samples(tmp_path)
    assert [row["formula"] for row in samples] == ["A1", "B1", "C1"]
    assert [row["phase"] for row in samples] == ["dqn_warmup", "dqn_warmup", "dqn_train"]
    assert [int(row["training_budget_used"]) for row in samples] == [1, 2, 3]
    assert [float(row["reward"]) for row in samples] == [1, 2, 3]
    assert [float(row["dp_mean"]) for row in samples] == [10, 20, 30]
    assert [float(row["obj_delta_mean"]) for row in samples] == pytest.approx([0.01, 0.02, 0.03])
    assert env.starts == ["A", "B", "A", "X", "C", "C", "C"]
    assert predictor.waves == [["A", "B"], ["A", "X"], ["C"], ["C", "C"]]
    assert predictor.scalar == predictor.raw == []
    state = saved(tmp_path)
    assert state["training_budget_state"]["counted"] == len(samples)
    assert state["training_budget_state"]["attempted"] == 6


@pytest.mark.parametrize("stop_kind", ["duplicate_guard", "partial_batch_cap"])
def test_pg_samples_are_durable_before_the_partial_batch_update(monkeypatch, tmp_path, stop_kind):
    predictor, _, kwargs = pg_setup(monkeypatch, [0, 1, 0, 0, 2])
    predictor.prediction_record = _record_completed_prediction
    writer = TrainingSampleWriter(tmp_path / "training_samples.csv")
    saved_before_update = []
    original_step = torch.optim.Adam.step

    def inspect_then_step(optimizer, *args, **kw):
        # Reading a separately opened file here proves rows were persisted
        # before the optimizer update, including for an incomplete batch.
        rows = _samples(tmp_path)
        saved_before_update.append(rows)
        assert [int(row["training_budget_used"]) for row in rows] == [1, 2]
        assert [float(row["reward"]) for row in rows] == [1, 2]
        return original_step(optimizer, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", inspect_then_step)
    options = ({"train_max_duplicate_attempts": 2} if stop_kind == "duplicate_guard"
               else {"max_train_attempts": 3})
    message = "2 consecutive duplicate" if stop_kind == "duplicate_guard" else "budget is incomplete"
    with pytest.raises(RuntimeError, match=message):
        training.train_pg(**kwargs, train_count_duplicates=False, training_samples=writer,
                          checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 0},
                          **options)

    samples = _samples(tmp_path)
    assert len(saved_before_update) == 1
    assert samples == saved_before_update[0]
    assert [row["phase"] for row in samples] == ["pg_episode", "pg_episode"]
    assert [float(row["dp_mean"]) for row in samples] == [10, 20]
    assert [float(row["obj_delta_mean"]) for row in samples] == pytest.approx([0.01, 0.02])
    assert predictor.waves == ([["A", "B"], ["A", "A"]]
                               if stop_kind == "duplicate_guard" else [["A", "B", "A"]])
    assert predictor.scalar == predictor.raw == []
    assert saved(tmp_path)["training_budget_state"]["counted"] == 2


def test_dqn_permutation_replay_does_not_export_synthetic_samples(tmp_path):
    from rl_matdesign.env import CompositionEnv

    predictor = ChangingPredictor()
    env = CompositionEnv(
        species_set=["Fe", "Ni", "Co"], fraction_set=["0.25", "0.50"],
        n_components=3, total_units=4,
        state_featurizer=lambda _: np.array([0.0, 1.0]),
        reward_fn=_TerminalReward(predictor, "fraction"),
    )
    writer = TrainingSampleWriter(tmp_path / "training_samples.csv")
    _, _, logs = run_dqn(
        env, tmp_path, elem_feats_scaled=np.eye(3), fraction_set=env.fraction_set,
        n_warmup_eps=2, dqn_num_train_eps=1, augment_permutations=2,
        checkpoint_cfg={"path": str(tmp_path / "checkpoint.pt"), "freq": 1},
        training_samples=writer,
    )
    samples = _samples(tmp_path)
    assert len(samples) == len(predictor.calls) == len(logs) == 3
    assert [float(row["reward"]) for row in samples] == [1.25, 2.25, 3.25]
    assert [int(row["training_budget_used"]) for row in samples] == [1, 2, 3]
    assert len(saved(tmp_path)["buffer"]) > 3 * env.n_components
