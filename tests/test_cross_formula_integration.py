"""Real worker IPC through deferred terminal rewards and DQN training."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from rl_matdesign import training
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.timing import PredictorTimer
from tests.test_cross_formula_training import ScriptedEnv
from tests.test_parallel_structures import pool_factory


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_training_crosses_formula_boundary_on_four_worker_processes(
    pool_factory, tmp_path, monkeypatch, target_mode,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11")
    module = tmp_path / "fake_modules" / "composition_batch_builder.py"
    module.write_text('''
from parallel_test_builder import Builder as Base

class Builder(Base):
    def build(self, candidate, *, n_configs=1, rng=None):
        return super().build({
            "id": next(iter(candidate)), "mode": "barrier", "barrier_count": 4,
        }, n_configs=n_configs, rng=rng)
''')
    root = tmp_path / "training_workers"
    inner = StructureScorePredictor({
        "builder": "composition_batch_builder:Builder", "test_root": str(root),
        "n_random_configs": 2, "predict_cache_size": 1,
        "structure_parallel": {
            "devices": "auto", "timeout_s": 20, "startup_timeout_s": 20,
        },
        "properties": [
            {"name": "delta", "predictor": "dp_property", "models": ["unused"],
             "direction": "max", "objective": "mean"},
            {"name": "pure", "predictor": "dp_classification", "models": ["unused"],
             "output_index": 1, "direction": "max", "objective": "mean"},
        ],
        "reward": {
            "type": "gated",
            "gates": [{"name": "synthesis", "score": {
                "type": "weighted_sum", "terms": {"pure": 1}}, "min": .8}],
            "objective": {"type": "weighted_sum", "terms": {"delta": 1}},
            "failure": {"reward": -10},
        },
    })
    monkeypatch.setattr(inner, "_get_prop_models", lambda prop: ([], {}))
    ensembles = []

    def score(prop, structures, **kwargs):
        ids = [cell.info["candidate_id"] for cell in structures]
        assert len(ids) == 2 and len(set(ids)) == 1
        ensembles.append((prop["name"], ids[0]))
        return (float(ord(ids[0]) - ord("A") + 1), 0.) if prop["name"] == "delta" else (.9, 0.)

    monkeypatch.setattr(inner, "_score", score)
    timing_path = tmp_path / "episode_timing.csv"
    predictor = PredictorTimer(inner, episode_log_path=str(timing_path))
    env = ScriptedEnv(["A", "B", "A", "C"], predictor)
    try:
        _, _, rows = training.train_dqn_online(
            env=env, elem_feats_scaled=np.array([[.1, .2]]), fraction_set=["0.5"],
            device=torch.device("cpu"), hidden_dim=4, n_warmup_eps=2,
            dqn_num_train_eps=2, batch_size=1, dqn_grad_steps_per_ep=1,
            dqn_target_update_freq=2, dqn_eps_anneal_eps=10,
            dqn_target_mode=target_mode, timer=predictor,
        )
        assert [row["terminal_reward"] for row in rows] == [1, 2, 1, 3]
        assert rows[-1]["training_attempts"] == 4
        assert rows[-1]["training_unique_candidates"] == 3
        calls = [json.loads(line) for line in (root / "build_calls.jsonl").read_text().splitlines()]
        assert len(calls) == 6  # Two configurations for each unique formula.
        assert {row["candidate_id"] for row in calls[:4]} == {"A", "B"}
        assert len({row["pid"] for row in calls[:4]}) == 4
        assert {row["device"] for row in calls[:4]} == {"3", "7", "9", "11"}
        assert len(ensembles) == 6  # delta and classification each score three formulas.
        assert predictor.n_calls == 4 and predictor.n_unique == 3
        with timing_path.open() as stream:
            timings = list(csv.DictReader(stream))
        assert len(timings) == 4
        assert [int(row["config_attempted"]) for row in timings] == [2, 2, 0, 2]
        assert [int(row["config_succeeded"]) for row in timings] == [2, 2, 0, 2]
        assert [row["cache_hit"] for row in timings] == ["False", "False", "True", "False"]
        for row in timings:
            assert float(row["md_task_s"]) == 0  # This fake builder performs no MD/OPT.
            assert float(row["opt_task_s"]) == 0
            assert float(row["episode_wall_share_s"]) == pytest.approx(
                sum(float(row[key]) for key in ("rollout_s", "eval_wall_share_s", "rl_update_s")))
        assert float(timings[0]["rl_update_s"]) == 0
        assert float(timings[-1]["rl_update_s"]) > 0
        assert inner.per_objective_stats({"A": 1.})["delta"][0] == 1.
        assert len((root / "build_calls.jsonl").read_text().splitlines()) == 6
    finally:
        predictor.close()
