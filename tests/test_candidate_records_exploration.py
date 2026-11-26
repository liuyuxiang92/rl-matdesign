"""HPO archives preserve sampler evidence and settled scientific constraints."""
import json

import numpy as np
import pytest
import torch

from rl_matdesign import training
from rl_matdesign.utils.candidate_records import CandidateRecorder, episode_record
from rl_matdesign.utils.episode_batch import EpisodeSnapshot
from tests.test_hpo_candidate_runtime import SnapshotPredictor
from tests.test_cross_formula_training import ScriptedEnv


def test_episode_record_copies_sampler_metadata_without_changing_scientific_record():
    episode = EpisodeSnapshot([], (("A", 1.),), {}, "A1",
        metadata={"exploration_target_used": .7, "exploration_entropy": .6,
                  "exploration_entropy_by_stage": {"0": .6, "1": None}, "unrelated": "skip"},
        prediction_record={"reward": 3., "dp_mean": 3., "gate_passed": True})
    record = episode_record(episode, "generation")
    assert record["exploration_target_used"] == .7
    assert record["exploration_entropy"] == .6
    assert record["exploration_entropy_by_stage"] == {"0": .6, "1": None}
    assert "unrelated" not in record
    assert "exploration_target_used" not in record["prediction_record"]
    episode.metadata["exploration_entropy_by_stage"]["0"] = 0.
    assert record["exploration_entropy_by_stage"]["0"] == .6


def test_recorder_forwards_postfilter_result_for_unique_and_duplicate_snapshots(tmp_path):
    recorder = CandidateRecorder(tmp_path, 3, ratio_limit=3.)
    checks = []
    recorder.constraint_check = lambda episode: checks.append(episode.formula) or False
    for _ in range(2):
        episode = EpisodeSnapshot([], (("A", 1.),), {}, "A1",
            metadata={"exploration_target_used": .7},
            prediction_record={"reward": 3., "dp_mean": 3., "gate_passed": True})
        recorder.start()
        recorder.success(episode)
        assert episode.metadata["post_constraint_passed"] is False
        assert training._exploration_observation(episode)["feasible"] is False
    assert checks == ["A1", "A1"]
    archived = [json.loads(line) for line in recorder.records_path.read_text().splitlines()]
    assert len(archived) == 1
    assert archived[0]["constraint_passed"] is False
    assert archived[0]["post_constraint_passed"] is False
    assert archived[0]["exploration_target_used"] == .7
    assert recorder.budget.attempts_started == 2
    assert recorder.budget.unique_evaluated == 1


@pytest.mark.parametrize("batch_size", [1, 20])
def test_hpo_generation_uses_postfilter_once_and_preserves_sampler_diagnostics(tmp_path, monkeypatch, batch_size):
    predictor = SnapshotPredictor(batch_size)
    env = ScriptedEnv(["A", "B", "C"], predictor)
    recorder = CandidateRecorder(tmp_path, 3, phase="generation")
    checks = []
    recorder.constraint_check = lambda episode: checks.append(episode.formula) or episode.formula != "A1"
    env.hpo_recorder = recorder

    def rollout(**kwargs):
        training._rollout_random_episode(kwargs["env"])
        kwargs["env"].exploration_diagnostics = {
            "exploration_entropy": .8,
            "exploration_entropy_by_stage": {"0": .8, "1": None},
        }

    monkeypatch.setattr(training, "_rollout_policy_episode", rollout)
    rows = training.generate_candidates(
        env=env, predictor=predictor, scaler=None, device=torch.device("cpu"),
        elem_feats_scaled=np.zeros((1, 1)), action_values=["1"],
        qnet=torch.nn.Linear(1, 1), num_gen_eps=3, gen_exploration_target=.7,
    )
    assert checks == ["A1", "B1", "C1"]
    assert not predictor.raw
    # A fails the postfilter, B fails its recorded reward gate, C is feasible.
    assert [row["constraint_passed"] for row in rows] == [False, False, True]
    assert [row["post_constraint_passed"] for row in rows] == [False, True, True]
    assert all(row["exploration_target_used"] == .7 for row in rows)
    assert all(row["exploration_entropy"] == .8 for row in rows)
    assert all(row["exploration_entropy_by_stage"] == {"0": .8, "1": None} for row in rows)
    assert env.exploration_state["window"]["attempts"] == 3
    assert env.exploration_state["window"]["feasible"] == 1
    archived = [json.loads(line) for line in recorder.records_path.read_text().splitlines()]
    assert [row["constraint_passed"] for row in archived] == [False, False, True]
    assert all(row["exploration_entropy_by_stage"] == {"0": .8, "1": None} for row in archived)
