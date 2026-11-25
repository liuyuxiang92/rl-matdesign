"""Gradient probes reuse one paid pilot without learning or HPO discoveries."""
import json
import random

import numpy as np

import pytest
import torch

from rl_matdesign import calibration_run
from rl_matdesign.configuration import build_env, build_predictor
from rl_matdesign.utils.seeding import set_global_seed
from rl_matdesign.utils.timing import PredictorTimer
from tests.test_experiment_reproducibility import _popart_config


CONTRACT = {"evaluation": {"fixture": "gradient-pilot-v1"}, "learning": {"method": "dqn"}}


class UnspentRecorder:
    def checkpoint_state(self):
        return {"budget": {"attempts_started": 0, "unique_evaluated": 0, "pending": 0}}

    def start(self):
        raise AssertionError("Pilot must not charge an HPO attempt")

    def success(self, *args):
        raise AssertionError("Pilot must not enter the candidate archive")


def build_case(out, normalization="popart", *, append=False, changed_grid=False):
    out.mkdir(exist_ok=True)
    cfg = _popart_config()
    cfg["value_normalization"] = normalization
    if changed_grid:
        cfg["groups"][0]["species_set"] = list(reversed(cfg["groups"][0]["species_set"]))
    inner = build_predictor(cfg, seed=13)
    timer = PredictorTimer(inner, eval_seed=13, episode_log_path=str(out / "episode_timing.csv"),
                           append=append)
    env, _, _ = build_env(cfg, timer)
    env.hpo_recorder = UnspentRecorder()
    return env, timer


def prepare(out, env, predictor, **kwargs):
    return calibration_run.prepare_reward_calibration(
        env=env, predictor=predictor, out=out, contract=CONTRACT,
        total_budget=9, attempt_limit=1000, gradient_clipping="clipping", **kwargs,
    )


def ledger(out):
    return [json.loads(line) for line in (out / "evaluations.jsonl").read_text().splitlines()]


def test_popart_gradient_probes_reuse_same_hundred_and_finalize_rewards(tmp_path):
    set_global_seed(17)
    env, predictor = build_case(tmp_path)
    episodes = prepare(tmp_path, env, predictor)
    assert len(episodes) == len(ledger(tmp_path)) == 100
    assert all(row["reward_status"] == "pending" for row in ledger(tmp_path))
    observations = json.loads((tmp_path / "reward_calibration_observations.json").read_text())
    assert [episode.path[-1].reward for episode in episodes] == [row["reward"] for row in observations]
    assert all(episode.prediction_record["reward"] == episode.path[-1].reward for episode in episodes)
    assert all(episode.metadata["gradient_calibration"] for episode in episodes)
    saved = torch.load(tmp_path / calibration_run.GRADIENT_CALIBRATION_ARTIFACT, weights_only=False)
    assert saved["episodes_hash"] == calibration_run._episodes_hash(episodes)
    assert saved["source_budget"]["evaluations"] == 100
    assert saved["accounting"] == "outside_training_budget"
    checkpoint = torch.load(tmp_path / calibration_run.CALIBRATION_STATE, weights_only=False)
    assert checkpoint["hpo_state"]["budget"]["attempts_started"] == 0
    # The saved untrained rollout still has the old placeholder; only the
    # detached probe snapshots receive finalized rewards.
    assert all(episode.path[-1].reward == 0 for episode, _ in checkpoint["entries"])


def test_raw_gradient_pilot_preserves_reward_and_stays_outside_learning(tmp_path):
    set_global_seed(17)
    env, predictor = build_case(tmp_path, "none")
    from rl_matdesign.utils.seeding import capture_rng_state
    initial_rng = capture_rng_state()
    episodes = prepare(tmp_path, env, predictor)
    actual_draws = (random.random(), np.random.random(), torch.rand(1))
    from rl_matdesign.utils.seeding import restore_rng_state
    restore_rng_state(initial_rng)
    expected_draws = (random.random(), np.random.random(), torch.rand(1))
    assert actual_draws[:2] == expected_draws[:2]
    assert torch.equal(actual_draws[2], expected_draws[2])
    assert len(episodes) == len(ledger(tmp_path)) == 100
    assert all(episode.path[-1].reward == episode.prediction[0] for episode in episodes)
    assert [episode.path[-1].reward for episode in episodes] == [row["reward"] for row in ledger(tmp_path)]
    assert not (tmp_path / calibration_run.CALIBRATION_ARTIFACT).exists()
    assert not (tmp_path / "training_samples.csv").exists()
    report = json.loads((tmp_path / calibration_run.CALIBRATION_BUDGET).read_text())
    assert report["valid_candidates"] == report["evaluations"] == 100
    assert report["accounting"] == "outside_training_budget"


@pytest.mark.parametrize("changed_grid", [False, True])
def test_loaded_calibration_reuses_compatible_episodes_or_collects_once(tmp_path, changed_grid):
    source = tmp_path / "source"
    set_global_seed(17)
    env, predictor = build_case(source)
    original = prepare(source, env, predictor)
    dest = tmp_path / "dest"
    env, predictor = build_case(dest, changed_grid=changed_grid)
    episodes = prepare(dest, env, predictor, load_path=source / calibration_run.CALIBRATION_ARTIFACT)
    assert len(episodes) == 100
    assert len(ledger(dest)) == (100 if changed_grid else 0)
    if not changed_grid:
        assert calibration_run._episodes_hash(episodes) == calibration_run._episodes_hash(original)
        report = json.loads((dest / "gradient_calibration_reuse.json").read_text())
        assert report["new_evaluations"] == 0
        assert report["source_budget"]["evaluations"] == 100
    else:
        report = json.loads((dest / calibration_run.CALIBRATION_BUDGET).read_text())
        assert report["evaluations"] == 100


def test_loaded_reward_without_episodes_collects_only_one_gradient_pilot(tmp_path):
    source = tmp_path / "source"
    env, predictor = build_case(source)
    calibration_run.prepare_reward_calibration(
        env=env, predictor=predictor, out=source, contract=CONTRACT,
        total_budget=9, attempt_limit=1000,
    )
    assert not (source / calibration_run.GRADIENT_CALIBRATION_ARTIFACT).exists()
    # A portable calibration JSON alone contains no trajectories to reuse.
    (source / calibration_run.CALIBRATION_STATE).unlink()
    dest = tmp_path / "dest"
    env, predictor = build_case(dest)
    episodes = prepare(dest, env, predictor, load_path=source / calibration_run.CALIBRATION_ARTIFACT)
    assert len(episodes) == len(ledger(dest)) == 100
    assert [row["reward"] for row in ledger(dest)] == [e.path[-1].reward for e in episodes]


def test_clipping_reuses_completed_reward_only_pilot_checkpoint(tmp_path):
    source = tmp_path / "source"
    env, predictor = build_case(source)
    calibration_run.prepare_reward_calibration(
        env=env, predictor=predictor, out=source, contract=CONTRACT,
        total_budget=9, attempt_limit=1000,
    )
    expected = json.loads((source / "reward_calibration_observations.json").read_text())
    dest = tmp_path / "dest"
    env, predictor = build_case(dest)
    episodes = prepare(dest, env, predictor, load_path=source / calibration_run.CALIBRATION_ARTIFACT)
    assert len(episodes) == 100
    assert ledger(dest) == []
    assert [episode.path[-1].reward for episode in episodes] == [row["reward"] for row in expected]
    report = json.loads((dest / "gradient_calibration_reuse.json").read_text())
    assert report["source"].endswith("calibration_checkpoint.pt")
    assert report["source_budget"]["evaluations"] == 100


@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_no_new_pilot_for_learner_resume_or_generation(tmp_path, normalization):
    env, predictor = build_case(tmp_path, normalization)
    prepare(tmp_path, env, predictor)
    before = len(ledger(tmp_path))
    for flag in ("resume_learner", "only_generate"):
        assert prepare(tmp_path, env, predictor, **{flag: True}) is None
    assert len(ledger(tmp_path)) == before


@pytest.mark.parametrize("after_completion", [False, True])
def test_raw_pilot_resume_uses_paid_trajectories_exactly(tmp_path, monkeypatch, after_completion):
    full = tmp_path / "full"
    set_global_seed(17)
    env, predictor = build_case(full, "none")
    expected = prepare(full, env, predictor)
    resumed = tmp_path / "resumed"
    set_global_seed(17)
    env, predictor = build_case(resumed, "none")
    save = calibration_run._save_checkpoint

    def interrupt(path, payload):
        save(path, payload)
        if (payload["frozen"] is not None if after_completion else len(payload["entries"]) == 7):
            raise KeyboardInterrupt

    monkeypatch.setattr(calibration_run, "_save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(resumed, env, predictor)
    monkeypatch.setattr(calibration_run, "_save_checkpoint", save)
    env, predictor = build_case(resumed, "none", append=True)
    actual = prepare(resumed, env, predictor, resume_pilot=True)
    assert len(ledger(resumed)) == 100
    assert [e.terminal_key for e in actual] == [e.terminal_key for e in expected]
    assert [e.path[-1].reward for e in actual] == [e.path[-1].reward for e in expected]
    assert json.loads((resumed / calibration_run.CALIBRATION_BUDGET).read_text()) == json.loads(
        (full / calibration_run.CALIBRATION_BUDGET).read_text())


def test_gradient_pilot_resume_refuses_advanced_cost_report(tmp_path, monkeypatch):
    env, predictor = build_case(tmp_path, "none")
    save = calibration_run._save_checkpoint

    def interrupt(path, payload):
        save(path, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(calibration_run, "_save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path, env, predictor)
    report_path = tmp_path / calibration_run.CALIBRATION_BUDGET
    report = json.loads(report_path.read_text())
    report["attempts_started"] += 1
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(calibration_run, "_save_checkpoint", save)
    env, predictor = build_case(tmp_path, "none", append=True)
    with pytest.raises(ValueError, match="budget report differs"):
        prepare(tmp_path, env, predictor, resume_pilot=True)


def test_corrupted_saved_probe_episodes_fail_without_new_evaluations(tmp_path):
    source = tmp_path / "source"
    env, predictor = build_case(source)
    prepare(source, env, predictor)
    path = source / calibration_run.GRADIENT_CALIBRATION_ARTIFACT
    saved = torch.load(path, weights_only=False)
    saved["episodes"][0].path[-1].reward += 1
    torch.save(saved, path)
    dest = tmp_path / "dest"
    env, predictor = build_case(dest)
    with pytest.raises(ValueError, match="content hash mismatch"):
        prepare(dest, env, predictor, load_path=source / calibration_run.CALIBRATION_ARTIFACT)
    assert len(ledger(dest)) == 0


def test_hpo_restart_accepts_loaded_reward_during_partial_gradient_pilot(tmp_path, monkeypatch):
    from rl_matdesign.hpo.restart import _validate_pilot_checkpoint

    source = tmp_path / "source"
    env, predictor = build_case(source)
    calibration_run.prepare_reward_calibration(
        env=env, predictor=predictor, out=source, contract=CONTRACT,
        total_budget=9, attempt_limit=1000,
    )
    dest = tmp_path / "dest"
    env, predictor = build_case(dest)
    artifact_path = dest / calibration_run.CALIBRATION_ARTIFACT
    artifact_path.write_bytes((source / calibration_run.CALIBRATION_ARTIFACT).read_bytes())
    save = calibration_run._save_checkpoint

    def interrupt(path, payload):
        save(path, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(calibration_run, "_save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(dest, env, predictor, load_path=artifact_path)
    saved = torch.load(dest / calibration_run.CALIBRATION_STATE, weights_only=False)
    assert saved["frozen"] is None
    assert saved["loaded_reward_calibration"] == json.loads(artifact_path.read_text())
    budget = dict(target_unique=9, attempts_started=0, unique_evaluated=0,
                  pending=0, failed_attempts=0, preparation_only_attempts=0)
    _validate_pilot_checkpoint(dest, saved, budget, set())
    bad = json.loads(artifact_path.read_text())
    bad["observation_count"] += 1
    artifact_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Frozen reward calibration differs"):
        _validate_pilot_checkpoint(dest, saved, budget, set())
