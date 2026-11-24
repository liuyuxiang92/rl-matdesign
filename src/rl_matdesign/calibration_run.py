"""Collect and freeze reward calibration at a resumable pre-training boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from .training import _rollout_random_episode, _save_checkpoint
from .utils.episode_batch import snapshot_episode
from .utils.execution_contract import require_execution_contract
from .utils.generation_resume import predictor_checkpoint_fields, restore_predictor_state
from .utils.rollout_errors import EvaluationProtocolError, InfeasibleEpisode
from .utils.seeding import restore_rng_state

CALIBRATION_STATE = "calibration_checkpoint.pt"
CALIBRATION_ARTIFACT = "reward_calibration.json"
CALIBRATION_BUDGET = "reward_calibration_budget.json"
CALIBRATION_ACCOUNTING = "outside_training_budget"


def _write_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _draw_candidate(env):
    """Capture the same terminal payload used by the ordinary reward adapter."""
    payloads = []

    def capture(payload):
        payloads.append(payload)
        # This transient placeholder is never exported or inserted into replay.
        return 0.0

    original_reward = env.reward_fn
    env.reward_fn = capture
    try:
        _rollout_random_episode(env)
        if len(payloads) != 1 or not env.path:
            raise EvaluationProtocolError("Calibration rollout has no terminal candidate")
        return payloads[0], snapshot_episode(env)
    finally:
        env.reward_fn = original_reward


def prepare_reward_calibration(
    *, env, predictor, out, contract, total_budget, attempt_limit,
    only_generate=False, resume_learner=False, resume_pilot=False, load_path=None,
):
    """Freeze a separate prediction pilot before any learner budget is consumed.

    Pilot observations never enter replay, training seen sets, or HPO candidate
    archives. Their costs are recorded separately from the training allowance.
    """
    out = Path(out)
    artifact_path = out / CALIBRATION_ARTIFACT
    checkpoint_path = out / CALIBRATION_STATE
    recorder = getattr(env, "hpo_recorder", None)
    if only_generate or resume_learner or load_path is not None:
        source = Path(load_path) if load_path else artifact_path
        if not source.is_file():
            raise ValueError(f"Frozen reward calibration required: {source}")
        predictor.load_reward_calibration(json.loads(source.read_text()))
        # A run using another grid's compatible artifact keeps a local copy.
        if not only_generate and not resume_learner and source != artifact_path:
            _write_json(artifact_path, predictor.reward_calibration_state_dict())
        return

    needed = predictor.reward_calibration_num_candidates
    entries = []
    attempts = 0
    evaluations = 0
    misses = 0
    frozen = None

    def budget_report():
        return {
            "schema_version": 1, "accounting": CALIBRATION_ACCOUNTING,
            "target_candidates": needed, "valid_candidates": len(entries),
            "attempts_started": attempts, "evaluations": evaluations,
            "status": "COMPLETED" if frozen is not None else "RUNNING",
        }

    if resume_pilot:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved.get("type") != "reward_calibration"
                or saved.get("accounting") != CALIBRATION_ACCOUNTING
                or saved.get("num_candidates") != needed
                or saved.get("total_budget") != total_budget):
            raise ValueError("Calibration resume requires the original pilot and training allowance")
        require_execution_contract(saved["execution_contract"], contract)
        if predictor.evaluation_log_size() != saved["evaluation_log_size"]:
            raise ValueError("Evaluation log differs from the last settled calibration checkpoint")
        entries = saved["entries"]
        attempts = saved["attempts"]
        evaluations = saved["evaluations"]
        misses = saved["misses"]
        frozen = saved["frozen"]
        report_path = out / CALIBRATION_BUDGET
        if not report_path.is_file() or json.loads(report_path.read_text()) != budget_report():
            raise ValueError("Calibration budget report differs from the last settled checkpoint")
        restore_predictor_state(predictor, saved)
        restore_rng_state(saved["rng_state"])
    elif checkpoint_path.exists() or artifact_path.exists():
        raise ValueError("Calibration files already exist. Resume this run or choose a new output directory.")

    def save():
        payload = {
            "type": "reward_calibration", "execution_contract": contract,
            "accounting": CALIBRATION_ACCOUNTING, "num_candidates": needed,
            "total_budget": total_budget, "attempts": attempts,
            "evaluations": evaluations, "misses": misses, "entries": entries,
            "frozen": frozen, "evaluation_log_size": predictor.evaluation_log_size(),
            **predictor_checkpoint_fields(predictor),
        }
        if recorder is not None:
            payload["hpo_state"] = recorder.checkpoint_state()
        _write_json(out / CALIBRATION_BUDGET, budget_report())
        _save_checkpoint(str(checkpoint_path), payload)

    predictor.set_phase("calibration")
    seen = {episode.terminal_key for episode, _ in entries}
    observed = {obs["candidate_key"] for _, obs in entries}
    while len(entries) < needed:
        # Bound consecutive failed/duplicate attempts independently of RL work.
        if misses >= attempt_limit:
            raise ValueError(f"Reward calibration unresolved after {attempts} attempts "
                             f"and {evaluations} evaluations ({len(entries)}/{needed} valid distinct candidates)")
        attempts += 1
        started = time.perf_counter()
        try:
            candidate, episode = _draw_candidate(env)
        except InfeasibleEpisode:
            misses += 1
            save()
            continue
        if episode.terminal_key in seen:
            misses += 1
            save()
            continue

        rollout_s = time.perf_counter() - started
        evaluations += 1
        started = time.perf_counter()
        try:
            obs = predictor.calibration_observation(candidate)
        except EvaluationProtocolError:
            raise
        except Exception as exc:
            predictor.record_episode({"phase": "calibration", "status": "failed",
                                      "formula": episode.formula, "rollout_s": rollout_s})
            misses += 1
            save()
            print(f"[WARN] Calibration evaluation failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        eval_s = time.perf_counter() - started
        if obs["candidate_key"] in observed:
            misses += 1
            save()
            continue
        timings = predictor.last_evaluation_timings
        episode.timing = {**(timings[0] if len(timings) == 1 else {}),
                          "evaluation_wave_id": attempts, "evaluation_wave_size": 1,
                          "rollout_s": rollout_s, "eval_wall_share_s": eval_s,
                          "episode_wall_share_s": rollout_s + eval_s}
        episode.prediction_record = predictor.calibration_prediction_record(obs)
        seen.add(episode.terminal_key)
        observed.add(obs["candidate_key"])
        entries.append((episode, obs))
        misses = 0
        save()

    if frozen is None:
        frozen = predictor.fit_reward_calibration([obs for _, obs in entries])
        # A single checkpoint binds the frozen state to the exact original raw
        # rows and trajectories. Its JSON exports can be regenerated on resume.
        save()
    else:
        predictor.load_reward_calibration(frozen)

    finalized_rows = []
    for _, obs in entries:
        scored = predictor.score_calibration_observation(obs)
        reward = scored["reward"]
        finalized_rows.append({"observation": obs, "reward_status": "final", "reward": reward})
    _write_json(out / "reward_calibration_observations.json", finalized_rows)
    _write_json(artifact_path, frozen)
    predictor.set_phase("train")
    predictor.mark("calibration_end")
    print(f"[INFO] Froze reward calibration from {len(entries)} distinct candidates "
          f"after {evaluations} evaluations (outside the training budget)", flush=True)
