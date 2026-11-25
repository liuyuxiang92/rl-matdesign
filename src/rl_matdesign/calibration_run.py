"""Collect and freeze reward calibration at a resumable pre-training boundary."""
from __future__ import annotations

import json
import os
import copy
import hashlib
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
import time

import torch

from .training import _rollout_random_episode, _save_checkpoint
from .utils.episode_batch import EpisodeSnapshot, snapshot_episode
from .utils.execution_contract import require_execution_contract
from .utils.generation_resume import predictor_checkpoint_fields, restore_predictor_state
from .utils.rollout_errors import EvaluationProtocolError, InfeasibleEpisode
from .utils.seeding import capture_rng_state, restore_rng_state
from .utils.model_contract import environment_contract
from .utils.candidate_records import candidate_identity

CALIBRATION_STATE = "calibration_checkpoint.pt"
CALIBRATION_ARTIFACT = "reward_calibration.json"
CALIBRATION_BUDGET = "reward_calibration_budget.json"
CALIBRATION_ACCOUNTING = "outside_training_budget"
GRADIENT_CALIBRATION_ARTIFACT = "gradient_calibration_episodes.pt"
GRADIENT_CALIBRATION_CANDIDATES = 100


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


def _episodes_hash(episodes):
    return hashlib.sha256(json.dumps(_json_value(episodes), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _pilot_identity(env, contract, calibration):
    evaluation = copy.deepcopy(contract.get("evaluation", {}))
    evaluation.pop("reward_calibration", None)
    return {"evaluation": evaluation, "environment": environment_contract(env),
            "reward_calibration": calibration.get("content_hash") if calibration else None}


def _validate_probe_episodes(episodes):
    if not isinstance(episodes, list) or len(episodes) != GRADIENT_CALIBRATION_CANDIDATES:
        raise ValueError("Gradient calibration requires exactly 100 saved episodes")
    keys = set()
    for episode in episodes:
        if not isinstance(episode, EpisodeSnapshot) or not episode.path:
            raise ValueError("Gradient calibration artifact lacks episode trajectories")
        key = candidate_identity(episode.terminal_key)
        if key in keys or any(not math.isfinite(float(step.reward)) for step in episode.path):
            raise ValueError("Gradient calibration requires distinct finite-reward episodes")
        keys.add(key)


def _save_probe_episodes(out, episodes, identity, *, source, budget):
    _validate_probe_episodes(episodes)
    payload = {"schema_version": 1, "purpose": "gradient_clipping",
               "identity": identity, "episodes": episodes,
               "episodes_hash": _episodes_hash(episodes), "source": source,
               "accounting": CALIBRATION_ACCOUNTING, "source_budget": copy.deepcopy(budget)}
    path = Path(out) / GRADIENT_CALIBRATION_ARTIFACT
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return copy.deepcopy(episodes)


def _reuse_probe_episodes(source, out, identity, predictor):
    path = Path(source).parent / GRADIENT_CALIBRATION_ARTIFACT
    if not path.is_file():
        # A reward-only run already paid for and saved all pilot trajectories.
        # Re-score those exact observations under its frozen reward instead of
        # purchasing a second pilot merely because clipping was disabled then.
        path = Path(source).parent / CALIBRATION_STATE
        if not path.is_file():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        frozen = checkpoint.get("frozen")
        evaluation = copy.deepcopy(checkpoint.get("execution_contract", {}).get("evaluation", {}))
        evaluation.pop("reward_calibration", None)
        checkpoint_identity = checkpoint.get("pilot_identity") or {
            "evaluation": evaluation, "environment": checkpoint.get("environment_contract"),
            "reward_calibration": frozen.get("content_hash") if isinstance(frozen, dict) else None,
        }
        if checkpoint_identity != identity:
            print("[INFO] Saved reward pilot trajectories lack a matching verified scientific/state/action "
                  "contract; collecting one new 100-candidate gradient pilot.", flush=True)
            return None
        entries = checkpoint.get("entries")
        if (checkpoint.get("type") != "reward_calibration" or frozen is None
                or checkpoint.get("accounting") != CALIBRATION_ACCOUNTING
                or not isinstance(entries, list) or len(entries) != GRADIENT_CALIBRATION_CANDIDATES):
            raise ValueError("Saved reward pilot is not a completed 100-candidate calibration")
        budget = {"schema_version": 1, "accounting": CALIBRATION_ACCOUNTING,
                  "target_candidates": GRADIENT_CALIBRATION_CANDIDATES,
                  "valid_candidates": len(entries), "attempts_started": checkpoint["attempts"],
                  "evaluations": checkpoint["evaluations"], "status": "COMPLETED"}
        report_path = path.parent / CALIBRATION_BUDGET
        if not report_path.is_file() or json.loads(report_path.read_text()) != budget:
            raise ValueError("Saved reward pilot cost report differs from its checkpoint")
        episodes = []
        for episode, observation in entries:
            episode = copy.deepcopy(episode)
            if checkpoint.get("purpose") != "gradient_clipping":
                scored = predictor.score_calibration_observation(observation)
                episode.path[-1].reward = float(scored["reward"])
                episode.prediction_record = copy.deepcopy(scored["prediction_record"])
            episode.metadata["gradient_calibration"] = True
            episodes.append(episode)
        saved = {"schema_version": 1, "purpose": "gradient_clipping", "identity": identity,
                 "episodes": episodes, "episodes_hash": _episodes_hash(episodes),
                 "source_budget": budget}
    else:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != 1 or saved.get("purpose") != "gradient_clipping":
        raise ValueError("Unknown gradient calibration episode artifact")
    episodes = saved.get("episodes")
    _validate_probe_episodes(episodes)
    if saved.get("episodes_hash") != _episodes_hash(episodes):
        raise ValueError("Gradient calibration episodes content hash mismatch")
    if saved.get("identity") != identity:
        print("[INFO] Saved gradient pilot uses a different scientific/state/action contract; "
              "collecting one new 100-candidate pilot.", flush=True)
        return None
    result = _save_probe_episodes(out, episodes, identity, source=str(path.resolve()),
                                 budget=saved.get("source_budget"))
    _write_json(Path(out) / "gradient_calibration_reuse.json", {
        "schema_version": 1, "source": str(path.resolve()),
        "episodes_hash": saved["episodes_hash"], "reused_candidates": len(result),
        "new_evaluations": 0, "source_budget": saved.get("source_budget"),
        "accounting": CALIBRATION_ACCOUNTING,
    })
    print("[INFO] Reused 100 gradient pilot trajectories without new evaluations", flush=True)
    return result


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


def _prepare_gradient_pilot(*, env, predictor, out, contract, total_budget,
                            attempt_limit, resume_pilot, calibration):
    """Use the unchanged terminal reward for a separate, resumable probe pilot."""
    out = Path(out)
    checkpoint_path = out / CALIBRATION_STATE
    entries, attempts, evaluations, misses, frozen = [], 0, 0, 0, None
    initial_rng_state = capture_rng_state()
    recorder = getattr(env, "hpo_recorder", None)
    identity = _pilot_identity(env, contract, calibration)

    def budget_report():
        return {"schema_version": 1, "accounting": CALIBRATION_ACCOUNTING,
                "target_candidates": GRADIENT_CALIBRATION_CANDIDATES,
                "valid_candidates": len(entries), "attempts_started": attempts,
                "evaluations": evaluations,
                "status": "COMPLETED" if frozen is not None else "RUNNING"}

    if resume_pilot:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved.get("type") != "reward_calibration"
                or saved.get("purpose") != "gradient_clipping"
                or saved.get("accounting") != CALIBRATION_ACCOUNTING
                or saved.get("num_candidates") != GRADIENT_CALIBRATION_CANDIDATES
                or saved.get("total_budget") != total_budget
                or saved.get("pilot_identity") != identity):
            raise ValueError("Gradient pilot resume requires its original scientific and learning contract")
        require_execution_contract(saved["execution_contract"], contract)
        if saved.get("initial_rng_state") is None:
            raise ValueError("Gradient pilot checkpoint lacks its initial RNG state")
        initial_rng_state = saved["initial_rng_state"]
        if predictor.evaluation_log_size() != saved["evaluation_log_size"]:
            raise ValueError("Evaluation log differs from the last settled calibration checkpoint")
        entries, attempts, evaluations, misses, frozen = (
            saved[name] for name in ("entries", "attempts", "evaluations", "misses", "frozen"))
        report = out / CALIBRATION_BUDGET
        if not report.is_file() or json.loads(report.read_text()) != budget_report():
            raise ValueError("Calibration budget report differs from the last settled checkpoint")
        restore_predictor_state(predictor, saved)
        restore_rng_state(saved["rng_state"])
    elif checkpoint_path.exists() or (out / GRADIENT_CALIBRATION_ARTIFACT).exists():
        raise ValueError("Gradient pilot files already exist. Resume this run or choose a new output directory.")

    def save():
        payload = {
            "type": "reward_calibration", "purpose": "gradient_clipping",
            "execution_contract": contract, "pilot_identity": identity,
            "accounting": CALIBRATION_ACCOUNTING,
            "num_candidates": GRADIENT_CALIBRATION_CANDIDATES,
            "total_budget": total_budget, "attempts": attempts,
            "evaluations": evaluations, "misses": misses, "entries": entries,
            "frozen": frozen, "loaded_reward_calibration": calibration,
            "initial_rng_state": initial_rng_state,
            "evaluation_log_size": predictor.evaluation_log_size(),
            **predictor_checkpoint_fields(predictor),
        }
        if recorder is not None:
            payload["hpo_state"] = recorder.checkpoint_state()
        _write_json(out / CALIBRATION_BUDGET, budget_report())
        _save_checkpoint(str(checkpoint_path), payload)

    predictor.set_phase("calibration")
    seen = {candidate_identity(episode.terminal_key) for episode, _ in entries}
    while len(entries) < GRADIENT_CALIBRATION_CANDIDATES:
        if misses >= attempt_limit:
            raise ValueError(f"Gradient calibration unresolved after {attempts} attempts and "
                             f"{evaluations} evaluations ({len(entries)}/100 valid distinct candidates)")
        attempts += 1
        started = time.perf_counter()
        try:
            candidate, episode = _draw_candidate(env)
        except InfeasibleEpisode:
            misses += 1
            save()
            continue
        key = candidate_identity(episode.terminal_key)
        if key in seen:
            misses += 1
            save()
            continue
        rollout_s = time.perf_counter() - started
        evaluations += 1
        started = time.perf_counter()
        # The terminal adapter preserves the existing scientific reward and
        # predictor timing/ledger. HPO charging belongs to EpisodeBatchEvaluator,
        # which is deliberately not used for this outside-budget pilot.
        reward_fn = env.reward_fn
        old_capture = getattr(reward_fn, "capture_prediction_records", None)
        if old_capture is not None:
            reward_fn.capture_prediction_records = True
        try:
            reward = float(reward_fn(candidate))
            if not math.isfinite(reward):
                raise ValueError("Gradient pilot returned a nonfinite terminal reward")
        except EvaluationProtocolError:
            raise
        except Exception as exc:
            predictor.record_episode({"phase": "calibration", "status": "failed",
                                      "formula": episode.formula, "rollout_s": rollout_s})
            misses += 1
            save()
            print(f"[WARN] Gradient pilot evaluation failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        finally:
            if old_capture is not None:
                reward_fn.capture_prediction_records = old_capture
        eval_s = time.perf_counter() - started
        episode.path[-1].reward = reward
        predictions = getattr(reward_fn, "last_predictions", [])
        episode.prediction = copy.deepcopy(predictions[0]) if len(predictions) == 1 else None
        records = getattr(reward_fn, "last_prediction_records", [])
        episode.prediction_record = copy.deepcopy(records[0]) if len(records) == 1 else {}
        episode.metadata["gradient_calibration"] = True
        timings = predictor.last_evaluation_timings
        episode.timing = {**(timings[0] if len(timings) == 1 else {}),
                          "evaluation_wave_id": attempts, "evaluation_wave_size": 1,
                          "rollout_s": rollout_s, "eval_wall_share_s": eval_s,
                          "episode_wall_share_s": rollout_s + eval_s}
        entries.append((episode, {"candidate_key": key, "reward": reward}))
        seen.add(key)
        misses = 0
        save()
    if frozen is None:
        frozen = calibration or {"gradient_calibration_complete": True}
        save()
    if calibration is not None:
        _write_json(out / CALIBRATION_ARTIFACT, calibration)
    episodes = _save_probe_episodes(out, [episode for episode, _ in entries], identity,
                                    source="fixed_reward_pilot", budget=budget_report())
    predictor.set_phase("train")
    predictor.mark("calibration_end")
    # A gradient-only pilot must not move the learner's sampling/initialization
    # RNG stream. During an interrupted pilot, rng_state still resumes the
    # actual pilot position; its separately saved initial state is restored only
    # after all 100 evaluations settle. The existing PopArt pilot is unchanged.
    restore_rng_state(initial_rng_state)
    print(f"[INFO] Prepared 100 gradient pilot candidates after {evaluations} evaluations "
          "(outside the training budget)", flush=True)
    return episodes


def prepare_reward_calibration(
    *, env, predictor, out, contract, total_budget, attempt_limit,
    only_generate=False, resume_learner=False, resume_pilot=False, load_path=None,
    gradient_clipping="none",
):
    """Freeze a separate prediction pilot before any learner budget is consumed.

    Pilot observations never enter replay, training seen sets, or HPO candidate
    archives. Their costs are recorded separately from the training allowance.
    Adaptive clipping returns these same 100 finalized trajectories for gradient
    probes; it never adds them to learner replay or performs an optimizer update.
    """
    out = Path(out)
    artifact_path = out / CALIBRATION_ARTIFACT
    checkpoint_path = out / CALIBRATION_STATE
    recorder = getattr(env, "hpo_recorder", None)
    probing = gradient_clipping == "clipping"
    reward_enabled = getattr(predictor, "reward_calibration_enabled", True)
    if resume_pilot:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("purpose") == "gradient_clipping":
            if not probing:
                raise ValueError("Resuming a gradient pilot requires adaptive clipping")
            calibration = saved.get("loaded_reward_calibration")
            if calibration is not None:
                predictor.load_reward_calibration(calibration)
                if artifact_path.is_file() and json.loads(artifact_path.read_text()) != calibration:
                    raise ValueError("Loaded reward calibration differs from the gradient pilot checkpoint")
            return _prepare_gradient_pilot(
                env=env, predictor=predictor, out=out, contract=contract,
                total_budget=total_budget, attempt_limit=attempt_limit,
                resume_pilot=True, calibration=calibration,
            )
    if not reward_enabled:
        if load_path is not None:
            raise ValueError("Reward calibration is disabled; cannot load its artifact")
        if only_generate or resume_learner or not probing:
            return None
        return _prepare_gradient_pilot(
            env=env, predictor=predictor, out=out, contract=contract,
            total_budget=total_budget, attempt_limit=attempt_limit,
            resume_pilot=resume_pilot, calibration=None,
        )
    if only_generate or resume_learner or load_path is not None:
        source = Path(load_path) if load_path else artifact_path
        if not source.is_file():
            raise ValueError(f"Frozen reward calibration required: {source}")
        predictor.load_reward_calibration(json.loads(source.read_text()))
        # A run using another grid's compatible artifact keeps a local copy.
        if not only_generate and not resume_learner and source != artifact_path and not probing:
            _write_json(artifact_path, predictor.reward_calibration_state_dict())
        if only_generate or resume_learner or not probing:
            return None
        calibration = predictor.reward_calibration_state_dict()
        reused = _reuse_probe_episodes(source, out, _pilot_identity(env, contract, calibration), predictor)
        if reused is not None:
            if source != artifact_path:
                _write_json(artifact_path, calibration)
            return reused
        return _prepare_gradient_pilot(
            env=env, predictor=predictor, out=out, contract=contract,
            total_budget=total_budget, attempt_limit=attempt_limit,
            resume_pilot=False, calibration=calibration,
        )

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
            "environment_contract": environment_contract(env),
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
    episodes = []
    for episode, obs in entries:
        scored = predictor.score_calibration_observation(obs)
        reward = scored["reward"]
        finalized_rows.append({"observation": obs, "reward_status": "final", "reward": reward})
        if probing:
            episode = copy.deepcopy(episode)
            episode.path[-1].reward = float(reward)
            episode.prediction_record = copy.deepcopy(scored["prediction_record"])
            episode.metadata["gradient_calibration"] = True
            episodes.append(episode)
    _write_json(out / "reward_calibration_observations.json", finalized_rows)
    _write_json(artifact_path, frozen)
    predictor.set_phase("train")
    predictor.mark("calibration_end")
    print(f"[INFO] Froze reward calibration from {len(entries)} distinct candidates "
          f"after {evaluations} evaluations (outside the training budget)", flush=True)
    if probing:
        return _save_probe_episodes(out, episodes, _pilot_identity(env, contract, frozen),
                                    source="reward_calibration_pilot", budget=budget_report())
    return None
