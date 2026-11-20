"""Read-only storage/accounting audit before restarting a trusted local campaign.

This does not run predictors, instantiate networks, repair files, or restore RNGs.
Runtime/source/scenario compatibility remains the worker/controller's responsibility.
Unfinished checkpoints are PyTorch pickle files: inspect only trusted local outputs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
import json
from pathlib import Path

from .artifacts import (
    CALIBRATION_CHECKPOINT, REWARD_CALIBRATION, digest, file_digest, read_json,
    resumable_checkpoint, reward_calibration_requested,
)

_CLASSIFICATIONS = ("CACHED", "SYNCHRONIZED", "RECOVERABLE_TERMINAL", "BLOCKED")
_TERMINAL = {"COMPLETED", "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}
_RETRYABLE_RESULTS = {"INTERRUPTED", "INCOMPLETE", "SYSTEM_ERROR"}
_COUNTERS = ("attempts_started", "unique_evaluated", "target_unique", "max_attempts",
             "pending", "failed_attempts", "preparation_only_attempts")


def _mapping(value, description):
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _read(path):
    return _mapping(read_json(path), str(path))


def _validate_hashes(result):
    hashes = _mapping(result.get("artifact_hashes", {}), "artifact_hashes")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Recorded artifact is missing or changed: {path}")


def _validate_records(directory, ledger):
    for key in _COUNTERS:
        value = ledger.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Candidate ledger {key} must be a nonnegative integer")
    unique, target = ledger["unique_evaluated"], ledger["target_unique"]
    if target < 1 or unique > target or ledger["attempts_started"] < unique:
        raise ValueError("Candidate ledger counts are inconsistent")
    ratio = Decimal(str(ledger.get("ratio_limit")))
    if not ratio.is_finite() or ratio <= 1:
        raise ValueError("Candidate ledger ratio_limit is invalid")
    records_path = directory / "candidate_records.jsonl"
    # Read every line, including a partial last line; never truncate a crash tail.
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    identities = []
    for record in records:
        _mapping(record, "Candidate record")
        identity = record.get("identity")
        if not isinstance(identity, str):
            raise ValueError("Candidate archive is missing canonical string identities")
        identities.append(identity)
    if len(records) != unique or len(set(identities)) != unique:
        raise ValueError("Candidate archive identities/count differ from the candidate ledger")
    return set(identities)


def _validate_final_models(directory, method, result=None):
    if method in {"a2c", "reinforce"}:
        weight = "policy.pt"
    elif method and method.startswith("dqn"):
        weight = "qnet.pt"
    else:
        candidates = [name for name in ("policy.pt", "qnet.pt") if (directory / name).is_file()]
        if len(candidates) != 1:
            raise ValueError("Completed training lacks an unambiguous final policy.pt/qnet.pt")
        weight = candidates[0]
    names = [weight, "std_scaler.bin"]
    config_path = directory / "config.yaml"
    if config_path.is_file():
        import yaml
        cfg = _mapping(yaml.safe_load(config_path.read_text()), "Job configuration")
        if reward_calibration_requested(cfg):
            names.append(REWARD_CALIBRATION)
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Completed training is missing its final model/scaler or reward calibration: {path}")
        if result is not None and result.get("artifact_hashes"):
            if str(path) not in result["artifact_hashes"]:
                raise ValueError(f"Completed training has no recorded fingerprint for final artifact: {path}")


def _validate_pilot_checkpoint(directory, saved, ledger, identities):
    """A settled pilot has trajectories and the oracle ledger before model setup."""
    _require_fields(saved, ("execution_contract", "total_budget", "attempts", "evaluations",
                            "misses", "entries", "evaluation_log_size"), "Calibration checkpoint")
    if saved.get("type") != "reward_calibration" or saved["total_budget"] != ledger["target_unique"]:
        raise ValueError("Calibration checkpoint type or training allowance differs from the ledger")
    counters = [saved[key] for key in ("attempts", "evaluations", "misses", "evaluation_log_size")]
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("Calibration checkpoint counters must be nonnegative integers")
    if (saved["attempts"] != ledger["attempts_started"] or
            not len(identities) <= saved["evaluations"] <= saved["attempts"] or
            saved["misses"] > saved["attempts"]):
        raise ValueError("Calibration checkpoint counters differ from synchronized HPO accounting")

    from ..utils.candidate_records import candidate_identity

    entries = saved["entries"]
    if not isinstance(entries, list) or len(entries) != len(identities):
        raise ValueError("Calibration checkpoint trajectories differ from candidate archive count")
    entry_keys, observation_keys = set(), set()
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Calibration checkpoint entry needs a trajectory and a raw observation")
        episode, obs = entry
        if not getattr(episode, "path", None):
            raise ValueError("Calibration checkpoint entry lacks its terminal trajectory")
        observation = _mapping(obs, "Calibration observation")
        key = observation.get("candidate_key")
        if not isinstance(key, str) or not key or key in observation_keys:
            raise ValueError("Calibration checkpoint needs distinct physical observations")
        observation_keys.add(key)
        entry_keys.add(candidate_identity(episode.terminal_key))
    if entry_keys != identities:
        raise ValueError("Calibration checkpoint trajectories differ from candidate archive identities")
    evaluation_log = directory / "evaluations.jsonl"
    if not evaluation_log.is_file() or evaluation_log.stat().st_size != saved["evaluation_log_size"]:
        raise ValueError("Evaluation log differs from the last settled calibration checkpoint")
    artifact_path = directory / REWARD_CALIBRATION
    if artifact_path.exists():
        artifact = _read(artifact_path)
        frozen = _mapping(saved.get("frozen"), "Frozen calibration checkpoint state")
        hashed = digest({key: value for key, value in frozen.items() if key != "content_hash"})
        if artifact != frozen or hashed != frozen.get("content_hash"):
            raise ValueError("Frozen reward calibration differs from the synchronized checkpoint")


def _require_fields(state, names, description):
    missing = [name for name in names if state.get(name) is None]
    if missing:
        raise ValueError(f"{description} is missing resumable state: {', '.join(missing)}")


def _validate_checkpoint(directory, ledger, identities, method):
    import torch

    phase = ledger["phase"]
    path = resumable_checkpoint(directory, phase)
    if not path.is_file():
        raise ValueError(f"Consumed attempts have no resumable checkpoint: {path}; paid cost cannot be reset")
    saved = _mapping(torch.load(path, map_location="cpu", weights_only=False), "Checkpoint")
    if saved.get("resume_safe") is False:
        raise ValueError("Diagnostic partial checkpoint is not safe for training restart")
    _require_fields(saved, ("hpo_state", "rng_state", "predictor_ledger"), "Checkpoint")
    hpo = _mapping(saved["hpo_state"], "Checkpoint hpo_state")
    budget = _mapping(hpo.get("budget"), "Checkpoint candidate budget")
    if hpo.get("phase") != phase:
        raise ValueError("Checkpoint phase differs from candidate ledger")
    if (budget.get("target_unique") != ledger["target_unique"]
            or Decimal(str(budget.get("ratio_limit"))) != Decimal(str(ledger["ratio_limit"]))):
        raise ValueError("Checkpoint budget/efficiency limit differs from candidate ledger")
    if any(budget.get(key) != ledger[key] for key in _COUNTERS) or budget.get("pending") != 0:
        raise ValueError("Candidate ledger advanced beyond checkpoint or is unsynchronized; refusing to erase already spent attempts")
    if file_digest(directory / "candidate_records.jsonl") != hpo.get("records_sha256"):
        raise ValueError("Candidate archive differs from synchronized checkpoint; paid observations were preserved")
    seen = hpo.get("seen")
    if (not isinstance(seen, (list, tuple)) or len(seen) != len(identities)
            or set(seen) != identities):
        raise ValueError("Checkpoint identities differ from candidate archive")
    rng = _mapping(saved["rng_state"], "Checkpoint RNG state")
    if rng.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint RNG state")
    _require_fields(rng, ("python", "numpy", "torch_cpu", "torch_cuda"), "Checkpoint RNG state")
    predictor = _mapping(saved["predictor_ledger"], "Checkpoint predictor ledger")
    if predictor.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint predictor ledger")
    _require_fields(predictor, ("n_calls", "seen", "t_predict_s", "phase", "phase_costs"),
                    "Checkpoint predictor ledger")
    if phase == "generation":
        if saved.get("type") != "hpo_generation":
            raise ValueError("Generation checkpoint has an invalid type")
        contract = _mapping(saved.get("generation_contract"), "Generation sampling contract")
        _require_fields(contract, ("execution", "sampling", "frozen_artifacts"), "Generation contract")
        return
    if path.name == CALIBRATION_CHECKPOINT:
        _validate_pilot_checkpoint(directory, saved, ledger, identities)
        return
    scaler = directory / "std_scaler.bin"
    if not scaler.is_file() or scaler.stat().st_size == 0:
        raise ValueError(f"Training restart requires the original scaler: {scaler}")
    _require_fields(saved, ("training_budget_state", "execution_contract", "environment_contract",
                            "scaler_fingerprint", "episodes_completed"), "Training checkpoint")
    from ..utils.training_budget import TrainingBudget
    training_budget = _mapping(saved["training_budget_state"], "Training budget state")
    restored_budget = TrainingBudget(count_duplicates=False,
                   max_duplicate_attempts=training_budget.get("max_duplicate_attempts", 1000),
                   state=training_budget)
    from ..utils.candidate_records import candidate_identity
    if (restored_budget.counted != ledger["unique_evaluated"]
            or {candidate_identity(key) for key in training_budget["seen_keys"]} != identities):
        raise ValueError("Training budget identities/count differ from synchronized HPO accounting")
    if method in {"a2c", "reinforce"} and (saved.get("type") != "pg" or saved.get("rl_method") != method):
        raise ValueError("Training checkpoint algorithm differs from campaign method")
    if method and method.startswith("dqn") and saved.get("type") != "dqn":
        raise ValueError("Training checkpoint algorithm differs from campaign method")
    if saved.get("type") == "dqn":
        _require_fields(saved, ("qnet_state", "opt_state", "buffer",
                                "training_episodes_completed", "eps"), "DQN checkpoint")
        learning = saved["execution_contract"].get("learning", {})
        target_mode = learning.get("dqn_target_mode", "mc" if method == "dqn_mc" else "bootstrap")
        networks, optimizers = ("qnet_state",), ("opt_state",)
        if target_mode != "mc":
            _require_fields(saved, ("target_net_state",), "DQN bootstrap checkpoint")
            networks += ("target_net_state",)
    elif saved.get("type") == "pg":
        _require_fields(saved, ("policy_state", "opt_actor_state", "iteration", "visit_counts",
                                "entropy_coef_eff", "rl_method"), "PG checkpoint")
        networks, optimizers = ("policy_state",), ("opt_actor_state",)
        if saved["rl_method"] == "a2c":
            _require_fields(saved, ("value_net_state", "opt_critic_state"), "A2C checkpoint")
            networks += ("value_net_state",)
            optimizers += ("opt_critic_state",)
    else:
        raise ValueError("Training checkpoint has no supported full training state (weights-only restart refused)")
    for name in networks:
        if not _mapping(saved[name], name):
            raise ValueError(f"Checkpoint network state is empty: {name}")
    for name in optimizers:
        optimizer = _mapping(saved[name], name)
        _require_fields(optimizer, ("state", "param_groups"), name)


def _inspect_job(directory, method):
    row = {"job_id": directory.name, "jobdir": str(directory), "classification": "BLOCKED"}
    try:
        result_path = directory / "result.json"
        result = _read(result_path) if result_path.exists() else None
        if result and result.get("status") in {"INVALID_RECORDS", "QUALITY_CALIBRATION_REQUIRED"}:
            raise ValueError("Stored job has {}: {}".format(result["status"], result.get("reason") or result.get("error") or "inspect its preserved records"))
        if result and result.get("status") not in _RETRYABLE_RESULTS:
            if not isinstance(result.get("status"), str):
                raise ValueError("Result has no status")
            if not result.get("artifact_hashes"):
                raise ValueError("Completed result has no artifact fingerprints; intact storage cannot be confirmed")
            if result.get("execution_failure") or (result["status"] == "COMPLETED" and result.get("returncode") not in (None, 0)):
                raise ValueError("Completed result contains a recorded worker failure; restart cannot clear it")
            _validate_hashes(result)
            if result.get("phase") == "train" and result.get("eligible_for_selection"):
                _validate_final_models(directory, method, result)
            row.update(classification="CACHED", reason="Completed result and recorded artifact hashes are intact")
            return row
        if result and result.get("finalization_pending"):
            _validate_hashes(result)
        ledger_path = directory / "candidate_budget.json"
        if not ledger_path.exists():
            if (any((directory / name).exists() for name in (
                    "candidate_records.jsonl", "checkpoint.pt", "generation_checkpoint.pt", CALIBRATION_CHECKPOINT))
                    or (result or {}).get("ledger", {}).get("attempts_started", 0)):
                raise ValueError("Candidate accounting is incomplete: original candidate_budget.json is missing")
            row.update(classification="SYNCHRONIZED", reason="Job has no candidate accounting yet; no recorded attempts need replay")
            return row
        ledger = _read(ledger_path)
        row.update(phase=ledger.get("phase"), attempts_started=ledger.get("attempts_started"),
                   unique_evaluated=ledger.get("unique_evaluated"))
        if ledger.get("phase") not in ("train", "generation"):
            raise ValueError("Candidate ledger has an invalid phase")
        identities = _validate_records(directory, ledger)
        if ledger.get("status") in _TERMINAL:
            execution_path = directory / "execution.json"
            execution = _read(execution_path) if execution_path.exists() else {}
            if ledger["status"] == "COMPLETED" and (execution.get("status") == "SYSTEM_ERROR"
                    or execution.get("returncode") not in (None, 0)):
                raise ValueError("Terminal ledger has a recorded worker failure in execution.json; consumed cost cannot be reset")
            if result and (result.get("execution_failure") or (ledger["status"] == "COMPLETED" and (
                    result.get("status") == "SYSTEM_ERROR" or result.get("returncode") not in (None, 0)))):
                raise ValueError("Terminal ledger has a recorded worker failure; failure and consumed cost cannot be cleared by restart")
            if ledger["pending"]:
                raise ValueError("Terminal candidate ledger contains pending attempts")
            if ledger["status"] == "COMPLETED":
                if ledger["unique_evaluated"] != ledger["target_unique"]:
                    raise ValueError("COMPLETED candidate ledger has not reached its target")
                if ledger["phase"] == "train":
                    _validate_final_models(directory, method)
            row.update(classification="RECOVERABLE_TERMINAL", reason="Terminal candidate artifacts can be finalized without new sampling")
        elif not ledger["attempts_started"]:
            raise ValueError("Initialized accounting has zero consumed attempts; inspect the initialization failure (no automatic reset)")
        else:
            _validate_checkpoint(directory, ledger, identities, method)
            row.update(classification="SYNCHRONIZED", reason="Full checkpoint, candidate archive and paid-attempt ledger agree")
    except Exception as exc:
        row.update(classification="BLOCKED", reason=f"{type(exc).__name__}: {exc}")
    return row


def inspect_restart(out) -> dict:
    """Classify every existing job without writes, repair, evaluation or sampling.

    ``ready`` certifies storage/accounting only, not hardware, current source,
    runtime dependencies or scenario compatibility. Run after workers have stopped.
    """
    directory = Path(out).expanduser().resolve()
    errors, method = [], None
    if not directory.is_dir():
        errors.append(f"HPO output directory does not exist: {directory}")
    manifest = directory / "manifest.json"
    if manifest.exists():
        try:
            method = _read(manifest).get("plan", {}).get("method")
        except Exception as exc:
            errors.append(f"Cannot read campaign manifest: {type(exc).__name__}: {exc}")
    jobs = [_inspect_job(path, method) for path in sorted((directory / "jobs").glob("*")) if path.is_dir()]
    counts = Counter(row["classification"] for row in jobs)
    return {"out": str(directory), "ready": not errors and not counts["BLOCKED"],
            "job_counts": {name: counts[name] for name in _CLASSIFICATIONS}, "jobs": jobs,
            "errors": errors, "scope": "Read-only storage/accounting preflight; runtime/source compatibility is checked separately"}
