"""RL training functions for rl_matdesign experiments.

All functions are domain-agnostic: they operate on a :class:`MultiGroupEnv`
and a :class:`~rl_matdesign.predictors.base.PropertyPredictor` and contain
no material-system-specific logic.

Online DQN supports TD and Monte Carlo targets. DQN and the A2C critic may
fit PopArt-normalized targets while exposing predictions in return units.
Policy-gradient training standardizes actor advantages across the batch.
"""
from __future__ import annotations

import collections
import copy
import json
import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .env_multigroup import MultiGroupEnv
from .action_features import value_features, value_feature_dim
from .predictors.base import PropertyPredictor
from .value_normalization import POPART_PROFILE
from .exploration import ExplorationController, entropy_probabilities, normalized_entropy
from .utils.training_budget import TrainingBudget
from .utils.stage_timing import checkpoint_stage, checkpoint_prepared, record_checkpoint
from .utils.checkpoint_serialization import save_checkpoint_payload
from .utils.rollout_errors import InfeasibleEpisode, EvaluationProtocolError
from .utils.seeding import capture_rng_state, restore_rng_state
from .utils.generation_resume import predictor_checkpoint_fields
from .utils.model_contract import bind_scaler, environment_contract, scaler_fingerprint
from .utils.prediction_records import metadata_record, objective_record
from .utils.episode_batch import (
    CandidateEvaluationFailed, EpisodeBatchEvaluator, training_wave_limit,
)


# ---------------------------------------------------------------------------
# Candidate-evaluation failures (build/predictor errors unrelated to whether
# the composition itself is valid — e.g. every random MD/relax realization
# failing for a structure builder). Raised by ``reward_fn``/``mg_reward_fn``
# (scripts/run_experiment.py); rollout call sites below catch it and retry
# with a fresh candidate instead of either crashing the run or burning a
# real episode-budget slot on a candidate that was never actually scored.
# ---------------------------------------------------------------------------

# How many consecutive candidates may fail evaluation before a rollout call
# site gives up and lets the failure surface as a hard error. A handful of
# retries absorbs occasional transient/geometric failures (e.g. a random
# interstitial placement, or a random solid-solution decoration, that happens
# to destabilize an MD run); exhausting this many in a row means something
# systemic is wrong (a broken binary path, a missing model file, ...) that no
# amount of retrying will fix — better to fail loudly than spend the rest of
# the training budget on candidates that never get scored.
_MAX_CANDIDATE_RETRIES = 10


def _exploration_observation(episode) -> dict:
    """Read fixed scientific utility from the completed evaluation snapshot only."""
    record = episode.prediction_record or {}
    constraints = [value for name, value in record.items()
                   if name in {"gate_passed", "primary_ok", "constraint_passed"}
                   or (name.startswith("gate_") and name.endswith("_passed"))]
    if "post_constraint_passed" in getattr(episode, "metadata", {}):
        constraints.append(episode.metadata["post_constraint_passed"])
    if any(isinstance(value, (bool, np.bool_)) and not value for value in constraints):
        feasible = False
    elif record.get("prediction_record_error") or any(
            not isinstance(value, (bool, np.bool_)) for value in constraints):
        feasible = None
    else:
        feasible = True  # An ungated scalar reward has no feasibility constraint.
    quality = next((record[name] for name in
                    ("reward_standardized_utility", "reward_objective", "reward")
                    if record.get(name) is not None),
                   float(episode.path[-1].reward) if episode.path else None)
    return {"key": repr(episode.terminal_key), "quality": quality,
            "feasible": feasible, "successful": True}


def _flush_exploration(controller, pending, env) -> dict:
    """Apply outcome feedback after the behavior collection block has completed."""
    diagnostics = controller.observe_block(pending) if pending else controller.diagnostics()
    pending.clear()
    env.exploration_state = controller.state_dict()
    env.exploration_seen_keys = list(controller.seen_keys)
    return diagnostics


def _exploration_metrics(controller):
    return {"exploration_" + key: value for key, value in controller.diagnostics().items()}


def _exploration_failure(pending):
    def record(_failure, _phase):
        pending.append({"key": None, "quality": None, "feasible": None,
                        "successful": False})
    return record


def _restore_exploration(initial_target, adaptive, state, *, resuming=False):
    controller = ExplorationController(initial_target=initial_target, adaptive=adaptive)
    if resuming and state is None:
        raise ValueError("Exact resume requires exploration_state; old exploration checkpoints cannot be resumed")
    if state is not None:
        controller.load_state_dict(state)
    return controller


def _record_sampling_entropy(diagnostics, stage, probabilities):
    if diagnostics is None:
        return
    per_stage = diagnostics.setdefault("exploration_entropy_by_stage", {})
    per_stage[str(stage)] = normalized_entropy(probabilities)
    observed = [value for value in per_stage.values() if value is not None]
    diagnostics["exploration_entropy"] = float(np.mean(observed)) if observed else None


def _sampler_probabilities(scores, target, model=None):
    if not np.isfinite(scores).all():
        from .utils.candidate_budget import NumericalTrainingError
        raise NumericalTrainingError("Nonfinite network scores during action selection")
    normalizer = getattr(model, "value_normalizer", None)
    scale = float(normalizer.scale) if normalizer is not None else None
    return entropy_probabilities(scores, target, scale=scale)


def _sync_training_device(device) -> None:
    """Measure completed CUDA updates, not only asynchronous launch time."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _episode_timing_row(episode, update_s: float = 0.0) -> dict:
    timing = dict(episode.timing)
    timing.update({
        "formula": episode.formula,
        "status": "ok",
        "rl_update_s": update_s,
        "episode_wall_share_s": timing.get("episode_wall_share_s", 0.0) + update_s,
    })
    return timing


def _check_hpo_numerical(env, *losses, models=()):
    if getattr(env, "hpo_recorder", None) is None:
        return
    from .utils.candidate_budget import NumericalTrainingError
    if any(not math.isfinite(float(loss)) for loss in losses):
        raise NumericalTrainingError("Nonfinite loss in HPO training")
    if any(not torch.isfinite(parameter).all().item()
           for model in models if model is not None for parameter in model.parameters()):
        raise NumericalTrainingError("Nonfinite network parameters in HPO training")


def _record_episode_timing(timer, row) -> None:
    record = getattr(timer, "record_episode", None)
    if callable(record):
        record(row)


def _rollout_with_retry(rollout_fn: Callable[[], None]) -> None:
    """Call ``rollout_fn()`` (which must drive one full episode via
    ``env.initialize()`` + ``env.step()`` to a terminal reward); on
    :class:`CandidateEvaluationFailed`, discard the attempt and retry with a
    fresh candidate — up to :data:`_MAX_CANDIDATE_RETRIES` times — so the
    caller's episode-budget counter never advances for a candidate that
    failed evaluation.
    """
    for attempt in range(_MAX_CANDIDATE_RETRIES + 1):
        try:
            rollout_fn()
            return
        except (CandidateEvaluationFailed, InfeasibleEpisode) as exc:
            if attempt == _MAX_CANDIDATE_RETRIES:
                raise RuntimeError(
                    f"{_MAX_CANDIDATE_RETRIES + 1} consecutive candidates all "
                    "failed evaluation -- giving up rather than spending the "
                    f"rest of the run on unscored candidates. Last failure: {exc}"
                ) from exc
            kind = "infeasible construction" if isinstance(exc, InfeasibleEpisode) else "candidate evaluation failed"
            print(f"[WARN] {kind} (retry "
                  f"{attempt + 1}/{_MAX_CANDIDATE_RETRIES}), discarding and "
                  f"trying a new candidate: {exc}")


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------

def _make_loss_fn(name: str) -> torch.nn.Module:
    name = (name or "smoothl1").lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name in ("smoothl1", "huber"):
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unknown DQN loss: {name!r} (expected 'mse' or 'smoothl1')")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: str, data: dict, numbered_path: str | None = None) -> dict:
    """Atomically write a checkpoint dict.

    If *numbered_path* is given, data is written there and *path* is updated
    as a relative symlink so ``checkpoint.pt`` always resolves to the latest.
    """
    import os
    checkpoint_prepared()
    target = numbered_path if numbered_path else path
    tmp = target + ".tmp"
    measurements = save_checkpoint_payload(tmp, {**data, "rng_state": capture_rng_state()})
    replace_started = time.perf_counter()
    os.replace(tmp, target)
    if numbered_path:
        link_tmp = path + ".lnk"
        if os.path.lexists(link_tmp):
            os.remove(link_tmp)
        os.symlink(os.path.basename(numbered_path), link_tmp)
        os.replace(link_tmp, path)
    measurements["checkpoint_replace_s"] = time.perf_counter() - replace_started
    record_checkpoint(target, measurements)
    from .utils.hpo_stop import checkpoint_committed
    checkpoint_committed(data)
    return measurements


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def objective_from_mean_std(
    mean: float, std: float, objective: str = "mean", k: float = 1.0
) -> float:
    """Compute scalar reward from predictor (mean, std) output."""
    if objective == "mean":
        return mean
    elif objective == "mean_minus_kstd":
        return mean - k * std
    elif objective == "mean_plus_kstd":
        return mean + k * std
    else:
        raise ValueError(f"Unknown objective '{objective}'.")


# ---------------------------------------------------------------------------
# Element feature precomputation (Magpie)
# ---------------------------------------------------------------------------

def _precompute_elem_features(
    species_set: List[str],
    featurizer: Callable,
    env=None,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """Encode action labels, supplementing chemical identity with descriptors.

    Structured environments supply their own label table. Their position/field
    identifiers must never be interpreted as chemical formulae by Magpie.
    """
    custom = getattr(env, "action_features", None)
    if callable(custom):
        table = np.asarray(custom(), dtype=float)
        if table.ndim != 2 or table.shape[0] != len(species_set) or not np.isfinite(table).all():
            raise ValueError("Environment action label features must be a finite 2-D table")
        return table, None
    raw = np.asarray([featurizer(el) for el in species_set], dtype=float)
    elem_scaler = StandardScaler()
    elem_scaler.fit(raw)
    return np.concatenate((np.eye(len(species_set)), elem_scaler.transform(raw)), axis=1), elem_scaler


# ---------------------------------------------------------------------------
# Replay buffer helpers (classical DQN)
# ---------------------------------------------------------------------------

def _attach_mc_returns(rows: list, path: list, gamma: float) -> None:
    """Attach the discounted Monte-Carlo return G to each buffer row in place.

    Non-bootstrap regression target, used only when ``dqn_target_mode == 'mc'``
    (mirrors the reference npj DQN). ``rows`` are in episode order, aligned 1:1
    with ``path``. With a terminal-only reward this reduces to
    ``G_t = gamma**(T-1-t) * R_terminal``; the backward recursion below stays
    correct even if a config ever emits intermediate rewards.
    """
    G = 0.0
    for k in reversed(range(len(path))):
        G = float(path[k].reward) + gamma * G   # terminal step: G = reward
        rows[k]["mc_return"] = G


def _action_value_record(step, alphabet) -> dict:
    index = int(np.argmax(step.action_comp_onehot))
    features = value_features(alphabet, [index])[0]
    encoding = getattr(alphabet, "encoding", None)
    spec = encoding.specs[index] if encoding is not None else None
    numeric = (float(spec.value) if spec.kind == "numeric" else None) if spec is not None else float(alphabet[index])
    return {"a_comp_idx": index, "a_comp_features": features, "a_comp_val": numeric}


def _row_value_features(row):
    return row["a_comp_features"] if "a_comp_features" in row else [row["a_comp_val"]]


def add_episode_to_buffer(
    path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    gamma: float = 0.9,
) -> None:
    """Convert a completed episode path into buffer rows and append.

    Each row stores:
    - Raw (unscaled) s_mat and s_mat_next for lazy scaling at batch time.
    - Element index and comp scalar value instead of one-hot vectors.
    - ``next_allowed_idx``: list of (elem_idx, comp_idx) pairs for the next
      step, used to compute the max-Q bootstrap target.
    """
    rows: List[dict] = []
    for k, step in enumerate(path):
        done = k == len(path) - 1
        next_step = path[k + 1] if not done else None
        next_allowed_idx: List[Tuple[int, int]] = []
        if not done:
            for (elem_oh, comp_oh) in next_step.allowed_actions:
                next_allowed_idx.append(
                    (int(np.argmax(elem_oh)), int(np.argmax(comp_oh)))
                )
        rows.append({
            "s_mat_raw":       np.asarray(step.state_material_features, dtype=float),
            "s_step":          np.asarray(step.state_step_onehot, dtype=float),
            "a_elem_idx":      int(np.argmax(step.action_elem_onehot)),
            **_action_value_record(step, action_values),
            "reward":          float(step.reward),
            "s_mat_next_raw":  np.asarray(next_step.state_material_features, dtype=float)
                               if not done else np.zeros(len(step.state_material_features), dtype=float),
            "s_step_next":     np.asarray(next_step.state_step_onehot, dtype=float)
                               if not done else np.zeros(len(step.state_step_onehot), dtype=float),
            "next_allowed_idx": next_allowed_idx,
            "done":            done,
        })
    # --- BEGIN mc-target experiment (removable) ---
    _attach_mc_returns(rows, path, gamma)
    # --- END mc-target experiment ---
    buffer.extend(rows)


def _row_fingerprint(row: dict) -> tuple:
    """Hashable fingerprint of a buffer row, used for within-episode dedup."""
    return (
        tuple(row["s_mat_raw"].tolist()),
        tuple(row["s_step"].tolist()),
        int(row["a_elem_idx"]),
        tuple(np.asarray(_row_value_features(row)).tolist()),
        bool(row["done"]),
    )


def _path_to_rows(path: list, action_values: List[str], gamma: float = 0.9) -> List[dict]:
    """Convert an env.path into row dicts (matching add_episode_to_buffer)."""
    rows: List[dict] = []
    n = len(path)
    for k, step in enumerate(path):
        done = k == n - 1
        next_step = path[k + 1] if not done else None
        next_allowed_idx: List[Tuple[int, int]] = []
        if not done:
            for (elem_oh, comp_oh) in next_step.allowed_actions:
                next_allowed_idx.append(
                    (int(np.argmax(elem_oh)), int(np.argmax(comp_oh)))
                )
        rows.append({
            "s_mat_raw":       np.asarray(step.state_material_features, dtype=float),
            "s_step":          np.asarray(step.state_step_onehot, dtype=float),
            "a_elem_idx":      int(np.argmax(step.action_elem_onehot)),
            **_action_value_record(step, action_values),
            "reward":          float(step.reward),
            "s_mat_next_raw":  np.asarray(next_step.state_material_features, dtype=float)
                               if not done else np.zeros(len(step.state_material_features), dtype=float),
            "s_step_next":     np.asarray(next_step.state_step_onehot, dtype=float)
                               if not done else np.zeros(len(step.state_step_onehot), dtype=float),
            "next_allowed_idx": next_allowed_idx,
            "done":            done,
        })
    _attach_mc_returns(rows, path, gamma)
    return rows


# Warn only once when there are insufficient alternative action orders.
_AUG_WARNED_NO_ALTS = False


def _augment_episode_in_buffer(
    env,
    original_path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    *,
    K: int,
    gamma: float = 0.9,
    terminal_key=None,
) -> None:
    """Insert *original_path* plus up to *K* permutation-augmented copies.

    Always inserts the original first (preserving exact ``K=0`` behavior).
    For each augmentation, re-drives the env with a permuted action sequence
    using ``env.step``, which keeps featurization, allowed_actions, and the
    constraint-filter outputs consistent with production code. The expensive
    predictor call is avoided by temporarily swapping ``env.reward_fn`` for
    a constant returning the original terminal reward.

    Dedup is scoped to this episode's augmentation pass: rows whose
    fingerprint already exists in the original or earlier-augmented rows
    of *this* episode are skipped. Cross-episode coincidences are kept,
    since they are legitimate independent observations.
    """
    global _AUG_WARNED_NO_ALTS

    # Always insert the original path first (zero risk when K == 0).
    add_episode_to_buffer(original_path, buffer, elem_feats_scaled, action_values, gamma=gamma)
    if K <= 0:
        return

    N = len(original_path)
    if N < 2:
        return

    permutable = list(range(N))

    import itertools

    if len(permutable) < 2:
        if not _AUG_WARNED_NO_ALTS:
            print(
                f"[WARN] dqn_augment_permutations: only {math.factorial(len(permutable))} "
                "permutation(s) of the permutable positions; augmentation is a no-op "
                "for this episode shape.",
                flush=True,
            )
            _AUG_WARNED_NO_ALTS = True
        return

    perm_count = math.factorial(len(permutable))
    alternatives_count = perm_count - 1
    if alternatives_count == 0:
        return
    num_to_sample = min(K, alternatives_count)
    if K > alternatives_count and not _AUG_WARNED_NO_ALTS:
        print(
            f"[WARN] dqn_augment_permutations: requested K={K} but only "
            f"{alternatives_count} alternative permutation(s) exist; using "
            f"K={alternatives_count}.",
            flush=True,
        )
        _AUG_WARNED_NO_ALTS = True

    identity = tuple(permutable)
    if len(permutable) <= 8:
        all_perms = list(itertools.permutations(permutable))
        all_perms.remove(identity)
        random.shuffle(all_perms)
        chosen_perms = all_perms[:num_to_sample]
    else:
        # Long sequences must not allocate N! trajectories to draw K examples.
        seen_perms = {identity}
        chosen_perms = []
        while len(chosen_perms) < num_to_sample:
            perm = tuple(random.sample(permutable, len(permutable)))
            if perm not in seen_perms:
                seen_perms.add(perm)
                chosen_perms.append(perm)

    # Build the action sequence from original_path.
    original_actions = [
        (step.action_elem_onehot, step.action_comp_onehot)
        for step in original_path
    ]

    # Build fingerprint set seeded with the original rows we just inserted.
    seen_fingerprints: set = set()
    for row in list(buffer)[-N:]:
        seen_fingerprints.add(_row_fingerprint(row))

    # Stash the original terminal reward and predictor.
    original_R = float(original_path[-1].reward)
    saved_reward_fn = env.reward_fn
    original_key = env.terminal_comp_key() if terminal_key is None else terminal_key

    try:
        env.reward_fn = lambda _formula: original_R
        for perm in chosen_perms:
            permuted_actions = [original_actions[i] for i in perm]

            env.initialize()
            try:
                for action in permuted_actions:
                    env.step(action)
            except (ValueError, RuntimeError) as _e:
                # An ordering that the env rejects (e.g., a constraint we
                # didn't anticipate): silently skip this augmentation.
                continue

            if not _env_terminal(env) or env.terminal_comp_key() != original_key:
                continue
            for row in _path_to_rows(env.path, action_values, gamma=gamma):
                fp = _row_fingerprint(row)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                buffer.append(row)
    finally:
        try:
            env.initialize()
            for action in original_actions:
                env.step(action)
        finally:
            env.reward_fn = saved_reward_fn


def _compute_td_target(
    row: dict,
    target_net: torch.nn.Module,
    s_mat_scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: torch.device,
    gamma: float,
    target_cache=None,
) -> float:
    """TD target for one buffer row: r if done, else r + γ·max_a' Q_target(s', a')."""
    if row["done"] or not row["next_allowed_idx"]:
        return float(row["reward"])
    cache_key = target_cache.key(row) if target_cache is not None else None
    if target_cache is not None:
        cached = target_cache.get(cache_key)
        if cached is not None:
            return float(row["reward"]) + gamma * cached
    next_idxs = row["next_allowed_idx"]
    n = len(next_idxs)
    ns_mat = s_mat_scaler.transform(row["s_mat_next_raw"].reshape(1, -1))[0]
    a_e = np.asarray([elem_feats_scaled[ei] for ei, _ in next_idxs], dtype=float)
    a_c = value_features(action_values, [fi for _, fi in next_idxs])
    ns_mat_b = np.repeat(ns_mat.reshape(1, -1), n, axis=0)
    ns_step_b = np.repeat(row["s_step_next"].reshape(1, -1), n, axis=0)
    with torch.no_grad():
        q_next = target_net(
            torch.tensor(ns_mat_b,  dtype=torch.float32, device=device),
            torch.tensor(ns_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_e,       dtype=torch.float32, device=device),
            torch.tensor(a_c,       dtype=torch.float32, device=device),
        ).reshape(-1)
    max_q = float(q_next.max().item())
    if target_cache is not None:
        target_cache.put(cache_key, max_q)
    return float(row["reward"]) + gamma * max_q


def _dqn_loss_terms(
    qnet: torch.nn.Module,
    batch: list,
    target_net: torch.nn.Module | None,
    s_mat_scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    loss_fn: torch.nn.Module,
    device: torch.device,
    gamma: float,
    target_mode: str = "bootstrap",
    target_cache=None,
    update_normalization: bool = True,
    target_values=None,
):
    """Fit one minibatch of raw returns in the selected value coordinates.

    ``target_mode='bootstrap'`` (default) uses the one-step TD target
    ``r + gamma*max_a' Q_target(s',a')``. ``target_mode='mc'`` regresses to the
    fixed discounted Monte-Carlo return stored on each row (no bootstrap).
    """
    if target_values is not None:
        y = target_values
    elif target_mode == "mc":
        y = [float(r["mc_return"]) for r in batch]
    else:
        y = [
            _compute_td_target(r, target_net, s_mat_scaler, elem_feats_scaled,
                               action_values, device, gamma, target_cache=target_cache)
            for r in batch
        ]
    s_mat_b  = torch.tensor(
        s_mat_scaler.transform(np.asarray([r["s_mat_raw"] for r in batch], dtype=float)),
        dtype=torch.float32, device=device,
    )
    s_step_b = torch.tensor(
        np.asarray([r["s_step"] for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    a_elem_b = torch.tensor(
        np.asarray([elem_feats_scaled[r["a_elem_idx"]] for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    a_comp_b = torch.tensor(
        np.asarray([_row_value_features(r) for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    qnet.train()
    inputs = (s_mat_b, s_step_b, a_elem_b, a_comp_b)
    normalizer = getattr(qnet, "value_normalizer", None)
    if normalizer is None:
        targets = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = qnet(*inputs)
    else:
        raw_targets = torch.tensor(y, dtype=torch.float64, device=device).reshape(-1, 1)
        targets = (qnet.update_value_normalization(raw_targets) if update_normalization
                   else normalizer.normalize(raw_targets, dtype=next(qnet.parameters()).dtype))
        prediction = qnet.forward_normalized(*inputs)
    loss = loss_fn(prediction, targets)
    return loss, prediction, targets, raw_targets if normalizer is not None else targets


def _dqn_gradient_step(
    qnet, batch, target_net, s_mat_scaler, elem_feats_scaled, action_values,
    optimizer, loss_fn, device, gamma, target_mode="bootstrap", target_cache=None,
    diagnostics=None, gradient_clipper=None,
) -> float:
    loss, prediction, targets, raw_targets = _dqn_loss_terms(
        qnet, batch, target_net, s_mat_scaler, elem_feats_scaled, action_values,
        loss_fn, device, gamma, target_mode=target_mode, target_cache=target_cache,
    )
    optimizer.zero_grad(set_to_none=True)
    if gradient_clipper is not None:
        _require_finite_loss(loss)
    loss.backward()
    clip_info = None
    before = None
    if gradient_clipper is not None:
        clip_info = _prepare_gradient_clipping(gradient_clipper, qnet)
        before = [p.detach().clone() for p in qnet.parameters()]
    optimizer.step()
    if gradient_clipper is not None:
        _require_finite_optimizer(qnet, optimizer)
        gradient_clipper.commit(clip_info)
        if diagnostics is not None:
            diagnostics.update(_clip_diagnostics("q", clip_info, qnet, before))
    normalizer = getattr(qnet, "value_normalizer", None)
    if diagnostics is not None and normalizer is not None:
        raw_error = (normalizer.decode(prediction.detach()) - raw_targets).abs().mean()
        loss_value, diagnostics["value_raw_mae"] = torch.stack((loss.detach(), raw_error)).tolist()
        return loss_value
    return float(loss.item())


def _prepare_gradient_clipping(controller, model):
    from .gradient_clipping import NonfiniteGradientError
    from .utils.candidate_budget import NumericalTrainingError
    try:
        return controller.prepare(model.parameters())
    except NonfiniteGradientError as exc:
        raise NumericalTrainingError(str(exc)) from exc


def _require_finite_loss(*losses):
    from .utils.candidate_budget import NumericalTrainingError
    if any(not torch.isfinite(loss.detach()).all().item() for loss in losses):
        raise NumericalTrainingError("Nonfinite loss before gradient clipping; incomplete update")


def _require_finite_optimizer(model, optimizer):
    from .utils.candidate_budget import NumericalTrainingError
    tensors = [p.detach() for p in model.parameters()]
    tensors += [v for state in optimizer.state.values() for v in state.values()
                if isinstance(v, torch.Tensor)]
    if any(not torch.isfinite(v).all().item() for v in tensors):
        raise NumericalTrainingError("Nonfinite optimizer result; retain the last complete checkpoint")


def _clip_diagnostics(prefix, info, model, before):
    displacement = sum(float((p.detach().double() - old.double()).square().sum().item())
                       for p, old in zip(model.parameters(), before)) ** .5
    return {f"{prefix}_grad_norm": info["raw_norm"],
            f"{prefix}_grad_cap": info["cap"],
            f"{prefix}_grad_factor": info["coefficient"],
            f"{prefix}_grad_clipped": int(info["clipped"]),
            f"{prefix}_grad_history": info["history_count"],
            f"{prefix}_grad_threshold_source": info["threshold_source"],
            f"{prefix}_grad_provisional_startup": int(info["provisional_startup"]),
            f"{prefix}_optimizer_update_norm": displacement}


# ---------------------------------------------------------------------------
# Classical online DQN training
# ---------------------------------------------------------------------------

def train_dqn_online(
    *,
    env: MultiGroupEnv,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: torch.device,
    hidden_dim: int = 128,
    n_warmup_eps: int = 500,
    dqn_num_train_eps: int = 20000,
    dqn_buffer_size: int = 50000,
    batch_size: int = 256,
    dqn_grad_steps_per_ep: int = 5,
    dqn_target_update_freq: int = 100,
    exploration_target: float = 0.7,
    exploration_adaptive: bool = True,
    gamma: float = 0.9,
    lr: float = 1e-3,
    loss_name: str = "smoothl1",
    checkpoint_cfg: Optional[dict] = None,
    resume_state: Optional[dict] = None,
    augment_permutations: int = 0,
    timer: Optional[object] = None,
    dqn_target_mode: str = "bootstrap",
    train_count_duplicates: bool = False,
    train_max_duplicate_attempts: int = 1000,
    training_samples=None,
    value_normalization: str = "none",
    value_normalization_profile: str = POPART_PROFILE,
    gradient_clipping: str = "none",
    gradient_calibration_episodes=None,
) -> Tuple[torch.nn.Module, StandardScaler, List[dict]]:
    """Online MC/bootstrap DQN with entropy-controlled legal-action sampling.

    Random rewarded warmup fits the feature scaler and fills replay. Online
    episodes retain the selected MC/TD target, replay and target-copy schedules.
    The exploration controller observes completed scientific outcomes only after
    each prefetched wave drains, and is restored exactly with optimizer/replay.
    Unique budgets still learn from repeated episodes; attempts remain observable.
    """
    if resume_state is None and n_warmup_eps < 1:
        raise ValueError("Fresh DQN training requires dqn_warmup_eps >= 1 to fit the state scaler")
    from .model import QRegressor
    if gradient_clipping not in ("none", "clipping"):
        raise ValueError("gradient_clipping must be none or clipping")
    if gradient_clipping == "clipping" and resume_state is None:
        from .gradient_calibration import _pilot
        _pilot(gradient_calibration_episodes)

    budget_state = resume_state.get("training_budget_state") if resume_state else None
    if resume_state is not None and budget_state is None and not train_count_duplicates:
        raise ValueError(
            "Cannot resume with train_count_duplicates=false: checkpoint has no "
            "training_budget_state containing previously seen candidates. Start a new run "
            "or resume with train_count_duplicates=true."
        )
    budget = TrainingBudget(
        count_duplicates=train_count_duplicates,
        max_duplicate_attempts=train_max_duplicate_attempts,
        state=budget_state,
    )
    budget_history_complete = bool(
        resume_state.get("training_budget_history_complete", budget_state is not None)
    ) if resume_state is not None else True
    if resume_state is not None and budget_state is None:
        print("[WARN] Legacy checkpoint has no training budget history: new diagnostic "
              "counters start at this resume; the existing episode schedule is preserved.", flush=True)
    if not train_count_duplicates:
        print("[INFO] Training budget counts new candidates across DQN warmup and training; "
              "duplicate episodes still enter replay and receive gradient updates.", flush=True)

    if dqn_target_mode == "mc":
        print(
            "[INFO] DQN target_mode='mc': regressing to fixed discounted "
            "Monte-Carlo returns (no bootstrap); target network and "
            "dqn_target_update_freq are inert.",
            flush=True,
        )
        if augment_permutations > 0:
            raise ValueError(
                "dqn_target_mode='mc' does not support permutation augmentation "
                "(augment_permutations>0) in the current benchmark protocol. "
                "Set dqn_augment_permutations=0 for the "
                "mc comparison run."
            )

    exploration = _restore_exploration(
        exploration_target, exploration_adaptive,
        (resume_state or {}).get("exploration_state"),
        resuming=resume_state is not None and resume_state.get("exact_resume", True),
    )
    if resume_state is None or not resume_state.get("exact_resume", True):
        exploration.seed_seen(repr(ep.terminal_key) for ep in gradient_calibration_episodes or ())
        exploration.seed_seen(getattr(env, "exploration_seen_keys", ()))

    if training_samples is not None:
        training_samples.start(
            budget.counted, history_complete=budget_history_complete,
            session_id=resume_state.get("training_sample_session_id") if resume_state else None,
        )
    exploration_pending = []
    evaluator = EpisodeBatchEvaluator(
        env, max_retries=_MAX_CANDIDATE_RETRIES, timer=timer,
        capture_predictions=True, failure_callback=_exploration_failure(exploration_pending),
    )

    # Warmup pays one real predictor call per episode (see below), so those
    # episodes are part of the training cost and belong in the log. Collected
    # here because `metrics` isn't built until after the warmup branch.
    warmup_rows: List[dict] = []

    if resume_state is not None:
        scaler     = resume_state["scaler"]
        qnet       = resume_state["qnet"]
        target_net = resume_state["target_net"] if dqn_target_mode == "bootstrap" else None
        optimizer  = resume_state["optimizer"]
        buffer     = resume_state["buffer"]
        start_ep   = int(resume_state.get("start_ep", 0))
        train_attempts = int(resume_state.get("training_episodes_completed", start_ep))
        loss_fn    = _make_loss_fn(loss_name)
        print(
            f"[INFO] DQN resume: start_ep={start_ep}, entropy_target={exploration.target:.4f}, "
            f"buffer_rows={len(buffer)}",
            flush=True,
        )
    else:
        buffer: collections.deque = collections.deque(maxlen=dqn_buffer_size)

        if augment_permutations > 0:
            print(
                f"[INFO] DQN warmup: {n_warmup_eps} random episodes with real rewards "
                f"(K={augment_permutations} permutation augmentations per episode)...",
                flush=True,
            )
        else:
            print(f"[INFO] DQN warmup: {n_warmup_eps} random episodes with real rewards...")
        pbar = tqdm(total=n_warmup_eps, desc="DQN warmup")
        warmup_counted = 0
        warmup_attempts = 0
        while warmup_counted < n_warmup_eps:
            budget.check_can_sample()
            episode = evaluator.next(
                lambda: _rollout_random_episode(env),
                limit=training_wave_limit(budget, n_warmup_eps - warmup_counted),
                retry_scalar=_rollout_with_retry,
                phase="dqn_warmup",
            )
            counted = budget.record(episode.terminal_key)
            warmup_counted += int(counted)
            warmup_attempts += 1
            if episode.path:
                warmup_rows.append({
                    "phase": "dqn_warmup",
                    "episode": warmup_attempts,
                    "return": float(episode.path[-1].reward),
                    "terminal_reward": float(episode.path[-1].reward),
                    "exploration_target_used": 1.0,
                    "terminal_comp_key": str(episode.terminal_key),
                    "budget_counted": counted,
                    "training_budget_history_complete": budget_history_complete,
                    **budget.snapshot(),
                    **_episode_timing_row(episode),
                })
                _record_episode_timing(timer, warmup_rows[-1])
                if training_samples is not None:
                    training_samples.record(episode, warmup_rows[-1])
            if augment_permutations > 0:
                _augment_episode_in_buffer(
                    env, episode.path, buffer, elem_feats_scaled, action_values,
                    K=augment_permutations, gamma=gamma, terminal_key=episode.terminal_key,
                )
            else:
                add_episode_to_buffer(episode.path, buffer, elem_feats_scaled, action_values, gamma=gamma)
            pbar.update(int(counted))
        pbar.close()
        # Unlike PG warmup (which neutralises reward_fn), DQN warmup pays a real
        # predictor call per episode. The mark makes that cost visible.
        if timer is not None:
            timer.mark("warmup_end")

        s_mat_all = np.asarray([r["s_mat_raw"] for r in buffer], dtype=float)
        scaler = bind_scaler(StandardScaler().fit(s_mat_all), env)
        if checkpoint_cfg and checkpoint_cfg.get("scaler_path"):
            # Mid-training checkpoints must be usable even if training later stops
            # at the duplicate guard, before the runner's final artifact save.
            import joblib
            joblib.dump(scaler, checkpoint_cfg["scaler_path"])
        state_dim = int(s_mat_all.shape[1])
        elem_dim  = int(elem_feats_scaled.shape[1])
        step_dim  = env.max_steps
        print(f"[INFO] s_mat scaler fitted on {len(s_mat_all)} warmup rows.")

        qnet = QRegressor(
            state_dim=state_dim, step_dim=step_dim,
            elem_dim=elem_dim, frac_dim=value_feature_dim(action_values), hidden_dim=hidden_dim,
            value_normalization=value_normalization,
            value_normalization_profile=value_normalization_profile,
        ).to(device)
        target_net = copy.deepcopy(qnet).eval() if dqn_target_mode == "bootstrap" else None
        optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
        loss_fn = _make_loss_fn(loss_name)
        start_ep = 0
        train_attempts = 0
        # Random warmup is an archive seed, not an adaptive-policy observation.
        exploration.seed_seen(repr(key) for key in budget.state_dict()["seen_keys"])
        exploration_pending.clear()

    gradient_clipper = None
    if gradient_clipping == "clipping":
        from .gradient_clipping import AdaptiveGradientClipper
        if resume_state is not None:
            saved_clip = resume_state.get("gradient_clipping_state")
            if saved_clip is None:
                raise ValueError("Exact clipping resume requires gradient_clipping_state")
            gradient_clipper = AdaptiveGradientClipper.from_state_dict(saved_clip, network="q")
        else:
            from .gradient_calibration import calibrate_dqn_gradients, write_gradient_report
            gradient_clipper, report = calibrate_dqn_gradients(
                qnet=qnet, target_net=target_net, episodes=gradient_calibration_episodes,
                scaler=scaler, elem_feats_scaled=elem_feats_scaled, action_values=action_values,
                loss_fn=loss_fn, device=device, gamma=gamma, target_mode=dqn_target_mode,
            )
            write_gradient_report(checkpoint_cfg, report)
    elif gradient_clipping != "none":
        raise ValueError("gradient_clipping must be none or clipping")

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    if checkpoint_cfg:
        ckpt_path    = checkpoint_cfg.get("path")
        ckpt_freq    = int(checkpoint_cfg.get("freq", 0))
    if resume_state and resume_state.get("rng_state") is not None:
        restore_rng_state(resume_state["rng_state"])

    metrics: List[dict] = []
    metrics.extend(warmup_rows)

    _flush_exploration(exploration, exploration_pending, env)
    if start_ep >= dqn_num_train_eps:
        print(
            f"[WARN] DQN: start_ep ({start_ep}) >= dqn_num_train_eps ({dqn_num_train_eps}); "
            "skipping training loop.",
            flush=True,
        )
        return qnet, scaler, metrics

    recorder = getattr(env, "hpo_recorder", None)
    if recorder is not None:
        recorder.partial_checkpoint_factory = lambda: {
            "type": "dqn", "qnet_state": qnet.state_dict(),
            "target_net_state": target_net.state_dict() if target_net is not None else None, "opt_state": optimizer.state_dict(),
            "gradient_clipping_state": gradient_clipper.state_dict() if gradient_clipper else None,
            "buffer": list(buffer), "scaler": scaler,
            "exploration_state": exploration.state_dict(),
            "training_budget_state": budget.state_dict(),
        }

    pbar = tqdm(total=dqn_num_train_eps, initial=start_ep, desc="DQN train")
    from .utils.td_target_cache import TDTargetCache
    target_cache = TDTargetCache() if dqn_target_mode == "bootstrap" else None

    def _rollout_entropy_episode() -> None:
        _rollout_policy_episode(
            env=env, qnet=qnet, scaler=scaler, device=device,
            elem_feats_scaled=elem_feats_scaled, action_values=action_values,
            exploration_target=exploration.target,
        )

    completed = start_ep
    while completed < dqn_num_train_eps:
        budget.check_can_sample()
        qnet.eval()
        episode = evaluator.next(
            _rollout_entropy_episode,
            limit=training_wave_limit(
                budget, dqn_num_train_eps - completed,
                checkpoint_freq=ckpt_freq if ckpt_path else 0,
                completed_attempts=train_attempts,
            ),
            retry_scalar=_rollout_with_retry,
            metadata={"exploration_target_used": exploration.target},
            phase="dqn_train",
        )
        counted = budget.record(episode.terminal_key)
        completed += int(counted)
        train_attempts += 1

        # Persist the evaluated sample before augmentation or updates can fail.
        if training_samples is not None:
            training_samples.record(episode, {
                "phase": "dqn_train", "episode": train_attempts,
                "iteration": train_attempts, "budget_counted": counted,
                "training_budget_history_complete": budget_history_complete,
                "terminal_comp_key": str(episode.terminal_key), **budget.snapshot(),
            })

        _sync_training_device(device)
        update_started = time.perf_counter()
        episode_reward = float(episode.path[-1].reward)
        if augment_permutations > 0:
            _augment_episode_in_buffer(
                env, episode.path, buffer, elem_feats_scaled, action_values,
                K=augment_permutations, gamma=gamma, terminal_key=episode.terminal_key,
            )
        else:
            add_episode_to_buffer(episode.path, buffer, elem_feats_scaled, action_values, gamma=gamma)

        # 2. Gradient steps.
        mean_loss = float("nan")
        optimizer_steps = 0
        fit_diagnostics = {}
        clip_updates = []
        value_normalizer = getattr(qnet, "value_normalizer", None)
        if len(buffer) >= batch_size:
            buf_list = list(buffer)
            losses = []
            raw_errors = []
            for _ in range(dqn_grad_steps_per_ep):
                _batch = random.sample(buf_list, batch_size)
                losses.append(_dqn_gradient_step(
                    qnet, _batch, target_net, scaler,
                    elem_feats_scaled, action_values, optimizer, loss_fn, device, gamma,
                    target_mode=dqn_target_mode,
                    target_cache=target_cache,
                    diagnostics=fit_diagnostics if value_normalizer is not None or gradient_clipper else None,
                    gradient_clipper=gradient_clipper,
                ))
                if gradient_clipper is not None:
                    clip_updates.append(dict(fit_diagnostics))
                if value_normalizer is not None:
                    raw_errors.append(fit_diagnostics["value_raw_mae"])
            _check_hpo_numerical(env, *losses, models=(qnet,))
            mean_loss = float(np.mean(losses))
            optimizer_steps = len(losses)
            if raw_errors:
                fit_diagnostics["value_raw_mae"] = float(np.mean(raw_errors))

        # 3. Hard target-net copy.
        if target_cache is not None and train_attempts % dqn_target_update_freq == 0:
            target_net.load_state_dict(qnet.state_dict())
            target_net.eval()
            target_cache.clear()

        exploration_pending.append(_exploration_observation(episode))
        if not evaluator.pending:
            _flush_exploration(exploration, exploration_pending, env)
        _sync_training_device(device)
        update_s = time.perf_counter() - update_started

        _row = {
            "phase": "dqn_train",
            "iteration": train_attempts,
            "episode": train_attempts,
            "budget_counted": counted,
            "training_budget_history_complete": budget_history_complete,
            "terminal_comp_key": str(episode.terminal_key),
            **budget.snapshot(),
            "return": episode_reward,
            # Identical to `return` for DQN (it already logs the undiscounted
            # terminal reward), but present so every phase carries the column that
            # is comparable across methods — see the PG rows, where they differ.
            "terminal_reward": episode_reward,
            "train_loss": mean_loss,
            "optimizer_steps": optimizer_steps,
            **_exploration_metrics(exploration),
            "exploration_target_used": episode.metadata["exploration_target_used"],
            "exploration_entropy": episode.metadata.get("exploration_entropy"),
            "exploration_entropy_by_stage": json.dumps(episode.metadata.get("exploration_entropy_by_stage", {}), sort_keys=True),
            "buffer_rows": len(buffer),
            **_episode_timing_row(episode, update_s),
        }
        if gradient_clipper is not None and clip_updates:
            for field in ("q_grad_norm", "q_grad_cap", "q_grad_factor", "q_optimizer_update_norm"):
                values = [item[field] for item in clip_updates if item[field] is not None]
                _row[field + "_mean"] = float(np.mean(values)) if values else None
                _row[field + "_max"] = max(values) if values else None
            _row["q_grad_clipped_count"] = sum(item["q_grad_clipped"] for item in clip_updates)
            _row["q_grad_clipped_fraction"] = _row["q_grad_clipped_count"] / len(clip_updates)
            _row["q_grad_provisional_startup_count"] = sum(
                item["q_grad_provisional_startup"] for item in clip_updates
            )
            _row["q_grad_threshold_source_last"] = clip_updates[-1]["q_grad_threshold_source"]
            _row["q_grad_history_last"] = clip_updates[-1]["q_grad_history"]
        if value_normalizer is not None:
            _row.update({key: value for key, value in fit_diagnostics.items() if not key.startswith("q_")})
            _row.update(value_normalizer.diagnostics("value_popart"))
            if target_cache is not None:
                _row.update(target_net.value_normalizer.diagnostics("target_popart"))
        if timer is not None:
            _row.update(timer.snapshot())
        metrics.append(_row)
        _record_episode_timing(timer, _row)
        pbar.update(int(counted))
        pbar.set_postfix(
            target=f"{exploration.target:.3f}",
            loss=f"{mean_loss:.4f}" if not math.isnan(mean_loss) else "warmup",
            ret=f"{episode_reward:.3f}",
        )

        # 5. Periodic checkpoint.
        budget_stop = not train_count_duplicates and (
            completed >= dqn_num_train_eps
            or budget.consecutive_duplicates >= train_max_duplicate_attempts
        )
        if ckpt_path and ckpt_freq > 0 and (train_attempts % ckpt_freq == 0 or budget_stop):
            # HPO resumes from the latest synchronized state; keeping every
            # replay-buffer snapshot makes disk usage grow with every attempt.
            # A direct atomic replacement also safely replaces an old symlink.
            _numbered = (None if getattr(env, "hpo_recorder", None) is not None
                         else ckpt_path.replace(".pt", f"-ep{train_attempts}.pt"))
            with checkpoint_stage("dqn_train", train_attempts):
                _save_checkpoint(ckpt_path, {
                    "type": "dqn",
                    "scaler_fingerprint": scaler_fingerprint(scaler),
                    "execution_contract": (checkpoint_cfg or {}).get("execution_contract"),
                    **predictor_checkpoint_fields(timer),
                    "environment_contract": environment_contract(env),
                    "episodes_completed": completed,
                    "training_episodes_completed": train_attempts,
                    "training_sample_session_id": training_samples.session_id if training_samples else None,
                    "training_budget_state": budget.state_dict(),
                    "hpo_state": env.hpo_recorder.checkpoint_state() if getattr(env, "hpo_recorder", None) is not None else None,
                    "training_budget_history_complete": budget_history_complete,
                    "qnet_state": qnet.state_dict(),
                    "target_net_state": target_net.state_dict() if target_net is not None else None,
                    "opt_state": optimizer.state_dict(),
                    "gradient_clipping_state": gradient_clipper.state_dict() if gradient_clipper else None,
                    "buffer": list(buffer),
                    "dqn_buffer_size": dqn_buffer_size,
                    "exploration_state": exploration.state_dict(),
                }, numbered_path=_numbered)
            tqdm.write(f"[INFO] DQN checkpoint saved at episode {train_attempts} "
                       f"(budget {completed}/{dqn_num_train_eps}) → {_numbered or ckpt_path}")

    pbar.close()
    return qnet, scaler, metrics


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Action selection (DQN)
# ---------------------------------------------------------------------------

def choose_action(
    *,
    model: torch.nn.Module,
    device: torch.device,
    s_material: np.ndarray,
    s_step: np.ndarray,
    allowed_actions: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    exploration_target: float = 0.7,
    sampling_diagnostics: Optional[dict] = None,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Sample legal Q actions at the requested normalized entropy."""
    n = len(allowed_actions)
    if not n:
        raise InfeasibleEpisode("Cannot choose an action from an empty legal set")

    a_elem = np.asarray(
        [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed_actions], dtype=float
    )
    a_comp = value_features(action_values, [int(np.argmax(a[1])) for a in allowed_actions])
    s_mat_b  = np.repeat(s_material.reshape(1, -1), n, axis=0)
    s_step_b = np.repeat(s_step.reshape(1, -1), n, axis=0)

    with torch.no_grad():
        q = model(
            torch.tensor(s_mat_b,  dtype=torch.float32, device=device),
            torch.tensor(s_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_elem,   dtype=torch.float32, device=device),
            torch.tensor(a_comp,   dtype=torch.float32, device=device),
        ).reshape(-1)

    probs = _sampler_probabilities(q.cpu().numpy().astype(float), exploration_target, model)
    _record_sampling_entropy(sampling_diagnostics, int(np.argmax(s_step)), probs)
    return allowed_actions[int(np.random.choice(n, p=probs))]


# ---------------------------------------------------------------------------
# Rollout functions
# ---------------------------------------------------------------------------

def _legal_actions(env):
    actions = env.allowed_actions()
    if not actions:
        raise InfeasibleEpisode(
            f"{type(env).__name__}: no legal continuation at step "
            f"{env.counter}/{env.max_steps}; check domain and active constraints"
        )
    return actions


def _env_terminal(env) -> bool:
    for name in ("is_terminal", "_is_terminal", "is_done"):
        marker = getattr(env, name, None)
        if marker is not None:
            return bool(marker() if callable(marker) else marker)
    return int(env.counter) >= int(env.max_steps)


def _episode_steps(env):
    """Allow STOP before the maximum horizon without inventing extra steps."""
    for index in range(env.max_steps):
        if _env_terminal(env):
            return
        yield index
    if not _env_terminal(env):
        raise RuntimeError("Environment did not terminate within its declared maximum steps")


def _rollout_random_episode(env: MultiGroupEnv) -> None:
    """Roll out one episode with uniformly random action selection."""
    env.initialize()
    env.exploration_diagnostics = {}
    for _ in _episode_steps(env):
        _legal_actions(env)
        a = env.sample_random_action()
        env.step(a)


def _rollout_policy_episode(
    *,
    env: MultiGroupEnv,
    qnet: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    exploration_target: float = 0.7,
) -> None:
    """Roll out one episode guided by a Q-network (used during generation)."""
    env.initialize()
    env.exploration_diagnostics = {}
    for _ in _episode_steps(env):
        allowed = _legal_actions(env)
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
        s_step = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            s_step[env.counter] = 1.0
        a = choose_action(
            model=qnet, device=device,
            s_material=s_mat, s_step=s_step,
            allowed_actions=allowed,
            elem_feats_scaled=elem_feats_scaled,
            action_values=action_values,
            exploration_target=exploration_target,
            sampling_diagnostics=env.exploration_diagnostics,
        )
        env.step(a)


def _fit_scaler_from_warmup(env: MultiGroupEnv, n_warmup_eps: int) -> StandardScaler:
    """Fit a StandardScaler on material features from random warmup episodes.

    The reward function is temporarily replaced with a no-op so warmup does not
    invoke the (potentially expensive) PropertyPredictor.  Used by PG methods
    only (DQN warmup is handled inside ``train_dqn_online``).
    """
    all_s_mat = []
    original_reward_fn = env.reward_fn
    env.reward_fn = lambda _f: 0.0
    pbar = tqdm(total=n_warmup_eps, desc="Warmup (scaler fit)")
    try:
        for _ in range(n_warmup_eps):
            recorder = getattr(env, "hpo_recorder", None)
            def preparation_rollout():
                if recorder is not None:
                    recorder.start()
                try:
                    _rollout_random_episode(env)
                except Exception:
                    if recorder is not None:
                        recorder.failure()
                    raise
                else:
                    if recorder is not None:
                        recorder.failure(preparation=True)
            _rollout_with_retry(preparation_rollout)
            for step in env.path:
                all_s_mat.append(np.asarray(step.state_material_features, dtype=float))
            pbar.update(1)
    finally:
        env.reward_fn = original_reward_fn
        pbar.close()
    scaler = StandardScaler()
    scaler.fit(np.asarray(all_s_mat, dtype=float))
    bind_scaler(scaler, env)
    print(f"[INFO] Scaler fitted on {len(all_s_mat)} warmup states.")
    return scaler


def _rollout_pg_episode(
    *,
    env: MultiGroupEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: torch.device,
) -> None:
    """Roll out the actor distribution exactly as reconstructed by its loss."""
    env.initialize()
    env.exploration_diagnostics = {}
    for _ in _episode_steps(env):
        allowed = _legal_actions(env)
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
        s_step = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            s_step[env.counter] = 1.0
        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = value_features(action_values, [int(np.argmax(a[1])) for a in allowed])
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)
        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch,  dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)
        if not torch.isfinite(logits).all().item():
            from .utils.candidate_budget import NumericalTrainingError
            raise NumericalTrainingError("Nonfinite actor logits during sampling")
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        _record_sampling_entropy(env.exploration_diagnostics, env.counter, probs)
        env.step(allowed[int(np.random.choice(n, p=probs))])


# ---------------------------------------------------------------------------
# Policy gradient training (REINFORCE / A2C)
# ---------------------------------------------------------------------------

def _episode_pg_terms(
    *,
    path,
    returns: List[float],
    policy: torch.nn.Module,
    value_net: Optional[torch.nn.Module],
    scaler: "StandardScaler",
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: "torch.device",
) -> tuple:
    """Build per-step policy-gradient *components* for one episode.

    Returns ``(logp_taken, advantage_raw, normalized_entropy, critic_loss)``
    rather than a finished actor loss. The advantage is deliberately left
    un-scaled and separate from ``logp`` so :func:`train_pg` can standardise it
    across the whole batch before forming ``-logp * adv`` — folding the two
    together here would make batch-level normalisation impossible.

    Log-probabilities and normalized entropies carry autograd state.
    Advantages are detached; forced decisions have no entropy term.
    """
    logp_taken: List[torch.Tensor] = []
    advantages_raw: List[float] = []
    entropy_terms: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []

    value_normalizer = getattr(value_net, "value_normalizer", None)
    for step, G_t in zip(path, returns):
        allowed = step.allowed_actions
        if not allowed:
            continue

        s_mat_raw = np.asarray(step.state_material_features, dtype=float)
        s_mat = scaler.transform(s_mat_raw.reshape(1, -1))[0]
        s_step = np.asarray(step.state_step_onehot, dtype=float)

        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = value_features(action_values, [int(np.argmax(a[1])) for a in allowed])
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

        s_mat_t  = torch.tensor(s_mat_batch,  dtype=torch.float32, device=device)
        s_step_t = torch.tensor(s_step_batch, dtype=torch.float32, device=device)
        a_elem_t = torch.tensor(a_elem_batch, dtype=torch.float32, device=device)
        a_comp_t = torch.tensor(a_comp_batch, dtype=torch.float32, device=device)

        logits    = policy(s_mat_t, s_step_t, a_elem_t, a_comp_t).reshape(-1)
        log_probs = torch.log_softmax(logits, dim=0)

        taken_elem = np.asarray(step.action_elem_onehot)
        taken_comp = np.asarray(step.action_comp_onehot)
        taken_idx = None
        for i, a in enumerate(allowed):
            if (np.array_equal(np.asarray(a[0]), taken_elem)
                    and np.array_equal(np.asarray(a[1]), taken_comp)):
                taken_idx = i
                break
        if taken_idx is None:
            continue

        if value_net is not None:
            s_mat_single  = torch.tensor(s_mat.reshape(1, -1),  dtype=torch.float32, device=device)
            s_step_single = torch.tensor(s_step.reshape(1, -1), dtype=torch.float32, device=device)
            inputs = (s_mat_single, s_step_single)
            if value_normalizer is None:
                prediction = value_net(*inputs).reshape(-1)[0]
                target = torch.tensor(G_t, dtype=torch.float32, device=device)
                value_raw = prediction.detach()
            else:
                prediction = value_net.forward_normalized(*inputs).reshape(-1)[0]
                raw_target = torch.tensor(G_t, dtype=torch.float64, device=device)
                target = value_normalizer.normalize(raw_target, dtype=prediction.dtype)
                value_raw = value_normalizer.decode(prediction.detach())
            critic_losses.append((prediction - target) ** 2)
            advantage = G_t - float(value_raw.item())
        else:
            # REINFORCE: no baseline. Batch standardisation in train_pg supplies
            # the variance reduction a critic would otherwise give.
            advantage = G_t

        logp_taken.append(log_probs[taken_idx])
        advantages_raw.append(float(advantage))
        if n > 1:
            # Compute this small entropy reduction in float64: finite float32
            # logit differences and H/log(n) gradients can otherwise overflow.
            entropy_log_probs = torch.log_softmax(logits.double(), dim=0)
            entropy_probs = torch.softmax(logits.double(), dim=0)
            safe_logs = entropy_log_probs.clamp_min(torch.finfo(torch.float32).min)
            entropy_terms.append(-(entropy_probs * safe_logs).sum() / math.log(n))

    return logp_taken, advantages_raw, entropy_terms, critic_losses


def train_pg(
    *,
    policy: torch.nn.Module,
    value_net: Optional[torch.nn.Module],
    env: MultiGroupEnv,
    scaler: "StandardScaler",
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: "torch.device",
    num_iters: int,
    batch_eps: int,
    gamma: float = 0.99,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    exploration_target: float = 0.7,
    exploration_adaptive: bool = True,
    rl_method: str = "a2c",
    max_train_attempts: Optional[int] = None,
    train_count_duplicates: bool = False,
    train_max_duplicate_attempts: int = 1000,
    checkpoint_cfg: Optional[dict] = None,
    timer: Optional[object] = None,
    training_samples=None,
    gradient_clipping: str = "none",
    gradient_calibration_episodes=None,
) -> List[dict]:
    """On-policy actor/critic updates with normalized entropy feedback.

    Duplicate trajectories retain their scientific returns and participate in
    learning without visit penalties. Fixed or adaptive entropy targets share
    one controller; outcome feedback changes the target only between batches.
    The actor samples its own softmax, and its loss uses the identical masked
    distribution. Forced decisions contribute no entropy observation or bonus.
    """
    budget_state = checkpoint_cfg.get("training_budget_state") if checkpoint_cfg else None
    resuming = bool(checkpoint_cfg and (
        checkpoint_cfg.get("is_resume")
        or checkpoint_cfg.get("start_episode", 0)
        or checkpoint_cfg.get("opt_actor_state") is not None
    ))
    if gradient_clipping not in ("none", "clipping"):
        raise ValueError("gradient_clipping must be none or clipping")
    if gradient_clipping == "clipping" and not resuming:
        from .gradient_calibration import _pilot
        _pilot(gradient_calibration_episodes)
    if not train_count_duplicates and resuming and budget_state is None:
        raise ValueError(
            "Cannot resume with train_count_duplicates=false: this checkpoint "
            "has no training_budget_state and the previously seen compositions "
            "cannot be recovered reliably. Start a new run or retain "
            "train_count_duplicates=true."
        )
    budget = TrainingBudget(
        count_duplicates=train_count_duplicates,
        max_duplicate_attempts=train_max_duplicate_attempts,
        state=budget_state,
    )
    budget_target = budget.counted + max(0, int(num_iters)) * int(batch_eps)
    if checkpoint_cfg and checkpoint_cfg.get("total_budget") is not None:
        budget_target = int(checkpoint_cfg["total_budget"])
    if int(batch_eps) < 1:
        raise ValueError("batch_eps must be a positive integer")
    budget_history_complete = bool(checkpoint_cfg.get(
        "training_budget_history_complete", budget_state is not None,
    )) if resuming else True
    if resuming and budget_state is None:
        print("[WARN] Legacy checkpoint has no training budget history: new diagnostic "
              "counters start at this resume; the existing episode schedule is preserved.", flush=True)

    exploration = _restore_exploration(
        exploration_target, exploration_adaptive,
        (checkpoint_cfg or {}).get("exploration_state"), resuming=resuming,
    )
    if not resuming:
        exploration.seed_seen(repr(ep.terminal_key) for ep in gradient_calibration_episodes or ())
        exploration.seed_seen(getattr(env, "exploration_seen_keys", ()))

    if training_samples is not None:
        training_samples.start(
            budget.counted, history_complete=budget_history_complete,
            session_id=checkpoint_cfg.get("training_sample_session_id") if checkpoint_cfg else None,
        )
    metrics: List[dict] = []
    budget_start = budget.counted

    exploration_pending = []
    evaluator = EpisodeBatchEvaluator(
        env, max_retries=_MAX_CANDIDATE_RETRIES, timer=timer,
        capture_predictions=True, failure_callback=_exploration_failure(exploration_pending),
    )
    policy.train()
    opt_actor = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    if value_net is not None:
        value_net.train()
        opt_critic = torch.optim.Adam(value_net.parameters(), lr=lr_critic)
    else:
        opt_critic = None

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    if checkpoint_cfg:
        ckpt_path = checkpoint_cfg.get("path")
        ckpt_freq = int(checkpoint_cfg.get("freq", 0))
        if checkpoint_cfg.get("opt_actor_state") is not None:
            opt_actor.load_state_dict(checkpoint_cfg["opt_actor_state"])
        if opt_critic is not None and checkpoint_cfg.get("opt_critic_state") is not None:
            opt_critic.load_state_dict(checkpoint_cfg["opt_critic_state"])

    gradient_clippers = {}
    recorder = getattr(env, "hpo_recorder", None)
    if recorder is not None:
        recorder.partial_checkpoint_factory = lambda: {
            "type": "pg", "policy_state": policy.state_dict(),
            "value_net_state": value_net.state_dict() if value_net is not None else None,
            "opt_actor_state": opt_actor.state_dict(),
            "gradient_clipping_state": {name: ctl.state_dict() for name, ctl in gradient_clippers.items()},
            "opt_critic_state": opt_critic.state_dict() if opt_critic is not None else None,
            "scaler": scaler, "training_budget_state": budget.state_dict(),
            "exploration_state": exploration.state_dict(),
        }

    coef_eff = exploration.coefficient
    _flush_exploration(exploration, exploration_pending, env)

    if gradient_clipping == "clipping":
        from .gradient_clipping import AdaptiveGradientClipper
        if resuming:
            saved_clippers = (checkpoint_cfg or {}).get("gradient_clipping_state")
            expected = {"actor", "critic"} if value_net is not None else {"actor"}
            if not isinstance(saved_clippers, dict) or set(saved_clippers) != expected:
                raise ValueError("Exact clipping resume requires actor/critic gradient_clipping_state")
            gradient_clippers = {name: AdaptiveGradientClipper.from_state_dict(state, network=name)
                                 for name, state in saved_clippers.items()}
        else:
            from .gradient_calibration import calibrate_pg_gradients, write_gradient_report
            gradient_clippers, report = calibrate_pg_gradients(
                policy=policy, value_net=value_net, episodes=gradient_calibration_episodes,
                scaler=scaler, elem_feats_scaled=elem_feats_scaled, action_values=action_values,
                device=device, gamma=gamma, exploration_target=exploration_target,
                exploration_adaptive=exploration_adaptive,
            )
            write_gradient_report(checkpoint_cfg, report)
    elif gradient_clipping != "none":
        raise ValueError("gradient_clipping must be none or clipping")

    accepted = int(checkpoint_cfg.get("start_episode", 0)) if checkpoint_cfg else 0
    attempted = 0
    start_iteration = int(checkpoint_cfg.get("start_iteration", 0)) if checkpoint_cfg else 0
    update_idx = start_iteration
    iteration_in_run = 0

    total_updates = max(0, int(num_iters))
    if checkpoint_cfg and checkpoint_cfg.get("total_iterations") is not None:
        total_updates = max(0, int(checkpoint_cfg["total_iterations"]) - start_iteration)
    if checkpoint_cfg and checkpoint_cfg.get("rng_state") is not None:
        restore_rng_state(checkpoint_cfg["rng_state"])
    budget_label = "episodes" if train_count_duplicates else "new compositions"
    print(
        f"[INFO] PG: budget={total_updates * batch_eps} {budget_label}, "
        f"up to {batch_eps} sampled episodes/update; "
        f"train_count_duplicates={train_count_duplicates}."
    )

    outer_pbar = tqdm(
        total=total_updates if train_count_duplicates else budget_target - budget_start,
        desc=f"{rl_method.upper()} " + ("iters" if train_count_duplicates else "new compositions"),
    )
    while (iteration_in_run < total_updates and budget.counted < budget_target if train_count_duplicates
           else budget.counted < budget_target):
        # The previous batch has already learned from its duplicate episodes.
        # Only now may the duplicate guard stop the next batch.
        try:
            budget.check_can_sample()
        except RuntimeError:
            outer_pbar.close()
            raise
        if max_train_attempts is not None and attempted >= max_train_attempts:
            if not train_count_duplicates:
                outer_pbar.close()
                raise RuntimeError(
                    f"PG max_train_attempts={max_train_attempts} reached with "
                    f"only {budget.counted - budget_start}/{budget_target - budget_start} "
                    "new compositions; the training budget is incomplete."
                )
            break
        it = start_iteration + iteration_in_run
        iteration_in_run += 1
        batch_paths: List[list] = []
        batch_returns: List[List[float]] = []
        batch_terminal_keys: List[tuple] = []
        batch_target = exploration.target
        # One record per *sampled episode*, so the PG training-reward distribution is
        # comparable with DQN's (which logs every episode). The pg_train row below
        # only carries the batch mean, which hides the low-temperature tail entirely.
        batch_ep_rows: List[dict] = []

        collected = 0
        while collected < int(batch_eps):
            if budget.counted >= budget_target or (not train_count_duplicates and
                    budget.consecutive_duplicates >= train_max_duplicate_attempts):
                break
            if max_train_attempts is not None and attempted >= max_train_attempts:
                break
            wave_limit = training_wave_limit(
                budget, min(int(batch_eps) - collected, budget_target - budget.counted),
                attempts_remaining=(max_train_attempts - attempted
                                    if max_train_attempts is not None else None),
            )
            attempted += 1
            try:
                episode = evaluator.next(
                    lambda: _rollout_pg_episode(
                        env=env, policy=policy, scaler=scaler,
                        elem_feats_scaled=elem_feats_scaled, action_values=action_values,
                        device=device,
                    ),
                    limit=wave_limit,
                    retry_scalar=_rollout_with_retry,
                    phase="pg_episode",
                )
            except BaseException:
                # Collection can stop at the failure guard (or interruption)
                # before this on-policy batch reaches its learning update.
                # Preserve already-scored episodes with zero update cost.
                for row in batch_ep_rows:
                    _record_episode_timing(timer, row)
                raise
            path = episode.path
            if not path:
                continue

            G = 0.0
            returns: List[float] = []
            for step in reversed(path):
                G = float(step.reward) + gamma * G
                returns.append(G)
            returns.reverse()

            terminal_key = episode.terminal_key
            budget_counted = budget.record(terminal_key)
            if not train_count_duplicates and budget_counted:
                outer_pbar.update(1)
            exploration_pending.append(_exploration_observation(episode))

            batch_paths.append(path)
            batch_returns.append(returns)
            batch_terminal_keys.append(terminal_key)
            collected += 1
            accepted += 1
            batch_ep_rows.append({
                "phase": "pg_episode",
                "iteration": it + 1,
                "episode": accepted,
                "return": returns[0],
                "return_raw": returns[0],
                # `return` is the *discounted* return G_0 = gamma^(n-1) * r_T for a
                # terminal-only reward — the training signal, not a temperature.
                # DQN's `return` column is the undiscounted terminal reward, so the
                # two are NOT comparable without this column. Log it explicitly
                # rather than making every reader re-derive gamma from run_config.
                "terminal_reward": float(path[-1].reward),
                "exploration_target_used": batch_target,
                "exploration_entropy": episode.metadata.get("exploration_entropy"),
                "exploration_entropy_by_stage": json.dumps(episode.metadata.get("exploration_entropy_by_stage", {}), sort_keys=True),
                "terminal_comp_key": str(terminal_key),
                "budget_counted": budget_counted,
                **budget.snapshot(),
                "training_budget_history_complete": budget_history_complete,
                **_episode_timing_row(episode),
            })
            if training_samples is not None:
                training_samples.record(episode, batch_ep_rows[-1])

        if train_count_duplicates:
            outer_pbar.update(1)
        if not batch_paths:
            continue

        _sync_training_device(device)
        update_started = time.perf_counter()
        value_normalizer = getattr(value_net, "value_normalizer", None)
        if value_normalizer is not None:
            targets = [target for path, returns in zip(batch_paths, batch_returns)
                       for step, target in zip(path, returns) if step.allowed_actions]
            if targets:
                # Reparameterize once, before any critic autograd graph exists.
                value_net.update_value_normalization(
                    torch.tensor(targets, dtype=torch.float64, device=device)
                )
        all_logp: List[torch.Tensor] = []
        all_adv_raw: List[float] = []
        all_entropy: List[torch.Tensor] = []
        all_critic: List[torch.Tensor] = []
        for path, returns in zip(batch_paths, batch_returns):
            lp, adv, e_terms, c_terms = _episode_pg_terms(
                path=path,
                returns=returns,
                policy=policy,
                value_net=value_net,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                action_values=action_values,
                device=device,
            )
            all_logp.extend(lp)
            all_adv_raw.extend(adv)
            all_entropy.extend(e_terms)
            all_critic.extend(c_terms)

        if not all_logp:
            for row in batch_ep_rows:
                _record_episode_timing(timer, row)
            continue

        # --- advantage standardisation (the whole point; see docstring) -------
        adv_raw = torch.tensor(all_adv_raw, dtype=torch.float32, device=device)
        adv_mean = float(adv_raw.mean().item())
        adv_std  = float(adv_raw.std(unbiased=False).item())
        if adv_std > 1e-8:
            adv = (adv_raw - adv_mean) / adv_std
        else:
            # Every episode in the batch scored identically — there is no signal to
            # extract. Dividing by an epsilon here would amplify float noise into a
            # full-magnitude gradient, so emit a zero update instead.
            adv = torch.zeros_like(adv_raw)

        logp_t = torch.stack(all_logp)
        actor_loss = -(logp_t * adv).mean()
        entropy_bonus = (torch.stack(all_entropy).mean() if all_entropy
                         else actor_loss * 0.0)
        entropy_norm = float(entropy_bonus.detach()) if all_entropy else None
        coef_eff = exploration.coefficient
        total_actor_loss = actor_loss - coef_eff * entropy_bonus

        clip_diagnostics = {}
        critic_loss_val: float | str = ""
        if gradient_clippers:
            # Check both independent gradient paths before either Adam step.
            critic_loss = torch.stack(all_critic).mean() if opt_critic is not None and all_critic else None
            _require_finite_loss(total_actor_loss, *([critic_loss] if critic_loss is not None else []))
            opt_actor.zero_grad(set_to_none=True)
            if opt_critic is not None:
                opt_critic.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            if critic_loss is not None:
                critic_loss.backward()
            actor_info = _prepare_gradient_clipping(gradient_clippers["actor"], policy)
            actor_before = [p.detach().clone() for p in policy.parameters()]
            critic_info = (_prepare_gradient_clipping(gradient_clippers["critic"], value_net)
                           if critic_loss is not None else None)
            critic_before = ([p.detach().clone() for p in value_net.parameters()]
                             if critic_info is not None else None)
            opt_actor.step()
            if critic_loss is not None:
                opt_critic.step()
                critic_loss_val = float(critic_loss.item())
            _require_finite_optimizer(policy, opt_actor)
            if critic_loss is not None:
                _require_finite_optimizer(value_net, opt_critic)
            gradient_clippers["actor"].commit(actor_info)
            clip_diagnostics.update(_clip_diagnostics("actor", actor_info, policy, actor_before))
            if critic_info is not None:
                gradient_clippers["critic"].commit(critic_info)
                clip_diagnostics.update(_clip_diagnostics("critic", critic_info, value_net, critic_before))
        else:
            _check_hpo_numerical(env, total_actor_loss.item())
            opt_actor.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            opt_actor.step()

            critic_loss_val: float | str = ""
            if opt_critic is not None and all_critic:
                critic_loss = torch.stack(all_critic).mean()
                _check_hpo_numerical(env, critic_loss.item())
                opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                opt_critic.step()
                critic_loss_val = float(critic_loss.item())

        _check_hpo_numerical(env, models=(policy, value_net))
        _sync_training_device(device)
        update_s = time.perf_counter() - update_started
        for row in batch_ep_rows:
            row["rl_update_s"] = update_s / len(batch_ep_rows)
            row["episode_wall_share_s"] += row["rl_update_s"]
            _record_episode_timing(timer, row)

        # Coefficient and target changes affect only the next on-policy batch.
        exploration.update_entropy(entropy_norm)
        _flush_exploration(exploration, exploration_pending, env)
        update_idx += 1
        ep_actor_loss = float(actor_loss.item())
        ep_entropy    = float(entropy_bonus.item())
        mean_return_raw = float(np.mean([rs[0] for rs in batch_returns])) if batch_returns else 0.0
        # Undiscounted terminal reward, in the property's own units — the only
        # column comparable with DQN's `return` (see the pg_episode row above).
        mean_terminal_reward = (float(np.mean([float(p[-1].reward) for p in batch_paths]))
                                if batch_paths else 0.0)

        _row = {
            "phase": "pg_train",
            "iteration": it + 1,
            "update": update_idx,
            "actor_optimizer_steps": 1,
            "critic_optimizer_steps": int(opt_critic is not None and bool(all_critic)),
            "episode": accepted,
            "batch_eps": int(batch_eps),
            "batch_eps_actual": len(batch_paths),
            "return": mean_return_raw,
            "return_raw": mean_return_raw,
            "terminal_reward": mean_terminal_reward,
            "unique_comps_seen": budget.unique,
            "exploration_target_used": batch_target,
            **_exploration_metrics(exploration),
            "terminal_comp_key": str(batch_terminal_keys[-1]) if batch_terminal_keys else "",
            "actor_loss": ep_actor_loss,
            "entropy": ep_entropy,
            "entropy_norm": entropy_norm,
            "exploration_coefficient_used": coef_eff,
            "exploration_coefficient": exploration.coefficient,
            "adv_mean_raw": adv_mean,
            "adv_std_raw": adv_std,
            "critic_loss": critic_loss_val,
            "rl_update_s": update_s,
            **budget.snapshot(),
            "training_budget_history_complete": budget_history_complete,
        }
        _row.update(clip_diagnostics)
        if value_normalizer is not None:
            _row.update(value_normalizer.diagnostics("value_popart"))
            _row["value_raw_mae"] = float(np.mean(np.abs(all_adv_raw)))
        if timer is not None:
            _row.update(timer.snapshot())
        metrics.append(_row)
        metrics.extend(batch_ep_rows)

        outer_pbar.set_postfix(
            ret=f"{mean_return_raw:.3f}",
            actor=f"{ep_actor_loss:.3f}",
            ent=f"{ep_entropy:.3f}",
            upd=update_idx,
        )

        stopping_incomplete = not train_count_duplicates and budget.counted < budget_target and (
            budget.consecutive_duplicates >= train_max_duplicate_attempts
            or (max_train_attempts is not None and attempted >= max_train_attempts)
        )
        budget_finished = not train_count_duplicates and budget.counted >= budget_target
        if ckpt_path and (
            (ckpt_freq > 0 and (it + 1) % ckpt_freq == 0)
            or stopping_incomplete or budget_finished
        ):
            # HPO needs the latest synchronized state, not iteration history.
            _numbered = (None if getattr(env, "hpo_recorder", None) is not None
                         else ckpt_path.replace(".pt", f"-iter{it + 1}.pt"))
            with checkpoint_stage("pg_train", it + 1):
                _save_checkpoint(ckpt_path, {
                    "type": "pg",
                    "scaler_fingerprint": scaler_fingerprint(scaler),
                    "execution_contract": (checkpoint_cfg or {}).get("execution_contract"),
                    **predictor_checkpoint_fields(timer),
                    "budget_target": budget_target,
                    "environment_contract": environment_contract(env),
                    "rl_method": rl_method,
                    "episodes_completed": accepted,
                    "iteration": it + 1,
                    "policy_state": policy.state_dict(),
                    "value_net_state": value_net.state_dict() if value_net is not None else None,
                    "opt_actor_state": opt_actor.state_dict(),
                    "gradient_clipping_state": {name: ctl.state_dict() for name, ctl in gradient_clippers.items()},
                    "opt_critic_state": opt_critic.state_dict() if opt_critic is not None else None,
                    "exploration_state": exploration.state_dict(),
                    "training_sample_session_id": training_samples.session_id if training_samples else None,
                    "training_budget_state": budget.state_dict(),
                    "hpo_state": env.hpo_recorder.checkpoint_state() if getattr(env, "hpo_recorder", None) is not None else None,
                    **budget.snapshot(),
                    "training_budget_history_complete": budget_history_complete,
                }, numbered_path=_numbered)
            tqdm.write(f"[INFO] PG checkpoint saved at iter {it + 1} → {_numbered or ckpt_path}")

    outer_pbar.close()
    print(
        f"[INFO] PG training: {update_idx - start_iteration} updates over "
        f"{attempted} sampled episodes this invocation; "
        f"budget used {budget.counted - budget_start}/{budget_target - budget_start} "
        f"{budget_label} ({accepted} cumulative episodes)."
    )
    return metrics


# ---------------------------------------------------------------------------
# Candidate generation with shared entropy-target sampling
# ---------------------------------------------------------------------------

def _pg_single_episode_generate(
    *,
    env: MultiGroupEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    device: torch.device,
    exploration_target: float = 0.7,
) -> None:
    """Generate with a frozen actor and entropy-controlled legal actions."""
    env.initialize()
    env.exploration_diagnostics = {}
    for _ in _episode_steps(env):
        allowed = _legal_actions(env)
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
        s_step = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            s_step[env.counter] = 1.0

        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = value_features(action_values, [int(np.argmax(a[1])) for a in allowed])
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch,  dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)

        probs = _sampler_probabilities(logits.cpu().numpy().astype(float), exploration_target)
        _record_sampling_entropy(env.exploration_diagnostics, env.counter, probs)
        idx = int(np.random.choice(n, p=probs))
        env.step(allowed[idx])


def generate_candidates(
    *,
    env: MultiGroupEnv,
    predictor: PropertyPredictor,
    scaler: StandardScaler,
    device: torch.device,
    elem_feats_scaled: np.ndarray,
    action_values: List[str],
    policy: Optional[torch.nn.Module] = None,
    qnet: Optional[torch.nn.Module] = None,
    num_gen_eps: int = 200,
    gen_exploration_target: float = 0.7,
    exploration_adaptive: bool = True,
    exploration_seen_keys=None,
    k: float = 1.0,
    max_attempts: Optional[int] = None,
    charge_filter: bool = False,
    charge_use_pauling: bool = False,
) -> List[dict]:
    """Generate unique candidates with a resumable entropy-target sampler.

    Training-seen identities seed global novelty without removing them from the
    generation export. Scientific rewards and reporting remain unchanged.
    """
    if policy is None and qnet is None:
        raise ValueError("Provide either policy (PG) or qnet (DQN).")
    use_pg = policy is not None

    if use_pg:
        policy.eval()
    else:
        qnet.eval()

    exploration = ExplorationController(initial_target=gen_exploration_target, adaptive=exploration_adaptive)
    exploration.seed_seen(exploration_seen_keys if exploration_seen_keys is not None
                          else getattr(env, "exploration_seen_keys", ()))
    exploration_pending = []
    evaluator = EpisodeBatchEvaluator(
        env, max_retries=_MAX_CANDIDATE_RETRIES, timer=predictor,
        capture_predictions=True, failure_callback=_exploration_failure(exploration_pending),
    )

    def rollout_generation():
        if use_pg:
            _pg_single_episode_generate(
                env=env, policy=policy, scaler=scaler,
                elem_feats_scaled=elem_feats_scaled, action_values=action_values,
                device=device, exploration_target=exploration.target,
            )
        else:
            _rollout_policy_episode(
                env=env, qnet=qnet, scaler=scaler, device=device,
                elem_feats_scaled=elem_feats_scaled, action_values=action_values,
                exploration_target=exploration.target,
            )

    recorder = getattr(env, "hpo_recorder", None)
    if recorder is not None:
        # HPO consumes the immutable evaluation snapshot before scientific
        # postfilters; never query a stochastic/expensive predictor again.
        if num_gen_eps != recorder.budget.target_unique:
            raise ValueError("HPO generation requires num_gen_eps=target_unique")
        from .utils.generation_resume import predictor_snapshot, restore_generation
        contract = getattr(env, "hpo_generation_contract", None)
        resumed = getattr(env, "hpo_generation_resume", None)
        if resumed is not None:
            restore_generation(predictor, resumed, contract)
            if "exploration_state" not in resumed:
                raise ValueError("Exact generation resume requires exploration_state")
            exploration.load_state_dict(resumed["exploration_state"])
        checkpoint_path = str(recorder.path / "generation_checkpoint.pt")

        def checkpoint_generation():
            if evaluator.pending or recorder.budget.pending:
                return
            state = predictor_snapshot(predictor)
            if state is None or contract is None:
                return  # Direct library callers may not expose a resumable predictor.
            _save_checkpoint(checkpoint_path, {
                "type": "hpo_generation", "hpo_state": recorder.checkpoint_state(),
                "exploration_state": exploration.state_dict(),
                "generation_contract": contract, "predictor_ledger": state["ledger"],
                "predictor_caches": state["caches"], "predictor_extra": state["extra"],
            })

        checkpoint_generation()
        while (recorder.budget.unique_evaluated < recorder.budget.target_unique
               or evaluator.pending):
            try:
                episode = evaluator.next(
                    rollout_generation,
                    limit=max(1, recorder.budget.target_unique - recorder.budget.unique_evaluated),
                    phase="generate", metadata={"exploration_target_used": exploration.target},
                )
            except (CandidateEvaluationFailed, InfeasibleEpisode):
                if not evaluator.pending:
                    _flush_exploration(exploration, exploration_pending, env)
                checkpoint_generation()
                continue
            exploration_pending.append(_exploration_observation(episode))
            if not evaluator.pending:
                _flush_exploration(exploration, exploration_pending, env)
            _record_episode_timing(predictor, {
                **_episode_timing_row(episode), "phase": "generate",
                "episode": recorder.budget.attempts_started,
                "exploration_target_used": episode.metadata.get("exploration_target_used"),
                **_exploration_metrics(exploration),
            })
            checkpoint_generation()
        # Reconstruct the export from authoritative unique snapshots. This
        # includes prior invocation rows without resampling or charging twice.
        return [{"formula": row["formula"], **row["prediction_record"],
                 **{key: value for key, value in row.items()
                    if key.startswith("exploration_") or key in
                    {"constraint_passed", "post_constraint_passed"}}}
                for row in (json.loads(line) for line in recorder.records_path.read_text().splitlines())]

    use_predict_raw = hasattr(predictor, "predict_raw")
    use_check_phase = hasattr(predictor, "check_phase")
    use_per_obj_stats = hasattr(predictor, "per_objective_stats")

    seen_comp_keys: set = set()
    rows: List[dict] = []

    n_target = num_gen_eps
    accepted = 0
    attempted = 0
    dup_rejected = 0
    charge_rejected = 0
    seen_nonneutral: set = set()  # unique non-neutral keys, for diagnosis
    max_att = max_attempts or (n_target * 20)

    pbar = tqdm(total=n_target, desc="Generate")
    while accepted < n_target and attempted < max_att:
        attempted += 1

        try:
            episode = evaluator.next(
                rollout_generation,
                limit=min(n_target - accepted, max_att - attempted + 1),
                phase="generate", metadata={"exploration_target_used": exploration.target},
            )
        except (CandidateEvaluationFailed, InfeasibleEpisode) as exc:
            # Unlike training, generation already has a generous, separate
            # attempts budget (max_att) distinct from `n_target` -- treat
            # this exactly like a rejected duplicate/infeasible candidate
            # (already counted via `attempted += 1` above) rather than
            # retrying without counting.
            print(f"[WARN] generation candidate failed evaluation, "
                  f"skipping: {exc}")
            if not evaluator.pending:
                _flush_exploration(exploration, exploration_pending, env)
            continue

        _record_episode_timing(predictor, {
            **_episode_timing_row(episode),
            "phase": "generate", "episode": attempted,
        })
        comp     = episode.composition
        comp_key = episode.terminal_key

        if comp_key in seen_comp_keys:
            dup_rejected += 1
            exploration_pending.append(_exploration_observation(episode))
            if not evaluator.pending:
                _flush_exploration(exploration, exploration_pending, env)
            continue

        # Charge-neutrality gate on the WHOLE built formula (covers
        # builder-derived anions, e.g. SSE). Gated entirely by config: when
        # charge_filter is False this block is skipped (no smact import).
        # Checked BEFORE adding to seen_comp_keys and BEFORE the (cached)
        # predictor call so that (a) repeated non-neutral picks are counted
        # as charge rejections — not duplicates — giving a clean diagnosis of
        # why yield is low, and (b) we don't pay a predictor call on a
        # candidate we're about to drop.
        if charge_filter:
            from .constraints.charge import charge_neutral

            # Prefer the builder's full composition label (host/derived
            # anions the env never sees); fall back to the env pick-only
            # formula. Same resolution as the accepted-row path below.
            _formula = episode.formula
            _cf = getattr(predictor, "composition_formula", None)
            if callable(_cf):
                try:
                    _full = _cf(comp)
                    if _full:
                        _formula = _full
                except Exception:
                    pass
            try:
                neutral = charge_neutral(_formula, use_pauling=charge_use_pauling)
            except Exception:
                neutral = True  # lenient: never crash generation on a bad parse
            if not neutral:
                seen_nonneutral.add(comp_key)
                charge_rejected += 1
                observation = _exploration_observation(episode)
                observation["feasible"] = False
                exploration_pending.append(observation)
                if not evaluator.pending:
                    _flush_exploration(exploration, exploration_pending, env)
                continue

        # Reward from the episode (env.reward_fn already called in env.step).
        reward = float(episode.path[-1].reward) if episode.path else 0.0

        # Raw predictor values for CSV columns. OOHCatalystPredictor caches
        # internally so this is usually a cache hit when env.reward_fn
        # already ran -- but for a non-caching predictor, or a builder
        # whose cache doesn't cover this call shape, it is a genuinely
        # separate call and can independently fail (e.g. a re-run MD
        # realization). Skip the candidate rather than crash generation;
        # don't mark comp_key as seen, so a later attempt on the same
        # composition can still succeed and be reported.
        try:
            if use_predict_raw:
                raw_mean, std = predictor.predict_raw(comp)
            elif episode.prediction is not None:
                raw_mean, std = episode.prediction
            else:
                raw_mean, std = predictor.predict(comp)
            per_stats = predictor.per_objective_stats(comp) if use_per_obj_stats else {}
            details_fn = getattr(predictor, "reward_details", None)
            details = details_fn(comp) if callable(details_fn) else {}
            metadata_fn = getattr(predictor, "candidate_metadata", None)
            metadata = metadata_fn(comp) if callable(metadata_fn) else {}
            metadata_fields = metadata_record(metadata)
            candidate_json = json.dumps(comp, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except EvaluationProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - any predictor/builder failure
            print(f"[WARN] generation candidate failed fetching raw stats, "
                  f"skipping: {type(exc).__name__}: {exc}")
            exploration_pending.append({"key": repr(comp_key), "quality": None,
                                        "feasible": None, "successful": False})
            if not evaluator.pending:
                _flush_exploration(exploration, exploration_pending, env)
            continue

        seen_comp_keys.add(comp_key)

        # Prefer the builder's full composition label (includes host/derived
        # elements the env never sees — e.g. SSE's surviving S, derived Br,
        # charge-balanced Li); fall back to the env's pick-only formula.
        formula = episode.formula
        comp_formula = getattr(predictor, "composition_formula", None)
        if callable(comp_formula):
            try:
                full = comp_formula(comp)
                if full:
                    formula = full
            except Exception:
                pass

        row: dict = {
            "formula": formula,
            "reward": reward,
            "exploration_target_used": episode.metadata.get("exploration_target_used"),
            "exploration_entropy": episode.metadata.get("exploration_entropy"),
            "candidate_json": candidate_json,
            **metadata_fields,
            **objective_record(raw_mean, per_stats, details),
        }
        if not use_per_obj_stats:
            row["dp_std"] = std
            row["dp_mean_minus_std"] = None if std is None else raw_mean - k * std
        if use_check_phase:
            phase_ok, phase_label = predictor.check_phase(comp)
            row["primary_ok"] = bool(phase_ok)
            row["primary_label"] = phase_label or ""

        observation = _exploration_observation(episode)
        if use_check_phase and not row["primary_ok"]:
            observation["feasible"] = False
        exploration_pending.append(observation)
        if not evaluator.pending:
            _flush_exploration(exploration, exploration_pending, env)
        row.update(_exploration_metrics(exploration))
        rows.append(row)
        accepted += 1
        pbar.update(1)
        pbar.set_postfix(
            attempts=attempted, dups=dup_rejected, charge=charge_rejected
        )

    pbar.close()
    rate = accepted / max(attempted, 1)
    print(
        f"[INFO] Generated {accepted}/{n_target} candidates "
        f"({attempted} attempts, {dup_rejected} dups, "
        f"{charge_rejected} non-neutral [{len(seen_nonneutral)} unique], "
        f"rate={rate:.3f})"
    )
    if charge_filter and accepted < n_target:
        print(
            f"[INFO] Shortfall diagnosis: of {attempted} attempts, "
            f"{dup_rejected} were duplicates and {charge_rejected} were "
            f"non-charge-neutral. "
            + (
                "Duplication dominates: the policy keeps proposing the same "
                "few compositions (inspect exploration_target diagnostics, or the "
                "neutral subspace may simply be small)."
                if dup_rejected >= charge_rejected
                else "Charge-neutrality dominates: most proposed compositions "
                "aren't neutral on the 0.05 grid (the neutral subspace is the "
                "bottleneck, not exploration)."
            )
        )

    return rows
