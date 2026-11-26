"""Gradient-only evaluation of the separate pilot on disposable network copies.

Each candidate contributes one mean trajectory-loss gradient. All copies stay at
the same initial weights; value coordinates are calibrated once on the full pilot.
No optimizer exists here and no pilot path is inserted into learner replay.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .gradient_clipping import (
    AdaptiveGradientClipper, NonfiniteGradientError,
    global_gradient_norm, gradient_clipping_profile,
)
from .utils.candidate_budget import NumericalTrainingError
from .utils.seeding import capture_rng_state, restore_rng_state


@contextmanager
def _preserve_rng():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _pilot(episodes):
    if episodes is None or len(episodes) != 100:
        raise ValueError("Gradient clipping initialization requires exactly 100 separate calibration candidates")
    if len({repr(ep.terminal_key) for ep in episodes}) != 100:
        raise ValueError("Gradient calibration candidates must be distinct")
    if any(not ep.path or any(not math.isfinite(float(step.reward)) for step in ep.path)
           for ep in episodes):
        raise ValueError("Gradient calibration requires finalized finite trajectory rewards")
    return episodes


def _validate_targets(targets):
    if any(not math.isfinite(value) for value in targets):
        raise NumericalTrainingError("Nonfinite computed gradient-calibration targets")


def _update_probe_coordinates(probe, targets, device):
    if getattr(probe, "value_normalizer", None) is not None:
        try:
            probe.update_value_normalization(torch.tensor(targets, dtype=torch.float64, device=device))
        except ValueError as exc:
            # Finite targets can still overflow the variance or compensated head.
            # Keep structural PopArt/configuration errors distinct from this case.
            if str(exc).startswith("Nonfinite PopArt"):
                raise NumericalTrainingError("Nonfinite gradient-calibration value coordinates") from exc
            raise


def _probe_gradient_norm(probe):
    try:
        return global_gradient_norm(probe.parameters())
    except NonfiniteGradientError as exc:
        raise NumericalTrainingError(str(exc)) from exc


def _report(controllers, targets, started):
    values = np.asarray(targets, dtype=np.float64)
    return {"schema_version": 1, "mode": "clipping", "profile": gradient_clipping_profile(),
            "accounting": "outside_training_budget", "candidate_count": 100,
            "optimizer_updates": 0, "additional_predictor_calls": 0,
            "probe_unit": "mean loss over one candidate trajectory; unchanged network copies",
            "target_mean": float(values.mean()), "target_std": float(values.std()),
            "target_count": int(values.size), "elapsed_s": time.perf_counter() - started,
            "networks": {name: controller.state_dict() for name, controller in controllers.items()}}


def write_gradient_report(checkpoint_cfg, report):
    path = (checkpoint_cfg or {}).get("path")
    if path:
        destination = Path(path).parent / "gradient_calibration.json"
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(destination)


def calibrate_dqn_gradients(*, qnet, target_net, episodes, scaler, elem_feats_scaled,
                            action_values, loss_fn, device, gamma, target_mode):
    from .training import _path_to_rows, _compute_td_target, _dqn_loss_terms

    started = time.perf_counter()
    episodes = _pilot(episodes)
    with _preserve_rng():
        probe = copy.deepcopy(qnet)
        target_probe = copy.deepcopy(target_net) if target_net is not None else None
        groups = [_path_to_rows(ep.path, action_values, gamma) for ep in episodes]
        targets = [[float(row["mc_return"]) if target_mode == "mc" else
                    _compute_td_target(row, target_probe, scaler, elem_feats_scaled,
                                       action_values, device, gamma)
                    for row in rows] for rows in groups]
        flat_targets = [value for group in targets for value in group]
        _validate_targets(flat_targets)
        _update_probe_coordinates(probe, flat_targets, device)
        norms = []
        for rows, values in zip(groups, targets):
            probe.zero_grad(set_to_none=True)
            loss, _, _, _ = _dqn_loss_terms(
                probe, rows, target_probe, scaler, elem_feats_scaled, action_values,
                loss_fn, device, gamma, target_mode=target_mode,
                update_normalization=False, target_values=values,
            )
            if not torch.isfinite(loss).item():
                raise NumericalTrainingError("Nonfinite DQN gradient-calibration loss")
            loss.backward()
            norms.append(_probe_gradient_norm(probe))
        controller = AdaptiveGradientClipper(norms, network="q")
        report = _report({"q": controller}, flat_targets, started)
        report["target_mode"] = target_mode
        return controller, report


def calibrate_pg_gradients(*, policy, value_net, episodes, scaler, elem_feats_scaled,
                           action_values, device, gamma, exploration_target=.7,
                           exploration_adaptive=True):
    from .training import _episode_pg_terms
    from .exploration import ExplorationController

    started = time.perf_counter()
    episodes = _pilot(episodes)
    with _preserve_rng():
        actor = copy.deepcopy(policy)
        critic = copy.deepcopy(value_net) if value_net is not None else None
        returns = []
        for episode in episodes:
            values = []
            value = 0.0
            for step in reversed(episode.path):
                value = float(step.reward) + gamma * value
                values.append(value)
            returns.append(list(reversed(values)))
        flat_targets = [value for episode, values in zip(episodes, returns)
                        for step, value in zip(episode.path, values) if step.allowed_actions]
        if not flat_targets:
            raise ValueError("Gradient calibration has no legal policy actions")
        _validate_targets(flat_targets)
        _update_probe_coordinates(critic, flat_targets, device)

        def terms(episode, values):
            return _episode_pg_terms(path=episode.path, returns=values, policy=actor,
                                     value_net=critic, scaler=scaler, elem_feats_scaled=elem_feats_scaled,
                                     action_values=action_values, device=device)

        # One fixed pilot-wide actor scaling, so single-step episodes retain a
        # meaningful advantage instead of centering each candidate to zero.
        all_adv = []
        with torch.no_grad():
            for episode, values in zip(episodes, returns):
                _, advantages, _, _ = terms(episode, values)
                all_adv.extend(advantages)
        adv_tensor = torch.tensor(all_adv, dtype=torch.float32, device=device)
        mean, std = adv_tensor.mean(), adv_tensor.std(unbiased=False)
        # The first real update uses this coefficient. The disposable probe must
        # neither advance the learner controller nor use a different loss.
        effective_coef = ExplorationController(
            initial_target=exploration_target, adaptive=exploration_adaptive,
        ).coefficient
        actor_norms, critic_norms = [], []
        for episode, values in zip(episodes, returns):
            actor.zero_grad(set_to_none=True)
            if critic is not None:
                critic.zero_grad(set_to_none=True)
            logp, advantages, entropies, critic_losses = terms(episode, values)
            if not logp:
                raise ValueError("Gradient calibration candidate has no legal policy terms")
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            scaled = (advantages - mean) / std if float(std) > 1e-8 else torch.zeros_like(advantages)
            loss = -(torch.stack(logp) * scaled).mean()
            if entropies:
                loss = loss - effective_coef * torch.stack(entropies).mean()
            if not torch.isfinite(loss).item():
                raise NumericalTrainingError("Nonfinite actor gradient-calibration loss")
            loss.backward()
            actor_norms.append(_probe_gradient_norm(actor))
            if critic is not None:
                loss = torch.stack(critic_losses).mean()
                if not torch.isfinite(loss).item():
                    raise NumericalTrainingError("Nonfinite critic gradient-calibration loss")
                loss.backward()
                critic_norms.append(_probe_gradient_norm(critic))
        controllers = {"actor": AdaptiveGradientClipper(actor_norms, network="actor")}
        if critic is not None:
            controllers["critic"] = AdaptiveGradientClipper(critic_norms, network="critic")
        report = _report(controllers, flat_targets, started)
        report.update(actor_advantage_mean=float(mean), actor_advantage_std=float(std),
                      actor_entropy_coefficient=effective_coef)
        return controllers, report
