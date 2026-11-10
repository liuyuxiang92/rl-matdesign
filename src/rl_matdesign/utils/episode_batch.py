"""Bounded, ordered terminal-reward evaluation for sequential RL rollouts.

Only expensive rewards run in a batch. Environment mutations and neural-network
sampling stay on the caller thread; snapshots never retain the environment or
predictor. A queue is drained before the next policy collection wave starts.
"""
from __future__ import annotations

import copy
import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


def supports_partial_batch_prediction(predictor) -> bool:
    """Recognize the opt-in batch protocol without requiring plug-in inheritance."""
    batch = getattr(predictor, "batch_predict", None)
    if not callable(batch):
        return False
    if getattr(predictor, "supports_partial_batch", False):
        return True
    try:
        parameters = inspect.signature(batch).parameters
    except (TypeError, ValueError):
        return False
    return "return_exceptions" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class CandidateEvaluationFailed(Exception):
    """A valid candidate could not be scored; retry without spending budget."""


@dataclass
class EpisodeSnapshot:
    path: list
    terminal_key: tuple
    composition: Any
    formula: str
    metadata: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)


def snapshot_episode(env, metadata: dict | None = None) -> EpisodeSnapshot:
    composition_fn = getattr(env, "terminal_cation_fractions", None)
    return EpisodeSnapshot(
        path=copy.deepcopy(env.path),
        terminal_key=copy.deepcopy(env.terminal_comp_key()),
        composition=copy.deepcopy(composition_fn()) if callable(composition_fn) else None,
        formula=getattr(env, "terminal_formula", ""),
        metadata=dict(metadata or {}),
    )


class EpisodeBatchEvaluator:
    """Prefetch a bounded wave, then yield successful trajectories in order.

    ``limit`` must include all caller-specific budget/checkpoint headroom. The
    consecutive failure cap is added here. Passing ``retry_scalar`` enables the
    training retry policy; omitting it preserves generation's failed-attempt
    counting. Batch size one calls the original rollout directly.
    """

    def __init__(self, env, *, max_retries: int = 10, timer=None, capture_predictions=False):
        self.env = env
        self.reward_fn = getattr(env, "reward_fn", None)
        self.size = 1
        self._refresh_size()
        self.max_retries = max_retries
        self.failures = 0
        self.pending = deque()
        self.timer = timer
        self.wave_id = 0
        self.capture_predictions = bool(capture_predictions)
        if hasattr(self.reward_fn, "capture_prediction_records"):
            self.reward_fn.capture_prediction_records = self.capture_predictions

    def _record_failure(self, failure, phase):
        record = getattr(self.timer, "record_episode", None)
        if callable(record):
            record({
                **getattr(failure, "episode_timing", {}),
                "phase": phase, "status": "failed", "episode": None,
                "formula": getattr(failure, "episode_formula", ""),
                "rl_update_s": 0.0,
            })

    def _failed_formula(self):
        # A rollout may fail before a custom environment reaches a valid
        # terminal state. Diagnostics must preserve the original exception.
        try:
            return getattr(self.env, "terminal_formula", "")
        except Exception:
            return ""

    def _timing(self, rollout_s, eval_share, count, details=None):
        return {
            **(details or {}),
            "evaluation_wave_id": self.wave_id,
            "evaluation_wave_size": count,
            "rollout_s": rollout_s,
            "eval_wall_share_s": eval_share,
            "rl_update_s": 0.0,
            "episode_wall_share_s": rollout_s + eval_share,
        }

    def _scalar_rollout(self, rollout, phase):
        self.wave_id += 1
        reset = getattr(self.reward_fn, "reset_evaluation_timing", None)
        if callable(reset):
            reset()
        started = time.perf_counter()
        failure = None
        try:
            rollout()
        except CandidateEvaluationFailed as exc:
            failure = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            evaluation = float(getattr(self.reward_fn, "last_evaluation_wall_s", 0.0))
            details = getattr(self.reward_fn, "last_evaluation_timings", [])
            self._scalar_timing = self._timing(
                max(0.0, elapsed - evaluation), evaluation, 1,
                details[0] if len(details) == 1 else {},
            )
            if failure is not None:
                failure.episode_timing = self._scalar_timing
                failure.episode_formula = self._failed_formula()
                self._record_failure(failure, phase)

    def _refresh_size(self):
        size_fn = getattr(self.reward_fn, "evaluation_batch_size", None)
        self.size = size_fn() if callable(size_fn) else 1
        if type(self.size) is not int or self.size < 1:
            raise ValueError("evaluation_batch_size must be a positive integer")
        if self.size > 1 and not callable(getattr(self.reward_fn, "evaluate_many", None)):
            raise ValueError("batched rewards require an evaluate_many adapter")

    def _collect(self, rollout: Callable[[], None], count: int, metadata: dict | None):
        self.wave_id += 1
        outcomes = []
        payloads = []
        indices = []
        for _ in range(count):
            started = time.perf_counter()
            captured = []

            def defer(payload):
                captured.append(copy.deepcopy(payload))
                return 0.0

            self.env.reward_fn = defer
            try:
                rollout()
                if len(captured) != 1 or not self.env.path:
                    raise RuntimeError("deferred rollout must produce exactly one terminal reward")
                episode = snapshot_episode(self.env, metadata)
            except CandidateEvaluationFailed as exc:
                exc.episode_formula = self._failed_formula()
                exc.episode_timing = self._timing(time.perf_counter() - started, 0.0, count)
                outcomes.append(exc)
                continue
            finally:
                self.env.reward_fn = self.reward_fn
            indices.append(len(outcomes))
            episode.timing = self._timing(time.perf_counter() - started, 0.0, count)
            payloads.append(captured[0])
            outcomes.append(episode)
        if payloads:
            started = time.perf_counter()
            rewards = list(self.reward_fn.evaluate_many(payloads))
            elapsed = time.perf_counter() - started
            predictions = getattr(self.reward_fn, "last_prediction_records", [])
            if len(predictions) != len(payloads):
                predictions = [{} for _ in payloads]
            details = getattr(self.reward_fn, "last_evaluation_timings", [])
            if len(details) != len(payloads):
                details = [{} for _ in payloads]
            if len(rewards) != len(payloads):
                raise RuntimeError("batched rewards returned the wrong number of results")
            for index, reward, timing, prediction in zip(indices, rewards, details, predictions):
                episode = outcomes[index]
                episode.timing.update(timing)
                if self.capture_predictions:
                    episode.prediction = copy.deepcopy(prediction)
                if isinstance(reward, Exception):
                    reward.episode_timing = episode.timing
                    reward.episode_formula = episode.formula
                    outcomes[index] = reward
                else:
                    outcomes[index].path[-1].reward = float(reward)
            # A wave is one shared wall interval. Charge every attempted
            # episode an equal share, including failed and cached candidates;
            # task-seconds in details remain each candidate's actual work.
            share = elapsed / len(outcomes)
            for outcome in outcomes:
                timing = (outcome.episode_timing if isinstance(outcome, Exception)
                          else outcome.timing)
                timing["eval_wall_share_s"] = share
                timing["episode_wall_share_s"] = timing["rollout_s"] + share
        return outcomes

    def next(self, rollout: Callable[[], None], *, limit: int,
             retry_scalar: Callable | None = None,
             metadata: dict | None = None, phase: str = "") -> EpisodeSnapshot:
        if limit < 1:
            raise ValueError("evaluation wave has no remaining budget")
        while True:
            if not self.pending:
                # A predictor can change its recommendation between waves.
                # Never change collection behavior while prefetched paths remain.
                self._refresh_size()
                if self.size == 1 and self.failures == 0:
                    if retry_scalar is None:
                        self._scalar_rollout(rollout, phase)
                    else:
                        retry_scalar(lambda: self._scalar_rollout(rollout, phase))
                    episode = snapshot_episode(self.env, metadata)
                    episode.timing = dict(self._scalar_timing)
                    predictions = getattr(self.reward_fn, "last_prediction_records", [])
                    if self.capture_predictions and len(predictions) == 1:
                        episode.prediction = copy.deepcopy(predictions[0])
                    return episode
                count = min(self.size, limit, self.max_retries + 1 - self.failures)
                self.pending.extend(self._collect(rollout, count, metadata))
            outcome = self.pending.popleft()
            if not isinstance(outcome, Exception):
                self.failures = 0
                return outcome
            if not isinstance(outcome, CandidateEvaluationFailed):
                raise outcome
            self._record_failure(outcome, phase)
            if retry_scalar is None:
                # Generation counts failed candidates in its own attempts budget.
                self.failures = 0
                raise outcome
            self.failures += 1
            if self.failures > self.max_retries:
                raise RuntimeError(
                    f"{self.max_retries + 1} consecutive candidates all failed evaluation -- "
                    "giving up rather than spending the rest of the run on unscored "
                    f"candidates. Last failure: {outcome}"
                ) from outcome
            print(f"[WARN] candidate evaluation failed (retry "
                  f"{self.failures}/{self.max_retries}), discarding and "
                  f"trying a new candidate: {outcome}")


def training_wave_limit(budget, remaining: int, *, attempts_remaining: int | None = None,
                        checkpoint_freq: int = 0, completed_attempts: int = 0) -> int:
    """Conservative cap: no possible outcome can overrun a stop boundary."""
    limit = int(remaining)
    if not budget.count_duplicates:
        limit = min(limit, budget.max_duplicate_attempts - budget.consecutive_duplicates)
    if attempts_remaining is not None:
        limit = min(limit, int(attempts_remaining))
    if checkpoint_freq > 0:
        limit = min(limit, checkpoint_freq - completed_attempts % checkpoint_freq)
    return limit
