"""Bounded, ordered terminal-reward evaluation for sequential RL rollouts.

Only expensive rewards run in a batch. Environment mutations and neural-network
sampling stay on the caller thread; snapshots never retain the environment or
predictor. A queue is drained before the next policy collection wave starts.
"""
from __future__ import annotations

import copy
import inspect
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

    def __init__(self, env, *, max_retries: int = 10):
        self.env = env
        self.reward_fn = getattr(env, "reward_fn", None)
        self.size = 1
        self._refresh_size()
        self.max_retries = max_retries
        self.failures = 0
        self.pending = deque()

    def _refresh_size(self):
        size_fn = getattr(self.reward_fn, "evaluation_batch_size", None)
        self.size = size_fn() if callable(size_fn) else 1
        if type(self.size) is not int or self.size < 1:
            raise ValueError("evaluation_batch_size must be a positive integer")
        if self.size > 1 and not callable(getattr(self.reward_fn, "evaluate_many", None)):
            raise ValueError("batched rewards require an evaluate_many adapter")

    def _collect(self, rollout: Callable[[], None], count: int, metadata: dict | None):
        outcomes = []
        payloads = []
        indices = []
        for _ in range(count):
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
                outcomes.append(exc)
                continue
            finally:
                self.env.reward_fn = self.reward_fn
            indices.append(len(outcomes))
            payloads.append(captured[0])
            outcomes.append(episode)
        if payloads:
            rewards = list(self.reward_fn.evaluate_many(payloads))
            if len(rewards) != len(payloads):
                raise RuntimeError("batched rewards returned the wrong number of results")
            for index, reward in zip(indices, rewards):
                if isinstance(reward, Exception):
                    outcomes[index] = reward
                else:
                    outcomes[index].path[-1].reward = float(reward)
        return outcomes

    def next(self, rollout: Callable[[], None], *, limit: int,
             retry_scalar: Callable | None = None,
             metadata: dict | None = None) -> EpisodeSnapshot:
        if limit < 1:
            raise ValueError("evaluation wave has no remaining budget")
        while True:
            if not self.pending:
                # Auto-tuning can shrink the pool after the preceding wave.
                # Never change collection behavior while prefetched paths remain.
                self._refresh_size()
                if self.size == 1 and self.failures == 0:
                    if retry_scalar is None:
                        rollout()
                    else:
                        retry_scalar(rollout)
                    return snapshot_episode(self.env, metadata)
                count = min(self.size, limit, self.max_retries + 1 - self.failures)
                self.pending.extend(self._collect(rollout, count, metadata))
            outcome = self.pending.popleft()
            if not isinstance(outcome, Exception):
                self.failures = 0
                return outcome
            if not isinstance(outcome, CandidateEvaluationFailed):
                raise outcome
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
