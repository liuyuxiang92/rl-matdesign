"""Wall-clock + predictor-cost instrumentation for method comparisons.

The point of this module is to answer "which RL method reaches a good candidate
for the least *total* cost", where cost includes the reward-model call — which,
for a DeepMD ensemble or a geometry-optimised structure score, dominates
everything the RL machinery does.

Two pieces:

* :class:`PredictorTimer` — a transparent proxy around any ``PropertyPredictor``.
  Every training reward flows through ``predictor.predict`` (the ``reward_fn``
  closures in ``run_experiment.build_env``, invoked only at an episode's terminal
  step), so wrapping the predictor object once captures every reward call in
  every phase, for every env type and every method.
* :class:`PhaseTimer` — accumulates wall-clock seconds per named phase
  (``setup`` / ``warmup`` / ``train`` / ``generate``).

Both are written to ``<out>/timing.json`` by ``run_experiment.py``; the
per-episode cumulative counters also land in ``training_log.csv`` so a
best-reward-vs-time curve can be plotted after the fact without re-running.
"""
from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .composition import composition_key
from .episode_batch import supports_partial_batch_prediction
from .rollout_errors import EvaluationProtocolError
from .seeding import isolated_rng, candidate_seed


def candidate_key(candidate: Any) -> tuple:
    """Use the same exact structured identity as prediction and domain caches.

    Domain adapters own physical normalization. Global decimal rounding could
    merge distinct fine-grid values or numeric experimental conditions.
    """
    return composition_key(candidate)


class PredictorTimer:
    """Counting/timing proxy around a ``PropertyPredictor``.

    Wraps ``predict`` / ``predict_raw`` / ``batch_predict`` and delegates every
    other attribute to the wrapped predictor.  Delegation is load-bearing:
    ``run_experiment`` reads ``predictor._cache`` for DQN checkpointing and
    ``training.generate_candidates`` probes ``predict_raw`` /
    ``per_objective_stats`` / ``check_phase``.  ``__getattr__`` hands back the
    inner objects, so those paths keep working untouched.

    ``n_unique`` is counted against this wrapper's own key set rather than the
    predictor's internal cache, so it is meaningful even for predictors that
    cache under a different attribute name (``StructureScorePredictor``) or
    don't cache at all (the ``dummy`` predictor).  ``n_calls - n_unique`` is
    therefore the number of calls that a correctly-keyed cache could serve.
    """

    EPISODE_COLUMNS = (
        "timing_session", "phase", "episode", "formula", "status", "wall_elapsed_s",
        "evaluation_wave_id", "evaluation_wave_size", "rollout_s",
        "eval_wall_share_s", "rl_update_s", "episode_wall_share_s",
        "md_task_s", "opt_task_s", "structure_other_task_s", "property_s",
        "config_attempted", "config_succeeded", "cache_hit", "timing_incomplete",
    )

    def __init__(self, inner: Any, t0: Optional[float] = None, *,
                 episode_log_path: Optional[str] = None, append: bool = False,
                 eval_seed: int = 0) -> None:
        self._inner = inner
        self._t0 = time.perf_counter() if t0 is None else float(t0)
        self.n_calls = 0
        self.n_unique = 0
        self.t_predict_s = 0.0
        self._prior_predict_s = 0.0
        self.best_mean: Optional[float] = None
        self._seen: set = set()
        self._marks: List[Dict[str, Any]] = []
        self.eval_seed = int(eval_seed)
        self.phase = "train"
        self._phase_costs: Dict[str, Any] = {}
        self._diagnostic_requests = 0
        self._diagnostic_failures = 0
        self._ledger_path = None
        self.last_evaluation_timings: List[Dict[str, Any]] = []
        self._episode_log_path = Path(episode_log_path) if episode_log_path else None
        self._timing_session = uuid.uuid4().hex
        self._episode_totals: Dict[str, Any] = {
            "timing_session": self._timing_session, "n_records": 0, "n_failed": 0,
        }
        if self._episode_log_path is not None:
            self._episode_log_path.parent.mkdir(parents=True, exist_ok=True)
            prefix = "generation_" if self._episode_log_path.name.startswith("generation_") else ""
            self._ledger_path = self._episode_log_path.with_name(prefix + "evaluations.jsonl")
            if not append:
                self._ledger_path.write_text("")
            if not append or not self._episode_log_path.exists() or not self._episode_log_path.stat().st_size:
                with self._episode_log_path.open("w", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=self.EPISODE_COLUMNS).writeheader()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _observation(self, candidate, result=None, error=None, timing=None) -> None:
        cost = self._phase_costs.setdefault(self.phase, {
            "requests": 0, "successes": 0, "failures": 0, "seen": set(),
            "best_reward": None, "known_cache_hits": 0, "cache_status_known": 0,
        })
        cost["requests"] += 1
        key = candidate_key(candidate)
        unique = result is not None and key not in cost["seen"]
        if result is not None:
            cost["successes"] += 1
            cost["seen"].add(key)
            reward = float(result[0])
            if cost["best_reward"] is None or reward > cost["best_reward"]:
                cost["best_reward"] = reward
        else:
            cost["failures"] += 1
        timing = timing or {}
        if isinstance(timing.get("cache_hit"), bool):
            cost["cache_status_known"] += 1
            cost["known_cache_hits"] += int(timing["cache_hit"])
        row = {"phase": self.phase, "request": cost["requests"],
               "candidate_key": repr(key), "candidate": candidate,
               "status": "ok" if result is not None else "failed",
               "new_successful_candidate": bool(unique), "unique": len(cost["seen"]),
               "reward": result[0] if result is not None else None,
               "best_reward": cost["best_reward"], "timing": timing,
               "error": None if error is None else f"{type(error).__name__}: {error}"}
        cached = getattr(self._inner, "cached_outputs", None)
        outputs = cached(candidate) if result is not None and callable(cached) else None
        row["outputs"] = {name: value.to_dict() for name, value in outputs.items()} if outputs else None
        backend = getattr(self._inner, "cost_counters", None)
        row["backend_costs"] = backend() if callable(backend) else None
        row["invocation"] = self._timing_session
        if self._ledger_path is not None:
            row.update(wall_elapsed_s=self.t_wall_s, prediction_elapsed_s=self.t_predict_s)
            with self._ledger_path.open("a") as stream:
                stream.write(json.dumps(row, default=repr, allow_nan=False) + "\n")

    def state_dict(self) -> dict:
        """Cost ledger is separate from bounded prediction caches."""
        return {"schema_version": 1, "n_calls": self.n_calls, "seen": sorted(self._seen, key=repr),
                "t_predict_s": self.t_predict_s, "best_mean": self.best_mean,
                "diagnostic_requests": self._diagnostic_requests,
                "diagnostic_failures": self._diagnostic_failures,
                "phase": self.phase, "phase_costs": {name: {**deepcopy(cost), "seen": sorted(cost["seen"], key=repr)}
                    for name, cost in self._phase_costs.items()}}

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1:
            raise EvaluationProtocolError("Unsupported predictor ledger checkpoint")
        self.n_calls, self._seen = int(state["n_calls"]), set(state["seen"])
        self.n_unique = len(self._seen)
        self.t_predict_s, self.best_mean = float(state["t_predict_s"]), state["best_mean"]
        self._prior_predict_s = self.t_predict_s
        self.phase = state["phase"]
        self._diagnostic_requests = state.get("diagnostic_requests", 0)
        self._diagnostic_failures = state.get("diagnostic_failures", 0)
        self._phase_costs = {name: {**deepcopy(cost), "seen": set(cost["seen"])}
                             for name, cost in state["phase_costs"].items()}

    def _take_evaluation_timings(self, count: int) -> None:
        rows = getattr(self._inner, "last_evaluation_timings", None)
        self.last_evaluation_timings = (
            [dict(row) if isinstance(row, dict) else {} for row in rows]
            if isinstance(rows, (list, tuple)) and len(rows) == count
            else [{} for _ in range(count)]
        )

    def record_episode(self, row: Dict[str, Any]) -> None:
        """Persist one attempt immediately, including failed or cached attempts.

        Wall shares are attributable elapsed time. Task seconds sum overlapping
        configuration work and must never be added to those wall shares.
        """
        record = {**row, "wall_elapsed_s": self.t_wall_s,
                  "timing_session": self._timing_session}
        self._episode_totals["n_records"] += 1
        self._episode_totals["n_failed"] += int(row.get("status") == "failed")
        for key in ("rollout_s", "eval_wall_share_s", "rl_update_s", "episode_wall_share_s",
                    "md_task_s", "opt_task_s", "structure_other_task_s", "property_s"):
            value = row.get(key)
            if value is None:
                missing = key + "_unavailable_records"
                self._episode_totals[missing] = self._episode_totals.get(missing, 0) + 1
            else:
                self._episode_totals[key] = self._episode_totals.get(key, 0.0) + float(value)
        if self._episode_log_path is not None:
            with self._episode_log_path.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.EPISODE_COLUMNS,
                               extrasaction="ignore").writerow(record)

    # ------------------------------------------------------------------
    # Timed pass-throughs
    # ------------------------------------------------------------------

    def _record(self, candidate: Any, mean: float, dt: float) -> None:
        self.n_calls += 1
        self.t_predict_s += dt
        key = candidate_key(candidate)
        if key not in self._seen:
            self._seen.add(key)
            self.n_unique += 1
        try:
            m = float(mean)
        except (TypeError, ValueError):
            return
        # NaN never wins a comparison, so a bad prediction can't poison the max.
        if m == m and (self.best_mean is None or m > self.best_mean):
            self.best_mean = m

    def predict(self, candidate: Any) -> Tuple[float, float]:
        self.last_evaluation_timings = []
        t = time.perf_counter()
        try:
            with isolated_rng(candidate_seed(self.eval_seed, candidate)):
                out = self._inner.predict(candidate)
            if not math.isfinite(float(out[0])):
                raise ValueError("Predictor returned a nonfinite reward")
        except BaseException as exc:
            self.t_predict_s += time.perf_counter() - t
            self._take_evaluation_timings(1)
            self._observation(candidate, error=exc, timing=self.last_evaluation_timings[0])
            raise
        self._take_evaluation_timings(1)
        dt = time.perf_counter() - t
        self._record(candidate, out[0], dt)
        self._observation(candidate, out, timing=self.last_evaluation_timings[0])
        return out

    def batch_predict(self, candidates: List[Any], *, return_exceptions: bool = False) -> List[Any]:
        candidates = list(candidates)
        self.last_evaluation_timings = []
        if not candidates:
            return []
        # Merely having batch_predict says nothing about its random protocol.
        # Only adapters guaranteeing candidate-local, batch-invariant sampling
        # opt into one shared call; all other plugins retain scalar semantics.
        if not getattr(self._inner, "candidate_seeded_batch", False):
            results, timings = [], []
            for candidate in candidates:
                try:
                    results.append(self.predict(candidate))
                except EvaluationProtocolError:
                    raise
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    results.append(exc)
                finally:
                    timings.extend(self.last_evaluation_timings)
            self.last_evaluation_timings = timings
            return results
        started = time.perf_counter()
        fatal = None
        try:
            with isolated_rng(self.eval_seed):
                batch = self._inner.batch_predict
                out = list(batch(candidates, return_exceptions=True)
                           if supports_partial_batch_prediction(self._inner) else batch(candidates))
            if len(out) != len(candidates):
                raise EvaluationProtocolError("batch_predict must return one result per candidate")
        except BaseException as exc:
            out = [exc] * len(candidates)
            if not isinstance(exc, Exception) or isinstance(exc, EvaluationProtocolError):
                fatal = exc
        elapsed = time.perf_counter() - started
        self._take_evaluation_timings(len(candidates))
        share = elapsed / len(candidates)
        for index, (candidate, result, timing) in enumerate(zip(candidates, out, self.last_evaluation_timings)):
            if not isinstance(result, BaseException):
                try:
                    if not math.isfinite(float(result[0])):
                        raise ValueError("Predictor returned a nonfinite reward")
                except (TypeError, ValueError, IndexError) as exc:
                    result = out[index] = exc
            if isinstance(result, BaseException):
                self.t_predict_s += share
                self._observation(candidate, error=result, timing=timing)
                if isinstance(result, EvaluationProtocolError):
                    fatal = result
            else:
                self._record(candidate, result[0], share)
                self._observation(candidate, result, timing=timing)
        # The full batch has already run. Preserve its entire cost record before
        # surfacing a protocol error; never charge only the prefix before it.
        if fatal is not None:
            raise fatal
        if not return_exceptions:
            for result in out:
                if isinstance(result, Exception):
                    raise result
        return out

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define itself, so
        # `_inner`, the counters and the methods above never land here.
        # `predict_raw` is routed through here on purpose: if the inner
        # predictor lacks it, AttributeError propagates and `hasattr` stays
        # correctly False (training.generate_candidates branches on that).
        inner = object.__getattribute__(self, "_inner")
        attr = getattr(inner, name)
        if name == "predict_raw":
            def _timed_predict_raw(candidate: Any) -> Tuple[float, float]:
                t = time.perf_counter()
                self._diagnostic_requests += 1
                out, error = None, None
                try:
                    with isolated_rng(candidate_seed(self.eval_seed, candidate)):
                        out = attr(candidate)
                except BaseException as exc:
                    self._diagnostic_failures += 1
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    elapsed = time.perf_counter() - t
                    self.t_predict_s += elapsed
                    if self._ledger_path is not None:
                        with self._ledger_path.open("a") as stream:
                            stream.write(json.dumps({"purpose": "reporting", "phase": self.phase,
                                "candidate_key": repr(candidate_key(candidate)), "raw_result": out,
                                "elapsed_s": elapsed, "error": error,
                                "invocation": self._timing_session}, default=repr, allow_nan=False) + "\n")
                return out
            return _timed_predict_raw
        return attr

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def t_wall_s(self) -> float:
        """Seconds since the timer was created (i.e. since run start)."""
        return time.perf_counter() - self._t0

    def snapshot(self) -> Dict[str, Any]:
        """Live counters, for embedding in a metrics row."""
        return {
            "t_wall": round(self.t_wall_s, 4),
            "t_predict_cum": round(self.t_predict_s, 4),
            "n_predict_calls": self.n_calls,
            "n_predict_unique": self.n_unique,
            "best_reward_so_far": self.best_mean,
        }

    def mark(self, phase: str) -> None:
        """Record a named boundary (end of warmup, end of training, ...)."""
        snap = self.snapshot()
        snap["phase"] = phase
        self._marks.append(snap)

    def summary(self) -> Dict[str, Any]:
        """Aggregate predictor-cost report for ``timing.json``."""
        n = self.n_calls
        backend = getattr(self._inner, "cost_counters", None)
        return {
            "t_predict_s": round(self.t_predict_s, 4),
            "invocation_predict_s": round(self.t_predict_s - self._prior_predict_s, 4),
            "backend_costs_this_invocation": backend() if callable(backend) else None,
            "n_calls": n,
            "diagnostic_requests": self._diagnostic_requests,
            "diagnostic_failures": self._diagnostic_failures,
            "n_unique": self.n_unique,
            "n_repeated_requests": n - self.n_unique,
            "n_cache_hits": sum(cost["known_cache_hits"] for cost in self._phase_costs.values()),
            "cache_status_known_requests": sum(cost["cache_status_known"] for cost in self._phase_costs.values()),
            "cache_hit_rate": (sum(cost["known_cache_hits"] for cost in self._phase_costs.values()) /
                               sum(cost["cache_status_known"] for cost in self._phase_costs.values()))
                              if any(cost["cache_status_known"] for cost in self._phase_costs.values()) else None,
            "phases": {name: {**{key: value for key, value in cost.items() if key != "seen"},
                               "unique_successes": len(cost["seen"]),
                               "duplicates": cost["successes"] - len(cost["seen"])}
                       for name, cost in self._phase_costs.items()},
            "mean_s_per_call": round(self.t_predict_s / n, 6) if n else 0.0,
            "mean_s_per_unique": (round(self.t_predict_s / self.n_unique, 6)
                                  if self.n_unique else 0.0),
            "best_reward": self.best_mean,
            "episode_timings": dict(self._episode_totals),
        }

    @property
    def marks(self) -> List[Dict[str, Any]]:
        return list(self._marks)


class PhaseTimer:
    """Accumulate wall-clock seconds under named phases.

    Usage::

        phases = PhaseTimer()
        with phases("setup"):
            ...
        phases.totals  # {"setup": 12.4, ...}

    Re-entering a phase name adds to its total rather than replacing it.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def __call__(self, name: str) -> "PhaseTimer":
        self._stack.append((name, time.perf_counter()))
        return self

    def __enter__(self) -> "PhaseTimer":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        name, t = self._stack.pop()
        self.totals[name] = round(self.totals.get(name, 0.0) + (time.perf_counter() - t), 4)
        return False
