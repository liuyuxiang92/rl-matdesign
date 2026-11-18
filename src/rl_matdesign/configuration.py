"""Shared YAML loading and environment construction for RL and baselines."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import time

from .utils.episode_batch import CandidateEvaluationFailed, supports_partial_batch_prediction
from .utils.rollout_errors import EvaluationProtocolError
from .config_schema import (
    load_yaml_config,
    validate_config,
)


def load_config(path: str) -> dict:
    """Read and validate authored settings without materializing defaults."""
    cfg = load_yaml_config(path)
    validate_config(cfg)
    return cfg


def build_predictor(cfg: dict, seed: int = None):
    from rl_matdesign.registry import build_reward
    return build_reward(cfg, seed=seed)


def build_constraint_filter(cfg: dict, env=None):
    """Build the list of optional scientific filters for one group."""
    from rl_matdesign.registry import build_constraints
    return build_constraints(cfg, env=env)


class _TerminalReward:
    """One conversion/error policy for immediate and deferred terminal rewards."""

    def __init__(self, predictor, *, k: float = 1.0):
        self.predictor = predictor
        self.last_evaluation_timings = []
        self.last_evaluation_wall_s = 0.0
        self.last_predictions = []
        self.k = float(k)
        self.capture_prediction_records = False
        self.last_prediction_records = []

    def reset_evaluation_timing(self):
        self.last_evaluation_timings = []
        self.last_evaluation_wall_s = 0.0
        self.last_predictions = []
        self.last_prediction_records = []

    def _prediction_record(self, candidate, result):
        if not self.capture_prediction_records:
            return {}
        from .utils.prediction_records import prediction_record
        try:
            return prediction_record(self.predictor, candidate, result, k=self.k)
        except Exception as exc:
            # Optional diagnostics must not discard a successfully scored
            # candidate or turn an unavailable statistic into another inference.
            return {"prediction_record_error": f"{type(exc).__name__}: {exc}"}

    def _prediction_timings(self, count):
        rows = getattr(self.predictor, "last_evaluation_timings", None)
        if isinstance(rows, (list, tuple)) and len(rows) == count:
            return [dict(row) if isinstance(row, dict) else {} for row in rows]
        return [{} for _ in range(count)]

    def evaluation_batch_size(self) -> int:
        size = getattr(self.predictor, "evaluation_batch_size", None)
        return size() if callable(size) else 1

    @staticmethod
    def _candidate(payload):
        if not isinstance(payload, Mapping):
            raise TypeError("Terminal candidate must be a structured mapping")
        return deepcopy(payload)

    def _failure(self, payload, exc):
        return CandidateEvaluationFailed(
            f"predictor.predict failed for candidate {payload!r}: "
            f"{type(exc).__name__}: {exc}"
        )

    @staticmethod
    def _mean(result):
        mean, _ = result
        mean = float(mean)
        if not math.isfinite(mean):
            raise ValueError(f"predictor returned nonfinite reward: {mean}")
        return mean

    def __call__(self, payload) -> float:
        self.reset_evaluation_timing()
        started = time.perf_counter()
        candidate = self._candidate(payload)
        try:
            result = self.predictor.predict(candidate)
            mean = self._mean(result)
            self.last_predictions = [tuple(result)]
        except EvaluationProtocolError:
            raise
        except Exception as exc:
            raise self._failure(payload, exc) from exc
        finally:
            self.last_evaluation_wall_s = time.perf_counter() - started
            self.last_evaluation_timings = self._prediction_timings(1)
        self.last_prediction_records = [self._prediction_record(candidate, result)]
        return mean

    def evaluate_candidate(self, candidate) -> float:
        """Evaluate the environment's canonical mapping without formula parsing."""
        return self(candidate)

    def evaluate_many(self, payloads) -> list:
        self.reset_evaluation_timing()
        started = time.perf_counter()
        self.last_evaluation_timings = [{} for _ in payloads]
        results = [None] * len(payloads)
        self.last_predictions = [None] * len(payloads)
        self.last_prediction_records = [{} for _ in payloads]
        candidates = [self._candidate(payload) for payload in payloads]
        indices = list(range(len(candidates)))
        if not candidates:
            self.last_evaluation_wall_s = time.perf_counter() - started
            return results
        batch_fn = getattr(self.predictor, "batch_predict", None)
        use_batch = supports_partial_batch_prediction(self.predictor)
        # The timing proxy may implement a batch as scalar calls. In that case
        # capture each result here before the next call overwrites plug-in state.
        if self.capture_prediction_records and getattr(self.predictor, "scalar_batch_fallback", False):
            use_batch = False
        if use_batch:
            try:
                predicted = list(batch_fn(candidates, return_exceptions=True))
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                predicted = [exc] * len(candidates)
            if len(predicted) != len(candidates):
                raise EvaluationProtocolError("predictor.batch_predict returned the wrong number of results")
            timings = self._prediction_timings(len(candidates))
            records = [
                {} if isinstance(result, Exception) else self._prediction_record(candidate, result)
                for candidate, result in zip(candidates, predicted)
            ]
        else:
            # A scalar plug-in need only implement predict; do not assume it
            # inherits the Protocol's default batch implementation.
            predicted = []
            timings = []
            records = []
            for candidate in candidates:
                try:
                    predicted.append(self.predictor.predict(candidate))
                except EvaluationProtocolError:
                    raise
                except Exception as exc:
                    predicted.append(exc)
                timings.extend(self._prediction_timings(1))
                result = predicted[-1]
                # A scalar plug-in may only retain its latest evaluation.
                records.append({} if isinstance(result, Exception)
                               else self._prediction_record(candidate, result))
        for index, timing in zip(indices, timings):
            self.last_evaluation_timings[index] = timing
        for index, result, record in zip(indices, predicted, records):
            try:
                if isinstance(result, Exception):
                    raise result
                results[index] = self._mean(result)
                self.last_predictions[index] = tuple(result)
                self.last_prediction_records[index] = record
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                results[index] = self._failure(payloads[index], exc)
        self.last_evaluation_wall_s = time.perf_counter() - started
        return results


class _SingleChemicalCandidate:
    """Keep existing formula-only predictors' flat input at the adapter boundary."""

    def __init__(self, name, *, empirical=False):
        self.name = name
        self.empirical = empirical

    def assemble_candidate(self, candidate):
        amounts = {label: value for label, value in candidate[self.name].items() if value != 0}
        if self.empirical:
            from functools import reduce
            from math import gcd
            divisor = reduce(gcd, (abs(int(value)) for value in amounts.values()), 0) or 1
            amounts = {label: int(value) // divisor for label, value in amounts.items() if value}
        return amounts

    def design_schema(self):
        return {"version": 1, "kind": "single_chemical_candidate", "group": self.name,
                "empirical_formula": self.empirical}


def build_env(cfg, predictor):
    """Compile the two public domains without exposing structure recipes in groups."""
    from .env_multigroup import MultiGroupEnv
    from .peptide_design import compile_groups

    if not cfg.get("groups"):
        raise ValueError("A design needs a non-empty groups list")
    groups, adapter = compile_groups(cfg)
    if adapter is not None:
        validate = getattr(adapter, "validate_constraints", None)
        hook = getattr(predictor, "sequence_constraints", None)
        if callable(validate) and callable(hook):
            validate(hook("peptide", list(adapter.tokens)))
    if (adapter is None and len(groups) == 1 and groups[0].get("kind", "composition") == "composition"
            and not cfg.get("structure")):
        from .amounts import _menu
        amount_spec = groups[0]["amount_set"]
        menu = (tuple(value for values in amount_spec.values() for value in _menu(values))
                if isinstance(amount_spec, dict) and set(amount_spec) != {"min", "max", "step"}
                else _menu(amount_spec))
        empirical = (groups[0].get("total") is None and all(value.denominator == 1 and value >= 0 for value in menu))
        adapter = _SingleChemicalCandidate(groups[0]["name"], empirical=empirical)
    reward_fn = _TerminalReward(predictor, k=float(cfg.get("k", 1.0)))
    env = MultiGroupEnv(groups=groups, reward_fn=reward_fn, candidate_adapter=adapter)
    for spec, inner in zip(groups, env._inners):
        # Some filters need the domain's exact lattice tables; attach only after
        # construction. Prior groups are supplied at each action by the wrapper.
        inner.filters = build_constraint_filter(spec, env=inner)
    structure = cfg.get("structure", {})
    if structure.get("builder") == "pipeline":
        from .design_domains import bind_pipeline_domains
        bind_pipeline_domains(structure, groups, env)
        env.requires_builder_formula = True
    env.design_spec = {"schema_version": 3, "groups": deepcopy(cfg["groups"])}
    if structure:
        env.design_spec["structure"] = deepcopy(structure)
    return env, "multi_group", cfg
