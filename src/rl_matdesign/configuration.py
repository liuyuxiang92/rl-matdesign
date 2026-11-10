"""Shared YAML loading and environment construction for RL and baselines."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
import time
import warnings

from .env import CompositionEnv
from .env_integer import IntegerRatioEnv, normalize_integer_group_spec
from .utils.episode_batch import CandidateEvaluationFailed, supports_partial_batch_prediction
from .utils.rollout_errors import EvaluationProtocolError
from .config_schema import (
    ALIASES as _FLAG_ALIASES,
    load_yaml_config,
    normalize_aliases,
    validate_config,
)


def _apply_flag_aliases(cfg: dict) -> dict:
    """Compatibility import for the shared, non-mutating alias normalizer."""
    return normalize_aliases(cfg)


def load_config(path: str) -> dict:
    """Read and validate authored settings without materializing defaults."""
    cfg = normalize_aliases(load_yaml_config(path))
    validate_config(cfg)
    if "generated_columns" in cfg:
        from .utils.metrics import validate_generated_columns
        validate_generated_columns(cfg["generated_columns"])
    return _normalize_integer_domains(cfg)


def _normalize_integer_domains(cfg: dict) -> dict:
    """Resolve known legacy domain declarations before recording run settings."""
    if "groups" not in cfg:
        return cfg
    return {**cfg, "groups": [normalize_integer_group_spec(group)
                              if group.get("kind") == "integer" else group
                              for group in cfg["groups"]]}


# ---------------------------------------------------------------------------
# Predictor / constraint factories — thin wrappers around the registry.
# YAML keys 'predictor' and 'constraint_filter' accept either a built-in
# short name or a fully-qualified 'pkg.module:ClassName' for user plug-ins.
# See src/rl_matdesign/registry.py for the dispatch logic.
# ---------------------------------------------------------------------------

def build_predictor(cfg: dict, seed: int = None):
    from rl_matdesign.registry import build_reward
    return build_reward(cfg, seed=seed)


def build_constraint_filter(cfg: dict, env=None):
    from rl_matdesign.registry import build_constraints
    return build_constraints(cfg, env=env)


# Fallback reward for a terminal formula that fails to even PARSE as a valid
# composition (env_type == "integer_ratio" only) -- this indicates a bug in
# the env, not a legitimate build/predictor failure, so unlike a
# CandidateEvaluationFailed (see reward_fn/mg_reward_fn below, which retries
# with a fresh candidate instead) there is no obviously-better candidate to
# retry with; score it as a very bad episode and move on. Large enough to be
# unambiguous against any of this repo's real reward scales (Kelvin-scale
# sintering temps, eV-scale energies) without risking overflow in downstream
# mean/std or CSV arithmetic.
_FAILED_CANDIDATE_REWARD = -2000.0


class _TerminalReward:
    """One conversion/error policy for immediate and deferred terminal rewards."""

    def __init__(self, predictor, env_type: str, anion_formula: str = "", *, k: float = 1.0):
        self.predictor = predictor
        self.env_type = env_type
        self.anion_formula = anion_formula
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

    def _candidate(self, payload):
        if isinstance(payload, Mapping):
            return deepcopy(payload)
        if self.env_type == "multi_group":
            return payload
        if self.env_type == "integer_ratio":
            from pymatgen.core.composition import Composition
            try:
                comp = dict(Composition(payload).fractional_composition.as_dict())
                return {str(k): float(v) for k, v in comp.items()}
            except Exception:
                return None  # Preserve the existing invalid-formula penalty.
        parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", payload)
        comp = {el: float(frac) for el, frac in parts}
        if self.anion_formula:
            for el, _ in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", self.anion_formula):
                comp.pop(el, None)
        return comp

    def _failure(self, payload, exc):
        label = "candidate" if self.env_type == "multi_group" else "formula"
        return CandidateEvaluationFailed(
            f"predictor.predict failed for {label} {payload!r}: "
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
        if candidate is None:
            self.last_evaluation_timings = [{}]
            self.last_evaluation_wall_s = time.perf_counter() - started
            self.last_prediction_records = [{
                "reward": _FAILED_CANDIDATE_REWARD,
                "prediction_record_error": "Invalid terminal formula; no property prediction",
            }]
            return _FAILED_CANDIDATE_REWARD
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
        results = [_FAILED_CANDIDATE_REWARD] * len(payloads)
        self.last_predictions = [None] * len(payloads)
        self.last_prediction_records = [{
            "reward": _FAILED_CANDIDATE_REWARD,
            "prediction_record_error": "Invalid terminal formula; no property prediction",
        } for _ in payloads]
        candidates, indices = [], []
        for index, payload in enumerate(payloads):
            candidate = self._candidate(payload)
            if candidate is not None:
                indices.append(index)
                candidates.append(candidate)
                self.last_prediction_records[index] = {}
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
            # A legacy plug-in need only implement predict; do not assume it
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


def build_env(cfg: dict, predictor):
    """Build the env + reward_fn + constraint filter used by an experiment run.

    Single source of truth for the search space and constraints, shared by
    ``main()`` and the baseline optimizers (``scripts/baselines/*``) so that
    "valid candidate" means exactly the same thing for RL, BO, and GA.

    Every config uses ``groups:`` — there is no separate ``env_type`` public
    knob. Returns ``(env, env_type, flat_cfg)`` where ``env_type`` is
    ``"fraction"`` / ``"integer_ratio"`` / ``"multi_group"`` (an *internal*
    label describing which class ``env`` ended up being, derived from the
    groups rather than read from the config) and ``flat_cfg`` is ``cfg``
    except when a lone group is collapsed to a flat env (then the group's
    params are merged in). The returned env already has its ``phase_filter``
    attached (per-group for ``multi_group``).
    """
    cfg = _normalize_integer_domains(cfg)
    if "generated_columns" in cfg:
        from .utils.metrics import validate_generated_columns
        validate_generated_columns(cfg["generated_columns"])
    if not cfg.get("groups"):
        raise ValueError(
            "Config has no 'groups:' — every scenario is expressed as one or "
            "more sublattice groups now (kind: composition / categorical / "
            "integer / sequence). See any configs/*.yaml "
            "for examples."
        )

    from rl_matdesign.env_multigroup import normalize_group_spec

    groups = cfg["groups"]
    env_type = "multi_group"
    flat_cfg = cfg

    # Single-group auto-collapse: a lone `composition` or `integer` group is
    # transparently built as the flat CompositionEnv/IntegerRatioEnv — same
    # chemistry, but a flat `{el: frac}` candidate instead of
    # `{group: {el: frac}}` — so a single-sublattice scenario can be written in
    # the groups syntax without breaking flat predictors/builders (substitute,
    # rf_magpie). `categorical` single groups keep group semantics and stay on
    # MultiGroupEnv (no flat equivalent).
    collapse_kind = str(groups[0].get("kind", "composition")) if len(groups) == 1 else None
    if collapse_kind in ("composition", "integer"):
        gspec = normalize_group_spec(groups[0])
        # Group params (species_set/fraction_set/ratio_set/bounds/...) override;
        # top-level keys (anion_formula, episode_style, builder/reward, ...) are kept.
        flat_cfg = {**cfg, **gspec}
        env_type = "integer_ratio" if collapse_kind == "integer" else "fraction"
        print(f"[INFO] single {collapse_kind} group "
              f"{groups[0].get('name', '?')!r} -> collapsed to flat "
              f"{'IntegerRatioEnv' if collapse_kind == 'integer' else 'CompositionEnv'}")

    reward_fn = _TerminalReward(predictor, env_type, cfg.get("anion_formula", ""),
                               k=float(cfg.get("k", 1.0)))

    # Build env first (without filter), then build the filter using env's
    # feasibility tables (_allowed_units, _possible_sums_by_k), then attach.
    # This is required for the ooh_phase filter and avoids the previous probe
    # episode that burned RNG state before warmup (Bug 3).
    if env_type == "integer_ratio":
        env = IntegerRatioEnv(
            species_set=flat_cfg["species_set"],
            ratio_set=flat_cfg["ratio_set"],
            n_components=flat_cfg.get("n_components", 5),
            reward_fn=reward_fn,
            phase_filter=None,
            token_requirements=flat_cfg.get("token_requirements"),
        )
    elif env_type == "multi_group":
        from rl_matdesign.env_multigroup import MultiGroupEnv

        # Normalize each group's fraction_set/total_units, then build the
        # per-group constraint instance (env=None — multi-group filters mask
        # by element/level/prior-group).
        built_groups = []
        for g in groups:
            gspec = normalize_group_spec(g)
            if gspec.get("kind") == "sequence":
                # Retained parent IDs belong to the predictor's chemical
                # universe but never silently become replacement actions.
                from .sequence import _token_list
                tokens = _token_list(gspec.get("tokens"), label="sequence tokens")
                parent = (_token_list(gspec["parent"], label="sequence parent", unique=False)
                          if "parent" in gspec else [])
                hook = getattr(predictor, "sequence_constraints", None)
                if callable(hook):
                    inferred = hook(gspec.get("name", ""), list(dict.fromkeys(tokens + parent)))
                    if not isinstance(inferred, list) or any(not isinstance(bound, dict) for bound in inferred):
                        raise ValueError("sequence_constraints must return a list of count-constraint mappings")
                    configured = gspec.get("count_constraints", [])
                    if not isinstance(configured, list):
                        raise ValueError("count_constraints must be a list")
                    bounds = deepcopy(configured)
                    for bound in inferred:
                        if bound not in bounds:
                            bounds.append(deepcopy(bound))
                    gspec["count_constraints"] = bounds
            gspec["constraint_filter"] = build_constraint_filter(gspec, env=None)
            built_groups.append(gspec)
        env = MultiGroupEnv(groups=built_groups, reward_fn=reward_fn)
    else:
        env = CompositionEnv(
            species_set=flat_cfg["species_set"],
            fraction_set=flat_cfg["fraction_set"],
            anion_formula=flat_cfg.get("anion_formula", ""),
            n_components=flat_cfg.get("n_components", 5),
            reward_fn=reward_fn,
            phase_filter=None,
            total_units=flat_cfg["total_units"],
            element_bounds=flat_cfg.get("element_bounds"),
            episode_style=flat_cfg.get("episode_style", "element_then_amount"),
        )

    # Single-group envs attach a top-level filter post-construction (some filters
    # need the env's feasibility tables). multi_group filters live per-group.
    # For a collapsed single group, the filter is built from the merged flat_cfg
    # (so the group's own filters: entries are preserved).
    if env_type != "multi_group":
        env.phase_filter = build_constraint_filter(flat_cfg, env=env)
        from .action_encoding import ValueSpec
        env.value_specs = [ValueSpec(float(value), "numeric") for value in env.fraction_set]

    # Keep declarative design constraints in artifact compatibility checks;
    # mutable filter caches and RNG state are not part of this contract.
    env.design_spec = {
        key: deepcopy(cfg[key]) for key in (
            "groups", "filters", "anion_formula", "episode_style",
            "scaffold_formula", "scaffold_poscar", "scaffold_site_symbol",
            "site_symbol", "constrain_training",
        ) if key in cfg
    }
    if env_type == "multi_group":
        # Include predictor-supplied intrinsic bounds in checkpoint contracts;
        # transient filter objects and caches must never enter serialization.
        env.design_spec["groups"] = [
            {key: deepcopy(value) for key, value in group.items() if key != "constraint_filter"}
            for group in built_groups
        ]
    return env, env_type, flat_cfg
