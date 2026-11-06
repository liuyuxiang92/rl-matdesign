"""Compose rewards from named scientific quantities while preserving their units.

Model definitions may be inline in ``properties`` or shared through top-level
``models`` and property ``model_ref``/``output`` selections. The existing registry
constructs one adapter per named model/member. Adapters exposing named
``ScientificOutput`` records take priority over legacy score/predict methods.

Engine-folded results preserve a rectangular model/configuration sample matrix.
Model spread, configuration spread and native backend uncertainty remain separate;
a singleton axis has no spread estimate. ``exp``/``pow10`` inverses act on every
raw sample before aggregation. Legacy self-folded mean/std becomes native
uncertainty, and cannot be nonlinearly transformed without the underlying samples.

The default objective is ``mean``. An explicit uncertainty penalty must select an
available statistic. Combined rewards return no synthetic standard deviation:
quantities in unrelated units cannot be averaged into reward uncertainty.

Structure objectives reference frozen named structure sources; each candidate's
source is sampled/relaxed once and reused by its objectives. Named model inference
is shared for equal candidate/source/conditions. Cached outputs, metadata, sample
diagnostics and the winning sweep condition are one evaluation record. Global RNG
streams are isolated from external leaf construction and candidate-local inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass, field, replace
import time

import numpy as np

from ..utils.rollout_errors import EvaluationProtocolError
from .outputs import (OutputSchema, ModelSamples, ScientificOutput, aggregate_samples,
                      inverse_transform_output)


_DIRECTION_ALIASES = {"min": -1.0, "minimize": -1.0, "max": +1.0, "maximize": +1.0}
# A third direction: hit a specific value rather than push it up or down. Needs
# `target_value`; `target_tolerance` is a deadband inside which the term is at its
# maximum (0), so the objective stops competing once the requirement is met.
_TARGET_DIRECTION = "target"
_TARGET_KEYS = ("target_value", "target_tolerance")
_OBJECTIVES = ("mean", "mean_minus_kstd", "mean_plus_kstd")
# Per-property output transforms, applied to each ensemble member before the
# mean/std fold (so heads that regress log(value) report the real value).
_TRANSFORMS = ("none", "identity", "exp", "pow10")
_UNCERTAINTIES = ("model_spread", "config_spread", "native_uncertainty", "pooled_spread")


def normalize_model_references(cfg: Dict[str, Any]) -> Tuple[Dict[str, dict], List[dict]]:
    """Validate named model ownership and return detached effective properties.

    This performs no model construction or file I/O. Model definitions own model
    arguments; property references own reward/output selection and source use.
    Repeating a model-owned field at a reference is rejected even if equal.
    """
    definitions = deepcopy(cfg.get("models", {}))
    if not isinstance(definitions, dict):
        raise EvaluationProtocolError("Top-level models must be a mapping of model IDs to definitions")
    for model_id, definition in definitions.items():
        if not isinstance(model_id, str) or not model_id.strip() or not isinstance(definition, dict):
            raise EvaluationProtocolError("Each named model needs a nonempty ID and a mapping definition")
        if not definition.get("predictor") or "model_ref" in definition:
            raise EvaluationProtocolError(f"Model {model_id!r} needs predictor and cannot reference another model")
        forbidden = {"direction", "weight", "scale", "objective", "source", "transform", "uncertainty",
                     "target_value", "target_tolerance"} & definition.keys()
        if forbidden:
            raise EvaluationProtocolError(f"Model {model_id!r} contains property-owned settings: {sorted(forbidden)}")
    properties = cfg.get("properties", [])
    if not isinstance(properties, list) or any(not isinstance(prop, dict) for prop in properties):
        raise EvaluationProtocolError("properties must be a list of mappings")
    used, resolved = set(), []
    for original in properties:
        prop = deepcopy(original)
        model_ref = prop.get("model_ref")
        if model_ref is not None:
            if not isinstance(model_ref, str) or model_ref not in definitions:
                raise EvaluationProtocolError(f"Unknown model_ref {model_ref!r}")
            duplicates = {"predictor", "model", "models", "dp_models", "head"} & prop.keys()
            if duplicates:
                raise EvaluationProtocolError(f"model_ref {model_ref!r} cannot repeat model configuration: {sorted(duplicates)}")
            conflicts = set(prop) & set(definitions[model_ref])
            if conflicts:
                raise EvaluationProtocolError(f"model_ref {model_ref!r} repeats model-defined fields: {sorted(conflicts)}")
            used.add(model_ref)
            prop = {**definitions[model_ref], **prop}
        resolved.append(prop)
    if set(definitions) - used:
        raise EvaluationProtocolError(f"Unused named model definitions: {sorted(set(definitions) - used)}")
    return definitions, resolved


class _Statistics(dict):
    """Legacy scalar view retaining its paired named scientific outputs."""

    def __init__(self, values=(), *, outputs=None):
        super().__init__(values)
        self.outputs = outputs or {}

def _empty_timing(*, cache_hit=False):
    return {"md_task_s": 0.0, "opt_task_s": 0.0, "structure_other_task_s": 0.0,
            "property_s": 0.0, "config_attempted": 0, "config_succeeded": 0,
            "cache_hit": cache_hit, "timing_incomplete": False}


def _add_timing(target, source):
    for name in ("md_task_s", "opt_task_s", "structure_other_task_s",
                 "property_s", "config_attempted", "config_succeeded"):
        value = source.get(name, 0)
        target[name] = (None if target.get(name) is None or value is None
                        else target.get(name, 0) + value)
    if source.get("sources"):
        target.setdefault("sources", {}).update(deepcopy(source["sources"]))
    target["timing_incomplete"] = bool(
        target.get("timing_incomplete") or source.get("timing_incomplete"))


@dataclass(frozen=True)
class _EvaluationRecord:
    """Statistics and diagnostics from the same successful evaluation."""

    stats: Dict[str, Tuple[float, Optional[float]]]
    metadata: Dict[str, Any]
    source_diagnostics: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, ScientificOutput] = field(default_factory=dict)


class StructureScorePredictor:
    supports_partial_batch = True
    candidate_seeded_batch = True

    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        self._resources: List[Any] = []
        self._closed = False
        try:
            self._initialize(deepcopy(cfg), seed=seed)
        except BaseException:
            self.close()
            raise

    def _own(self, resource: Any) -> Any:
        self._resources.append(resource)
        return resource

    def _initialize(self, cfg: Dict[str, Any], *, seed: Optional[int]) -> None:
        import secrets
        from ..registry import resolve_builder

        self.cfg = cfg
        self.k: float = float(cfg.get("k", 1.0))
        self.n_random_configs: int = int(cfg.get("n_random_configs", 1))
        from numbers import Integral
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
            raise ValueError("Evaluation seed must be a nonnegative integer")
        self._seed = int(seed) if seed is not None else secrets.randbits(64)

        # Optional inner optimization over a shared operating condition (e.g.
        # temperature). For each candidate we score every property at each sweep
        # value and keep the single value maximizing the combined reward. The
        # value is substituted into each property's fparam `null` placeholder(s).
        sweep = cfg.get("sweep")
        self.sweep: Optional[Dict[str, Any]] = None
        if sweep is not None:
            values = sweep.get("values")
            if not values:
                raise ValueError(
                    "sweep needs a non-empty 'values' list (e.g. temperatures to "
                    "try for each composition)."
                )
            self.sweep = {
                "name": str(sweep.get("name", "sweep")),
                "values": [float(v) for v in values],
            }

        raw_props = cfg.get("properties")
        if not raw_props:
            raise ValueError(
                "reward needs a non-empty 'properties' list. Each entry is one "
                "objective: {name, predictor: dp_energy|dp_property|dp_classification|rf_magpie|<fqn>, "
                "direction: max|min, ...}."
            )

        model_definitions, resolved_props = normalize_model_references(cfg)
        shared_instances = {}
        self._model_declarations = {}
        self.properties: List[Dict[str, Any]] = []
        self._source_builders: Dict[str, Any] = {}
        seen: set[str] = set()
        for i, p in enumerate(resolved_props):
            model_ref = p.get("model_ref")
            model_definition = model_definitions.get(model_ref)
            name = str(p.get("name", f"prop{i}"))
            if name in seen:
                raise ValueError(
                    f"properties[{i}].name = {name!r} duplicates an earlier "
                    "objective (used as the CSV column prefix); names must be unique."
                )
            seen.add(name)

            physical_cfg = dict(model_definition if model_definition is not None else p)
            for field_name in ("name", "model_ref", "direction", "target_value", "target_tolerance",
                               "weight", "scale", "objective", "source", "transform", "uncertainty",
                               "output", "unit", "representation"):
                physical_cfg.pop(field_name, None)
            self._model_declarations[model_ref if model_ref is not None else name] = physical_cfg
            predictor_name = str(p.get("predictor", "")).strip()
            if not predictor_name:
                raise ValueError(
                    f"properties[{i}] ({name!r}) needs a 'predictor' — one of "
                    "'dp_energy', 'dp_property', 'dp_classification', a registry leaf (e.g. 'rf_magpie', "
                    "'ooh'), or an FQN 'pkg.mod:Class'."
                )
            is_dp_backend = predictor_name in ("dp_energy", "dp_property", "dp_classification")

            # direction is REQUIRED — the sign (lower vs higher is better) must be
            # explicit, never implied.
            direction_raw = p.get("direction")
            if direction_raw is None:
                raise ValueError(
                    f"properties[{i}] ({name!r}) needs an explicit 'direction' "
                    "(max or min)."
                )
            direction = str(direction_raw).lower()
            is_target = direction == _TARGET_DIRECTION
            if not is_target and direction not in _DIRECTION_ALIASES:
                raise ValueError(
                    f"properties[{i}].direction = {direction!r}; expected one of "
                    f"{sorted(_DIRECTION_ALIASES) + [_TARGET_DIRECTION]}."
                )

            # `direction: target` -> hit a value. Kept as its own spec rather than a
            # sign, because it is not a sign: see _property_value.
            target_spec = None
            if is_target:
                if p.get("target_value") is None:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) has direction: target but no "
                        "'target_value' -- the value to hit, in the property's real "
                        "units (e.g. target_value: 1.34 for a 1.34 eV bandgap). Note "
                        "this is NOT the leaf's own 'target'/'model' key, which "
                        "selects the checkpoint."
                    )
                try:
                    target_value = float(p["target_value"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"properties[{i}].target_value = {p['target_value']!r} is not "
                        "a number."
                    ) from None
                target_tol = float(p.get("target_tolerance", 0.0))
                if target_tol < 0.0:
                    raise ValueError(
                        f"properties[{i}].target_tolerance must be >= 0 (got {target_tol})."
                    )
                target_spec = {"value": target_value, "tolerance": target_tol}
            else:
                stray = [k for k in _TARGET_KEYS if k in p]
                if stray:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) sets {stray} but direction="
                        f"{direction!r}. Set 'direction: target' to hit a specific "
                        "value, or remove those keys -- they would otherwise be "
                        "silently ignored."
                    )

            scale = float(p.get("scale", 1.0))
            if scale == 0.0:
                raise ValueError(f"properties[{i}].scale must be non-zero (got 0).")

            objective = str(p.get("objective", "mean"))
            if objective not in _OBJECTIVES:
                raise ValueError(
                    f"properties[{i}].objective = {objective!r}; expected one of "
                    f"{list(_OBJECTIVES)}."
                )

            # `null` entries in fparam mark the slot filled by the sweep value.
            fparam = p.get("fparam")
            sweep_slots = (
                [j for j, x in enumerate(fparam) if x is None]
                if isinstance(fparam, (list, tuple)) else []
            )
            if sweep_slots and self.sweep is None:
                raise ValueError(
                    f"properties[{i}] ({name!r}) has a null placeholder in fparam "
                    f"but no top-level 'sweep' is configured. Add a 'sweep' block "
                    "(e.g. {name: temperature, values: [...]}) or fill the fparam."
                )

            transform = str(p.get("transform", "none")).lower()
            if transform not in _TRANSFORMS:
                raise ValueError(
                    f"properties[{i}].transform = {transform!r}; expected one of "
                    f"{sorted(_TRANSFORMS)}. Use 'exp' for heads that emit log(value)."
                )

            uncertainty = p.get("uncertainty")
            if uncertainty is not None and uncertainty not in _UNCERTAINTIES:
                raise EvaluationProtocolError(f"properties[{i}].uncertainty must be one of {_UNCERTAINTIES}")
            representation = p.get("representation", "ln" if transform == "exp" else
                                   "log10" if transform == "pow10" else "physical")
            try:
                OutputSchema(name, p.get("unit"), representation)
            except ValueError as exc:
                raise EvaluationProtocolError(str(exc)) from exc

            if is_dp_backend:
                models = p.get("models") or p.get("dp_models")
                if not models:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) with predictor={predictor_name!r} "
                        "needs 'models' (list of DeepMD checkpoint paths)."
                    )
                backend = predictor_name.removeprefix("dp_")
                if backend == "classification":
                    if not isinstance(models, (list, tuple)):
                        raise ValueError(f"properties[{i}].models must be a list of paths.")
                    aggregator = p.get("output_aggregator", "index")
                    if aggregator not in ("index", "mean", "max"):
                        raise ValueError(f"properties[{i}].output_aggregator must be index, mean, or max.")
                    index = p.get("output_index")
                    if isinstance(index, list):
                        from ..utils.dp_eval import validate_classification_indices

                        validate_classification_indices(index, output_aggregator=aggregator)
                    elif aggregator == "index" and (
                        isinstance(index, bool) or not isinstance(index, int) or index < 0
                    ):
                        raise ValueError(
                            f"properties[{i}] dp_classification needs an explicit non-negative "
                            "integer output_index or a list of unique class indices "
                            "matching the model's class order."
                        )
                    if transform not in ("none", "identity"):
                        raise ValueError("dp_classification probabilities require transform: none.")
                    if p.get("head") is not None and not isinstance(p["head"], str):
                        raise ValueError("dp_classification head must be a branch name string.")
                    if "output_kind" in p:
                        raise ValueError(
                            "dp_classification always calls predict_proba(); remove output_kind."
                        )
                instance = None
                is_structure = True
            else:
                # Registry leaf or FQN. `model:` may be ONE path or a LIST of paths;
                # each path becomes its own leaf instance, because a leaf represents
                # exactly one model. That is what makes "one model -> std == 0" fall
                # out of the arithmetic instead of needing a flag (see _score).
                from ..registry import resolve_predictor

                models = None
                model_spec = p.get("model", p.get("models"))
                if isinstance(model_spec, (list, tuple)):
                    model_paths = [str(m) for m in model_spec]
                    if not model_paths:
                        raise ValueError(
                            f"properties[{i}] ({name!r}) has an empty 'model' list."
                        )
                elif model_spec is None:
                    model_paths = [None]          # leaf needs no model path
                else:
                    model_paths = [str(model_spec)]

                instance = shared_instances.get(model_ref) if model_ref is not None else None
                if instance is None:
                    instance = []
                    for j, path in enumerate(model_paths):
                        leaf_cfg = dict(model_definition if model_definition is not None else p)
                        if path is not None:
                            leaf_cfg["model"] = path
                        # Model-reference identity is independent of property declaration order.
                        import hashlib
                        namespace = self._key({"reference": model_ref, "definition": physical_cfg})
                        child_seed = int.from_bytes(hashlib.sha256(
                            repr(("model-init-v1", self._seed, namespace, j)).encode()).digest()[:4], "big")
                        from ..utils.seeding import isolated_rng
                        with isolated_rng(child_seed):
                            instance.append(self._own(resolve_predictor(predictor_name, leaf_cfg, seed=child_seed)))
                    if model_ref is not None:
                        shared_instances[model_ref] = instance

                # A leaf that can score structures is a structure objective (gets the
                # built cells); otherwise it is a composition objective (gets the
                # candidate). Auto-detected from the leaf, no YAML flag.
                probe = instance[0]
                is_structure = any(callable(getattr(probe, method, None)) for method in
                                   ("score_structures", "predict_structures", "predict_structure_outputs"))
                backend = "structure_fqn" if is_structure else "composition"

            declared_paths = list(models) if models else model_paths
            if len(declared_paths) != len(set(declared_paths)):
                raise EvaluationProtocolError(f"Property {name!r} repeats a checkpoint; duplicated models do not form an ensemble")
            self.properties.append({
                "name": name,
                "model_ref": model_ref,
                "model_key": model_ref if model_ref is not None else name,
                "output": p.get("output"),
                "unit": p.get("unit"),
                "unit_declared": "unit" in p,
                "representation": representation,
                "representation_declared": "representation" in p,
                "uncertainty": uncertainty,
                "predictor": predictor_name,
                "backend": backend,
                "instance": instance,
                "models": list(models) if models else None,
                "head": p.get("head"),
                "output_index": (
                    list(p["output_index"])
                    if backend == "classification" and isinstance(p.get("output_index"), list)
                    else int(p.get("output_index", 0))
                ),
                "output_aggregator": str(p.get("output_aggregator", "index")),
                "energy_per_atom": bool(p.get("energy_per_atom", True)),
                "fparam": fparam,
                "sweep_slots": sweep_slots,
                "aparam": p.get("aparam"),
                "direction": None if is_target else _DIRECTION_ALIASES[direction],
                "target": target_spec,
                "weight": float(p.get("weight", 1.0)),
                "scale": scale,
                "objective": objective,
                "transform": transform,
                "n_random_configs": int(p.get("n_random_configs", self.n_random_configs)),
            })

        self._output_memo = {}
        self._metadata_memo = {}
        self._active_leaf_metadata = {}
        self._active_outputs = {}

        from ..rewards import RewardCombiner

        self._reward_combiner = RewardCombiner(
            cfg.get("reward"), {p["name"]: p["weight"] for p in self.properties}
        )

        # Any structure predictor present? Only then do we need a builder / base
        # POSCAR — a pure composition-predictor config (e.g. rf_magpie) builds
        # nothing.
        self._has_structure = any(p["backend"] != "composition" for p in self.properties)

        from ..structure_sources import normalize_structure_sources
        self.source_plan = normalize_structure_sources(
            cfg, needs_structure=[p["name"] for p in self.properties if p["backend"] != "composition"],
        )
        from ..structure_sources import resolve_source_assets
        from pathlib import Path
        resolved_sources = []
        self._source_asset_references = {}
        self._source_asset_snapshots = {}
        self._source_runtime_cwd = str(Path.cwd())
        for source in self.source_plan.sources:
            frozen, references = resolve_source_assets(source)
            resolved_sources.append(frozen)
            self._source_asset_references[source.source_id] = references
            for reference in references:
                if reference["status"] == "resolved":
                    path = reference["path"]
                    stat = Path(path).stat()
                    self._source_asset_snapshots[path] = (stat.st_dev, stat.st_ino, stat.st_size,
                                                         stat.st_mtime_ns, stat.st_ctime_ns)
        self.source_plan = replace(self.source_plan, sources=tuple(resolved_sources))
        self._property_sources = self.source_plan.property_sources
        for spec in self.source_plan.sources:
            source_cfg = spec.effective_config
            init_seed = spec.seed_for(None, eval_seed=self._seed, purpose="builder-init")
            if source_cfg.get("structure_parallel") is not None:
                from ..utils.parallel_structures import ParallelStructureBuilder
                builder = ParallelStructureBuilder(spec.builder, source_cfg, seed=init_seed)
            else:
                builder = resolve_builder(spec.builder, source_cfg, seed=init_seed)
            self._source_builders[spec.source_id] = self._own(builder)
        self._calc_cache: Dict[str, Any] = {}     # name -> ase calculators (energy backend)
        self._prop_cache: Dict[str, Tuple[List[Any], Dict[str, int]]] = {}  # name -> property models
        self._geo_calcs: Dict[str, Any] = {}
        self._parallel_models_ready = False
        # Persistent per-composition stats cache (LRU). Keyed by the canonical
        # candidate key, it makes each unique composition build+relax+score
        # exactly ONCE for the predictor's lifetime — so training repeats and
        # duplicate generation rollouts of the same composition are free (the
        # single biggest saving when geo_opt is expensive). `predict_cache_size`
        # caps entries (<=0 = unbounded). Entries include paired metadata/sample
        # diagnostics, so memory also depends on ensemble size. A cap of 1 keeps
        # only the most recent entry.
        from collections import OrderedDict

        self._stats_cache: "OrderedDict[tuple, _EvaluationRecord]" = OrderedDict()
        self._stats_cache_size: int = int(cfg.get("predict_cache_size", 200000))
        # Retain the most recent evaluation wave until consumers have logged its
        # raw/property statistics, even when the persistent LRU is smaller.
        self._batch_stats: Dict[tuple, _EvaluationRecord] = {}
        self.last_evaluation_timings: List[dict] = []
        self._materialization_timings: List[dict] = []
        self._last_build_timing: dict = {}
        self._materialization_diagnostics: List[dict] = []
        self._last_build_diagnostics: dict = {}
        self._freeze_model_assets()
        self._dp_cost = {"ensemble_evaluation_calls": 0, "model_configuration_pairs_attempted": 0,
                         "network_forwards": None}

    # ------------------------------------------------------------------
    # PropertyPredictor protocol
    # ------------------------------------------------------------------

    def _reward_details(
        self, stats: Dict[str, Tuple[float, float]], *, raw: bool = False,
    ) -> Dict[str, Any]:
        """Use one reward rule for training, generation, sweeps and diagnostics."""
        # Classifier gates use a fixed one-std margin across model means.
        # Keep this independent of objective direction/scale/k and of `raw`,
        # so training, CSV diagnostics and raw_combine agree on feasibility.
        gate_values = {
            p["name"]: stats[p["name"]][0] - (
                (getattr(stats, "outputs", {}).get(p["name"]).model_spread or 0.0)
                if p["backend"] == "classification" and p["name"] in getattr(stats, "outputs", {})
                else (stats[p["name"]][1] or 0.0) if p["backend"] == "classification" else 0.0
            )
            for p in self.properties
        }
        values = {
            p["name"]: self._property_value(
                p, stats[p["name"]][0], 0.0 if raw else stats[p["name"]][1]
            ) / p["scale"]
            for p in self.properties
        }
        return self._reward_combiner.evaluate(gate_values, values)

    def _combine(self, stats: Dict[str, Tuple[float, float]]) -> float:
        return self._reward_details(stats)["reward"]

    def reward_details(self, candidate: Any) -> Dict[str, Any]:
        """Return objective and gate diagnostics using the cached property stats."""
        return self._reward_details(self._raw_stats(candidate))

    def _property_value(self, prop: Dict[str, Any], mean: float, std: float) -> float:
        """Signed 'higher is better' value for ONE objective, before weight/scale.

        max/min: the existing signed objective, unchanged. target: the deadbanded
        distance to ``target_value``, negated so closer is better, then pushed
        through the SAME objective helper -- so uncertainty handling is identical
        and consistent (``mean_minus_kstd`` becomes ``-e - k*std``, i.e. farther
        from target when less certain, which is the pessimistic reading).
        """
        from ..training import objective_from_mean_std

        if std is None:
            if prop["objective"] != "mean":
                raise EvaluationProtocolError(
                    f"Property {prop['name']!r} requests {prop['objective']} but its selected uncertainty is unavailable")
            std = 0.0  # Internal arithmetic for a mean-only reward, not a scientific uncertainty.
        tgt = prop["target"]
        if tgt is None:
            return objective_from_mean_std(
                prop["direction"] * float(mean), float(std), prop["objective"], self.k
            )
        e = max(0.0, abs(float(mean) - tgt["value"]) - tgt["tolerance"])
        return objective_from_mean_std(-e, float(std), prop["objective"], self.k)

    def predict(self, candidate: Any) -> Tuple[float, None]:
        stats = self._raw_stats(candidate)
        reward = self._combine(stats)
        return reward, None

    def predict_raw(self, candidate: Any) -> Tuple[float, None]:
        """Unpenalized objective with the same gates — for the dp_mean column."""
        return float(self.raw_combine(self._raw_stats(candidate))), None

    def per_objective_stats(self, candidate: Any) -> Dict[str, Tuple[float, Optional[float]]]:
        """Return mean and selected spread (None when unavailable) for each objective."""
        stats = self._raw_stats(candidate)
        return _Statistics(stats, outputs=dict(getattr(stats, "outputs", {})))

    def predict_outputs(self, candidate: Any) -> Dict[str, ScientificOutput]:
        """Return physical named outputs from the exact cached/winning evaluation."""
        return dict(self._evaluation(candidate).outputs)

    def batch_predict_outputs(self, candidates: List[Any]) -> List[Dict[str, ScientificOutput]]:
        candidates = list(candidates)
        self.batch_predict(candidates)
        return [dict(self._evaluation(candidate).outputs) for candidate in candidates]

    def cached_outputs(self, candidate: Any) -> Optional[Dict[str, ScientificOutput]]:
        """Read the existing paired record only; a miss never evaluates a model."""
        key = self._key(candidate)
        record = self._batch_stats.get(key, self._stats_cache.get(key))
        return None if record is None else dict(record.outputs)

    def sequence_constraints(self, group_name: str, tokens: List[str]) -> list[dict]:
        """Combine domain-owned count requirements before constructing an env.

        Token chemistry belongs to the leaf; the environment enforces the
        returned token sets. Every property/model requirement applies jointly.
        """
        constraints = []
        for prop in self.properties:
            for leaf in prop.get("instance") or []:
                hook = getattr(leaf, "sequence_constraints", None)
                if not callable(hook):
                    continue
                bounds = hook(group_name, list(tokens))
                if not isinstance(bounds, list) or any(not isinstance(b, dict) for b in bounds):
                    raise ValueError("sequence_constraints must return a list of count-constraint mappings")
                for bound in bounds:
                    if bound not in constraints:
                        constraints.append(deepcopy(bound))
        return constraints

    def candidate_metadata(self, candidate: Any) -> dict:
        """Return diagnostics paired with the cached prediction, never requery a leaf.

        A single objective/model preserves its field names. Multiple objectives
        or models are namespaced by objective name and model index. A batch keeps
        complete records until its consumers finish, even with a smaller LRU.
        """
        return deepcopy(self._evaluation(candidate).metadata)

    def _collect_candidate_metadata(self, candidate: Any) -> dict:
        """Snapshot leaf diagnostics immediately after scoring this candidate.

        A leaf's optional hook must describe its matching prediction, without
        performing an independent stochastic prediction. Snapshotting here keeps
        later cache eviction or another candidate from changing the record.
        """
        result = {}
        for prop in self.properties:
            leaves = prop.get("instance") or []
            model_metadata = []
            has_metadata = False
            for index, leaf in enumerate(leaves):
                snapshot_key = (prop["name"], index)
                if snapshot_key in self._active_leaf_metadata:
                    metadata = self._active_leaf_metadata[snapshot_key]
                    if metadata is None:
                        model_metadata.append({})
                        continue
                    if len(self.properties) == 1 and len(leaves) == 1:
                        return deepcopy(metadata)
                    model_metadata.append(deepcopy(metadata))
                    has_metadata = True
                    continue
                hook = getattr(leaf, "candidate_metadata", None)
                if not callable(hook):
                    model_metadata.append({})
                    continue
                metadata = hook(candidate)
                if not isinstance(metadata, dict):
                    raise ValueError("candidate_metadata must return a mapping")
                if len(self.properties) == 1 and len(leaves) == 1:
                    return deepcopy(metadata)
                model_metadata.append(metadata)
                has_metadata = True
            if has_metadata:
                result[prop["name"]] = model_metadata
        return deepcopy(result)

    def evaluation_batch_size(self) -> int:
        """Suggested candidate wave size across all declared sources."""
        sizes = [1]
        for spec in self.source_plan.sources:
            recommend = getattr(self._source_builders[spec.source_id],
                                "recommended_candidate_batch_size", None)
            if callable(recommend):
                sizes.append(recommend(spec.n_random_configs))
        return max(sizes)

    def batch_predict(
        self, candidates: List[Any], *, return_exceptions: bool = False,
    ) -> List[Any]:
        """Build unique uncached candidates together; score them in input order.

        GPU workers only handle construction/relaxation. Property inference and
        cache mutation remain in the parent process. A failed candidate can be
        returned as an exception without discarding other candidates' results.
        """
        self._validate_source_assets()
        candidates = list(candidates)
        self.last_evaluation_timings = [_empty_timing() for _ in candidates]
        if not candidates:
            return []
        # Preserve recent diagnostics while checking for hits in this new wave.
        previous_batch = self._batch_stats
        self._batch_stats = {}
        results: Dict[tuple, Any] = {}
        pending: Dict[tuple, Any] = {}
        keys: List[Any] = []
        timings = {}
        for candidate in candidates:
            try:
                key = self._key(candidate)
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                keys.append(exc)
                continue
            keys.append(key)
            try:
                if key in results or key in pending:
                    continue
                record = self._stats_cache.get(key, previous_batch.get(key))
                if record is None:
                    pending[key] = candidate
                else:
                    results[key] = self._prediction_from_stats(record.stats)
                    timings[key] = _empty_timing(cache_hit=True)
                    self._batch_stats[key] = record
                    if key in self._stats_cache:
                        self._stats_cache.move_to_end(key)
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                results[key] = exc

        if pending:
            self._materialization_timings = [_empty_timing() for _ in pending]
            try:
                inputs = self._materialize_structures_many(list(pending.values()))
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                # A pool crash/timeout closes that pool. Nothing from an
                # incomplete wave is misreported as a successful evaluation.
                inputs = [exc] * len(pending)
            for index, ((key, _candidate), prop_inputs) in enumerate(zip(pending.items(), inputs)):
                timing = dict(self._materialization_timings[index])
                timings[key] = timing
                try:
                    if isinstance(prop_inputs, Exception):
                        raise prop_inputs
                    started = time.perf_counter()
                    try:
                        record = self._evaluate_inputs(
                            prop_inputs, _candidate, self._materialization_diagnostics[index])
                    finally:
                        timing["property_s"] = time.perf_counter() - started
                    results[key] = self._prediction_from_stats(record.stats)
                    self._cache_evaluation(key, record)
                    self._batch_stats[key] = record
                except EvaluationProtocolError:
                    raise
                except Exception as exc:
                    results[key] = exc
        seen = set()
        for index, key in enumerate(keys):
            if isinstance(key, Exception):
                continue
            if key in seen:
                # A coalesced duplicate did not launch another physical task.
                self.last_evaluation_timings[index] = _empty_timing(
                    cache_hit=not isinstance(results[key], Exception))
            else:
                self.last_evaluation_timings[index] = timings.get(key, _empty_timing())
                seen.add(key)
        out = [key if isinstance(key, Exception) else results[key] for key in keys]
        if not return_exceptions:
            for result in out:
                if isinstance(result, Exception):
                    raise result
        return out

    def _prediction_from_stats(self, stats: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        reward = self._combine(stats)
        return reward, None

    def close(self) -> None:
        """Close every owned resource exactly once, including failed constructors."""
        if self._closed:
            return
        self._closed = True
        seen = set()
        # Resources are registered immediately after construction, before any
        # subsequent validation or initialization can fail.
        for resource in reversed(self._resources):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def source_identity(self) -> Dict[str, Any]:
        """Frozen declarations and the resolved seed for evaluation provenance."""
        from ..structure_sources import SOURCE_SEED_VERSION
        return {"eval_seed": self._seed, "seed_version": SOURCE_SEED_VERSION,
                "sources": [spec.to_schema() for spec in self.source_plan.sources],
                "property_sources": self.source_plan.property_sources,
                "migration_notes": list(self.source_plan.migration_notes),
                "asset_resolution": deepcopy(self._source_asset_references),
                "asset_coverage": "declared known fields; external builder fields may require an adapter"}

    def _freeze_model_assets(self):
        from pathlib import Path
        from ..utils.provenance import sha256_file
        self._model_assets = {}
        for model_id, definition in self._model_declarations.items():
            references = []
            predictor = str(definition.get("predictor", ""))
            for field_name in ("model", "models", "dp_models"):
                declared = definition.get(field_name)
                paths = declared if isinstance(declared, (list, tuple)) else [declared]
                for item in paths:
                    if not isinstance(item, str):
                        continue
                    path = Path(item).expanduser()
                    reference = {"field": field_name, "declared": item}
                    if predictor.startswith("dp_") or path.is_absolute() or predictor in ("rf_magpie", "lookup"):
                        path = path.resolve()
                        reference["path"] = str(path)
                        if path.is_file():
                            reference.update(status="resolved", sha256=sha256_file(path), size=path.stat().st_size)
                            stat = path.stat()
                            self._source_asset_snapshots[str(path)] = (
                                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                        else:
                            reference["status"] = "missing"
                    else:
                        reference.update(path=None, status="unresolved")
                    references.append(reference)
            self._model_assets[model_id] = references

    def cost_counters(self) -> Dict[str, Any]:
        """Read cumulative backend counters; shared instances are counted once.

        A DP ensemble evaluation is one helper invocation for one candidate and
        source/condition. Model/configuration pairs are requested scientific
        evaluations, not a count of network forward calls or GPU kernel launches.
        Unknown network work is None, never an invented zero.
        """
        fields = ("requests", "model_batches", "model_candidates", "cache_hits", "batch_duplicates")
        leaves, seen = [], {}
        for prop in self.properties:
            for index, leaf in enumerate(prop.get("instance") or []):
                if id(leaf) in seen:
                    record = seen[id(leaf)]
                    record["properties"].append(prop["name"])
                    continue
                hook = getattr(leaf, "cache_info", None)
                info = hook() if callable(hook) else None
                counters = {name: (info.get(name) if isinstance(info, dict) else None) for name in fields}
                record = {"model": prop["model_key"], "member": index,
                          "properties": [prop["name"]],
                          "status": "reported" if isinstance(info, dict) else "unknown",
                          "counters": deepcopy(counters)}
                seen[id(leaf)] = record
                leaves.append(record)
        return {"schema_version": 1, "leaf_backends": leaves, "dp": dict(self._dp_cost),
                "dp_counter_definition": "ensemble helper attempts and requested model/configuration pairs; network forwards unknown"}

    def _count_dp_work(self, prop, structures):
        self._dp_cost["ensemble_evaluation_calls"] += 1
        self._dp_cost["model_configuration_pairs_attempted"] += len(prop["models"]) * len(structures)

    def oracle_identity(self) -> Dict[str, Any]:
        """Physical sources, model declarations and frozen asset evidence, without rewards."""
        self._validate_source_assets()
        adapters = {}
        for prop in self.properties:
            key = prop["model_key"]
            if key in adapters:
                continue
            evidence = []
            for leaf in prop.get("instance") or []:
                hook = getattr(leaf, "oracle_identity", None)
                identity = hook() if callable(hook) else getattr(leaf, "identity", None)
                if callable(identity):
                    identity = identity()
                evidence.append(deepcopy(identity) if identity is not None else {
                    "adapter": f"{type(leaf).__module__}:{type(leaf).__qualname__}",
                    "coverage": "declared configuration only; external dependencies are unverified"})
            adapters[key] = evidence
        return {"schema_version": 1, "engine": "structure_score", "sources": self.source_identity(),
                "models": deepcopy(self._model_declarations), "model_assets": deepcopy(self._model_assets),
                "adapters": adapters,
                "outputs": {prop["name"]: {key: deepcopy(prop[key]) for key in
                    ("model_key", "output", "unit", "representation", "transform", "fparam", "aparam",
                     "head", "output_index", "output_aggregator", "energy_per_atom")} for prop in self.properties},
                "coverage": "Known model/template files are hashed at initialization; unknown adapter dependencies require adapter evidence"}

    def _validate_source_assets(self) -> None:
        from pathlib import Path
        if str(Path.cwd()) != self._source_runtime_cwd and any(
            reference.get("path") and not Path(reference["declared"]).is_absolute()
            for references in [*self._source_asset_references.values(), *getattr(self, "_model_assets", {}).values()]
            for reference in references
        ):
            raise EvaluationProtocolError("Runtime directory changed after source initialization; create a new predictor")
        for references in [*self._source_asset_references.values(), *getattr(self, "_model_assets", {}).values()]:
            for reference in references:
                if reference.get("status") == "missing" and reference.get("path") and Path(reference["path"]).exists():
                    raise EvaluationProtocolError("A previously missing oracle asset appeared after initialization; create a new predictor")
        for path, expected in self._source_asset_snapshots.items():
            try:
                stat = Path(path).stat()
                current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            except OSError as exc:
                raise EvaluationProtocolError(f"Structure source asset disappeared: {path}") from exc
            if current != expected:
                raise EvaluationProtocolError(f"Structure source asset changed after initialization: {path}; create a new predictor")

    def source_diagnostics(self, candidate: Any) -> Dict[str, Any]:
        """Sample/build/relax diagnostics paired with the cached statistics."""
        return deepcopy(self._evaluation(candidate).source_diagnostics)

    def composition_formula(self, candidate: Any) -> Optional[str]:
        """A unique physical formula is available only with one source."""
        if len(self.source_plan.sources) != 1:
            return None
        builder = self._source_builders[self.source_plan.sources[0].source_id]
        fn = getattr(builder, "composition_formula", None)
        return fn(candidate) if callable(fn) else None

    def score_breakdown(self, candidate: Any) -> Dict[str, Any]:
        """Diagnostic: per-sweep-value property stats + combined reward.

        Builds (and relaxes) the structure **once**, then scores every property
        at each sweep value (or just once if no sweep). Returns::

            {"rows": [{"value": v, "stats": {name: (mean, std)}, "reward": r}, ...],
             "best": {"value": v*, "stats": {...}, "reward": r*},
             "structures": {prop_name: [relaxed Atoms, ...]}}

        ``value`` is ``None`` when no sweep is configured. ``structures`` are the
        exact (relaxed) cells that were scored, so a caller can save/inspect them
        without re-building or re-relaxing. Used by ``scripts/evaluate_lips.py``.
        """
        self._validate_source_assets()
        prop_structures = self._materialize_structures(candidate)
        diagnostics = deepcopy(self._materialization_diagnostics[0])
        values = self.sweep["values"] if self.sweep is not None else [None]
        rows: List[Dict[str, Any]] = []
        self._evaluation_candidate = self._canonical_candidate(candidate)
        self._output_memo = {}
        self._metadata_memo = {}
        for v in values:
            self._active_outputs = {}
            self._active_leaf_metadata = {}
            if v is None:                       # no sweep — score each property once
                stats = {
                    p["name"]: self._score(p, prop_structures[p["name"]])
                    for p in self.properties
                }
            else:
                stats = {
                    p["name"]: self._score(
                        p, prop_structures[p["name"]], fparam=self._fparam_at(p, v)
                    )
                    for p in self.properties
                }
            stats = _Statistics(stats, outputs=self._active_outputs.copy())
            details = self._reward_details(stats)
            rows.append({
                "value": v, "stats": stats, "reward": details["reward"],
                "reward_details": details, "outputs": self._active_outputs.copy(),
            })
        best = max(rows, key=lambda r: r["reward"])
        return {"rows": rows, "best": best, "structures": prop_structures, "sources": diagnostics}

    def raw_combine(self, stats: Dict[str, Tuple[float, float]]) -> float:
        """Apply the configured reward rule without objective std penalties.

        For the default weighted sum this preserves the historical dp_mean.
        For gated rewards it retains classification mean-minus-std scores and
        the same failure branch; individual physical means remain available
        through per_objective_stats.
        """
        return self._reward_details(stats, raw=True)["reward"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _materialize_structures(self, candidate: Any) -> Dict[str, Any]:
        """Build each source once, then route its ensemble to its properties."""
        self._materialization_timings = [_empty_timing()]
        self._materialization_diagnostics = [{}]
        self._preload_parallel_models()
        inputs = {p["name"]: candidate for p in self.properties if p["backend"] == "composition"}
        for spec in self.source_plan.sources:
            try:
                structures = self._build_structures(spec.source_id, candidate)
            finally:
                self._record_source_timing(0, spec.source_id, self._last_build_timing)
            for name, source_id in self._property_sources.items():
                if source_id == spec.source_id:
                    inputs[name] = structures
        return inputs

    def _record_source_timing(self, index, source_id, record):
        target = self._materialization_timings[index]
        _add_timing(target, record)
        target.setdefault("sources", {})[source_id] = deepcopy(record)
        self._materialization_diagnostics[index][source_id] = deepcopy(self._last_build_diagnostics)

    def _geo_opt_enabled(self, source_id):
        go = self.source_plan.source(source_id).geo_opt
        return bool(go) and bool(go.get("enabled", True))

    @staticmethod
    def _canonical_candidate(candidate):
        """Give builders a detached mapping in identity order; retain positions."""
        from collections.abc import Mapping
        from ..utils.composition import composition_key
        if isinstance(candidate, Mapping):
            return {key: StructureScorePredictor._canonical_candidate(candidate[key])
                    for key in sorted(candidate, key=lambda k: repr(composition_key(k)))}
        if isinstance(candidate, (list, tuple)):
            values = [StructureScorePredictor._canonical_candidate(v) for v in candidate]
            return tuple(values) if isinstance(candidate, tuple) else values
        return deepcopy(candidate)

    def _stage_summary(self, source_id, builder, capture, elapsed, attempted, succeeded):
        from ..utils.stage_timing import summarize_stage_times
        cfg = self.source_plan.source(source_id).effective_config
        supported = getattr(builder, "timing_stages", ()) or ()
        supported = {supported} if isinstance(supported, str) else set(supported)
        geo_opt_enabled = self._geo_opt_enabled(source_id)
        if geo_opt_enabled and not getattr(builder, "handles_geo_opt", False):
            supported.add("opt")
        return summarize_stage_times(
            capture, elapsed,
            md_expected=bool(cfg.get("md") or getattr(builder, "md", None)),
            opt_expected=bool(geo_opt_enabled or cfg.get("opt") or getattr(builder, "opt", None)),
            supported_stages=supported, config_attempted=attempted, config_succeeded=succeeded)

    @staticmethod
    def _unknown_build_timing():
        return {"md_task_s": None, "opt_task_s": None, "structure_other_task_s": None,
                "config_attempted": None, "config_succeeded": 0, "timing_incomplete": True}

    def _build_structures(self, source_id: str, candidate: Any) -> List[Any]:
        from ..utils.structure import strict_relaxation_errors
        from ..utils.stage_timing import capture_stage_times
        spec = self.source_plan.source(source_id)
        builder = self._source_builders[source_id]
        candidate = self._canonical_candidate(candidate)
        self._last_build_timing = {}
        self._last_build_diagnostics = {}
        self._last_relaxation_failures = []
        self._last_surviving_samples = []
        native_records = None
        succeeded = 0
        started = time.perf_counter()
        with capture_stage_times() as capture:
            try:
                with strict_relaxation_errors():
                    seeded = getattr(builder, "build_seeded", None)
                    if callable(seeded):
                        structures = seeded(candidate, n_configs=spec.n_random_configs,
                                            source_spec=spec, eval_seed=self._seed)
                    elif callable(getattr(builder, "build_sample", None)):
                        from ..utils.structure_samples import execute_seeded_samples
                        structures = execute_seeded_samples(
                            builder, candidate, n_configs=spec.n_random_configs,
                            source_spec=spec, eval_seed=self._seed)
                    elif getattr(builder, "parallel_execution", False):
                        raise EvaluationProtocolError("Parallel structure builders require build_seeded/source sample-plan support")
                    else:
                        # Preserve an external builder's native ensemble call;
                        # a single source-local RNG cannot mix different queries.
                        rng = np.random.default_rng(spec.seed_for(candidate, eval_seed=self._seed))
                        structures = list(builder.build(candidate, n_configs=spec.n_random_configs, rng=rng))
                        native_records = self._annotate_native_samples(spec, candidate, structures)
                structures = self._usable_structures(source_id, structures)
                succeeded = len(structures)
                return structures
            finally:
                self._last_build_diagnostics = self._sample_diagnostics(builder, 0, native_records)
                if getattr(builder, "parallel_execution", False):
                    records = getattr(builder, "last_build_timings", [])
                    self._last_build_timing = (dict(records[0]) if len(records) == 1
                                               else self._unknown_build_timing())
                    self._last_build_timing["config_succeeded"] = succeeded
                else:
                    attempted = self._attempted_samples(self._last_build_diagnostics)
                    self._last_build_timing = self._stage_summary(
                        source_id, builder, capture, time.perf_counter() - started, attempted, succeeded)

    def _usable_structures(self, source_id: str, structures: Any) -> List[Any]:
        from ..utils.structure import strict_relaxation_errors
        builder = self._source_builders[source_id]
        structures = list(structures)
        self._last_relaxation_failures = []
        self._last_surviving_samples = []
        if self._geo_opt_enabled(source_id) and not getattr(builder, "handles_geo_opt", False):
            successful = []
            for index, atoms in enumerate(structures):
                try:
                    with strict_relaxation_errors():
                        successful.append(self._relax(atoms, source_id=source_id))
                except EvaluationProtocolError:
                    raise
                except Exception as exc:
                    record = deepcopy(getattr(atoms, "info", {}).get("_rl_matdesign_sample",
                                      {"source_id": source_id, "returned_index": index}))
                    record.update(status="relaxation_failed", error=f"{type(exc).__name__}: {exc}")
                    self._last_relaxation_failures.append(record)
                    print(f"[structure_score] source {source_id!r} config {index + 1}/{len(structures)} "
                          f"relaxation failed, skipping: {type(exc).__name__}: {exc}")
            structures = successful
        self._last_surviving_samples = [deepcopy(getattr(atoms, "info", {}).get("_rl_matdesign_sample",
                                       {"source_id": source_id, "returned_index": index}))
                                       for index, atoms in enumerate(structures)]
        if not structures:
            raise ValueError("No usable configurations remain for candidate")
        return structures

    def _preload_parallel_models(self) -> None:
        structure_props = [p for p in self.properties if p["backend"] != "composition"]
        if not self._parallel_models_ready and any(
            getattr(builder, "parallel_execution", False) for builder in self._source_builders.values()
        ):
            for prop in structure_props:
                if prop["backend"] == "energy":
                    self._get_calculators(prop)
                elif prop["backend"] in ("property", "classification"):
                    self._get_prop_models(prop)
            self._parallel_models_ready = True

    def _materialize_structures_many(self, candidates: List[Any]) -> List[Any]:
        self._materialization_timings = [_empty_timing() for _ in candidates]
        self._materialization_diagnostics = [{} for _ in candidates]
        self._preload_parallel_models()
        inputs: List[Any] = [
            {p["name"]: candidate for p in self.properties if p["backend"] == "composition"}
            for candidate in candidates
        ]
        for spec in self.source_plan.sources:
            source_id = spec.source_id
            builder = self._source_builders[source_id]
            active = [i for i, entry in enumerate(inputs) if not isinstance(entry, Exception)]
            if not active:
                break
            build_many = getattr(builder, "build_many_seeded", None)
            if not callable(build_many) and callable(getattr(builder, "build_sample", None)):
                from functools import partial
                from ..utils.structure_samples import execute_many_seeded_samples
                build_many = partial(execute_many_seeded_samples, builder)
            if not callable(build_many):
                # A legacy build_many accepting one advancing RNG is not a
                # reproducible batch capability. Keep its native scalar calls.
                for i in active:
                    try:
                        structures = self._build_structures(source_id, candidates[i])
                        for name, reference in self._property_sources.items():
                            if reference == source_id:
                                inputs[i][name] = structures
                    except EvaluationProtocolError:
                        raise
                    except Exception as exc:
                        inputs[i] = exc
                    finally:
                        self._record_source_timing(i, source_id, self._last_build_timing)
                continue
            from ..utils.structure import strict_relaxation_errors
            from ..utils.stage_timing import capture_stage_times
            try:
                with strict_relaxation_errors():
                    built = build_many([self._canonical_candidate(candidates[i]) for i in active],
                                       n_configs=spec.n_random_configs, source_spec=spec,
                                       eval_seed=self._seed)
                if len(built) != len(active):
                    raise EvaluationProtocolError("build_many_seeded must return one result per candidate")
            except EvaluationProtocolError:
                raise
            except Exception as exc:
                built = [exc] * len(active)
            records = getattr(builder, "last_build_timings", [])
            for offset, (i, structures) in enumerate(zip(active, built)):
                self._last_relaxation_failures = []
                self._last_surviving_samples = []
                record = (dict(records[offset]) if len(records) == len(active)
                          else self._unknown_build_timing())
                try:
                    if isinstance(structures, Exception):
                        raise structures
                    started = time.perf_counter()
                    survived = 0
                    with capture_stage_times() as capture:
                        try:
                            structures = self._usable_structures(source_id, structures)
                            survived = len(structures)
                        finally:
                            if record["opt_task_s"] is not None:
                                record["opt_task_s"] += capture.seconds.get("opt", 0.0)
                            if record["structure_other_task_s"] is not None:
                                record["structure_other_task_s"] += max(
                                    0.0, time.perf_counter() - started - capture.seconds.get("opt", 0.0))
                            record["config_succeeded"] = survived
                    for name, reference in self._property_sources.items():
                        if reference == source_id:
                            inputs[i][name] = structures
                except EvaluationProtocolError:
                    raise
                except Exception as exc:
                    inputs[i] = exc
                finally:
                    self._last_build_diagnostics = self._sample_diagnostics(builder, offset)
                    if record.get("config_attempted") is None:
                        record["config_attempted"] = self._attempted_samples(self._last_build_diagnostics)
                    self._record_source_timing(i, source_id, record)
        return inputs

    def _annotate_native_samples(self, spec, candidate, structures):
        """Native plugins expose returned samples, not individual task attempts."""
        records = []
        for index, atoms in enumerate(structures):
            record = {"source_id": spec.source_id, "source_fingerprint": spec.protocol_fingerprint,
                      "sample_index": index, "template_id": None,
                      "seed": spec.seed_for(candidate, eval_seed=self._seed),
                      "seed_scope": "native_ensemble", "identity_scope": "returned_ensemble"}
            info = getattr(atoms, "info", None)
            if isinstance(info, dict):
                previous = info.get("_rl_matdesign_sample")
                if previous is not None and previous != record:
                    raise EvaluationProtocolError("External builder returned conflicting _rl_matdesign_sample identity")
                info["_rl_matdesign_sample"] = deepcopy(record)
            records.append({**record, "status": "built"})
        return records

    @staticmethod
    def _attempted_samples(diagnostics):
        if diagnostics["sample_identity"] != "explicit_plan":
            return None
        statuses = [sample.get("status") for sample in diagnostics["samples"]]
        known = {"succeeded", "failed", "interrupted", "not_attempted", "plan_failed"}
        if any(status not in known for status in statuses):
            return None
        return sum(status in {"succeeded", "failed", "interrupted"} for status in statuses)

    def _sample_diagnostics(self, builder, offset, native_records=None):
        batches = getattr(builder, "last_sample_records", [])
        explicit = native_records is None and offset < len(batches)
        records = batches[offset] if explicit else (native_records or [])
        return {"sample_identity": "explicit_plan" if explicit else "native_ensemble",
                "samples": deepcopy(records),
                "parent_relaxation_failures": deepcopy(self._last_relaxation_failures),
                "surviving_samples": deepcopy(self._last_surviving_samples)}

    def _raw_stats(self, candidate: Any) -> Dict[str, Tuple[float, float]]:
        return self._evaluation(candidate).stats

    def _evaluation(self, candidate: Any) -> _EvaluationRecord:
        self._validate_source_assets()
        timing = _empty_timing()
        self.last_evaluation_timings = [timing]
        key = self._key(candidate)
        cached = self._stats_cache.get(key)
        if cached is not None:
            self._stats_cache.move_to_end(key)
            timing["cache_hit"] = True
            return cached
        if key in self._batch_stats:
            timing["cache_hit"] = True
            return self._batch_stats[key]

        self._materialization_timings = [_empty_timing()]
        try:
            prop_structures = self._materialize_structures(candidate)
        finally:
            _add_timing(timing, self._materialization_timings[0])
        started = time.perf_counter()
        try:
            record = self._evaluate_inputs(prop_structures, candidate, self._materialization_diagnostics[0])
        finally:
            timing["property_s"] = time.perf_counter() - started
        self._cache_evaluation(key, record)
        return record

    def _evaluate_inputs(self, prop_structures: Dict[str, Any], candidate: Any,
                         source_diagnostics: Optional[Dict[str, Any]] = None) -> _EvaluationRecord:
        self._output_memo = {}
        self._metadata_memo = {}
        self._evaluation_candidate = self._canonical_candidate(candidate)
        best_record = None
        best_reward = None
        values = self.sweep["values"] if self.sweep is not None else [None]
        for value in values:
            self._active_outputs = {}
            self._active_leaf_metadata = {}
            stats = _Statistics({p["name"]: (self._score(p, prop_structures[p["name"]])
                if value is None else self._score(p, prop_structures[p["name"]], fparam=self._fparam_at(p, value)))
                for p in self.properties}, outputs=self._active_outputs.copy())
            reward = self._combine(stats)
            if best_reward is None or reward > best_reward:
                best_reward = reward
                if self.sweep is not None:
                    stats[self.sweep["name"]] = (float(value), None)
                best_record = _EvaluationRecord(stats, self._collect_candidate_metadata(candidate),
                                                deepcopy(source_diagnostics or {}),
                                                self._active_outputs.copy())
        return best_record

    def _cache_evaluation(self, key: tuple, record: _EvaluationRecord) -> None:
        self._stats_cache[key] = record
        cap = self._stats_cache_size
        if cap and cap > 0:
            while len(self._stats_cache) > cap:
                self._stats_cache.popitem(last=False)   # evict least-recently used

    @staticmethod
    def _fparam_at(prop: Dict[str, Any], value: float) -> Any:
        """Return *prop*'s fparam with its ``null`` slots filled by *value*."""
        if not prop["sweep_slots"]:
            return prop["fparam"]      # T-independent property (or no fparam)
        out = list(prop["fparam"])
        for j in prop["sweep_slots"]:
            out[j] = value
        return out

    def _score(
        self, prop: Dict[str, Any], structures: Any, *, fparam: Any = None,
    ) -> Tuple[float, Optional[float]]:
        output, default_uncertainty = self._score_output(prop, structures, fparam=fparam)
        self._active_outputs[prop["name"]] = output
        kind = prop["uncertainty"] or default_uncertainty
        return output.mean, getattr(output, kind)

    @staticmethod
    def _detached_payload(payload, *, is_structure):
        if not is_structure:
            return deepcopy(payload)
        detached = []
        for atoms in payload:
            if callable(getattr(atoms, "get_chemical_symbols", None)):
                cell = atoms.copy()
                cell.info = deepcopy(atoms.info)
                cell.calc = None
            else:
                cell = deepcopy(atoms)
            detached.append(cell)
        return detached

    def _configuration_ids(self, prop, payload):
        if prop["backend"] == "composition":
            return ("candidate",)
        import json
        source = self._property_sources.get(prop["name"], prop["name"])
        ids = []
        for index, cell in enumerate(payload):
            sample = getattr(cell, "info", {}).get("_rl_matdesign_sample")
            ids.append(json.dumps(sample, sort_keys=True, separators=(",", ":"))
                       if sample is not None else f"{source}:returned:{index}")
        return tuple(ids)

    def _raw_schema(self, prop):
        # A legacy transform declares its input representation. A named adapter's
        # schema is authoritative and is validated separately below.
        return OutputSchema(prop["name"], None if prop["transform"] in ("exp", "pow10") else prop["unit"],
                            prop["representation"])

    def _transform_output(self, prop, output):
        transform = prop["transform"]
        if prop["representation_declared"] and output.schema.representation != prop["representation"]:
            raise EvaluationProtocolError(f"Property {prop['name']!r} representation conflicts with adapter output schema")
        if transform in ("none", "identity"):
            if prop["unit_declared"] and output.schema.unit != prop["unit"]:
                raise EvaluationProtocolError(f"Property {prop['name']!r} unit conflicts with adapter output schema")
        else:
            output = inverse_transform_output(output, transform, unit=prop["unit"])
        return replace(output, schema=replace(output.schema, name=prop["name"]))

    @staticmethod
    def _scientific_contract(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            raise EvaluationProtocolError(f"Invalid scientific output: {exc}") from exc

    def _score_output(self, prop, payload, *, fparam=None):
        if prop["backend"] in ("composition", "structure_fqn"):
            output, default = self._score_leaf_output(prop, payload, fparam=fparam)
        else:
            output = self._score_dp_output(prop, payload, fparam=fparam)
            default = "model_spread"
        return self._scientific_contract(self._transform_output, prop, output), default

    def _score_dp_output(self, prop, structures, *, fparam=None):
        if prop["backend"] == "energy":
            from ..utils.dp_eval import eval_energy_ase
            calculators = self._get_calculators(prop)
            self._count_dp_work(prop, structures)
            values = eval_energy_ase(structures, calculators,
                                     energy_per_atom=prop["energy_per_atom"])
        elif prop["backend"] == "property" and prop["model_ref"] is not None:
            from ..utils.dp_eval import eval_property_outputs, pick_scalar
            frame = fparam if fparam is not None else prop["fparam"]
            key = ("dp_outputs", prop["model_key"], self._property_sources.get(prop["name"]),
                   self._key({"fparam": frame, "aparam": prop["aparam"]}))
            if key not in self._output_memo:
                models, elem_to_type = self._get_prop_models(prop)
                self._count_dp_work(prop, structures)
                self._output_memo[key] = eval_property_outputs(
                    structures, models, elem_to_type, fparam=frame, aparam=prop["aparam"])
            raw = self._output_memo[key]
            values = [[self._scientific_contract(pick_scalar, vector, output_index=prop["output_index"],
                       output_aggregator=prop["output_aggregator"]) for vector in row] for row in raw]
        else:
            from ..utils.dp_eval import eval_classification_ensemble, eval_property_ensemble
            models, elem_to_type = self._get_prop_models(prop)
            evaluate = (eval_classification_ensemble if prop["backend"] == "classification"
                        else eval_property_ensemble)
            self._count_dp_work(prop, structures)
            values = evaluate(structures, models, elem_to_type,
                              output_index=prop["output_index"],
                              output_aggregator=prop["output_aggregator"],
                              fparam=fparam if fparam is not None else prop["fparam"], aparam=prop["aparam"])
        array = self._scientific_contract(np.asarray, values, dtype=float)
        expected = (len(prop["models"]), len(structures))
        if array.shape != expected:
            raise EvaluationProtocolError(f"Expected complete model/configuration matrix {expected}, got {array.shape}")
        ids = self._configuration_ids(prop, structures)
        return self._scientific_contract(aggregate_samples, self._raw_schema(prop), [
            self._scientific_contract(ModelSamples, f"{prop['model_key']}:{index}:{model}", ids, tuple(array[index]))
            for index, model in enumerate(prop["models"])])

    def _score_leaves(self, prop, payload):
        """Compatibility helper; all leaves use the same sample-aware output path."""
        return self._score(prop, payload)

    def _call_leaf(self, method, payload, prop, index, conditions):
        from ..utils.seeding import isolated_rng, candidate_seed
        candidate = getattr(self, "_evaluation_candidate", payload if prop["backend"] == "composition" else None)
        source_id = self._property_sources.get(prop["name"])
        source = ({"id": source_id, "protocol": self.source_plan.source(source_id).protocol_fingerprint}
                  if source_id is not None else None)
        definition = self._model_declarations[prop["model_key"]]
        namespace = {"version": 1, "candidate": candidate, "reference": prop["model_ref"],
                     "model": definition, "member": index, "source": source, "conditions": conditions}
        with isolated_rng(candidate_seed(self._seed, namespace)):
            result = method(payload, **conditions)
            leaf = prop["instance"][index]
            hook = getattr(leaf, "candidate_metadata", None)
            metadata = hook(candidate) if callable(hook) else None
            if metadata is not None and not isinstance(metadata, dict):
                raise EvaluationProtocolError("candidate_metadata must return a mapping")
            self._active_leaf_metadata[(prop["name"], index)] = deepcopy(metadata)
            return result

    def _score_leaf_output(self, prop, payload, *, fparam=None):
        from collections.abc import Mapping
        is_structure = prop["backend"] == "structure_fqn"
        ids = self._configuration_ids(prop, payload)
        sampled, folded, named = [], [], []
        conditions = {key: value for key, value in
                      (("fparam", fparam if fparam is not None else prop["fparam"]),
                       ("aparam", prop["aparam"])) if value is not None}
        for index, leaf in enumerate(prop["instance"]):
            native_outputs = getattr(leaf, "predict_structure_outputs" if is_structure else "predict_outputs", None)
            if callable(native_outputs):
                # A shared model produces all named quantities once for this exact
                # source and operating condition; reward weights/selectors do not
                # belong to the inference key.
                key = (id(leaf), self._property_sources.get(prop["name"]), self._key(conditions))
                if key not in self._output_memo:
                    import inspect
                    detached = self._detached_payload(payload, is_structure=is_structure)
                    try:
                        inspect.signature(native_outputs).bind(detached, **conditions)
                    except (TypeError, ValueError) as exc:
                        raise EvaluationProtocolError("Named-output adapter does not support the declared conditions") from exc
                    results = self._call_leaf(native_outputs, detached, prop, index, conditions)
                    if not isinstance(results, Mapping) or not results:
                        raise EvaluationProtocolError("predict_outputs must return a nonempty named ScientificOutput mapping")
                    for name, output in results.items():
                        if not isinstance(output, ScientificOutput) or name != output.schema.name:
                            raise EvaluationProtocolError("Named output keys must match ScientificOutput schema names")
                    declared = getattr(leaf, "output_schema", None)
                    if declared is not None and (not isinstance(declared, Mapping) or
                        set(declared) != set(results) or any(declared[name] != value.schema for name, value in results.items())):
                        raise EvaluationProtocolError("Adapter output_schema does not match its evaluated outputs")
                    self._output_memo[key] = dict(results)
                    self._metadata_memo[key] = deepcopy(self._active_leaf_metadata[(prop["name"], index)])
                self._active_leaf_metadata[(prop["name"], index)] = deepcopy(self._metadata_memo[key])
                results = self._output_memo[key]
                selected = prop["output"]
                if selected is None:
                    if len(results) != 1:
                        raise EvaluationProtocolError(f"Property {prop['name']!r} must select one named output from {sorted(results)}")
                    selected = next(iter(results))
                if selected not in results:
                    raise EvaluationProtocolError(f"Unknown output {selected!r}; available: {sorted(results)}")
                named.append(results[selected])
                continue
            if prop["output"] is not None:
                raise EvaluationProtocolError("output selection requires an adapter implementing predict_outputs")
            if conditions:
                raise EvaluationProtocolError("Legacy leaf interfaces cannot silently ignore fparam/aparam; use predict_outputs")
            detached = self._detached_payload(payload, is_structure=is_structure)
            score = getattr(leaf, "score_structures" if is_structure else "score", None)
            model_id = f"{prop['model_key']}:{index}"
            if callable(score):
                result = self._call_leaf(score, detached, prop, index, {})
                values = tuple(result) if is_structure else (result,)
                sampled.append(self._scientific_contract(ModelSamples, model_id, ids, values))
            else:
                predict = getattr(leaf, "predict_structures" if is_structure else "predict", None)
                if not callable(predict):
                    raise EvaluationProtocolError("Leaf must implement a named-output, score, or predict interface")
                result = self._call_leaf(predict, detached, prop, index, {})
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise EvaluationProtocolError("Legacy predict must return (mean, native_uncertainty)")
                mean, uncertainty = result
                folded.append(self._scientific_contract(ScientificOutput, self._raw_schema(prop), mean,
                                                        native_uncertainty=uncertainty))
        if named:
            if sampled or folded:
                raise EvaluationProtocolError("All members of a model ensemble must implement the same output protocol")
            if len(named) == 1:
                return named[0], "model_spread"
            if any(output.schema != named[0].schema for output in named[1:]):
                raise EvaluationProtocolError("Named ensemble members must declare the same output schema")
            if any(output.samples is None or output.native_uncertainty is not None for output in named):
                raise EvaluationProtocolError("Combining named model instances requires raw samples and no unresolved native uncertainty")
            rows = [replace(row, model_id=f"{index}:{row.model_id}") for index, output in enumerate(named)
                    for row in output.samples]
            return self._scientific_contract(aggregate_samples, named[0].schema, rows), "model_spread"
        if folded:
            if sampled or len(folded) != 1:
                raise EvaluationProtocolError("Multiple self-folding leaves require explicit raw samples; native uncertainties cannot be averaged")
            return folded[0], "native_uncertainty"
        return self._scientific_contract(aggregate_samples, self._raw_schema(prop), sampled), "model_spread"

    def _close_new_resources(self, start: int) -> None:
        earlier = {id(resource) for resource in self._resources[:start]}
        new = self._resources[start:]
        del self._resources[start:]
        for resource in reversed(new):
            if id(resource) in earlier:
                continue
            earlier.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _get_calculators(self, prop: Dict[str, Any]) -> List[Any]:
        name = prop["model_key"]
        if name not in self._calc_cache:
            from ..utils.dp_eval import load_ase_calculators
            owned_start = len(self._resources)
            try:
                calculators = []
                for model in prop["models"]:
                    calculators.extend(self._own(calc) for calc in
                                       load_ase_calculators([model], head=prop["head"]))
                self._calc_cache[name] = calculators
            except BaseException:
                self._close_new_resources(owned_start)
                raise
        return self._calc_cache[name]

    def _get_prop_models(self, prop: Dict[str, Any]) -> Tuple[List[Any], Dict[str, int]]:
        name = prop["model_key"]
        if name not in self._prop_cache:
            if prop["backend"] == "classification":
                from deepmd.infer import DeepEval
                factory = DeepEval
            else:
                from deepmd.pt.infer.deep_eval import DeepProperty
                factory = DeepProperty
            owned_start = len(self._resources)
            try:
                models = [
                    self._own(factory(model_file=p, auto_batch_size=False, head=prop["head"]))
                    for p in prop["models"]
                ]
                if prop["backend"] == "classification" and any(
                    not callable(getattr(model, "predict_proba", None)) for model in models
                ):
                    raise EvaluationProtocolError(
                        "dp_classification requires a classification head and a DeepMD version "
                        "providing predict_proba()."
                    )
                type_map = list(models[0].get_type_map())
                if any(list(model.get_type_map()) != type_map for model in models[1:]):
                    raise EvaluationProtocolError(f"Property {name!r}: ensemble models must share the same type_map.")
                self._prop_cache[name] = (models, {el: i for i, el in enumerate(type_map)})
            except BaseException:
                self._close_new_resources(owned_start)
                raise
        return self._prop_cache[name]

    def _relax(self, atoms: Any, *, source_id: str) -> Any:
        from ..utils.structure import relax_structure
        go = self.source_plan.source(source_id).geo_opt
        if source_id not in self._geo_calcs:
            from deepmd.calculator import DP as DPCalculator
            head = go.get("head")
            model = go.get("model", "models/DPA-3.1-3M.pt")
            self._geo_calcs[source_id] = self._own(
                DPCalculator(model=model, **({"head": head} if head else {})))
        return relax_structure(
            atoms, calc=self._geo_calcs[source_id],
            fmax=float(go.get("fmax", 0.001)), steps=int(go.get("steps", 1000)),
            relax_cell=bool(go.get("relax_cell", True)),
        )

    @staticmethod

    def _key(candidate: Any) -> tuple:
        """Canonical typed identity for composition, named fields or sequences.

        Mapping insertion order is irrelevant; sequence positions and the
        difference between string IDs and numerical values remain meaningful.
        """
        from ..utils.composition import composition_key

        return composition_key(candidate)
