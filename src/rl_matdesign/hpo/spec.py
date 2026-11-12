"""Strict HPO YAML loading, including compatibility for authored v1 files."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import warnings
from pathlib import Path

from ..config_schema import ConfigError, load_yaml_config, resolve_config
from .algorithm_profiles import METHODS, get_profile, validate_search_space, validate_parameter_value

_TRAIN_WEIGHTS = {"qualified_fraction": .45, "best_fraction_quality": .30,
                  "best_value_quality": .10, "sampling_efficiency": .15}
_GEN_WEIGHTS = {"qualified_fraction": .50, "best_fraction_quality": .40, "best_value_quality": .10}


def _mapping(value, context, allowed):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ConfigError(f"{context} must be a mapping with string keys")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ConfigError(f"Unknown fields in {context}: {sorted(unknown)}")
    return deepcopy(dict(value))


def _number(value, context, minimum=None, maximum=None, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{context} must be a finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ConfigError(f"{context} must be a finite number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{context} must be finite")
    if minimum is not None and (value <= minimum if exclusive else value < minimum):
        raise ConfigError(f"{context} must be {'>' if exclusive else '>='} {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{context} must be <= {maximum}")
    return value


def _integer(value, context, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _boolean(value, context):
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be boolean")
    return value


def _seeds(value, context):
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{context} must be a nonempty list of integer seeds")
    values = [_integer(v, context) for v in value]
    if len(values) != len(set(values)) or any(v >= 2**32 for v in values):
        raise ConfigError(f"{context} must contain distinct seeds < 2**32")
    return values


def _weights(value, context, defaults):
    raw = _mapping(value, context, defaults)
    result = {key: _number(raw.get(key, default), f"{context}.{key}", 0) for key, default in defaults.items()}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ConfigError(f"{context} must sum to 1")
    return result


def phase_efficiency(plan: Mapping, phase: str) -> dict:
    if phase == "training":
        phase = "train"
    if phase not in {"train", "generation"}:
        raise ConfigError("phase must be train or generation")
    settings = plan["efficiency"]
    return {**{key: settings[key] for key in ("max_attempts_per_unique", "comparison", "stop_when_completion_is_impossible")},
            **settings.get(phase, {})}


def _efficiency(raw):
    allowed = {"max_attempts_per_unique", "comparison", "stop_when_completion_is_impossible", "train", "generation"}
    raw = _mapping(raw, "efficiency", allowed)
    result = {
        "max_attempts_per_unique": _number(raw.get("max_attempts_per_unique", 1.5), "efficiency.max_attempts_per_unique", 1, exclusive=True),
        "comparison": raw.get("comparison", "lt"),
        "stop_when_completion_is_impossible": _boolean(raw.get("stop_when_completion_is_impossible", True), "efficiency.stop_when_completion_is_impossible"),
    }
    if result["comparison"] != "lt":
        raise ConfigError("efficiency.comparison must be lt (strict upper limit)")
    for phase in ("train", "generation"):
        overrides = _mapping(raw.get(phase, {}), f"efficiency.{phase}", {"max_attempts_per_unique"})
        result[phase] = {"max_attempts_per_unique": _number(overrides.get("max_attempts_per_unique", result["max_attempts_per_unique"]), f"efficiency.{phase}.max_attempts_per_unique", 1, exclusive=True)}
    return result


def _objective(raw, base):
    raw = _mapping(raw, "objective", {"field", "direction", "threshold", "threshold_inclusive", "unit", "normalization"})
    field = raw.get("field")
    if not isinstance(field, str) or not field.strip() or field != field.strip():
        raise ConfigError("objective.field must be a nonempty named property")
    direction = raw.get("direction")
    if direction not in {"minimize", "maximize"}:
        raise ConfigError("objective.direction must be minimize or maximize")
    threshold = _number(raw.get("threshold"), "objective.threshold")
    unit = raw.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit.strip() or unit != unit.strip()):
        raise ConfigError("objective.unit must be a nonempty string or null")
    matches = [p for p in base.get("properties", []) if p.get("name") == field]
    if base.get("properties") and len(matches) != 1:
        raise ConfigError(f"objective.field {field!r} must identify exactly one property in base_config")
    if matches and matches[0].get("unit") is not None and unit != matches[0]["unit"]:
        raise ConfigError(f"objective.unit must match base_config property unit {matches[0]['unit']!r}; no implicit unit conversion")
    norm = _mapping(raw.get("normalization", {"kind": "positive_threshold_ratio"}), "objective.normalization", {"kind", "scale"})
    kind = norm.get("kind")
    if kind == "positive_threshold_ratio":
        if "scale" in norm:
            raise ConfigError("positive_threshold_ratio does not accept scale")
        if threshold <= 0:
            raise ConfigError("positive_threshold_ratio requires a positive threshold; use sigmoid for zero or negative values")
    elif kind == "sigmoid":
        norm["scale"] = _number(norm.get("scale"), "objective.normalization.scale", 0, exclusive=True)
    else:
        raise ConfigError("objective.normalization.kind must be positive_threshold_ratio or sigmoid")
    return {"field": field, "direction": direction, "threshold": threshold,
            "threshold_inclusive": _boolean(raw.get("threshold_inclusive", False), "objective.threshold_inclusive"),
            "unit": unit, "normalization": norm}


def _budget(raw, objective):
    raw = _mapping(raw, "budget", {"mode", "strategy", "train", "generation", "tolerances", "upper_confirmation_levels", "space_size"})
    mode = raw.get("mode", "fraction_of_space")
    if mode not in {"fraction_of_space", "absolute_unique"}:
        raise ConfigError("budget.mode must be fraction_of_space or absolute_unique")
    if raw.get("strategy", "bidirectional_grid") != "bidirectional_grid":
        raise ConfigError("budget.strategy must be bidirectional_grid")
    result = {"mode": mode, "strategy": "bidirectional_grid"}
    if "space_size" in raw:
        result["space_size"] = _integer(raw["space_size"], "budget.space_size", 1)
    for phase in ("train", "generation"):
        allowed = {"factor", "min_unique"} | ({"initial_fraction", "min_fraction", "max_fraction"} if mode == "fraction_of_space" else {"initial_unique", "max_unique"})
        cfg = _mapping(raw.get(phase, {}), f"budget.{phase}", allowed)
        cfg["factor"] = _number(cfg.get("factor", 2), f"budget.{phase}.factor", 1, exclusive=True)
        cfg["min_unique"] = _integer(cfg.get("min_unique", 100), f"budget.{phase}.min_unique", 1)
        if mode == "fraction_of_space":
            for key, default in (("initial_fraction", .0001), ("min_fraction", .000001), ("max_fraction", .01)):
                cfg[key] = _number(cfg.get(key, default), f"budget.{phase}.{key}", 0, 1, exclusive=True)
            low, initial, high = (cfg[key] for key in ("min_fraction", "initial_fraction", "max_fraction"))
        else:
            cfg["initial_unique"] = _integer(cfg.get("initial_unique"), f"budget.{phase}.initial_unique", 1)
            cfg["max_unique"] = _integer(cfg.get("max_unique"), f"budget.{phase}.max_unique", 1)
            low, initial, high = (cfg[key] for key in ("min_unique", "initial_unique", "max_unique"))
        if not low <= initial <= high:
            raise ConfigError(f"budget.{phase} requires min <= initial <= max")
        result[phase] = cfg
    tol = _mapping(raw.get("tolerances", {}), "budget.tolerances", {"qualified_fraction", "best_fraction_mean", "best_value", "attempts_per_unique", "score"})
    tolerances = {key: _number(tol.get(key, default), f"budget.tolerances.{key}", 0)
                  for key, default in (("qualified_fraction", .02), ("attempts_per_unique", .10), ("score", .01))}
    for key in ("best_fraction_mean", "best_value"):
        value = tol.get(key, {"kind": "threshold_fraction", "value": .005})
        value = _mapping(value, f"budget.tolerances.{key}", {"kind", "value"})
        if value.get("kind") not in {"threshold_fraction", "absolute"}:
            raise ConfigError(f"budget.tolerances.{key}.kind must be threshold_fraction or absolute")
        value["value"] = _number(value.get("value"), f"budget.tolerances.{key}.value", 0)
        if value["kind"] == "threshold_fraction" and objective["threshold"] == 0:
            raise ConfigError("Zero threshold requires absolute quality tolerances")
        tolerances[key] = value
    result["tolerances"] = tolerances
    result["upper_confirmation_levels"] = _integer(raw.get("upper_confirmation_levels", 2), "budget.upper_confirmation_levels", 1)
    return result


def _search_settings(raw, generation=False):
    if generation:
        defaults = {"parameter": "gen_temperature", "initial_values": [0, .001, .01, .1, 1, 10, 100, 1000],
                    "max_extra_values_per_direction": 2, "fixed": {"gen_epsilon": 0, "gen_top_frac": 0, "exploration_gen_eps": 0},
                    "paired_seeds": [401, 503, 607]}
        raw = _mapping(raw, "generation_search", defaults)
        cfg = {**deepcopy(defaults), **raw}
        if cfg["parameter"] != "gen_temperature":
            raise ConfigError("generation_search.parameter must be gen_temperature")
        if not isinstance(cfg["initial_values"], list) or not cfg["initial_values"]:
            raise ConfigError("generation_search.initial_values must be a nonempty list")
        cfg["initial_values"] = [_number(v, "generation_search.initial_values", 0) for v in cfg["initial_values"]]
        if len(set(cfg["initial_values"])) != len(cfg["initial_values"]) or not any(v > 0 for v in cfg["initial_values"]):
            raise ConfigError("generation_search.initial_values requires distinct values and at least one positive temperature")
        cfg["initial_values"].sort()
        fixed = _mapping(cfg["fixed"], "generation_search.fixed", defaults["fixed"])
        cfg["fixed"] = {**defaults["fixed"], **fixed}
        for key, value in cfg["fixed"].items():
            if _number(value, f"generation_search.fixed.{key}") != 0:
                raise ConfigError(f"generation_search.fixed.{key} must be 0; temperature-only generation")
            cfg["fixed"][key] = 0
        cfg["paired_seeds"] = _seeds(cfg["paired_seeds"], "generation_search.paired_seeds")
        cfg["max_extra_values_per_direction"] = _integer(cfg["max_extra_values_per_direction"], "generation_search.max_extra_values_per_direction")
        return cfg
    defaults = {"trials": 30, "startup_random_trials": 10, "seeds": [11, 29], "finalists": 3,
                "validation_seeds": [101, 211, 307], "max_budget_revisions": 2,
                "additional_trials_per_revision": 10, "additional_random_trials": 5}
    raw = _mapping(raw, "training_search", defaults)
    cfg = {**defaults, **raw}
    for key in ("seeds", "validation_seeds"):
        cfg[key] = _seeds(cfg[key], f"training_search.{key}")
    if set(cfg["seeds"]) & set(cfg["validation_seeds"]):
        raise ConfigError("training_search.validation_seeds must be fresh, disjoint from training_search.seeds")
    for key in set(defaults) - {"seeds", "validation_seeds"}:
        cfg[key] = _integer(cfg[key], f"training_search.{key}", 0 if key in {"startup_random_trials", "max_budget_revisions", "additional_random_trials"} else 1)
    if cfg["startup_random_trials"] > cfg["trials"] or cfg["additional_random_trials"] > cfg["additional_trials_per_revision"]:
        raise ConfigError("Random startup trial counts must not exceed corresponding total trials")
    return cfg


_PATH_KEYS = {"model", "checkpoint", "model_path", "weights", "weights_path", "scaler", "scaler_path", "model_dir", "cache_dir",
              "base_poscar", "poscar", "poscar_template", "scaffold_poscar", "mgt_repo", "mgt_python", "config", "geo_opt_model", "path"}
_PATH_LIST_KEYS = {"dp_models", "model_paths", "checkpoint_paths"}


def _resolve_asset_paths(value, base_dir, key=None):
    """Resolve established filesystem fields; never reinterpret identifiers/FQNs."""
    def path(v):
        if not isinstance(v, str) or not v or "://" in v:
            return v
        p = Path(v).expanduser()
        return str((base_dir / p).resolve()) if not p.is_absolute() else str(p.resolve())
    if isinstance(value, Mapping):
        return {k: _resolve_asset_paths(v, base_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        if key in _PATH_LIST_KEYS:
            return [path(v) for v in value]
        return [_resolve_asset_paths(v, base_dir) for v in value]
    return path(value) if key in _PATH_KEYS else value


def _load_legacy_plan(path: str | Path, method: str | None = None) -> dict:
    """Load, validate and materialize a complete campaign configuration.

    No scientific model is loaded and no experiment is run. Authored YAML and
    default provenance remain available for manifests and reproducibility.
    """
    path = Path(path).expanduser().resolve()
    allowed = {"protocol", "base_config", "method", "parameter_profile", "objective", "metrics", "efficiency",
               "budget", "range_search", "search_space", "training_search", "generation_search"}
    raw = _mapping(load_yaml_config(path), "HPO configuration", allowed)
    if raw.get("protocol", "hpo_v1") != "hpo_v1":
        raise ConfigError("protocol must be hpo_v1")
    authored_method = raw.get("method")
    if method is not None and authored_method is not None and method != authored_method:
        raise ConfigError(f"CLI method {method!r} conflicts with YAML method {authored_method!r}")
    method = method or authored_method
    if method not in METHODS:
        raise ConfigError(f"method must be one of {METHODS}")
    if not isinstance(raw.get("base_config"), str) or not raw["base_config"].strip():
        raise ConfigError("base_config must name an existing scenario YAML")
    base_path = (path.parent / Path(raw["base_config"]).expanduser()).resolve()
    base_raw = load_yaml_config(base_path)
    base = resolve_config(_resolve_asset_paths(base_raw, base_path.parent)).config
    profile_ref = raw.get("parameter_profile", f"builtin:{method}")
    if not isinstance(profile_ref, str):
        raise ConfigError("parameter_profile must be a builtin name or YAML path")
    profile = get_profile(method)
    if profile_ref.startswith("builtin:"):
        if profile_ref != f"builtin:{method}":
            raise ConfigError("parameter_profile must match the selected method")
    else:
        profile_path = (path.parent / Path(profile_ref).expanduser()).resolve()
        custom = _mapping(load_yaml_config(profile_path), "parameter_profile", {"method", "search_space", "extension_choices"})
        if custom.get("method") != method:
            raise ConfigError("parameter_profile.method must match the selected method")
        custom_space = custom.get("search_space", {})
        if not isinstance(custom_space, Mapping):
            raise ConfigError("parameter_profile.search_space must be a mapping")
        profile["search_space"].update(custom_space)
        extensions = _mapping(custom.get("extension_choices", {}), "parameter_profile.extension_choices", profile["search_space"])
        for name, choices in extensions.items():
            if not isinstance(choices, list) or not choices:
                raise ConfigError(f"parameter_profile.extension_choices.{name} must be a nonempty list")
            choices = [validate_parameter_value(method, name, v) for v in choices]
            if len(set(choices)) != len(choices):
                raise ConfigError(f"parameter_profile.extension_choices.{name} contains duplicates")
            profile["extension_choices"][name] = sorted(choices)
        profile_ref = str(profile_path)
    override_space = raw.get("search_space", {})
    if not isinstance(override_space, Mapping):
        raise ConfigError("search_space must be a mapping")
    space = validate_search_space(method, {**profile["search_space"], **override_space})
    objective = _objective(raw.get("objective", {}), base)
    metrics = _mapping(raw.get("metrics", {}), "metrics", {"best_fraction", "training_weights", "generation_weights"})
    metrics = {"best_fraction": _number(metrics.get("best_fraction", .10), "metrics.best_fraction", 0, 1, exclusive=True),
               "training_weights": _weights(metrics.get("training_weights", {}), "metrics.training_weights", _TRAIN_WEIGHTS),
               "generation_weights": _weights(metrics.get("generation_weights", {}), "metrics.generation_weights", _GEN_WEIGHTS)}
    ranges = _mapping(raw.get("range_search", {}), "range_search", {"reference", "seeds", "confirmation_seed", "max_extra_values_per_direction", "score_improvement"})
    ranges = {"reference": ranges.get("reference", "initial_midpoint"), "seeds": _seeds(ranges.get("seeds", [11, 29]), "range_search.seeds"),
              "confirmation_seed": _integer(ranges.get("confirmation_seed", 47), "range_search.confirmation_seed"),
              "max_extra_values_per_direction": _integer(ranges.get("max_extra_values_per_direction", 2), "range_search.max_extra_values_per_direction"),
              "score_improvement": _number(ranges.get("score_improvement", .02), "range_search.score_improvement", 0)}
    if ranges["reference"] != "initial_midpoint":
        raise ConfigError("range_search.reference must be initial_midpoint")
    if ranges["confirmation_seed"] in ranges["seeds"] or ranges["confirmation_seed"] >= 2**32:
        raise ConfigError("range_search.confirmation_seed must be fresh and < 2**32")
    training_search = _search_settings(raw.get("training_search", {}))
    if set(training_search["validation_seeds"]) & (set(ranges["seeds"]) | {ranges["confirmation_seed"]}):
        raise ConfigError("training_search.validation_seeds must be disjoint from range-search seeds")
    generation_search = _search_settings(raw.get("generation_search", {}), generation=True)
    if len(generation_search["paired_seeds"]) != len(training_search["validation_seeds"]):
        raise ConfigError("generation_search.paired_seeds must pair one-to-one with training_search.validation_seeds")
    return {"protocol": "hpo_v1", "method": method, "parameter_profile": profile_ref,
            "base_config": str(base_path), "base_config_dir": str(base_path.parent),
            "base_config_raw": base_raw, "base_config_data": base, "source_path": str(path),
            "objective": objective, "metrics": metrics, "efficiency": _efficiency(raw.get("efficiency", {})),
            "budget": _budget(raw.get("budget", {}), objective), "range_search": ranges,
            "search_space": space, "extension_choices": profile["extension_choices"],
            "training_search": training_search, "generation_search": generation_search,
            "authored_config": raw,
            "sources": {key: "yaml" if key in raw else "default" for key in allowed}}


def load_plan(path: str | Path, method: str | None = None) -> dict:
    """Read the simplified public schema, while retaining authored v1 files."""
    path = Path(path).expanduser().resolve()
    raw = load_yaml_config(path)
    if isinstance(raw, Mapping) and (raw.get("protocol") == "hpo_v1" or
            not {"evaluation", "training", "generation", "random_seed"}.intersection(raw)
            and {"objective", "metrics", "efficiency", "budget", "range_search",
                 "search_space", "training_search", "generation_search", "parameter_profile"}.intersection(raw)):
        warnings.warn(
            "Legacy HPO v1 YAML retains its original scoring and seed settings. "
            "Migrate to evaluation/training/generation to use inferred scene objectives, "
            "inclusive thresholds, fixed quality mapping and shared repeat counts.",
            FutureWarning, stacklevel=2,
        )
        return _load_legacy_plan(path, method)
    from .spec_v2 import load_simplified_plan
    return load_simplified_plan(path, raw, method)
