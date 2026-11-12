"""Small public HPO schema, materialized into explicit campaign settings.

Only experiment choices are authored. Objective identity, invariant scoring
rules, generation strategy and reproducible phase seeds are resolved here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import random

from ..config_schema import ConfigError, load_yaml_config, resolve_config, selected_generation_parameter
from ..rewards import RewardCombiner
from .algorithm_profiles import METHODS, get_profile, validate_search_space
from .spec import (_GEN_WEIGHTS, _TRAIN_WEIGHTS, _efficiency, _integer, _mapping,
                   _number, _resolve_asset_paths, _weights)

_PUBLIC_KEYS = {"base_config", "method", "random_seed", "evaluation", "training", "generation"}
# Fixed protocol rules, intentionally absent from the user-facing YAML schema.
_BUDGET_FACTOR = 2.0
_RANGE_MAX_EXTRA_VALUES_PER_DIRECTION = 2
_GENERATION_MAX_EXTRA_VALUES_PER_DIRECTION = 2
_VALIDATION_TOP_CONFIGS = 3
_GENERATION_DEFAULTS = {
    "gen_temperature": [0, .001, .01, .1, 1, 10, 100, 1000],
    "gen_epsilon": [0, .01, .05, .1, .2, .4, .7, 1],
    "gen_top_frac": [.001, .01, .05, .1, .25, .5, 1],
}


def infer_objective(base: Mapping, threshold: float) -> dict:
    """Resolve a single named physical target, excluding constraint-only terms."""
    props = base.get("properties")
    if not isinstance(props, list) or not props:
        raise ConfigError("HPO requires named properties in base_config to infer its physical objective")
    named = {}
    weights = {}
    for index, prop in enumerate(props):
        if not isinstance(prop, Mapping):
            raise ConfigError(f"base_config.properties[{index}] must be a mapping")
        name = prop.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip() or name in named:
            raise ConfigError("base_config.properties must have distinct explicit nonempty names")
        named[name] = prop
        weights[name] = _number(prop.get("weight", 1.0), f"properties[{index}].weight")
    try:
        combiner = RewardCombiner(base.get("reward"), weights)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Cannot infer HPO objective from base_config: {exc}") from exc
    active = [(name, coefficient) for name, coefficient in combiner.objective_terms.items() if coefficient != 0]
    if len(active) != 1:
        raise ConfigError("HPO requires exactly one nonzero reward objective property; "
                          "multiple objectives or an all-zero objective cannot use one evaluation.threshold")
    name, coefficient = active[0]
    prop = named[name]
    aliases = {"min": "minimize", "minimize": "minimize", "max": "maximize", "maximize": "maximize"}
    direction = aliases.get(str(prop.get("direction", "")).lower())
    if direction is None:
        raise ConfigError(f"HPO property {name!r} needs direction min or max; direction target is not supported")
    scale = _number(prop.get("scale", 1.0), f"properties[{name!r}].scale")
    if scale == 0 or coefficient / scale <= 0:
        raise ConfigError(f"Reward coefficient/scale for {name!r} contradicts its declared direction; "
                          "make the physical min/max objective unambiguous in base_config")
    return {"field": name, "direction": direction, "threshold": threshold,
            "threshold_inclusive": True, "unit": prop.get("unit"),
            "normalization": {"kind": "sigmoid", "scale": None, "calibration": "reference"}}


def _phase_budget(value, context):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    mode = value.get("mode", "fraction_of_space")
    if mode not in {"fraction_of_space", "absolute_unique"}:
        raise ConfigError(f"{context}.mode must be fraction_of_space or absolute_unique")
    keys = {"mode"} | ({"min_fraction", "initial_fraction", "max_fraction"}
            if mode == "fraction_of_space" else {"min_unique", "initial_unique", "max_unique"})
    raw = _mapping(value, context, keys)
    result = {"mode": mode, "factor": _BUDGET_FACTOR}
    if mode == "fraction_of_space":
        result["min_unique"] = 1  # Only runtime learning/preparation feasibility may lift this floor.
        for key, default in (("min_fraction", .000001), ("initial_fraction", .0001), ("max_fraction", .01)):
            result[key] = _number(raw.get(key, default), f"{context}.{key}", 0, 1, exclusive=True)
        bounds = [result[key] for key in ("min_fraction", "initial_fraction", "max_fraction")]
    else:
        for key in ("min_unique", "initial_unique", "max_unique"):
            result[key] = _integer(raw.get(key), f"{context}.{key}", 1)
        bounds = [result[key] for key in ("min_unique", "initial_unique", "max_unique")]
    if not bounds[0] <= bounds[1] <= bounds[2]:
        raise ConfigError(f"{context} requires min <= initial <= max")
    return result


def _seed_schedule(master, repeats):
    rng = random.Random(master)
    used = set()
    def next_seed():
        while True:
            seed = rng.randrange(2**32)
            if seed not in used:
                used.add(seed)
                return seed
    return {"exploration": [next_seed() for _ in range(repeats)],
            "validation": [next_seed() for _ in range(repeats)]}


def load_simplified_plan(path: Path, value, method=None) -> dict:
    raw = _mapping(value, "HPO configuration", _PUBLIC_KEYS)
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
    master = _integer(raw.get("random_seed", 42), "random_seed")
    if master >= 2**32:
        raise ConfigError("random_seed must be < 2**32")

    evaluation = _mapping(raw.get("evaluation", {}), "evaluation",
                          {"threshold", "best_fraction", "score_tolerance", "max_attempts_per_unique"})
    threshold = _number(evaluation.get("threshold"), "evaluation.threshold")
    tolerance = _number(evaluation.get("score_tolerance", .01), "evaluation.score_tolerance", 0)
    objective = infer_objective(base, threshold)
    training = _mapping(raw.get("training", {}), "training",
                        {"repeats", "parameter_ranges", "score_weights", "budget", "search"})
    generation = _mapping(raw.get("generation", {}), "generation",
                          {"repeats", "score_weights", "initial_values", "budget"})
    repeats = _integer(training.get("repeats", 3), "training.repeats", 1)
    schedule = _seed_schedule(master, repeats)
    profile = get_profile(method)
    overrides = training.get("parameter_ranges", {})
    if not isinstance(overrides, Mapping):
        raise ConfigError("training.parameter_ranges must be a mapping")
    space = validate_search_space(method, {**profile["search_space"], **overrides})
    search = _mapping(training.get("search", {}), "training.search", {"max_rounds", "trials_per_round", "startup_random_trials"})
    rounds = _integer(search.get("max_rounds", 3), "training.search.max_rounds", 1)
    trials = _integer(search.get("trials_per_round", 30), "training.search.trials_per_round", 1)
    startup = _integer(search.get("startup_random_trials", 10), "training.search.startup_random_trials", 0)
    if startup > trials:
        raise ConfigError("training.search.startup_random_trials must not exceed trials_per_round")

    parameter = selected_generation_parameter(base_raw)
    initial_values = generation.get("initial_values", _GENERATION_DEFAULTS[parameter])
    if not isinstance(initial_values, list) or not initial_values:
        raise ConfigError("generation.initial_values must be a nonempty list")
    initial_values = [_number(value, "generation.initial_values", 0,
                             None if parameter == "gen_temperature" else 1) for value in initial_values]
    if len(set(initial_values)) != len(initial_values):
        raise ConfigError("generation.initial_values must contain distinct values")
    train_budget = _phase_budget(training.get("budget", {}), "training.budget")
    gen_budget = _phase_budget(generation.get("budget", {}), "generation.budget")
    budget_mode = train_budget["mode"] if train_budget["mode"] == gen_budget["mode"] else "per_phase"
    training_search = {"trials": trials, "startup_random_trials": startup,
                       "seeds": list(schedule["exploration"]),
                       "finalists": _VALIDATION_TOP_CONFIGS,
                       "validation_seeds": list(schedule["validation"]),
                       "max_budget_revisions": rounds - 1,
                       "additional_trials_per_revision": trials, "additional_random_trials": startup}
    return {
        "protocol": "hpo_v2", "method": method, "parameter_profile": f"builtin:{method}",
        "base_config": str(base_path), "base_config_dir": str(base_path.parent),
        "base_config_raw": base_raw, "base_config_data": base, "source_path": str(path),
        "random_seed": master, "seed_schedule": schedule,
        "objective": objective, "scoring_scope": "score_only",
        "metrics": {"best_fraction": _number(evaluation.get("best_fraction", .1), "evaluation.best_fraction", 0, 1, exclusive=True),
                    "training_weights": _weights(training.get("score_weights", {}), "training.score_weights", _TRAIN_WEIGHTS),
                    "generation_weights": _weights(generation.get("score_weights", {}), "generation.score_weights", _GEN_WEIGHTS)},
        "efficiency": _efficiency({"max_attempts_per_unique": evaluation.get("max_attempts_per_unique", 1.5)}),
        "budget": {"mode": budget_mode, "strategy": "bidirectional_grid", "train": train_budget,
                   "generation": gen_budget, "tolerances": {"score": tolerance}, "upper_confirmation_levels": 2},
        "range_search": {"reference": "initial_midpoint", "seeds": list(schedule["exploration"]),
                         "random_seed": master,
                         "max_extra_values_per_direction": _RANGE_MAX_EXTRA_VALUES_PER_DIRECTION,
                         "score_improvement": tolerance},
        "search_space": space, "extension_choices": profile["extension_choices"],
        "training_search": training_search,
        "generation_search": {"parameter": parameter, "initial_values": sorted(initial_values),
                              "max_extra_values_per_direction": _GENERATION_MAX_EXTRA_VALUES_PER_DIRECTION,
                              "fixed": {key: 0 for key in ("gen_epsilon", "gen_top_frac", "gen_temperature", "exploration_gen_eps")
                                        if key != parameter},
                              "repeats_per_model": _integer(generation.get("repeats", 3), "generation.repeats", 1),
                              "master_seed": master},
        "authored_config": deepcopy(raw),
        "sources": {key: "yaml" if key in raw else "default" for key in _PUBLIC_KEYS},
    }
