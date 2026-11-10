"""Lightweight experiment configuration contract, independent of ML backends.

Only shared experiment parameters are normalized here. Nested environment,
property, builder and filter options belong to their respective implementations;
accepting their containers does not claim to validate those scientific schemas.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
import math
from pathlib import Path
from typing import Any
import warnings

import yaml


class ConfigError(ValueError):
    """A configuration is ambiguous, malformed, or outside its contract."""


class ConfigDeprecationWarning(DeprecationWarning):
    """A supported configuration spelling should be migrated."""


ALIASES = {
    "buffer_size": "dqn_buffer_size",
    "num_train_eps": "dqn_num_train_eps",
    "grad_steps_per_ep": "dqn_grad_steps_per_ep",
    "target_update_freq": "dqn_target_update_freq",
    "eps_anneal_eps": "dqn_eps_anneal_eps",
    "eps_min": "dqn_eps_min",
    "entropy_coef": "pg_entropy_coef",
    "repeat_penalty_coef": "pg_repeat_penalty_coef",
    "repeat_penalty_shape": "pg_repeat_penalty_shape",
    "dp_seed": "eval_seed",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # Check authored entries before flattening YAML merges. A merged default
        # overridden by an explicit entry is legitimate, two authored keys are not.
        seen = {}
        for key_node, _ in node.value:
            key = ("<<" if key_node.tag == "tag:yaml.org,2002:merge"
                   else self.construct_object(key_node, deep=deep))
            try:
                if key in seen:
                    first = seen[key]
                    raise ConfigError(
                        f"Duplicate YAML key {key!r} at {key_node.start_mark}; "
                        f"first defined at {first}"
                    )
                seen[key] = key_node.start_mark
            except TypeError as exc:
                raise ConfigError(f"Unhashable YAML key at {key_node.start_mark}") from exc
        return super().construct_mapping(node, deep=deep)


def _mapping(value: Any, context: str) -> dict:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings")
    return deepcopy(dict(value))


def load_yaml_config(path: str | Path) -> dict:
    """Load one YAML mapping; reject duplicate keys at every nesting level."""
    with open(path, encoding="utf-8") as stream:
        try:
            value = yaml.load(stream, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    return _mapping(value, f"Configuration {path}")


def _normalize_aliases(raw: Mapping, source: str) -> tuple[dict, dict[str, str]]:
    cfg = _mapping(raw, source)
    sources = {key: source for key in cfg}
    for old, new in ALIASES.items():
        if old not in cfg:
            continue
        if new in cfg and PARAMETERS[new].normalize(old, cfg[old]) != PARAMETERS[new].normalize(new, cfg[new]):
            raise ConfigError(
                f"Conflicting config aliases {old!r}={cfg[old]!r} and "
                f"{new!r}={cfg[new]!r} in {source}; keep only {new!r}."
            )
        warnings.warn(
            f"Config key {old!r} is deprecated; use {new!r}.",
            ConfigDeprecationWarning, stacklevel=3,
        )
        if new not in cfg:
            cfg[new] = cfg[old]
            sources[new] = f"{source}:alias:{old}"
        del cfg[old]
        del sources[old]
    return cfg, sources


def normalize_aliases(raw: Mapping) -> dict:
    """Return a copy with canonical keys; conflicting spellings are errors."""
    return _normalize_aliases(raw, "yaml")[0]


@dataclass(frozen=True)
class Parameter:
    kind: type
    default: Any
    description: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: tuple[str, ...] = ()
    nullable: bool = False
    fallback: str | None = None

    def normalize(self, key: str, value: Any) -> Any:
        if value is None and self.nullable:
            return None
        if self.kind is bool:
            valid = isinstance(value, bool)
        elif self.kind is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif self.kind is float:
            # PyYAML's YAML 1.1 resolver reads common scientific notation such
            # as 1e-3 as text. Accept numerical strings only for numeric fields.
            valid = isinstance(value, (int, float, str)) and not isinstance(value, bool)
            if valid:
                try:
                    value = float(value)
                except (ValueError, OverflowError):
                    valid = False
        else:
            valid = isinstance(value, self.kind)
        if not valid:
            raise ConfigError(f"{key} must be {self.kind.__name__}, got {value!r}")
        if self.kind in (int, float):
            if self.kind is float and not math.isfinite(value):
                raise ConfigError(f"{key} must be finite, got {value!r}")
            if self.minimum is not None:
                invalid = value <= self.minimum if self.exclusive_minimum else value < self.minimum
                if invalid:
                    relation = ">" if self.exclusive_minimum else ">="
                    raise ConfigError(f"{key} must be {relation} {self.minimum}, got {value!r}")
            if self.maximum is not None and value > self.maximum:
                raise ConfigError(f"{key} must be <= {self.maximum}, got {value!r}")
        if self.choices and value not in self.choices:
            raise ConfigError(f"{key} must be one of {self.choices}, got {value!r}")
        return value


# These values are the existing experiment defaults, now specified in one place.
# Dynamic fallbacks are resolved before defaults, so merely adding pg_warmup_eps
# to the effective record cannot silently change DQN's 500-episode fallback.
PARAMETERS: dict[str, Parameter] = {
    "method": Parameter(str, "a2c", "Training algorithm; inactive algorithm settings are retained.", choices=("dqn", "reinforce", "a2c")),
    "eval_seed": Parameter(int, 0, "Random seed for property evaluation and structure sampling.", minimum=0, maximum=2**32 - 1),
    "train_seed": Parameter(int, None, "Training random seed.", minimum=0, maximum=2**32 - 1, nullable=True, fallback="eval_seed"),
    "gen_seed": Parameter(int, None, "Candidate-generation random seed.", minimum=0, maximum=2**32 - 1, nullable=True, fallback="eval_seed"),
    "constrain_training": Parameter(bool, True, "Apply configured constraints during training."),
    "train_count_duplicates": Parameter(bool, True, "Count repeated terminal candidates toward training budgets."),
    "train_budget": Parameter(int, None, "Total paid training candidates; derives algorithm episode limits when set.", minimum=1, nullable=True),
    "train_max_duplicate_attempts": Parameter(int, 1000, "Maximum consecutive duplicates in a unique-candidate budget.", minimum=1),
    "gamma": Parameter(float, 0.9, "Shared discount fallback for both algorithms.", minimum=0, maximum=1),
    "dqn_gamma": Parameter(float, 0.9, "DQN discount, overriding the shared fallback.", minimum=0, maximum=1, fallback="gamma"),
    "pg_gamma": Parameter(float, 0.9, "Policy-gradient discount, overriding the shared fallback.", minimum=0, maximum=1, fallback="gamma"),
    "dqn_hidden_dim": Parameter(int, 256, "DQN hidden-layer width.", minimum=1),
    "dqn_lr": Parameter(float, 1e-3, "DQN optimizer learning rate.", minimum=0, exclusive_minimum=True),
    "dqn_buffer_size": Parameter(int, 50000, "DQN replay capacity in transitions.", minimum=1),
    "dqn_eps_anneal_eps": Parameter(int, 10000, "DQN epsilon annealing episodes.", minimum=1),
    "dqn_eps_min": Parameter(float, 0.05, "DQN minimum exploration probability.", minimum=0, maximum=1),
    "dqn_num_train_eps": Parameter(int, 20000, "DQN training episode budget.", minimum=0),
    "dqn_warmup_eps": Parameter(int, 500, "DQN warmup; explicit pg_warmup_eps is a legacy fallback.", minimum=0, fallback="explicit pg_warmup_eps"),
    "dqn_batch_size": Parameter(int, 256, "DQN replay minibatch size.", minimum=1),
    "dqn_grad_steps_per_ep": Parameter(int, 5, "DQN gradient updates per training episode.", minimum=1),
    "dqn_target_update_freq": Parameter(int, 100, "DQN target-network update interval.", minimum=1),
    "dqn_loss": Parameter(str, "smoothl1", "DQN regression loss.", choices=("mse", "smoothl1")),
    "dqn_target_mode": Parameter(str, "bootstrap", "DQN regression target definition.", choices=("bootstrap", "mc")),
    "dqn_augment_permutations": Parameter(int, 0, "DQN replay trajectory permutation count.", minimum=0),
    "pg_warmup_eps": Parameter(int, 200, "Policy-gradient scaler warmup episodes.", minimum=0),
    "pg_num_iters": Parameter(int, 1000, "Policy-gradient training iterations.", minimum=0),
    "pg_batch_eps": Parameter(int, 15, "Episodes per policy-gradient update.", minimum=1),
    "pg_lr_actor": Parameter(float, 1e-3, "Actor optimizer learning rate.", minimum=0, exclusive_minimum=True),
    "pg_lr_critic": Parameter(float, 1e-3, "Critic optimizer learning rate.", minimum=0, exclusive_minimum=True),
    "pg_entropy_coef": Parameter(float, 0.01, "Normalized-entropy penalty coefficient.", minimum=0),
    "pg_entropy_min": Parameter(float, 0.3, "Normalized policy entropy floor.", minimum=0, maximum=1),
    "pg_repeat_penalty_coef": Parameter(float, 0.0, "Repeated-candidate policy penalty coefficient.", minimum=0),
    "pg_repeat_penalty_shape": Parameter(str, "log", "Repeated-candidate penalty growth.", choices=("log", "sqrt", "linear")),
    "num_gen_eps": Parameter(int, 200, "Requested exploitation candidates.", minimum=0),
    "exploration_gen_eps": Parameter(int, 0, "Additional random-exploration generation episodes.", minimum=0),
    "max_gen_attempts": Parameter(int, None, "Maximum candidate-generation attempts.", minimum=0, fallback="10 * num_gen_eps"),
    "gen_temperature": Parameter(float, 1.0, "Generation softmax temperature; zero selects greedy behavior.", minimum=0),
    "gen_top_frac": Parameter(float, 0.0, "Generation top-action fraction; zero disables top-fraction sampling.", minimum=0, maximum=1),
    "gen_epsilon": Parameter(float, 0.0, "Generation random-action probability.", minimum=0, maximum=1),
    "k": Parameter(float, 1.0, "Default uncertainty coefficient passed to property evaluation.", minimum=0),
}

# Ownership declarations, NOT complete nested schemas. These are existing flat
# options consumed by the named owners. The next schema layers can migrate them
# beside the owning implementation without changing the core training contract.
DELEGATED_KEYS = {
    "environment": frozenset({"groups", "anion_formula", "episode_style", "generated_columns"}),
    "structure_sources": frozenset({"structure_sources"}),
    "model_resources": frozenset({"models"}),
    "reward": frozenset({"predictor", "properties", "reward", "sweep", "share_structure", "n_random_configs", "predict_cache_size", "geo_opt", "structure_parallel"}),
    "builders": frozenset({"builder", "base_poscar", "poscar", "poscar_template", "site_symbol", "site_map", "valences", "halide_total", "host", "eligible_region", "eligible_region_fallback", "formula_units", "p_site_group", "s_site_group", "p_site_per_fu", "s_site_per_fu", "li_per_fu", "o_element", "cl_element", "a_site_symbol", "b_site_symbol", "a_dopant_group", "b_dopant_group", "a_defect_group", "interstitial_min_dist", "interstitial_max_attempts"}),
    "constraints": frozenset({"filters", "constraint_filter", "scaffold_poscar", "scaffold_site_symbol", "scaffold_formula", "smact_anions", "smact_anion", "smact_anion_charge", "smact_anion_stoich", "required_elements", "nonzero_ratio_at_last", "reserve_for_last", "phase_patterns", "target_phases"}),
    "ooh_predictor": frozenset({"dp_models", "objective", "ads_height", "ads_dz", "adsorbates", "geo_opt_model", "uncertainty", "output_index"}),
    "flat_model_predictors": frozenset({"model", "mgt_repo", "mgt_python", "config", "device", "cutoff", "max_neighbors", "atom_features", "triplet_endpoint", "triplet_pad_mode"}),
}
_DELEGATED_NAMES = frozenset().union(*DELEGATED_KEYS.values())
_RESERVED_PREFIXES = ("dqn_", "pg_", "train_", "gen_", "eval_")


def _has_flat_extension(cfg: Mapping) -> bool:
    # A property-level FQN receives that property's mapping, not the entire run;
    # it therefore does not excuse unknown keys at the experiment top level.
    for key in ("predictor", "builder"):
        if isinstance(cfg.get(key), str) and ":" in cfg[key]:
            return True
    for item in cfg.get("filters") or []:
        kind = item if isinstance(item, str) else item.get("type") if isinstance(item, Mapping) else None
        if isinstance(kind, str) and ":" in kind:
            return True
    return False


def validate_config(raw: Mapping) -> None:
    """Validate core values and ownership without materializing any defaults.

    Call :func:`normalize_aliases` first if accepting deprecated spellings.
    Extension schemas retain authority over their own values and nested keys.
    """
    cfg = _mapping(raw, "Configuration")
    for key, value in cfg.items():
        if key in PARAMETERS:
            PARAMETERS[key].normalize(key, value)
    for key in ("groups", "properties", "filters"):
        if key in cfg:
            if not isinstance(cfg[key], list):
                raise ConfigError(f"{key} must be a list")
            for index, item in enumerate(cfg[key]):
                if not isinstance(item, Mapping) and not (key == "filters" and isinstance(item, str)):
                    raise ConfigError(f"{key}[{index}] must be a mapping" + (" or filter name" if key == "filters" else ""))
    if "structure_sources" in cfg:
        # This owner is deliberately lightweight: checking its declaration does
        # not import a builder, resolve files, infer predictor capabilities or
        # materialize source defaults into the authored run configuration.
        from .structure_sources import StructureSourceSpec
        sources = _mapping(cfg["structure_sources"], "structure_sources")
        for name, definition in sources.items():
            try:
                StructureSourceSpec(name, definition)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"structure_sources[{name!r}]: {exc}") from exc
    for index, prop in enumerate(cfg.get("properties", [])):
        if "source" in prop:
            name = prop["source"]
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ConfigError(f"properties[{index}].source must be a nonempty source name without surrounding whitespace")
        for field_name in ("model_ref", "output", "unit"):
            if field_name not in prop or field_name == "unit" and prop[field_name] is None:
                continue
            value = prop[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"properties[{index}].{field_name} must be a nonempty string")
        for field_name, options in {
            "representation": ("physical", "ln", "log10", "pIC50"),
            "transform": ("none", "identity", "exp", "pow10"),
            "uncertainty": ("model_spread", "config_spread", "native_uncertainty", "pooled_spread"),
            "objective": ("mean", "mean_minus_kstd", "mean_plus_kstd"),
        }.items():
            if field_name in prop and prop[field_name] not in options:
                raise ConfigError(f"properties[{index}].{field_name} must be one of {options}")
    if isinstance(cfg.get("models"), Mapping) or any("model_ref" in prop for prop in cfg.get("properties", [])):
        from .predictors.structure_score import normalize_model_references
        try:
            normalize_model_references(cfg)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Named model definitions: {exc}") from exc
    elif "models" in cfg:
        # A list retains its existing flat-adapter meaning. It does not become
        # a named resource table or excuse an invalid model_ref.
        if not isinstance(cfg["models"], list) or cfg.get("properties"):
            raise ConfigError("models must be a named mapping; legacy lists belong only to flat predictors")
    if "constraint_filter" in cfg and cfg["constraint_filter"] is not None:
        raise ConfigError("constraint_filter no longer selects a filter; use filters: [{type: ...}].")
    flat_extension = _has_flat_extension(cfg)
    known = set(PARAMETERS) | set(_DELEGATED_NAMES)
    for key in cfg:
        if key in known:
            continue
        core_typo = get_close_matches(key, sorted(PARAMETERS), n=1, cutoff=0.85)
        if flat_extension and not key.startswith(_RESERVED_PREFIXES) and not core_typo:
            continue
        suggestions = get_close_matches(key, sorted(known | set(ALIASES)), n=1, cutoff=0.65)
        hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
        raise ConfigError(
            f"Unknown experiment key {key!r}.{hint} "
            "Property-specific options belong inside properties entries; "
            "a flat custom predictor/builder may own additional top-level options."
        )


@dataclass(frozen=True)
class ResolvedConfig:
    config: dict[str, Any]
    sources: dict[str, str]


def resolve_config(raw: Mapping, overrides: Mapping | None = None) -> ResolvedConfig:
    """Resolve explicit overrides over YAML, then fill fallbacks/defaults.

    ``None`` override values mean absent CLI options; ``False`` and ``0`` are
    explicit values. Alias conflicts are checked within each source separately,
    so a canonical CLI override can intentionally supersede an old YAML spelling.
    Both DQN and policy-gradient settings remain present regardless of method.
    """
    cfg, sources = _normalize_aliases(raw, "yaml")
    if overrides is not None:
        explicit = {key: value for key, value in _mapping(overrides, "Overrides").items() if value is not None}
        cli, cli_sources = _normalize_aliases(explicit, "override")
        cfg.update(cli)
        sources.update(cli_sources)
    validate_config(cfg)
    authored_keys = set(cfg)
    # Resolve dependencies while presence still means explicitly specified.
    if "dqn_warmup_eps" not in cfg and "pg_warmup_eps" in cfg:
        cfg["dqn_warmup_eps"] = cfg["pg_warmup_eps"]
        sources["dqn_warmup_eps"] = "derived:pg_warmup_eps"
    for key in ("dqn_gamma", "pg_gamma"):
        if key not in cfg and "gamma" in cfg:
            cfg[key] = cfg["gamma"]
            sources[key] = "derived:gamma"
    for key, spec in PARAMETERS.items():
        if key not in cfg and (spec.default is not None or spec.nullable):
            cfg[key] = deepcopy(spec.default)
            sources[key] = "default"
    if cfg["train_budget"] is not None:
        budget = cfg["train_budget"]
        if cfg["dqn_warmup_eps"] > budget:
            raise ConfigError("dqn_warmup_eps exceeds train_budget; reduce warmup or increase the shared budget")
        derived = {
            "dqn_num_train_eps": (budget - cfg["dqn_warmup_eps"], "derived:train_budget-dqn_warmup_eps"),
            "pg_num_iters": ((budget + cfg["pg_batch_eps"] - 1) // cfg["pg_batch_eps"],
                             "derived:ceil(train_budget/pg_batch_eps)"),
        }
        for key, (value, origin) in derived.items():
            if key in authored_keys and cfg[key] != value:
                raise ConfigError(f"{key}={cfg[key]} conflicts with train_budget={budget}, which requires {value}; "
                                  f"remove redundant {key} and let the shared budget determine it")
            cfg[key], sources[key] = value, origin
    for key in ("train_seed", "gen_seed"):
        if cfg.get(key) is None:
            cfg[key] = cfg["eval_seed"]
            sources[key] = "derived:eval_seed"
    if "max_gen_attempts" not in cfg:
        cfg["max_gen_attempts"] = 10 * cfg["num_gen_eps"]
        sources["max_gen_attempts"] = "derived:10*num_gen_eps"
    for key, spec in PARAMETERS.items():
        cfg[key] = spec.normalize(key, cfg[key])
    return ResolvedConfig(cfg, sources)


def schema_markdown() -> str:
    """Render the core parameter table from the executable definitions."""
    lines = ["| Parameter | Type | Default / fallback | Constraints | Meaning |", "|---|---|---|---|---|"]
    for key, spec in PARAMETERS.items():
        default = repr(spec.default)
        if spec.fallback:
            default = f"{spec.fallback}; otherwise {default}" if spec.default is not None else spec.fallback
        constraints = []
        if spec.minimum is not None:
            constraints.append(f"{'>' if spec.exclusive_minimum else '>='} {spec.minimum}")
        if spec.maximum is not None:
            constraints.append(f"<= {spec.maximum}")
        if spec.choices:
            constraints.append(", ".join(spec.choices))
        lines.append(f"| `{key}` | {spec.kind.__name__} | {default} | {'; '.join(constraints) or '—'} | {spec.description} |")
    return "\n".join(lines) + "\n"
