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

import yaml


class ConfigError(ValueError):
    """A configuration is ambiguous, malformed, or outside its contract."""


# Authored order selects one generation policy. These keys are intentionally not
# sorted: iteration over the source mapping, rather than this tuple, decides.
GENERATION_PARAMETERS = ("gen_temperature", "gen_top_frac", "gen_epsilon")
_GENERATION_METADATA = "generation_selection"


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
        result = super().construct_mapping(node, deep=deep)
        # SafeLoader flattens merged defaults before explicit entries, but dict
        # reassignment otherwise retains an inherited key's earlier position.
        # Move strategy keys on each occurrence so authored overrides remain
        # later than inherited generation fields.
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in GENERATION_PARAMETERS:
                result[key] = result.pop(key)
        return result


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


# Experiment defaults are defined once; each algorithm owns its warmup budget.
PARAMETERS: dict[str, Parameter] = {
    "method": Parameter(str, "a2c", "Training algorithm; inactive algorithm settings are retained.", choices=("dqn", "reinforce", "a2c")),
    "value_normalization": Parameter(str, "none", "Adaptive value-target normalization for DQN and A2C.", choices=("none", "popart")),
    "value_normalization_profile": Parameter(str, "popart_v1", "Versioned PopArt settings shared by DQN and A2C.", choices=("popart_v1",)),
    "eval_seed": Parameter(int, 0, "Random seed for property evaluation and structure sampling.", minimum=0, maximum=2**32 - 1),
    "train_seed": Parameter(int, None, "Training random seed.", minimum=0, maximum=2**32 - 1, nullable=True, fallback="eval_seed"),
    "gen_seed": Parameter(int, None, "Candidate-generation random seed.", minimum=0, maximum=2**32 - 1, nullable=True, fallback="eval_seed"),
    "hpo_recording": Parameter(bool, False, "Enable strict candidate accounting for an isolated HPO job."),
    "hpo_resume_generation": Parameter(bool, False, "Restore an exactly synchronized HPO generation checkpoint.", nullable=True),
    "hpo_phase": Parameter(str, "train", "HPO job phase.", choices=("train", "generation")),
    "hpo_target_unique": Parameter(int, None, "HPO target successfully evaluated distinct candidates.", minimum=1, nullable=True),
    "hpo_ratio_limit": Parameter(float, 1.5, "Strict HPO attempts/unique upper bound.", minimum=1, exclusive_minimum=True),
    "train_count_duplicates": Parameter(bool, False, "Count repeated terminal candidates toward training budgets."),
    "train_budget": Parameter(int, None, "Total paid training candidates; derives algorithm episode limits when set.", minimum=1, nullable=True),
    "train_max_duplicate_attempts": Parameter(int, 1000, "Maximum consecutive duplicates in a unique-candidate budget.", minimum=1),
    "gamma": Parameter(float, 0.9, "Discount factor used by DQN, REINFORCE and A2C.", minimum=0, maximum=1),
    "dqn_hidden_dim": Parameter(int, 256, "DQN hidden-layer width.", minimum=1),
    "dqn_lr": Parameter(float, 1e-3, "DQN optimizer learning rate.", minimum=0, exclusive_minimum=True),
    "dqn_buffer_size": Parameter(int, 50000, "DQN replay capacity in transitions.", minimum=1),
    "dqn_eps_anneal_eps": Parameter(int, 10000, "DQN epsilon annealing episodes.", minimum=1),
    "dqn_eps_min": Parameter(float, 0.05, "DQN minimum exploration probability.", minimum=0, maximum=1),
    "dqn_num_train_eps": Parameter(int, 20000, "DQN training episode budget.", minimum=0),
    "dqn_warmup_eps": Parameter(int, 500, "DQN random warmup episodes.", minimum=0),
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
    "num_gen_eps": Parameter(int, 200, "Requested distinct generated candidates.", minimum=0),
    "max_gen_attempts": Parameter(int, None, "Maximum candidate-generation attempts.", minimum=0, fallback="10 * num_gen_eps"),
    "gen_temperature": Parameter(float, 1.0, "Generation softmax temperature; zero selects greedy behavior.", minimum=0),
    "gen_top_frac": Parameter(float, 0.0, "Generation top-action fraction; zero disables top-fraction sampling.", minimum=0, maximum=1),
    "gen_epsilon": Parameter(float, 0.0, "Generation random-action probability.", minimum=0, maximum=1),
    "k": Parameter(float, 1.0, "Default uncertainty coefficient passed to property evaluation.", minimum=0),
}

# Nested structure and group implementations own their scientific options.
DELEGATED_KEYS = {
    "environment": frozenset({"groups"}),
    "structure": frozenset({"structure"}),
    "reward": frozenset({"properties", "reward"}),
}
_PROPERTY_OWNED_KEYS = frozenset({
    "predictor", "models", "model", "model_ref", "dp_models", "objective",
    "uncertainty", "output_index", "mgt_repo", "mgt_python", "config", "device",
    "cutoff", "max_neighbors", "atom_features", "triplet_endpoint", "triplet_pad_mode",
})
# Rejection only: retired keys cannot be mistaken for custom predictor options.
_RETIRED_KEYS = frozenset({
    "anion_formula", "episode_style", "structure_sources", "sweep", "share_structure",
    "builder", "base_poscar", "poscar", "poscar_template", "site_symbol", "site_map",
    "filters", "constraint_filter", "scaffold_formula", "scaffold_poscar", "scaffold_site_symbol",
    "smact_anions", "smact_anion", "smact_anion_charge", "smact_anion_stoich",
    "required_elements", "nonzero_ratio_at_last", "reserve_for_last", "phase_patterns", "target_phases",
    "n_random_configs", "geo_opt", "geo_opt_model", "structure_parallel", "ads_height", "ads_dz", "adsorbates",
    "application", "applications", "md", "opt", "constrain_training",
})
_DELEGATED_NAMES = frozenset().union(*DELEGATED_KEYS.values())


def _generation_selection(raw: Mapping, source: str = "yaml") -> tuple[str, str]:
    """Select before defaults, respecting a saved effective-value snapshot.

    Resolved configs contain all three policy fields and may be exported with
    alphabetically sorted keys. The snapshot preserves their original selection;
    an explicitly edited field supersedes it, including a newly supplied zero.
    """
    authored = [key for key in raw if key in GENERATION_PARAMETERS]
    for key in authored:
        PARAMETERS[key].normalize(key, raw[key])
    if _GENERATION_METADATA not in raw:
        return (authored[-1], source) if authored else ("gen_temperature", "default")

    metadata = _mapping(raw[_GENERATION_METADATA], _GENERATION_METADATA)
    if set(metadata) != {"parameter", "values", "source"}:
        raise ConfigError("generation_selection requires exactly parameter, values and source")
    selected = metadata["parameter"]
    if selected not in GENERATION_PARAMETERS:
        raise ConfigError(f"generation_selection.parameter must be one of {GENERATION_PARAMETERS}")
    if metadata["source"] not in ("yaml", "override", "default"):
        raise ConfigError("generation_selection.source must be yaml, override or default")
    values = _mapping(metadata["values"], "generation_selection.values")
    if set(values) != set(GENERATION_PARAMETERS):
        raise ConfigError("generation_selection.values must contain all three generation parameters")
    values = {key: PARAMETERS[key].normalize(key, values[key]) for key in GENERATION_PARAMETERS}
    if any(values[key] != 0 for key in GENERATION_PARAMETERS if key != selected):
        raise ConfigError("generation_selection.values must disable both unselected generation parameters")
    if not set(GENERATION_PARAMETERS).issubset(raw):
        raise ConfigError("generation_selection metadata requires all three effective generation fields")
    changed = [key for key in authored if PARAMETERS[key].normalize(key, raw[key]) != values[key]]
    return (changed[-1], source) if changed else (selected, metadata["source"])


def selected_generation_parameter(raw: Mapping) -> str:
    """Return the selected canonical generation field without adding defaults.

    In an authored YAML mapping the last explicit generation field wins. A
    resolved config's selection survives serialization and key sorting. Later
    value edits are treated as new explicit choices. When no field is authored,
    the existing temperature default is selected.
    """
    return _generation_selection(_mapping(raw, "Configuration"))[0]


def with_generation_selection(raw: Mapping, parameter: str) -> dict:
    """Return an exclusive, persistable selection for a synthesized run config.

    Unlike authored-order discovery, a job builder already knows which policy
    it intends to run. Its value is taken from ``raw`` (or its normal default),
    while both inactive values are set to zero before capturing the snapshot.
    """
    if parameter not in GENERATION_PARAMETERS:
        raise ConfigError(f"Generation parameter must be one of {GENERATION_PARAMETERS}")
    cfg = _mapping(raw, "Configuration")
    for key in GENERATION_PARAMETERS:
        cfg[key] = (PARAMETERS[key].normalize(key, cfg.get(key, PARAMETERS[key].default))
                    if key == parameter else 0.0)
    cfg[_GENERATION_METADATA] = {
        "parameter": parameter,
        "values": {key: cfg[key] for key in GENERATION_PARAMETERS},
        "source": "override",
    }
    return cfg


def validate_config(raw: Mapping) -> None:
    """Validate core values and ownership without materializing any defaults.

    Extension schemas retain authority over their own values and nested keys.
    """
    cfg = _mapping(raw, "Configuration")
    _generation_selection(cfg)
    for key, value in cfg.items():
        if key in PARAMETERS:
            PARAMETERS[key].normalize(key, value)
    if cfg.get("method") == "reinforce" and cfg.get("value_normalization") == "popart":
        raise ConfigError("value_normalization=popart requires DQN or A2C. REINFORCE has no value network")
    retired = set(cfg) & (_RETIRED_KEYS | _PROPERTY_OWNED_KEYS)
    if retired:
        raise ConfigError(f"Removed top-level parameters: {sorted(retired)}; use groups, structure and properties")
    for key in ("groups", "properties"):
        if key in cfg:
            if not isinstance(cfg[key], list):
                raise ConfigError(f"{key} must be a list")
            for index, item in enumerate(cfg[key]):
                if not isinstance(item, Mapping):
                    raise ConfigError(f"{key}[{index}] must be a mapping")
    if "properties" in cfg and not cfg["properties"]:
        raise ConfigError("properties must be a non-empty list")
    names = set()
    for index, group in enumerate(cfg.get("groups", [])):
        name = group.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ConfigError(f"groups[{index}].name must be a unique nonempty string")
        names.add(name)
        kind = group.get("kind", "composition")
        common = {"name", "kind", "n_steps", "filters"}
        fields = {
            "composition": common | {"species_set", "amount_set", "total", "unique", "element_bounds"},
            "categorical": common | {"choices", "labels", "values"},
        }
        if kind not in fields:
            raise ConfigError("Groups use only composition and categorical kinds")
        unknown = set(group) - fields[kind]
        if unknown:
            raise ConfigError(f"Group {name!r} has unused or removed parameters: {sorted(unknown)}")
        if kind == "composition" and not {"species_set", "amount_set", "n_steps"}.issubset(group):
            raise ConfigError(f"Composition group {name!r} needs species_set, amount_set and n_steps")
        if kind == "categorical" and ("choices" in group) == ("labels" in group or "values" in group):
            raise ConfigError(f"Categorical group {name!r} needs choices or labels/values")
        if kind == "categorical" and "choices" not in group and not {"labels", "values", "n_steps"}.issubset(group):
            raise ConfigError(f"Categorical group {name!r} needs labels, values and n_steps")
    if "structure" in cfg:
        structure = _mapping(cfg["structure"], "structure")
        if not isinstance(structure.get("builder"), str) or not structure["builder"].strip():
            raise ConfigError("structure.builder must name a builder")
        if structure["builder"] == "pipeline":
            from .predictors.builders.pipeline import validate_pipeline_config
            try:
                validate_pipeline_config(structure)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"structure: {exc}") from exc
        elif structure["builder"] != "peptide" and ":" not in structure["builder"]:
            raise ConfigError("structure.builder must be pipeline, peptide, or a custom class FQN")
    for index, prop in enumerate(cfg.get("properties", [])):
        if "weight" in prop:
            raise ConfigError(f"properties[{index}].weight was removed; put coefficients in reward terms")
        misplaced = set(prop) & (_RETIRED_KEYS | {"source", "model", "dp_models", "model_ref"})
        if misplaced:
            raise ConfigError(f"properties[{index}] has removed or structure-owned parameters: {sorted(misplaced)}")
        if prop.get("predictor") in {"ooh", "structure_score"}:
            raise ConfigError(f"properties[{index}].predictor must name a leaf predictor; use ooh_frames for OOH")
        if "models" in prop:
            models = prop["models"]
            if (not isinstance(models, list) or not models
                    or any(not isinstance(path, str) or not path.strip() for path in models)):
                raise ConfigError(f"properties[{index}].models must be a non-empty list of model paths")
        for field_name in ("output", "unit"):
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
    if cfg.get("properties"):
        from .rewards import RewardCombiner
        try:
            properties = [dict(prop, name=prop.get("name", f"prop{i}"))
                          for i, prop in enumerate(cfg["properties"])]
            combiner = RewardCombiner(cfg.get("reward"), [prop["name"] for prop in properties])
            if combiner.calibration_config or any("normalization" in prop for prop in properties):
                from .reward_calibration import RewardCalibration

                RewardCalibration(combiner, properties)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    known = set(PARAMETERS) | set(_DELEGATED_NAMES) | {_GENERATION_METADATA}
    for key in cfg:
        if key in known:
            continue
        suggestions = get_close_matches(key, sorted(known), n=1, cutoff=0.65)
        hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
        raise ConfigError(
            f"Unknown experiment key {key!r}.{hint} "
            "Property-specific options belong inside properties entries."
        )


@dataclass(frozen=True)
class ResolvedConfig:
    config: dict[str, Any]
    sources: dict[str, str]


def resolve_config(raw: Mapping, overrides: Mapping | None = None) -> ResolvedConfig:
    """Resolve explicit overrides over YAML, then fill fallbacks/defaults.

    ``None`` override values mean absent CLI options; ``False`` and ``0`` are
    explicit values. Only canonical parameter names are accepted.
    Both DQN and policy-gradient settings remain present regardless of method.
    """
    cfg = _mapping(raw, "Configuration")
    sources = {key: "yaml" for key in cfg}
    selected, selection_source = _generation_selection(cfg)
    if overrides is not None:
        explicit = {key: value for key, value in _mapping(overrides, "Overrides").items() if value is not None}
        cli = explicit
        cli_sources = {key: "override" for key in cli}
        if any(key in cli for key in GENERATION_PARAMETERS) or _GENERATION_METADATA in cli:
            selected, selection_source = _generation_selection(cli, "override")
        cfg.update(cli)
        sources.update(cli_sources)
    validate_config(cfg)
    authored_keys = set(cfg)
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
    # A generation policy is exclusive. Inactive fields cannot reactivate merely
    # because a default or an earlier authored field has a nonzero value.
    for key in GENERATION_PARAMETERS:
        if key != selected:
            cfg[key] = 0.0
            sources[key] = f"disabled:selection:{selected}"
    cfg[_GENERATION_METADATA] = {
        "parameter": selected,
        "values": {key: cfg[key] for key in GENERATION_PARAMETERS},
        "source": selection_source,
    }
    sources[_GENERATION_METADATA] = "derived:generation-selection"
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
