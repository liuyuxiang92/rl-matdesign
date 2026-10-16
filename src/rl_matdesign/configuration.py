"""Shared YAML loading and environment construction for RL and baselines."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import warnings

import yaml

from .env import CompositionEnv
from .env_integer import IntegerRatioEnv
from .training import CandidateEvaluationFailed


# Old YAML key → new YAML key. Old keys still work but emit a one-time
# deprecation warning. Conventions:
#   dqn_*  → used only by train_dqn_online
#   pg_*   → used only by train_pg (REINFORCE / A2C)
#   (no prefix) → shared across both methods
_FLAG_ALIASES = {
    "buffer_size":          "dqn_buffer_size",
    "num_train_eps":        "dqn_num_train_eps",
    "grad_steps_per_ep":    "dqn_grad_steps_per_ep",
    "target_update_freq":   "dqn_target_update_freq",
    "eps_anneal_eps":       "dqn_eps_anneal_eps",
    "eps_min":              "dqn_eps_min",
    "entropy_coef":         "pg_entropy_coef",
    "repeat_penalty_coef":  "pg_repeat_penalty_coef",
    "repeat_penalty_shape": "pg_repeat_penalty_shape",
}


def _apply_flag_aliases(cfg: dict) -> dict:
    """Map deprecated flag names to their new prefixed forms.

    If both old and new appear, the new value wins and a warning is emitted.
    """
    for old, new in _FLAG_ALIASES.items():
        if old in cfg:
            if new in cfg:
                warnings.warn(
                    f"Config key '{old}' is deprecated (renamed to '{new}'); "
                    f"both keys are present — using '{new}' and ignoring '{old}'.",
                    DeprecationWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Config key '{old}' is deprecated; rename to '{new}'.",
                    DeprecationWarning, stacklevel=2,
                )
                cfg[new] = cfg[old]
            del cfg[old]
    return cfg


def resolve_sequence_catalog_paths(cfg: dict, config_path: str) -> dict:
    """Copy config fields and resolve sequence CSVs relative to their source YAML.

    Call this before relocating a config (for example, into an HPO trial).
    Existing material/model/plugin path conventions remain unchanged.
    """
    cfg = deepcopy(cfg)
    groups = []
    for group in cfg.get("groups", []):
        tokens = group.get("tokens")
        if group.get("kind") == "sequence" and isinstance(tokens, dict) and tokens.get("csv"):
            catalog = Path(tokens["csv"]).expanduser()
            if not catalog.is_absolute():
                catalog = Path(config_path).resolve().parent / catalog
            # YAML aliases may share this mapping with unrelated plugin fields.
            group = {**group, "tokens": {**tokens, "csv": str(catalog.resolve())}}
        groups.append(group)
    if "groups" in cfg:
        cfg["groups"] = groups
    return cfg


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Direct build_env callers pass resolved paths.
    cfg = resolve_sequence_catalog_paths(cfg, path)
    return _apply_flag_aliases(cfg)


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
    if not cfg.get("groups"):
        raise ValueError(
            "Config has no 'groups:' — every scenario is expressed as one or "
            "more sublattice groups now (kind: composition / categorical / "
            "integer / sequence). See CLAUDE.md's proposed schema / any configs/*.yaml "
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

    def reward_fn(formula: str) -> float:
        if env_type == "integer_ratio":
            from pymatgen.core.composition import Composition
            try:
                comp = dict(Composition(formula).fractional_composition.as_dict())
                comp = {str(k): float(v) for k, v in comp.items()}
            except Exception:
                return _FAILED_CANDIDATE_REWARD
        else:
            import re
            parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", formula)
            comp = {el: float(frac) for el, frac in parts}
            anion_formula = cfg.get("anion_formula", "")
            if anion_formula:
                for el, _ in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", anion_formula):
                    comp.pop(el, None)
        # A valid, well-formed composition can still fail downstream (a
        # structure builder's random realizations all failing MD/relax, a
        # flaky external subprocess, ...) -- that must not crash a run that
        # may be hours into training, NOR silently consume a real
        # episode-budget slot on a candidate that was never actually scored.
        # Raise so the rollout call site (training.py) discards this attempt
        # and retries with a fresh candidate instead.
        try:
            mean, _ = predictor.predict(comp)
        except Exception as exc:  # noqa: BLE001 - any predictor/builder failure
            raise CandidateEvaluationFailed(
                f"predictor.predict failed for formula {formula!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return mean

    def mg_reward_fn(groups: dict) -> float:
        # MultiGroupEnv hands the predictor the structured {group: {el: frac}}
        # mapping; the (recipe) predictor assembles the full structure from it.
        # See reward_fn's matching try/except for why this raises rather than
        # returning a fallback reward.
        try:
            mean, _ = predictor.predict(groups)
        except Exception as exc:  # noqa: BLE001 - any predictor/builder failure
            raise CandidateEvaluationFailed(
                f"predictor.predict failed for candidate {groups!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return mean

    # Build env first (without filter), then build the filter using env's
    # feasibility tables (_allowed_units, _possible_sums_by_k), then attach.
    # This is required for the ooh_phase filter and avoids the previous probe
    # episode that burned RNG state before warmup (Bug 3).
    if env_type == "integer_ratio":
        env = IntegerRatioEnv(
            species_set=flat_cfg["species_set"],
            ratio_set=flat_cfg["ratio_set"],
            n_components=int(flat_cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=None,
        )
    elif env_type == "multi_group":
        from rl_matdesign.env_multigroup import MultiGroupEnv

        # Normalize each group's fraction_set/total_units, then build the
        # per-group constraint instance (env=None — multi-group filters mask
        # by element/level/prior-group).
        built_groups = []
        for g in groups:
            gspec = normalize_group_spec(g)
            gspec["constraint_filter"] = build_constraint_filter(gspec, env=None)
            built_groups.append(gspec)
        env = MultiGroupEnv(groups=built_groups, reward_fn=mg_reward_fn)
    else:
        env = CompositionEnv(
            species_set=flat_cfg["species_set"],
            fraction_set=flat_cfg["fraction_set"],
            anion_formula=flat_cfg.get("anion_formula", ""),
            n_components=int(flat_cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=None,
            total_units=int(flat_cfg["total_units"]),
            element_bounds=flat_cfg.get("element_bounds"),
            episode_style=flat_cfg.get("episode_style", "element_then_amount"),
        )

    # Single-group envs attach a top-level filter post-construction (some filters
    # need the env's feasibility tables). multi_group filters live per-group.
    # For a collapsed single group, the filter is built from the merged flat_cfg
    # (so the group's own filters: entries are preserved).
    if env_type != "multi_group":
        env.phase_filter = build_constraint_filter(flat_cfg, env=env)

    return env, env_type, flat_cfg
