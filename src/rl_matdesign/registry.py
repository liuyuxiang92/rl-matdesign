"""Plug-in registry for predictors and constraint filters.

Two extension paths:

1. **Short name** — `properties[].predictor: ooh_frames` or a `filters:` entry `type: smact_charge`
   in YAML resolves to a built-in factory registered in :data:`PREDICTORS` /
   :data:`CONSTRAINTS`.

2. **Fully-qualified name** — `predictor: "my_pkg.my_mod:MyClass"` loads
   ``MyClass`` from ``my_pkg.my_mod`` via :mod:`importlib`. The class is
   instantiated as ``MyClass(cfg, **ctx)`` where ``ctx`` is keyword-only
   context (``seed=...`` for predictors, ``env=...`` for constraints).

User predictors only need ``__init__(self, cfg, *, seed=None)`` and
``predict(self, composition) -> (mean, std)``. Zero edits to
``scripts/run_experiment.py``.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional


# Factory signature: (cfg: dict, **ctx: Any) -> instance
Factory = Callable[..., Any]


# ---------------------------------------------------------------------------
# Built-in predictor factories
# ---------------------------------------------------------------------------

def _make_structure_score(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.structure_score import StructureScorePredictor
    return StructureScorePredictor(cfg, seed=seed)


def _make_rf_magpie(cfg: dict, **_):
    from .predictors.rf_magpie import RFMagpiePredictor
    return RFMagpiePredictor(model_path=cfg["model"])


def _make_dummy(cfg: dict, *, seed: Optional[int] = None, **_):
    import random as _random
    from .utils.seeding import candidate_seed

    class _DummyPredictor:
        def predict(self, composition):
            rng = _random.Random(candidate_seed(0 if seed is None else seed, composition))
            return float(rng.gauss(0.5, 0.1)), float(abs(rng.gauss(0.05, 0.01)))

        def batch_predict(self, compositions):
            return [self.predict(c) for c in compositions]

    return _DummyPredictor()


# Leaf predictors selected by `properties[].predictor` inside the reward engine. `dp_energy`/`dp_property`/`dp_classification` are NOT here — they are
# structure-scoring branches handled internally by the engine (they need its
# build/relax machinery), not standalone composition leaves.
def _make_mgtransformer(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.mgtransformer import MGTransformerPredictor
    return MGTransformerPredictor(cfg, seed=seed)


def _make_lookup(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.lookup import LookupPredictor
    return LookupPredictor(cfg, seed=seed)


def _make_glassnet(cfg: dict, **_):
    from .predictors.glassnet import GlassNetPredictor
    # Only adapter-owned settings; property selectors and rewards stay in the
    # engine, and omitted settings use the audited adapter's own defaults.
    return GlassNetPredictor(**{key: cfg[key] for key in ("oxide_basis", "cache_size", "batch_size") if key in cfg})


PREDICTORS: Dict[str, Factory] = {
    "rf_magpie":          _make_rf_magpie,
    "ooh_frames":         lambda cfg, **kwargs: _load_fqn("rl_matdesign.predictors.ooh_frames:OOHFramesPredictor")(cfg, **kwargs),
    "mgtransformer":      _make_mgtransformer,     # bridge to ../MGTransformer
    "lookup":             _make_lookup,
    "glassnet":           _make_glassnet,
    "dummy":              _make_dummy,
}


# ---------------------------------------------------------------------------
# Built-in structure-builder factories (composition/groups -> ASE Atoms)
# ---------------------------------------------------------------------------


def _make_pipeline_builder(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.builders.pipeline import PipelineBuilder
    return PipelineBuilder(cfg, seed=seed)





BUILDERS: Dict[str, Factory] = {
    "pipeline": _make_pipeline_builder,
}


def resolve_builder(kind: str, cfg: dict, *, seed: Optional[int] = None):
    """Build a structure-builder from a short name or ``pkg.module:ClassName`` FQN."""
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, seed=seed)
    factory = BUILDERS.get(kind.lower())
    if factory is None:
        raise ValueError(
            f"Unknown builder {kind!r}. Built-ins: {sorted(BUILDERS)}. "
            f"For a custom builder, use the FQN form: 'pkg.module:ClassName'."
        )
    return factory(cfg, seed=seed)


# ---------------------------------------------------------------------------
# Built-in constraint factories
# ---------------------------------------------------------------------------

def _resolve_scaffold(cfg: dict):
    """Resolve fixed counts relative to the substituted formula unit."""
    scaffold_fixed, n_sites = None, 1
    if cfg.get("scaffold_poscar"):
        from .constraints.charge import template_scaffold
        scaffold_fixed, n_sites = template_scaffold(
            cfg["scaffold_poscar"], cfg["scaffold_site_symbol"])
    elif cfg.get("scaffold_formula"):
        from .constraints.charge import parse_formula
        scaffold_fixed = parse_formula(cfg["scaffold_formula"])
    return scaffold_fixed, n_sites


def _make_smact_charge(cfg: dict, **_):
    from .constraints.smact_filter import SMACTChargeFilter
    scaffold_fixed, n_sites = _resolve_scaffold(cfg)
    return SMACTChargeFilter(scaffold_fixed=scaffold_fixed, scaffold_n_sites=n_sites,
                             **{key: cfg[key] for key in ("allow_alloys", "tol") if key in cfg})


def _make_pauling_en(cfg: dict, **_):
    from .constraints.smact_filter import ElectronegativityFilter
    scaffold_fixed, n_sites = _resolve_scaffold(cfg)
    return ElectronegativityFilter(scaffold_fixed=scaffold_fixed, scaffold_n_sites=n_sites,
                                   **{key: cfg[key] for key in ("allow_alloys", "tol") if key in cfg})


def _cfg_names_constraint(cfg: dict, name: str) -> bool:
    """True if *cfg*'s ``filters:`` list configures constraint ``name``."""
    for f in cfg.get("filters") or []:
        if isinstance(f, str) and f == name:
            return True
        if isinstance(f, dict) and f.get("type") == name:
            return True
    return False


def _has_constraint(cfg: dict, name: str) -> bool:
    for g in cfg.get("groups") or []:
        if isinstance(g, dict) and _cfg_names_constraint(g, name):
            return True
    return False


def smact_charge_enabled(cfg: dict) -> bool:
    """True when a ``smact_charge`` constraint is configured in a group filter list."""
    return _has_constraint(cfg, "smact_charge")


def pauling_en_enabled(cfg: dict) -> bool:
    """True when a ``pauling_en`` constraint is configured in a group filter list."""
    return _has_constraint(cfg, "pauling_en")


def charge_check_enabled(cfg: dict) -> bool:
    """Single switch for the post-episode generation drop: True if either a
    ``smact_charge`` or ``pauling_en`` constraint is configured. When False the
    post-episode validation in ``generate_candidates`` does nothing (no smact
    import, output unchanged)."""
    return smact_charge_enabled(cfg) or pauling_en_enabled(cfg)


def charge_use_pauling(cfg: dict) -> bool:
    """True when the post-episode drop should also enforce Pauling EN (``pauling_en``)."""
    return pauling_en_enabled(cfg)




def _make_ooh_phase(cfg: dict, *, env=None, **_):
    if env is None:
        raise ValueError(
            "filter type 'ooh_phase' requires env (needs _allowed_units "
            "and _possible_sums_by_k). Build the env without a filter first, "
            "then call resolve_constraint(..., env=env)."
        )
    from abcde_ooh.constraints.phase_sampler import PhaseActionFilter
    return PhaseActionFilter(
        target_phase=cfg.get("target_phases", ["any"]),
        allowed_units=env._allowed_units,
        possible_sums_by_k=env._possible_sums_by_k,
        amount_scale=env.amount_scale,
    )




CONSTRAINTS: Dict[str, Factory] = {
    "smact_charge":      _make_smact_charge,
    "pauling_en":        _make_pauling_en,
    "ooh_phase":         _make_ooh_phase,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _load_fqn(kind: str) -> type:
    """Resolve ``"pkg.module:ClassName"`` → class object."""
    try:
        mod_path, cls_name = kind.rsplit(":", 1)
    except ValueError:
        raise ValueError(
            f"Bad FQN {kind!r}: expected 'pkg.module:ClassName' "
            "(use ':' to separate module path from class)."
        )
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import module {mod_path!r} for plug-in {kind!r}. "
            "Make sure the package is installed and on PYTHONPATH."
        ) from e
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module {mod_path!r} has no attribute {cls_name!r} for plug-in {kind!r}."
        ) from e


def resolve_predictor(kind: str, cfg: dict, *, seed: Optional[int] = None):
    """Build a predictor instance from a short name or FQN."""
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, seed=seed)
    factory = PREDICTORS.get(kind.lower())
    if factory is None:
        raise ValueError(
            f"Unknown predictor {kind!r}. "
            f"Built-ins: {sorted(PREDICTORS)}. "
            f"For a custom predictor, use the FQN form: 'pkg.module:ClassName'."
        )
    return factory(cfg, seed=seed)


def build_reward(cfg: dict, *, seed: Optional[int] = None):
    """Build the reward engine from the single supported properties format."""
    retired = {"predictor", "models", "model", "dp_models", "model_ref"} & cfg.keys()
    if retired:
        raise ValueError(
            f"Removed top-level model settings: {sorted(retired)}; "
            "use properties entries with predictor and models."
        )
    if not isinstance(cfg.get("properties"), list) or not cfg["properties"]:
        raise ValueError("Reward config requires a non-empty 'properties' list.")
    return _make_structure_score(cfg, seed=seed)


def resolve_constraint(kind: Optional[str], cfg: dict, *, env=None):
    """Build a constraint filter instance from a short name or FQN, or None."""
    if kind is None:
        return None
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, env=env)
    factory = CONSTRAINTS.get(kind)
    if factory is None:
        raise ValueError(
            f"Unknown filter type {kind!r}. "
            f"Built-ins: {sorted(CONSTRAINTS)}. "
            f"For a custom filter, use the FQN form: 'pkg.module:ClassName'."
        )
    owned = {
        "smact_charge": {"scaffold_poscar", "scaffold_site_symbol", "scaffold_formula", "allow_alloys", "tol"},
        "pauling_en": {"scaffold_poscar", "scaffold_site_symbol", "scaffold_formula", "allow_alloys", "tol"},
        "ooh_phase": {"target_phases"},
    }
    unused = set(cfg) - owned[kind]
    if unused:
        raise ValueError(f"Filter {kind!r} has unused parameters: {sorted(unused)}")
    return factory(cfg, env=env)


def build_constraints(cfg: dict, *, env=None) -> list:
    """Resolve a group's filters in declaration order; every filter must pass.

    An absent or empty list means no optional scientific filters. Each entry owns
    its options and resolves directly to a built-in or custom filter instance.
    """
    specs = cfg.get("filters", [])
    if not isinstance(specs, list):
        raise ValueError("'filters' must be a list of filter declarations.")
    filters = []
    for index, spec in enumerate(specs):
        if isinstance(spec, str):
            kind, options = spec, {}
        elif isinstance(spec, dict):
            options = dict(spec)
            kind = options.pop("type", None)
            if not kind:
                raise ValueError(f"filters[{index}] is missing 'type'.")
        else:
            raise ValueError(f"filters[{index}] must be a dict or a bare string.")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"filters[{index}].type must be a nonempty string.")
        filter_ = resolve_constraint(kind, options, env=env)
        if not callable(getattr(filter_, "filter_actions", None)):
            raise ValueError(f"filters[{index}].type={kind!r} must implement filter_actions().")
        filters.append(filter_)
    return filters
