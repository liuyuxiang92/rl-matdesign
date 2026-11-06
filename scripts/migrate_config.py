#!/usr/bin/env python3
"""One-time, reviewable migration of authored pre-0.2 experiment YAML.

This is deliberately outside the runtime configuration loader. It does not load
models, change checkpoint paths, choose scientific units, or write its input.
The default command prints a diff; writing requires a distinct --output path.
Comments are not preserved in the generated YAML, so retain the input as the
historical configuration and review the diff before replacing a working copy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import difflib
import json
from pathlib import Path
import sys
import warnings

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_matdesign.config_schema import (  # noqa: E402
    ALIASES, ConfigDeprecationWarning, DELEGATED_KEYS, load_yaml_config,
    normalize_aliases, validate_config,
)
from rl_matdesign.structure_sources import (  # noqa: E402
    LegacySourceMigrationRequired, normalize_structure_sources,
)


@dataclass
class MigrationResult:
    config: dict
    changes: list[str]
    reviews: list[dict]

    @property
    def ready(self):
        return not any(item["blocking"] for item in self.reviews)


_STRUCTURE_ADAPTERS = {"dp_energy", "dp_property", "dp_classification", "mgtransformer",
                       "rl_matdesign.predictors.mgtransformer:MGTransformerPredictor"}
_COMPOSITION_ADAPTERS = {"rf_magpie", "lookup", "glassnet", "dummy",
                         "rl_matdesign.predictors.lookup:LookupPredictor"}
_SOURCE_FIELDS = set(DELEGATED_KEYS["builders"]) | {"md", "opt", "n_random_configs", "geo_opt"}
_OBJECTIVES = {"mean", "mean_minus_kstd", "mean_plus_kstd"}


def migrate_config(raw, *, source_policy=None, source_name="default",
                   structure_properties=None, objective_overrides=None,
                   base_config=None):
    """Return a defensive migrated copy and unresolved scientific decisions.

    ``source_policy`` is explicit ``shared`` or ``independent`` consent for a
    legacy false setting. ``structure_properties`` declares external adapter
    capabilities without importing them. ``objective_overrides`` is a reviewed
    property-name mapping; it never guesses uncertainty from property names.
    A missing old objective is materialized as mean-minus-kstd, except when
    k=0 proves its reward identical to the mean. No reported units are invented.
    """
    if source_policy not in (None, "shared", "independent"):
        raise ValueError("source_policy must be shared or independent")
    overrides = dict(objective_overrides or {})
    if any(value not in _OBJECTIVES for value in overrides.values()):
        raise ValueError("objective overrides must name a supported objective")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigDeprecationWarning)
        cfg = normalize_aliases(raw)
    changes = [f"Rename {old} to {new}." for old, new in ALIASES.items() if old in raw]
    reviews = []

    def review(field, message, blocking=True):
        reviews.append({"field": field, "message": message, "blocking": blocking})

    if "search_space" in cfg:
        if base_config is not None:
            cfg["base_config"] = base_config
            changes.append("Replace base_config with the explicitly supplied reference.")
        return MigrationResult(cfg, changes, reviews)
    if base_config is not None:
        raise ValueError("base_config is only valid for an HPO control file")

    # Delegate the known integer terminal-domain conversion to its sole owner.
    for index, group in enumerate(cfg.get("groups", [])):
        field = f"groups[{index}]"
        kind = group.get("kind", "composition")
        if kind == "integer":
            from rl_matdesign.env_integer import normalize_integer_group_spec
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                normalized = normalize_integer_group_spec(group)
            if normalized != group:
                cfg["groups"][index] = group = normalized
                changes.append(f"{field}: express terminal token requirements independently of action order.")
        elif kind == "composition":
            style = group.get("episode_style")
            if style in ("element_then_amount", "fixed_order_amount"):
                if style == "fixed_order_amount":
                    group["n_components"] = len(group["species_set"])
                del group["episode_style"]
                changes.append(f"{field}: remove obsolete action-order spelling; preserve selected-slot count.")
            kept = []
            for rule in group.get("filters") or []:
                required = rule.get("required_elements", []) if isinstance(rule, dict) else []
                bounds = group.get("element_bounds", {})
                redundant = (isinstance(rule, dict) and rule.get("type") == "last_step_element"
                             and required and rule.get("reserve_for_last", True) is True
                             and all(element in bounds and float(bounds[element][0]) > 0 for element in required))
                if redundant:
                    changes.append(f"{field}: remove construction-order restriction already guaranteed by positive element bounds.")
                else:
                    kept.append(rule)
            if "filters" in group:
                group["filters"] = kept

    needed, external = [], []
    properties = cfg.get("properties", [])
    names = [str(prop.get("name", f"prop{i}")) for i, prop in enumerate(properties)]
    if set(overrides) - set(names):
        raise ValueError(f"Unknown objective override properties: {sorted(set(overrides) - set(names))}")
    for index, (name, prop) in enumerate(zip(names, properties)):
        field = f"properties[{index}]"
        predictor = prop.get("predictor")
        definition = cfg.get("models", {}).get(prop.get("model_ref"), {}) if isinstance(cfg.get("models"), dict) else {}
        predictor = predictor or definition.get("predictor")
        if predictor in _STRUCTURE_ADAPTERS:
            needed.append(name)
        elif predictor not in _COMPOSITION_ADAPTERS:
            external.append(name)
        if name in overrides:
            if prop.get("objective") != overrides[name]:
                changes.append(f"{field}: use explicitly reviewed objective {overrides[name]}.")
            prop["objective"] = overrides[name]
        elif "objective" not in prop:
            prop["objective"] = "mean" if float(cfg.get("k", 1.0)) == 0 else "mean_minus_kstd"
            changes.append(f"{field}: materialize old effective objective as {prop['objective']}.")
        if prop["objective"] != "mean":
            if "uncertainty" not in prop and predictor in _STRUCTURE_ADAPTERS:
                prop["uncertainty"] = "model_spread"
                changes.append(f"{field}: name the historical model-ensemble statistic explicitly.")
            paths = prop.get("models", prop.get("model", prop.get("dp_models")))
            if paths is None:
                paths = definition.get("models", definition.get("model", definition.get("dp_models")))
            count = len(paths) if isinstance(paths, (list, tuple)) else 1
            if prop.get("uncertainty") == "model_spread" and count < 2:
                review(field + ".objective", "One declared model has no model-spread estimate. Select mean explicitly or supply an independently justified ensemble/statistic.")
            elif "uncertainty" not in prop:
                review(field + ".uncertainty", "External adapter uncertainty protocol is unverified; select its documented statistic explicitly.")
        transform = prop.get("transform")
        if transform in ("exp", "pow10") and "representation" not in prop:
            prop["representation"] = "ln" if transform == "exp" else "log10"
            changes.append(f"{field}: record the input representation implied by {transform}.")
        if "unit" not in prop:
            review(field + ".unit", "Physical unit remains unspecified; verify training/model provenance before scientific reporting.", False)

    if structure_properties is not None:
        declared = set(structure_properties)
        if declared - set(names):
            raise ValueError(f"Unknown structure properties: {sorted(declared - set(names))}")
        if set(needed) - declared:
            raise ValueError("structure_properties cannot exclude a built-in structure adapter")
        needed = [name for name in names if name in declared]
    elif external and any(key in cfg for key in ("builder", "structure_sources", "base_poscar")):
        review("properties", "External adapter capabilities are unknown; provide the complete structure_properties list before moving its builder settings.")
        return MigrationResult(cfg, changes, reviews)

    if "structure_sources" not in cfg and needed:
        try:
            plan = normalize_structure_sources(cfg, needs_structure=needed)
            sources = {source.source_id: source.effective_config for source in plan.sources}
            refs = plan.property_sources
            for note in plan.migration_notes:
                if "ignored property" in note:
                    review("structure_sources", note, False)
        except LegacySourceMigrationRequired as exc:
            if source_policy is None:
                review("share_structure", "Independent legacy sampling requires explicit shared or independent source_policy.")
                return MigrationResult(cfg, changes, reviews)
            report = exc.migration_report
            sources, refs = report["structure_sources"], report["property_sources"]
            if source_policy == "shared":
                configurations = list(sources.values())
                if any(value != configurations[0] for value in configurations[1:]):
                    review("structure_sources", "Independent legacy builder definitions differ. Author the intended shared physical source explicitly; migration will not discard overrides.")
                    return MigrationResult(cfg, changes, reviews)
                sources = {source_name: configurations[0]}
                refs = {name: source_name for name in needed}
                changes.append("Explicitly merge independent ensembles into one source; this intentionally changes property sampling correlation.")
            else:
                changes.append("Preserve independent sampling with distinct named sources.")
        if list(sources) == ["default"] and source_name != "default":
            sources = {source_name: sources["default"]}
            refs = {name: source_name for name in needed}
        # The normalizer owns which external builder fields were effective.
        moved = set().union(*(set(value) for value in sources.values())) - {"groups"}
        for key in moved | _SOURCE_FIELDS | {"share_structure"}:
            cfg.pop(key, None)
        for name, prop in zip(names, properties):
            if name in refs:
                prop.setdefault("name", name)
                prop["source"] = refs[name]
                for key in _SOURCE_FIELDS | {"groups", "structure_parallel"}:
                    prop.pop(key, None)
        cfg["structure_sources"] = sources
        changes.append("Move effective builder/relaxation settings to named structure sources.")
    elif not needed and "share_structure" in cfg and (not external or structure_properties is not None):
        del cfg["share_structure"]
        changes.append("Remove unused share_structure setting for composition-only properties.")
    if "structure_sources" in cfg:
        for source in cfg["structure_sources"].values():
            if "groups" in cfg and source.get("groups") == cfg["groups"]:
                del source["groups"]
                changes.append("Let the structure source inherit identical top-level groups instead of duplicating the domain.")
            if source.get("geo_opt") == {}:
                del source["geo_opt"]
                changes.append("Omit the empty, disabled geometry-optimization declaration.")
        normalize_structure_sources(cfg, needs_structure=needed)
    validate_config(cfg)
    return MigrationResult(cfg, changes, reviews)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, help="Write migrated YAML to a NEW file after resolving blocking reviews")
    parser.add_argument("--report", type=Path, help="Write the complete structured migration report")
    parser.add_argument("--source-policy", choices=("shared", "independent"))
    parser.add_argument("--source-name", default="default")
    parser.add_argument("--structure-property", action="append", default=None)
    parser.add_argument("--objective", action="append", default=[], metavar="NAME=OBJECTIVE")
    parser.add_argument("--base-config", help="Explicit HPO base reference; never inferred from filenames")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.objective:
        name, separator, objective = item.partition("=")
        if not separator or not name or name in overrides:
            parser.error("--objective requires a unique NAME=OBJECTIVE")
        overrides[name] = objective
    try:
        result = migrate_config(load_yaml_config(args.config), source_policy=args.source_policy,
                                source_name=args.source_name, structure_properties=args.structure_property,
                                objective_overrides=overrides, base_config=args.base_config)
        rendered = yaml.safe_dump(result.config, sort_keys=False, allow_unicode=True)
        print("".join(difflib.unified_diff(args.config.read_text().splitlines(True), rendered.splitlines(True),
                                         fromfile=str(args.config), tofile="migrated.yaml")), end="")
        if args.report:
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump({**asdict(result), "ready": result.ready}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        for item in result.reviews:
            print(f"{'REQUIRED' if item['blocking'] else 'REVIEW'} {item['field']}: {item['message']}", file=sys.stderr)
        if args.output:
            if not result.ready:
                parser.error("Blocking migration reviews remain; no YAML output was written")
            if args.output.resolve() == args.config.resolve():
                parser.error("Input is never overwritten; choose a new --output path and review its diff")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        return 0 if result.ready else 2
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
