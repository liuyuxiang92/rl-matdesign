"""Immutable structure-source definitions and versioned local random seeds.

Normalization never loads a model/template, constructs a builder, starts a
worker, or changes prediction statistics. The optional asset resolver hashes
declared files before runtime evaluation. The
caller supplies which properties need structures after resolving its adapters.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any


SOURCE_SCHEMA_VERSION = 1
SOURCE_SEED_VERSION = 1
_SHARED_LEGACY_KEYS = {
    "builder", "base_poscar", "poscar", "poscar_template", "site_symbol",
    "n_random_configs", "structure_parallel",
}
_REWARD_KEYS = {
    "name", "properties", "predictor", "reward", "sweep", "k", "direction", "weight",
    "scale", "objective", "transform", "target_value", "target_tolerance",
    "predict_cache_size", "share_structure", "source", "structure_sources",
    "seed", "rng_seed", "eval_seed", "dp_seed", "train_seed", "gen_seed",
}
_RUNTIME_KEYS = {
    "structure_parallel", "device", "devices", "num_workers", "batch_size",
    "threads", "threads_per_worker", "workers_per_device", "reserve_devices",
    "timeout_s", "log_level", "verbose", "output_dir",
}


def _json_value(value: Any) -> Any:
    """Canonical JSON, including one representation for equal finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Source configuration and seed identity require finite numbers")
        return int(number) if number.is_integer() else number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Source configuration mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported source configuration value: {type(value).__name__}")


def _dump(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _physical(value: Any) -> Any:
    """Execution placement is not a physical sampling condition."""
    if isinstance(value, dict):
        return {key: _physical(item) for key, item in value.items() if key not in _RUNTIME_KEYS}
    if isinstance(value, list):
        return [_physical(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class StructureSourceSpec:
    """Frozen source declaration; exported dictionaries are defensive copies.

    The physical fingerprint initially covers configuration, including declared
    paths. It does NOT prove those paths still contain the same bytes. An asset
    resolver must call ``with_asset_hashes`` before evaluation to include verified
    content identities. Normalization deliberately performs no asset I/O.
    """

    source_id: str
    _config_json: str
    _asset_hashes_json: str

    def __init__(self, source_id: str, builder_config: Mapping[str, Any], *,
                 geo_opt: Mapping[str, Any] | None = None,
                 asset_hashes: Mapping[str, str] | None = None) -> None:
        source_id = _name(source_id, "source id")
        if not isinstance(builder_config, Mapping):
            raise ValueError(f"Source {source_id!r} must be a mapping")
        from .config_schema import PARAMETERS
        cfg = json.loads(_dump(builder_config))
        forbidden = set(cfg) & (_REWARD_KEYS | set(PARAMETERS))
        if forbidden:
            raise ValueError(f"Source {source_id!r} contains property/reward settings: {sorted(forbidden)}")
        cfg["builder"] = _name(cfg.get("builder", "substitute"), "builder")
        cfg["n_random_configs"] = _integer(cfg.get("n_random_configs", 1),
                                            "n_random_configs", minimum=1)
        if geo_opt is not None:
            if "geo_opt" in cfg and _dump(cfg["geo_opt"]) != _dump(geo_opt):
                raise ValueError(f"Source {source_id!r} has conflicting geo_opt definitions")
            cfg["geo_opt"] = json.loads(_dump(geo_opt))
        go = cfg.get("geo_opt")
        if go is None:
            cfg["geo_opt"] = {}
        elif not isinstance(go, dict):
            raise ValueError(f"Source {source_id!r} geo_opt must be a mapping")
        if "enabled" in cfg["geo_opt"] and not isinstance(cfg["geo_opt"]["enabled"], bool):
            raise ValueError("geo_opt.enabled must be a boolean")
        hashes = {} if asset_hashes is None else dict(asset_hashes)
        for name, digest in hashes.items():
            _name(name, "asset identity")
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("Asset identities require lowercase SHA-256 hex digests")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "_config_json", _dump(cfg))
        object.__setattr__(self, "_asset_hashes_json", _dump(hashes))

    @property
    def builder(self) -> str:
        return self.effective_config["builder"]

    @property
    def n_random_configs(self) -> int:
        return self.effective_config["n_random_configs"]

    @property
    def effective_config(self) -> dict:
        return json.loads(self._config_json)

    @property
    def builder_config(self) -> dict:
        cfg = self.effective_config
        cfg.pop("geo_opt")
        return cfg

    @property
    def geo_opt(self) -> dict:
        return self.effective_config["geo_opt"]

    @property
    def physical_protocol(self) -> dict:
        cfg = _physical(self.effective_config)
        # Search-domain menus and filters are not structure-generation physics.
        # These built-ins do not consume group declarations at all; SSE reads
        # only the named S-site's ordered slot labels and element identities.
        if self.builder in {"substitute", "site_pick", "defect_site"}:
            cfg.pop("groups", None)
        elif self.builder == "sse" and "groups" in cfg:
            s_group = str(cfg.get("s_site_group", "S_site"))
            cfg["groups"] = [
                {"name": group.get("name"), "choices": [
                    {key: choice[key] for key in ("name", "element") if key in choice}
                    for choice in group.get("choices", []) or []
                ]}
                for group in cfg["groups"] or [] if str(group.get("name", "")) == s_group
            ]
        return {"version": SOURCE_SCHEMA_VERSION, "config": cfg,
                "asset_sha256": json.loads(self._asset_hashes_json)}

    @property
    def protocol_fingerprint(self) -> str:
        return hashlib.sha256(_dump(self.physical_protocol).encode("utf-8")).hexdigest()

    def with_asset_hashes(self, asset_hashes: Mapping[str, str]) -> StructureSourceSpec:
        """Return a new declaration with resolver-verified asset content hashes."""
        return type(self)(self.source_id, self.effective_config, asset_hashes=asset_hashes)

    def to_schema(self) -> dict:
        return {"version": SOURCE_SCHEMA_VERSION, "source_id": self.source_id,
                "config": self.effective_config, "physical_protocol": self.physical_protocol,
                "protocol_fingerprint": self.protocol_fingerprint}

    def seed_for(self, candidate: Any, *, eval_seed: int, purpose: str = "build",
                 template_id: Any = None, sample_index: int | None = None) -> int:
        """Derive a uint64 seed, independent of call order or process scheduling.

        Use a distinct purpose for each stochastic operation. Backends limited
        to 32 bits may use ``seed_for(..., purpose='backend-name:32bit') % 2**32``;
        that deterministic mapping must be recorded by the adapter. This module
        never creates a second user-configurable random seed.
        """
        from .utils.composition import composition_key

        payload = {
            "version": SOURCE_SEED_VERSION,
            "eval_seed": _integer(eval_seed, "eval_seed"),
            "candidate": composition_key(candidate),
            "source_id": self.source_id,
            "physical_protocol": self.physical_protocol,
            "purpose": _name(purpose, "seed purpose"),
            "template_id": composition_key(template_id),
            "sample_index": None if sample_index is None else _integer(sample_index, "sample_index"),
        }
        digest = hashlib.sha256(_dump(payload).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class StructureSourcePlan:
    sources: tuple[StructureSourceSpec, ...]
    _property_sources_json: str
    migration_notes: tuple[str, ...] = ()

    @property
    def property_sources(self) -> dict[str, str]:
        return json.loads(self._property_sources_json)

    def source(self, source_id: str) -> StructureSourceSpec:
        for spec in self.sources:
            if spec.source_id == source_id:
                return spec
        raise KeyError(source_id)


class LegacySourceMigrationRequired(ValueError):
    """An old independent-sampling config needs an explicit source decision."""

    def __init__(self, report: dict) -> None:
        self._report_json = _dump(report)
        super().__init__(
            "share_structure: false requires explicit structure_sources and property source references. "
            "Inspect migration_report: separate sources preserve independent sampling; assigning "
            "the same source deliberately changes sampling correlation."
        )

    @property
    def migration_report(self) -> dict:
        return json.loads(self._report_json)


def _builder_keys() -> set[str]:
    from .config_schema import DELEGATED_KEYS

    return set(DELEGATED_KEYS["builders"]) | {"md", "opt", "groups", "n_random_configs", "geo_opt"}


def _legacy_builder_config(cfg: Mapping, *, shared: bool) -> dict:
    """Preserve declared builder inputs, excluding unrelated run/model settings."""
    from .config_schema import ALIASES, DELEGATED_KEYS, PARAMETERS

    known = set(PARAMETERS) | set(ALIASES) | set().union(*DELEGATED_KEYS.values())
    allowed = _builder_keys() | {"structure_parallel"}
    selected = {key: value for key, value in cfg.items() if key in allowed}
    if ":" in str(cfg.get("builder", "")):
        # A legacy FQN builder may own extra declared fields. Do not silently
        # drop them; explicit source definitions subsequently own these keys.
        selected.update({key: value for key, value in cfg.items()
                         if key not in known | _REWARD_KEYS})
    return selected


def normalize_structure_sources(cfg: Mapping[str, Any], *,
                                needs_structure: Iterable[str]) -> StructureSourcePlan:
    """Normalize source declarations without importing or probing any backend.

    ``needs_structure`` contains property names selected by the caller's resolved
    adapter capabilities. Dictionary equality never merges declared sources.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("Configuration must be a mapping")
    if isinstance(needs_structure, (str, bytes, Mapping)):
        raise ValueError("needs_structure must be an iterable of property names")
    needed = tuple(_name(name, "property name") for name in needs_structure)
    if len(set(needed)) != len(needed):
        raise ValueError("needs_structure contains duplicate property names")
    raw_props = cfg.get("properties", [])
    if not isinstance(raw_props, (list, tuple)) or any(not isinstance(p, Mapping) for p in raw_props):
        raise ValueError("properties must be a list of mappings")
    props = {}
    for index, prop in enumerate(raw_props):
        # Match StructureScorePredictor's existing generated objective names.
        name = _name(str(prop.get("name", f"prop{index}")), "property name")
        if name in props:
            raise ValueError(f"Duplicate property name {name!r}")
        props[name] = prop
    missing = set(needed) - props.keys()
    if missing:
        raise ValueError(f"Unknown structure properties: {sorted(missing)}")
    for name, prop in props.items():
        if name not in needed and "source" in prop:
            raise ValueError(f"Composition/sequence property {name!r} cannot reference a structure source")

    if "structure_sources" not in cfg:
        if any("source" in prop for prop in props.values()):
            raise ValueError("Property source references require structure_sources")
        if not needed:
            return StructureSourcePlan((), _dump({}))
        shared = cfg.get("share_structure", True)
        if not isinstance(shared, bool):
            raise ValueError("share_structure must be a boolean")
        if not shared:
            inherited = {key: cfg[key] for key in _SHARED_LEGACY_KEYS if key in cfg}
            suggestions, references = {}, {}
            for name in needed:
                effective = _legacy_builder_config({**inherited, **props[name]}, shared=False)
                # Old false did not inherit groups, and post-build relaxation
                # was global even if the property supplied its own geo_opt.
                effective.setdefault("groups", [])
                effective["geo_opt"] = cfg.get("geo_opt") or {}
                source_id = f"property:{name}"
                suggestions[source_id] = StructureSourceSpec(source_id, effective).effective_config
                references[name] = source_id
            raise LegacySourceMigrationRequired({
                "reason": "legacy_independent_sampling", "structure_sources": suggestions,
                "property_sources": references,
                "decision": "Keep distinct source names for independent ensembles, or explicitly share a name.",
                "ignored_top_level_builder_fields": sorted(set(_legacy_builder_config(cfg, shared=True)) -
                                                           _SHARED_LEGACY_KEYS - {"geo_opt"}),
            })
        source = StructureSourceSpec("default", _legacy_builder_config(cfg, shared=True))
        notes = ["Legacy shared structure settings normalized to source 'default'."]
        for name in needed:
            ignored = set(props[name]) & (_builder_keys() - {"groups"} | {"structure_parallel"})
            if ignored:
                notes.append(f"Legacy shared mode ignored property {name!r} builder fields: {sorted(ignored)}")
        return StructureSourcePlan((source,), _dump({name: "default" for name in needed}), tuple(notes))

    raw_sources = cfg["structure_sources"]
    if not isinstance(raw_sources, Mapping):
        raise ValueError("structure_sources must be a mapping of names to definitions")
    conflicting = set(cfg) & ((_builder_keys() - {"groups"}) | {"share_structure"})
    if conflicting:
        raise ValueError(f"Explicit structure_sources conflict with legacy top-level fields: {sorted(conflicting)}")
    sources = []
    for name, definition in raw_sources.items():
        if not isinstance(definition, Mapping):
            raise ValueError(f"Source {name!r} must be a mapping")
        resolved = dict(definition)
        if "structure_parallel" in cfg:
            resolved.setdefault("structure_parallel", cfg["structure_parallel"])
        # SSE uses group names/choice element labels when assembling O-form
        # slots. Keep this context without copying property/reward settings.
        if "groups" in cfg:
            resolved.setdefault("groups", cfg["groups"])
        sources.append(StructureSourceSpec(name, resolved))
    source_names = {spec.source_id for spec in sources}
    references = {}
    for name in needed:
        prop = props[name]
        inline = set(prop) & (_builder_keys() | {"structure_parallel"})
        if inline:
            raise ValueError(f"Property {name!r} has source-owned settings: {sorted(inline)}")
        reference = prop.get("source")
        if reference is None:
            if len(sources) != 1:
                raise ValueError(f"Property {name!r} must explicitly name a source when there is not exactly one")
            reference = sources[0].source_id
        if not isinstance(reference, str) or reference not in source_names:
            raise ValueError(f"Property {name!r} references unknown source {reference!r}")
        references[name] = reference
    unused = source_names - set(references.values())
    if unused:
        raise ValueError(f"Unused structure sources: {sorted(unused)}")
    return StructureSourcePlan(tuple(sources), _dump(references))


def resolve_source_assets(spec: StructureSourceSpec, *, base_dir: str | Path | None = None):
    """Freeze known physical files; explicitly report unknown plug-in ownership.

    Unlike normalization this optional runtime step performs file I/O. Built-in
    templates and model files are relative to runtime cwd. The post-build
    ``geo_opt`` calculator also belongs to this package. An external builder's
    relative paths may follow another convention and are never guessed.
    """
    from .utils.provenance import sha256_file

    base = Path.cwd() if base_dir is None else Path(base_dir).absolute()
    builtins = {"substitute", "sse", "site_pick", "defect_site"}
    template_keys = {"base_poscar", "poscar", "poscar_template"}
    model_keys = {"model", "models", "dp_model", "dp_models", "model_file", "model_files",
                  "model_path", "checkpoint", "checkpoint_path", "weights"}
    references = []

    def file(value, field, owner_known):
        if not isinstance(value, (str, Path)) or not str(value):
            return
        path = Path(value)
        record = {"field": field, "declared": str(value)}
        if not path.is_absolute() and not owner_known:
            references.append({**record, "status": "unresolved", "path": None,
                               "reason": "External builder owns relative path resolution"})
            return
        path = path if path.is_absolute() else base / path
        record["path"] = str(path.absolute())
        try:
            if path.is_file():
                record.update(status="resolved", sha256=sha256_file(path), size=path.stat().st_size)
            else:
                record.update(status="missing", reason="Not a readable regular file")
        except OSError as exc:
            record.update(status="unresolved", reason=str(exc))
        references.append(record)

    def visit(value, field, owner_known):
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            name = f"{field}.{key}" if field else key
            known = True if key == "geo_opt" else owner_known
            if key in template_keys:
                for index, entry in enumerate(child if isinstance(child, (list, tuple)) else [child]):
                    label = f"{name}[{index}]"
                    if isinstance(entry, Mapping):
                        for alias in ("path", "base_poscar", "poscar"):
                            if entry.get(alias):
                                file(entry[alias], f"{label}.{alias}", known)
                                break
                    else:
                        file(entry, label, known)
            elif key in model_keys:
                for index, entry in enumerate(child if isinstance(child, (list, tuple)) else [child]):
                    file(entry, f"{name}[{index}]", known)
            elif isinstance(child, Mapping):
                visit(child, name, known)

    visit(spec.effective_config, "", spec.builder in builtins)
    go = spec.geo_opt
    if go and go.get("enabled", True) and not go.get("model"):
        file("models/DPA-3.1-3M.pt", "geo_opt.model[default]", True)
    hashes = dict(spec.physical_protocol["asset_sha256"])
    for reference in references:
        hashes.pop(reference["field"], None)
    hashes.update({r["field"]: r["sha256"] for r in references if r["status"] == "resolved"})
    return spec.with_asset_hashes(hashes), references
