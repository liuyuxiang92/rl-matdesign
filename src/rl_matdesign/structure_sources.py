"""One immutable structure definition, deterministic samples, and asset evidence.

Resolving the singular structure declaration never reads assets or loads models.
The internal identity is always ``structure``; properties cannot select sources.
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

SOURCE_SCHEMA_VERSION = 2
SOURCE_SEED_VERSION = 1
_STRUCTURE_FIELDS = {"builder", "base_poscar", "n_random_configs", "steps", "structure_parallel",
                     "site_symbol", "site_map", "geo_opt", "md", "opt",
                     "ads_height", "ads_dz", "adsorbates", "geo_opt_model"}
_RUNTIME_KEYS = {"structure_parallel", "device", "devices", "num_workers", "batch_size",
                 "threads", "threads_per_worker", "reserve_devices",
                 "timeout_s", "log_level", "verbose", "output_dir"}

def _json_value(value: Any) -> Any:
    """Canonical JSON, including one representation for equal finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Structure configuration and seed identity require finite numbers")
        return int(number) if number.is_integer() else number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Structure configuration mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported structure configuration value: {type(value).__name__}")


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
class StructureSpec:
    """Frozen physical construction settings with a single internal sample ID."""
    _config_json: str
    _asset_hashes_json: str

    def __init__(self, builder_config: Mapping[str, Any], *, asset_hashes: Mapping[str, str] | None = None):
        if not isinstance(builder_config, Mapping):
            raise ValueError("structure must be a mapping")
        cfg = json.loads(_dump(builder_config))
        from .config_schema import PARAMETERS
        forbidden = set(cfg) & (set(PARAMETERS) | {"properties", "predictor", "reward", "groups"})
        if forbidden:
            raise ValueError(f"structure contains unrelated settings: {sorted(forbidden)}")
        cfg["builder"] = _name(cfg.get("builder"), "structure.builder")
        cfg["n_random_configs"] = _integer(cfg.get("n_random_configs", 1), "n_random_configs", minimum=1)
        if cfg["builder"] == "pipeline":
            from .predictors.builders.pipeline import validate_pipeline_config
            validate_pipeline_config(cfg)
        hashes = {} if asset_hashes is None else dict(asset_hashes)
        for name, digest in hashes.items():
            _name(name, "asset identity")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Asset identities require lowercase SHA-256 hex digests")
        object.__setattr__(self, "_config_json", _dump(cfg))
        object.__setattr__(self, "_asset_hashes_json", _dump(hashes))

    @property
    def source_id(self):
        return "structure"

    @property
    def builder(self):
        return self.effective_config["builder"]

    @property
    def n_random_configs(self):
        return self.effective_config["n_random_configs"]

    @property
    def effective_config(self):
        return json.loads(self._config_json)

    @property
    def physical_protocol(self):
        return {"version": SOURCE_SCHEMA_VERSION, "config": _physical(self.effective_config),
                "asset_sha256": json.loads(self._asset_hashes_json)}

    @property
    def protocol_fingerprint(self):
        return hashlib.sha256(_dump(self.physical_protocol).encode("utf-8")).hexdigest()

    def with_asset_hashes(self, asset_hashes):
        return type(self)(self.effective_config, asset_hashes=asset_hashes)

    def to_schema(self):
        return {"version": SOURCE_SCHEMA_VERSION, "source_id": self.source_id,
                "config": self.effective_config, "physical_protocol": self.physical_protocol,
                "protocol_fingerprint": self.protocol_fingerprint}

    def _physical_candidate(self, candidate):
        """Zero chemical amounts do not alter a pipeline's physical recipe."""
        cfg = self.effective_config
        if cfg["builder"] != "pipeline" or not isinstance(candidate, Mapping):
            return candidate
        chemical_groups = {value for step in cfg["steps"] for key, value in step.items()
                           if key.endswith(("_dopant_group", "_defect_group")) or key == "halide_group"}
        return {group: {label: amount for label, amount in values.items()
                        if not (isinstance(amount, Real) and not isinstance(amount, bool) and amount == 0)}
                if group in chemical_groups and isinstance(values, Mapping) else values
                for group, values in candidate.items()}

    def seed_for(self, candidate, *, eval_seed, purpose="build", template_id=None, sample_index=None):
        """Stable samples are independent of scheduling, call order, and reward settings."""
        from .utils.composition import composition_key
        payload = {"version": SOURCE_SEED_VERSION,
                   "eval_seed": _integer(eval_seed, "eval_seed"),
                   "candidate": composition_key(self._physical_candidate(candidate)), "source_id": self.source_id,
                   "physical_protocol": self.physical_protocol, "purpose": _name(purpose, "seed purpose"),
                   "template_id": composition_key(template_id),
                   "sample_index": None if sample_index is None else _integer(sample_index, "sample_index")}
        return int.from_bytes(hashlib.sha256(_dump(payload).encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class StructurePlan:
    structure: StructureSpec | None
    property_names: tuple[str, ...]

    @property
    def sources(self):
        """Execution protocol iteration, containing zero or one structure."""
        return () if self.structure is None else (self.structure,)

    @property
    def property_sources(self):
        return {name: "structure" for name in self.property_names}

    def source(self, source_id):
        if self.structure is None or source_id != "structure":
            raise KeyError(source_id)
        return self.structure


def resolve_structure_plan(cfg: Mapping[str, Any], *, needs_structure: Iterable[str]) -> StructurePlan:
    """Bind every structure-consuming property to the singular structure block."""
    if not isinstance(cfg, Mapping):
        raise ValueError("Configuration must be a mapping")
    removed = set(cfg) & {"structure_sources", "share_structure", "sweep"}
    if removed:
        raise ValueError(f"Removed structure settings: {sorted(removed)}; use structure and explicit properties")
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
        name = _name(str(prop.get("name", f"prop{index}")), "property name")
        if name in props:
            raise ValueError(f"Duplicate property name {name!r}")
        props[name] = prop
        if "source" in prop:
            raise ValueError(f"Property {name!r} implicitly uses the singular structure; source is unsupported")
        misplaced = set(prop) & _STRUCTURE_FIELDS
        if misplaced:
            raise ValueError(f"Property {name!r} has structure-owned fields: {sorted(misplaced)}")
    if set(needed) - props.keys():
        raise ValueError(f"Unknown structure properties: {sorted(set(needed) - props.keys())}")
    misplaced = set(cfg) & _STRUCTURE_FIELDS
    if misplaced:
        raise ValueError(f"Top-level settings conflict with structure ownership: {sorted(misplaced)}")
    structure = cfg.get("structure")
    if structure is None:
        if needed:
            raise ValueError("Structure properties require a structure block")
        return StructurePlan(None, ())
    if not isinstance(structure, Mapping):
        raise ValueError("structure must be a mapping")
    if structure.get("builder") == "peptide":
        if needed:
            raise ValueError("The peptide design assembler does not produce ASE structures")
        return StructurePlan(None, ())
    if not needed:
        raise ValueError("structure is declared but no property consumes structures")
    return StructurePlan(StructureSpec(structure), needed)


def resolve_source_assets(spec: StructureSpec, *, base_dir: str | Path | None = None):
    """Freeze known physical files; explicitly report unknown plug-in ownership.

    Unlike pure declaration resolution this optional runtime step performs file I/O. Built-in
    templates and model files are relative to runtime cwd. An external builder's
    relative paths may follow another convention and are never guessed.
    """
    from .utils.provenance import sha256_file

    base = Path.cwd() if base_dir is None else Path(base_dir).absolute()
    builtins = {"pipeline"}
    template_keys = {"base_poscar"}
    model_keys = {"model", "models", "model_file", "model_files",
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
            known = owner_known
            if key in template_keys:
                for index, entry in enumerate(child if isinstance(child, (list, tuple)) else [child]):
                    label = f"{name}[{index}]"
                    if isinstance(entry, Mapping):
                        for alias in ("path",):
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
            elif isinstance(child, (list, tuple)):
                for index, entry in enumerate(child):
                    if isinstance(entry, Mapping):
                        visit(entry, f"{name}[{index}]", known)

    visit(spec.effective_config, "", spec.builder in builtins)
    hashes = dict(spec.physical_protocol["asset_sha256"])
    for reference in references:
        hashes.pop(reference["field"], None)
    hashes.update({r["field"]: r["sha256"] for r in references if r["status"] == "resolved"})
    return spec.with_asset_hashes(hashes), references
