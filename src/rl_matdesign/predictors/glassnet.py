"""Composition-only, named physical outputs from the audited GlassNet model.

This leaf is deliberately not a scalar reward predictor. It never chooses an
objective, fabricates an uncertainty, downloads assets, or fits a scaler.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
from importlib import metadata, util
import json
import math
from numbers import Real
from pathlib import Path
import re
from threading import RLock
from typing import Any

from .outputs import OUTPUT_SCHEMA_VERSION, OutputSchema, ScientificOutput


GLASSPY_COMMIT = "ddc06240152749d14d01e0c4f7cec9e4e79c706d"
GLASSPY_VERSION = "0.6.0"
_SOURCE_SHA256 = "fcaff450cef398c4955ee36d00591e27bb750de904ece37c4145e495ac4a599d"
_WEIGHT_SHA256 = {
    "GlassNetMH.pth": "3a5d19fe43e6d97350b5f65151b2705c6cfe49c111cb45cf28933eeea2ee9a0c",
    "st-nn/Tg.pth": "1c9f881c6dc0eed4a3767f4c169edfb0abc53033e84b1e4f1a7f35d2976a41d4",
    "GlassNetMH_lc.p": "dd89a5f2dbed5c5d9144a83023385f3f5f697772ca2d822c5825685032c452e7",
    "st-nn/Tg_lc.p": "22d7146b6da3f53e79db1dc3ce4e3f92be6283603a911ab99fac441f09ba1134",
}
_AUDITED_SCALER_SHA256 = {
    "GlassNet_scaler_x.joblib": "da48e50966b684175e8b0d9b3e010efed4c71c914ea44834fcbee8faa526614e",
    "GlassNet_scaler_y.joblib": "e5178856e031ff8215743d6220b464cb2bfd33b77d44779dbea543e47b527d34",
}
DEFAULT_OXIDES = ("SiO2", "B2O3", "Al2O3", "Na2O", "K2O", "CaO", "MgO")
_OUTPUT_SCHEMA = {
    "CTEbelowTg": {
        "unit": "K^-1", "source_unit": "log10(K^-1)",
        "transform": "10**x", "head": "GlassNetMH", "uncertainty": None,
    },
    "Tg": {
        "unit": "K", "source_unit": "K", "transform": "identity",
        "head": "Tg", "uncertainty": None,
    },
    "T3": {
        "unit": "K", "source_unit": "K", "transform": "identity",
        "head": "GlassNetMH", "uncertainty": None,
        "description": "Temperature at viscosity 10^3 Pa.s",
    },
}
_Key = tuple[tuple[str, int, int], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(root: Path) -> str:
    """Hash the author's Python sources and chemical feature table, not pyc."""
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.csv")]):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _stock_assets() -> tuple[dict[str, Any], Path]:
    """Validate local assets *before* GlassPy's import-time download hook."""
    spec = util.find_spec("glasspy")
    if spec is None or spec.origin is None:
        raise ImportError(
            f"Install GlassPy from commit {GLASSPY_COMMIT} and prepare its local "
            "models/scalers before using GlassNet. See docs/glassnet_oracle.md."
        )
    root = Path(spec.origin).resolve().parent
    try:
        installed_version = metadata.version("glasspy")
    except metadata.PackageNotFoundError:
        # The audited checkout can be imported through PYTHONPATH. GlassPy uses
        # its pyproject version in this case; check it without importing GlassPy.
        project = root.parent / "pyproject.toml"
        content = project.read_text() if project.is_file() else ""
        match = re.search(r'^version\s*=\s*[\'"]([^\'"]+)', content, re.MULTILINE)
        installed_version = match.group(1) if match else "unknown"
    if installed_version != GLASSPY_VERSION or _source_digest(root) != _SOURCE_SHA256:
        raise RuntimeError(f"GlassNet requires the audited GlassPy source at {GLASSPY_COMMIT}.")

    from platformdirs import user_data_dir

    data_dir = Path(user_data_dir("GlassPy")).resolve()
    versions_path = data_dir / "versions.json"
    if not versions_path.is_file():
        raise FileNotFoundError(f"Prepare GlassPy assets first: missing {versions_path}.")
    versions = json.loads(versions_path.read_text())
    expected_versions = {
        "glasspy": GLASSPY_VERSION, "sklearn": metadata.version("scikit-learn"),
    }
    if not isinstance(versions, dict) or any(
        versions.get(key) != value for key, value in expected_versions.items()
    ):
        raise RuntimeError(
            "GlassPy cache versions do not match this environment; importing it "
            "could delete/redownload the cache. Prepare a compatible cache first."
        )
    hashes: dict[str, str] = {}
    for name in (*_WEIGHT_SHA256, *_AUDITED_SCALER_SHA256):
        path = data_dir / "models" / name
        if not path.is_file():
            raise FileNotFoundError(f"Prepare GlassNet assets first: missing {path}.")
        hashes[name] = _sha256(path)
        if name in _WEIGHT_SHA256 and hashes[name] != _WEIGHT_SHA256[name]:
            raise RuntimeError(f"GlassNet asset checksum mismatch: {path}.")
    dependencies = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightning"):
        dependencies[package] = metadata.version(package)
    return {
        "backend": "stock_glasspy", "glasspy_version": GLASSPY_VERSION,
        "source_commit": GLASSPY_COMMIT, "source_sha256": _SOURCE_SHA256,
        "assets_sha256": hashes, "dependencies": dependencies,
        "matches_audited_scalers": all(
            hashes[name] == digest for name, digest in _AUDITED_SCALER_SHA256.items()
        ),
    }, data_dir


class GlassNetPredictor:
    """Evaluate oxide mole amounts with one persistent CPU GlassNet instance.

    Parameters
    ----------
    oxide_basis
        Allowed binary oxide labels. Missing labels mean zero. Amounts are
        nonnegative mole ratios: fractions, mol.% and integer mole units are
        equivalent. Mass fractions and elemental atomic fractions are not.
    cache_size
        Maximum number of scalar result records in the per-instance LRU cache;
        zero disables persistent caching. This is not the unique-budget ledger.
    batch_size
        Maximum distinct candidates sent to one stock ``predict`` call.

    Notes
    -----
    ``_model_factory`` is a test seam. Injected models bypass asset validation
    and their identity explicitly says ``injected_unvalidated``. Stock inference
    uses its original feature engineering/scalers and ``st_models=['Tg']``.
    This class returns named physical values, not legacy reward/std tuples.
    """

    def __init__(
        self,
        *,
        oxide_basis: Sequence[str] = DEFAULT_OXIDES,
        cache_size: int = 4096,
        batch_size: int = 128,
        _model_factory: Callable[[], Any] | None = None,
    ) -> None:
        if isinstance(oxide_basis, (str, bytes)) or not oxide_basis:
            raise ValueError("oxide_basis must be a nonempty sequence of oxide labels.")
        if any(
            not isinstance(label, str)
            or re.fullmatch(r"(?!O(?:[0-9]|O|$))[A-Z][a-z]?(?:[1-9][0-9]*)?O(?:[1-9][0-9]*)?", label) is None
            for label in oxide_basis
        ):
            raise ValueError("oxide_basis requires binary oxide labels such as SiO2 or Na2O.")
        if len(set(oxide_basis)) != len(oxide_basis):
            raise ValueError("oxide_basis contains duplicate labels.")
        for name, value, minimum in (("cache_size", cache_size, 0), ("batch_size", batch_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self._oxide_basis = tuple(sorted(oxide_basis))
        self._cache_size = cache_size
        self._batch_size = batch_size
        self._model_factory = _model_factory
        self._model: Any = None
        self._identity: dict[str, Any] | None = None
        self._asset_identity: dict[str, Any] | None = None
        self._data_dir: Path | None = None
        self._cache: OrderedDict[_Key, dict[str, float]] = OrderedDict()
        self._lock = RLock()
        self._counts = dict(requests=0, cache_hits=0, batch_duplicates=0, model_batches=0, model_candidates=0)

    @property
    def oxide_basis(self) -> tuple[str, ...]:
        """Fixed named input basis for this model/cache lifetime."""
        return self._oxide_basis

    @property
    def cache_size(self) -> int:
        """Fixed maximum number of cached candidate records."""
        return self._cache_size

    @property
    def batch_size(self) -> int:
        """Fixed inference batching protocol for this oracle instance."""
        return self._batch_size

    @property
    def output_schema(self) -> dict[str, OutputSchema]:
        """Declare physical output units; raw head/conversion metadata is in identity."""
        return {name: OutputSchema(name, record['unit'], 'physical')
                for name, record in _OUTPUT_SCHEMA.items()}

    @property
    def identity(self) -> dict[str, Any]:
        """Return JSON-serializable provenance without loading neural weights."""
        with self._lock:
            if self._identity is None:
                if self._model_factory is None:
                    self._identity, self._data_dir = _stock_assets()
                    self._asset_identity = deepcopy(self._identity)
                else:
                    self._identity = {"backend": "injected_unvalidated"}
                self._identity.update({
                    "adapter_schema_version": 2, "oxide_basis": list(self.oxide_basis),
                    "scientific_output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "input_units": "oxide mole ratios",
                    "outputs": {name: schema.to_dict() for name, schema in self.output_schema.items()},
                    "output_metadata": deepcopy(_OUTPUT_SCHEMA),
                    "st_models": ["Tg"], "device": "cpu", "model_dtype": "float32",
                    "preprocessing": "stock physchem_featurizer, eaw, sum=1, official x/y scalers",
                    "batch_size": self.batch_size,
                })
            return deepcopy(self._identity)

    def canonical_key(self, composition: Mapping[str, Real]) -> _Key:
        """Preserve exact decimal ratios, independent of mapping/action order."""
        if not isinstance(composition, Mapping) or not composition:
            raise ValueError("Composition must be a nonempty mapping of oxides to mole amounts.")
        unknown = set(composition) - set(self.oxide_basis)
        if unknown:
            raise ValueError(f"Composition has labels outside oxide_basis: {sorted(map(str, unknown))}.")
        values: dict[str, Fraction] = {}
        for label, amount in composition.items():
            if isinstance(amount, bool) or not isinstance(amount, Real):
                raise ValueError(f"Mole amount for {label} must be a finite nonnegative number.")
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"Mole amount for {label} must be finite and nonnegative.")
            if amount:
                values[label] = Fraction(str(amount))
        total = sum(values.values(), Fraction(0))
        if total <= 0:
            raise ValueError("At least one oxide mole amount must be positive.")
        return tuple(
            (label, ratio.numerator, ratio.denominator)
            for label, value in sorted(values.items())
            for ratio in (value / total,)
        )

    def _load_model(self) -> Any:
        if self._model is None:
            _ = self.identity
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                # A manifest may be requested long before the first prediction.
                # Recheck before import/loading so it cannot describe old
                # scalers or a different cache directory than the loaded model.
                current_identity, current_dir = _stock_assets()
                if current_identity != self._asset_identity or current_dir != self._data_dir:
                    raise RuntimeError(
                        "GlassNet assets changed after identity was captured. "
                        "Create a new oracle and review its complete identity."
                    )
                from glasspy.predict import GlassNet
                from glasspy.predict.base import _BASEMODELPATH

                if Path(_BASEMODELPATH).resolve() != self._data_dir / "models":
                    raise RuntimeError("Already-imported GlassPy uses a different asset directory.")
                self._model = GlassNet(st_models=["Tg"])
        return self._model

    def _evaluate(self, keys: list[_Key]) -> list[dict[str, float]]:
        rows = []
        for key in keys:
            amounts = {label: float(Fraction(num, den) * 100) for label, num, den in key}
            rows.append([amounts.get(label, 0.0) for label in self.oxide_basis])
        model = self._load_model()
        self._counts["model_batches"] += 1
        self._counts["model_candidates"] += len(keys)
        frame = model.predict(rows, input_cols=list(self.oxide_basis), return_dataframe=True)
        try:
            columns = {name: list(frame[name]) for name in _OUTPUT_SCHEMA}
        except (KeyError, TypeError) as exc:
            raise ValueError("GlassNet did not return the required CTEbelowTg/Tg/T3 columns.") from exc
        if any(len(values) != len(keys) for values in columns.values()):
            raise ValueError("GlassNet returned an unexpected number of prediction rows.")
        results = []
        for index in range(len(keys)):
            values = {name: float(column[index]) for name, column in columns.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("GlassNet returned a nonfinite scientific prediction.")
            try:
                values["CTEbelowTg"] = 10.0 ** values["CTEbelowTg"]
            except OverflowError as exc:
                raise ValueError("GlassNet CTE conversion overflowed.") from exc
            if not all(math.isfinite(value) and value > 0 for value in values.values()):
                raise ValueError("GlassNet returned a nonpositive CTE or absolute temperature.")
            results.append(values)
        return results

    def predict_outputs(self, composition: Mapping[str, Real]) -> dict[str, ScientificOutput]:
        """Return named physical point predictions, with every spread absent."""
        return self.batch_predict_outputs([composition])[0]

    def batch_predict_outputs(
        self, compositions: Sequence[Mapping[str, Real]],
    ) -> list[dict[str, ScientificOutput]]:
        schemas = self.output_schema
        return [{name: ScientificOutput(schemas[name], value) for name, value in values.items()}
                for values in self._batch_predict_values(compositions)]

    def _batch_predict_values(
        self, compositions: Sequence[Mapping[str, Real]],
    ) -> list[dict[str, float]]:
        """Predict in order, coalescing duplicates and bounding inference batches.

        A failed model batch is not cached as a penalty or as a partial success.
        Earlier completed batches remain cached. Returned dicts cannot mutate
        cached records. No composition/structure list for the full search space
        is created; the caller controls the input/output list size.
        """
        if isinstance(compositions, (str, bytes)) or not isinstance(compositions, Sequence):
            raise ValueError("Batch input must be a sequence of named oxide compositions.")
        results = []
        with self._lock:
            for start in range(0, len(compositions), self.batch_size):
                keys = [self.canonical_key(c) for c in compositions[start:start + self.batch_size]]
                available: dict[_Key, dict[str, float]] = {}
                missing: dict[_Key, None] = {}
                self._counts["requests"] += len(keys)
                for key in keys:
                    if key in self._cache:
                        available[key] = self._cache[key]
                        self._cache.move_to_end(key)
                        self._counts["cache_hits"] += 1
                    elif key in missing:
                        self._counts["batch_duplicates"] += 1
                    else:
                        missing[key] = None
                if missing:
                    evaluated = self._evaluate(list(missing))
                    for key, values in zip(missing, evaluated):
                        available[key] = values
                for key in keys:
                    if self.cache_size:
                        # Recency follows request order, including duplicates,
                        # rather than the order of distinct inference rows.
                        self._cache[key] = available[key]
                        self._cache.move_to_end(key)
                        while len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
                    results.append(dict(available[key]))
        return results

    def cache_info(self) -> dict[str, int]:
        """Return request/work counters, not a scientific unique-query budget."""
        with self._lock:
            return {**self._counts, "cache_entries": len(self._cache), "cache_capacity": self.cache_size}

    def clear_cache(self) -> None:
        """Evict result records while retaining the model and work counters."""
        with self._lock:
            self._cache.clear()
