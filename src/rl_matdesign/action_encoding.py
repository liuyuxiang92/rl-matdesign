"""Typed action features shared by composition and assignment environments.

An action has a label index and a value index. Labels and categorical values
are identities, encoded without an artificial numerical ordering. Numeric values
retain their real magnitudes. This module does not choose legal actions or alter
training algorithms; environments own those decisions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def _typed(value: Any) -> list:
    """A JSON-safe identity preserving scalar types and sequence positions."""
    if value is None:
        return ["none"]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Action identities must contain finite numbers")
        return ["float", 0.0 if number == 0 else number]
    if isinstance(value, (list, tuple)):
        return ["list" if isinstance(value, list) else "tuple", [_typed(item) for item in value]]
    if isinstance(value, Mapping):
        items = [[_typed(key), _typed(item)] for key, item in value.items()]
        items.sort(key=lambda pair: _json(pair[0]))
        return ["mapping", items]
    raise TypeError(f"Unsupported action identity type: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def typed_identity(value: Any) -> str:
    """Return a stable typed identity, independent of mapping insertion order."""
    return _json(_typed(value))


def _restore(node: Any) -> Any:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("Invalid typed action identity in schema")
    tag = node[0]
    if tag == "none" and len(node) == 1:
        return None
    if len(node) != 2:
        raise ValueError("Invalid typed action identity in schema")
    value = node[1]
    if tag == "bool" and isinstance(value, bool):
        return value
    if tag == "str" and isinstance(value, str):
        return value
    if tag == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tag == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    if tag in ("list", "tuple") and isinstance(value, list):
        restored = [_restore(item) for item in value]
        return restored if tag == "list" else tuple(restored)
    if tag == "mapping" and isinstance(value, list):
        pairs = []
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid mapping identity in action schema")
            pairs.append((_restore(pair[0]), _restore(pair[1])))
        try:
            mapping = dict(pairs)
        except TypeError as exc:
            raise ValueError("Action schema mapping keys must be hashable") from exc
        if len(mapping) != len(pairs):
            raise ValueError("Duplicate mapping keys in action schema")
        return mapping
    raise ValueError(f"Invalid typed action identity {tag!r} in schema")


def _entries(values: Sequence[Any], name: str) -> list:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a nonempty sequence")
    if not values:
        raise ValueError(f"{name} must be nonempty")
    return deepcopy(list(values))


def _index(index: int, size: int) -> int:
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
        raise TypeError("Action index must be an integer, not a boolean")
    index = int(index)
    if not 0 <= index < size:
        raise IndexError(f"Action index {index} is outside [0, {size})")
    return index


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _schema_header(schema: Mapping, encoding: str, keys: set[str]) -> None:
    if not isinstance(schema, Mapping) or set(schema) != keys:
        raise ValueError(f"Invalid {encoding} action schema fields")
    if type(schema["version"]) is not int or schema["version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported {encoding} action schema version")
    if schema["encoding"] != encoding:
        raise ValueError(f"Expected action encoding {encoding!r}")


class LabelEncoding:
    """One-hot label identities in the environment's declared index order.

    Labels can be namespaced tuples such as ``('peptide', 'position', 3)``.
    Construction does not reorder them: existing action indices must keep their
    meaning. A reordered vocabulary therefore has a different schema.
    """

    def __init__(self, labels: Sequence[Any]) -> None:
        self._labels = _entries(labels, "labels")
        keys = [typed_identity(label) for label in self._labels]
        if len(set(keys)) != len(keys):
            raise ValueError("Action labels must have unique typed identities")
        self._indices = {key: index for index, key in enumerate(keys)}
        self._features = _readonly(np.eye(len(keys), dtype=float))

    @property
    def labels(self) -> tuple:
        return tuple(deepcopy(self._labels))

    @property
    def features(self) -> np.ndarray:
        return self._features.view()

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]

    def __len__(self) -> int:
        return len(self._labels)

    def index_of(self, label: Any) -> int:
        try:
            return self._indices[typed_identity(label)]
        except KeyError:
            raise KeyError(f"Unknown action label {label!r}") from None

    def encode(self, index: int) -> np.ndarray:
        return self._features[_index(index, len(self))].view()

    def to_schema(self) -> dict:
        return {"version": SCHEMA_VERSION, "encoding": "label_one_hot",
                "labels": [_typed(label) for label in self._labels]}

    @classmethod
    def from_schema(cls, schema: Mapping) -> LabelEncoding:
        _schema_header(schema, "label_one_hot", {"version", "encoding", "labels"})
        if not isinstance(schema["labels"], list):
            raise ValueError("Action schema labels must be a list")
        return cls([_restore(label) for label in schema["labels"]])


@dataclass(frozen=True)
class ValueSpec:
    """A value's scientific type and optional group/field namespace.

    Numeric values share one real-valued feature channel. Categorical values
    get independent identity channels, including categorical numbers. Namespace
    separates identical-looking tokens belonging to different value domains.
    """

    value: Any
    kind: str = "categorical"
    namespace: Any = None


def _value_spec(spec: ValueSpec) -> ValueSpec:
    if spec.kind not in ("numeric", "categorical"):
        raise ValueError("Action value kind must be 'numeric' or 'categorical'")
    value = deepcopy(spec.value)
    namespace = deepcopy(spec.namespace)
    typed_identity(namespace)
    if spec.kind == "numeric":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("Numeric action values must be real numbers, not strings or booleans")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Numeric action values must be finite")
        value = 0.0 if value == 0 else value
    else:
        typed_identity(value)
    return ValueSpec(value, spec.kind, namespace)


def _value_identity(spec: ValueSpec) -> str:
    return _json([spec.kind, _typed(spec.namespace), _typed(spec.value)])


class ValueEncoding:
    """Encode values with numerical magnitudes or unordered categorical identity.

    ``ValueEncoding(values, kind='numeric')`` produces an N-by-1 table.
    ``kind='categorical'`` produces N-by-N one-hot features. ``from_specs``
    supports a mixed axis: one numeric channel, if needed, followed by the
    categorical channels. No categorical token is converted to an ordinal.
    """

    def __init__(self, values: Sequence[Any], *, kind: str = "numeric", namespace: Any = None) -> None:
        entries = _entries(values, "values")
        self._initialize([ValueSpec(value, kind, namespace) for value in entries])

    def _initialize(self, specs: Sequence[ValueSpec]) -> None:
        entries = _entries(specs, "value specs")
        if any(not isinstance(spec, ValueSpec) for spec in entries):
            raise TypeError("from_specs requires ValueSpec entries")
        self._specs = tuple(_value_spec(spec) for spec in entries)
        keys = [_value_identity(spec) for spec in self._specs]
        if len(set(keys)) != len(keys):
            raise ValueError("Action values must have unique identities within each namespace and kind")
        self._indices = {key: index for index, key in enumerate(keys)}
        self._has_numeric = any(spec.kind == "numeric" for spec in self._specs)
        categorical = [i for i, spec in enumerate(self._specs) if spec.kind == "categorical"]
        width = int(self._has_numeric) + len(categorical)
        features = np.zeros((len(self._specs), width), dtype=float)
        for index, spec in enumerate(self._specs):
            if spec.kind == "numeric":
                features[index, 0] = spec.value
        for column, index in enumerate(categorical, start=int(self._has_numeric)):
            features[index, column] = 1.0
        self._features = _readonly(features)

    @classmethod
    def from_specs(cls, specs: Sequence[ValueSpec]) -> ValueEncoding:
        result = cls.__new__(cls)
        result._initialize(specs)
        return result

    @property
    def specs(self) -> tuple[ValueSpec, ...]:
        return deepcopy(self._specs)

    @property
    def features(self) -> np.ndarray:
        return self._features.view()

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]

    def __len__(self) -> int:
        return len(self._specs)

    def index_of(self, value: Any, *, kind: str | None = None, namespace: Any = None) -> int:
        if kind is None:
            kinds = {spec.kind for spec in self._specs}
            if len(kinds) != 1:
                raise ValueError("A mixed value axis requires an explicit kind for lookup")
            kind = next(iter(kinds))
        key = _value_identity(_value_spec(ValueSpec(value, kind, namespace)))
        try:
            return self._indices[key]
        except KeyError:
            raise KeyError(f"Unknown {kind} action value {value!r} in namespace {namespace!r}") from None

    def encode(self, index: int) -> np.ndarray:
        return self._features[_index(index, len(self))].view()

    def to_schema(self) -> dict:
        return {"version": SCHEMA_VERSION, "encoding": "typed_value_features",
                "values": [{"kind": spec.kind, "namespace": _typed(spec.namespace),
                            "value": _typed(spec.value)} for spec in self._specs]}

    @classmethod
    def from_schema(cls, schema: Mapping) -> ValueEncoding:
        _schema_header(schema, "typed_value_features", {"version", "encoding", "values"})
        if not isinstance(schema["values"], list):
            raise ValueError("Action schema values must be a list")
        specs = []
        for item in schema["values"]:
            if not isinstance(item, Mapping) or set(item) != {"kind", "namespace", "value"}:
                raise ValueError("Invalid value specification in action schema")
            specs.append(ValueSpec(_restore(item["value"]), item["kind"], _restore(item["namespace"])))
        return cls.from_specs(specs)


class ActionEncoding:
    """Feature tables for complete ``(label_index, value_index)`` actions."""

    def __init__(self, labels: Sequence[Any] | LabelEncoding,
                 values: Sequence[Any] | ValueEncoding, *, value_kind: str = "numeric") -> None:
        self.labels = labels if isinstance(labels, LabelEncoding) else LabelEncoding(labels)
        self.values = values if isinstance(values, ValueEncoding) else ValueEncoding(values, kind=value_kind)

    def encode(self, label_index: int, value_index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.labels.encode(label_index), self.values.encode(value_index)

    def encode_many(self, index_pairs: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        pairs = list(index_pairs)
        if not pairs:
            return (np.empty((0, self.labels.feature_dim), dtype=float),
                    np.empty((0, self.values.feature_dim), dtype=float))
        labels, values = [], []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each action must have a label index and a value index")
            label, value = self.encode(*pair)
            labels.append(label)
            values.append(value)
        return np.stack(labels), np.stack(values)

    def to_schema(self) -> dict:
        return {"version": SCHEMA_VERSION, "encoding": "label_value_pair",
                "labels": self.labels.to_schema(), "values": self.values.to_schema()}

    @classmethod
    def from_schema(cls, schema: Mapping) -> ActionEncoding:
        _schema_header(schema, "label_value_pair", {"version", "encoding", "labels", "values"})
        return cls(LabelEncoding.from_schema(schema["labels"]), ValueEncoding.from_schema(schema["values"]))
