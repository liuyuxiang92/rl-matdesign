"""Exact property lookup on a dense, named Cartesian grid.

Use ``predictor: rl_matdesign.predictors.lookup:LookupPredictor`` with a
``model`` path to an NPZ archive containing:

* ``field_names``: a nonempty, unique, one-dimensional Unicode string array;
* ``axis_0``, ..., ``axis_n-1``: nonempty, unique one-dimensional numeric or
  string arrays, corresponding to the names in ``field_names``;
* ``mean``: a finite real array with shape ``tuple(len(axis) for axis in axes)``;
* optional ``std``: a finite, nonnegative real scalar or array of that shape.

Candidates are either ``{field: value}`` or ``{group: {field: value}}`` with
exactly one group. Values must lie exactly on the stored grid; this predictor
does no interpolation, fitting, or nearest-neighbor matching. It returns raw
values and leaves reward direction and uncertainty penalties to the engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import numpy as np


class LookupPredictor:
    """Load one read-only table once and look up individual candidates."""

    def __init__(self, cfg: dict, *, seed: int | None = None) -> None:
        model = cfg.get("model")
        if not isinstance(model, (str, PathLike)) or not str(model):
            raise ValueError("LookupPredictor requires a nonempty 'model' path")

        archive = np.load(model, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("LookupPredictor model must be an NPZ archive")
        with archive:
            names = self._read(archive, "field_names")
            if names.ndim != 1 or names.dtype.kind != "U" or not names.size:
                raise ValueError("field_names must be a nonempty 1-D Unicode string array")
            self._field_names = tuple(names.tolist())
            if any(not name for name in self._field_names):
                raise ValueError("field_names must not contain empty names")
            if len(set(self._field_names)) != len(self._field_names):
                raise ValueError("field_names must contain unique names")
            self._field_set = frozenset(self._field_names)

            axes = []
            indices = []
            string_axes = []
            for i, name in enumerate(self._field_names):
                key = f"axis_{i}"
                axis = self._read(archive, key)
                if axis.ndim != 1 or not axis.size or axis.dtype.kind not in "iufUS":
                    raise ValueError(
                        f"{key} ({name!r}) must be a nonempty 1-D real numeric or string array"
                    )
                is_string = axis.dtype.kind in "US"
                if is_string:
                    # NPZ also supports fixed-width byte strings. Decode these
                    # explicitly; never stringify numbers into categorical keys.
                    if axis.dtype.kind == "S":
                        try:
                            axis = np.char.decode(axis, "utf-8")
                        except UnicodeDecodeError as exc:
                            raise ValueError(f"{key} must contain UTF-8 strings") from exc
                elif not np.isfinite(axis).all():
                    raise ValueError(f"{key} ({name!r}) must contain only finite values")
                index = {value: j for j, value in enumerate(axis.tolist())}
                if len(index) != len(axis):
                    raise ValueError(f"{key} ({name!r}) must contain unique values")
                axis.flags.writeable = False
                axes.append(axis)
                indices.append(index)
                string_axes.append(is_string)

            shape = tuple(len(axis) for axis in axes)
            self._mean = self._real_array(self._read(archive, "mean"), "mean")
            if self._mean.shape != shape:
                raise ValueError(f"mean shape {self._mean.shape} does not match grid shape {shape}")
            std = self._read(archive, "std") if "std" in archive.files else np.asarray(0.0)
            self._std = self._real_array(std, "std")
            if self._std.shape not in ((), shape):
                raise ValueError(f"std must be scalar or have grid shape {shape}")
            if (self._std < 0).any():
                raise ValueError("std must contain only nonnegative values")

        self._axes = tuple(axes)
        self._indices = tuple(indices)
        self._string_axes = tuple(string_axes)
        self._mean.flags.writeable = False
        self._std.flags.writeable = False

    @staticmethod
    def _read(archive, name: str) -> np.ndarray:
        if name not in archive.files:
            raise ValueError(f"LookupPredictor model is missing required array {name!r}")
        try:
            return archive[name]
        except ValueError as exc:
            raise ValueError(f"Cannot read {name!r} without pickle: {exc}") from exc

    @staticmethod
    def _real_array(values: np.ndarray, name: str) -> np.ndarray:
        if values.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a real numeric array")
        # Predictions are Python floats. Validate the representation we return,
        # including values that would overflow when converted from a wider dtype.
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result

    def predict(self, candidate: Mapping[str, Any]) -> tuple[float, float]:
        """Return ``(mean, std)`` for the exact field values in *candidate*."""
        if not isinstance(candidate, Mapping):
            raise ValueError("Lookup candidate must be a mapping of named fields")
        if len(candidate) == 1:
            group, values = next(iter(candidate.items()))
            if isinstance(values, Mapping):
                if not isinstance(group, str) or not group:
                    raise ValueError("Lookup candidate group name must be a nonempty string")
                candidate = values

        fields = set(candidate)
        if fields != self._field_set:
            missing = self._field_set - fields
            extra = fields - self._field_set
            raise ValueError(
                "Lookup candidate fields do not match model: "
                f"missing={sorted(missing)!r}, extra={sorted(extra, key=repr)!r}"
            )

        location = []
        for name, index, is_string in zip(
            self._field_names, self._indices, self._string_axes
        ):
            value = candidate[name]
            if is_string:
                valid = isinstance(value, str)
            else:
                valid = (
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                )
                if valid and isinstance(value, (float, np.floating)):
                    valid = bool(np.isfinite(value))
                if valid and isinstance(value, np.generic):
                    value = value.item()
            if not valid:
                kind = "a string" if is_string else "a finite real number (not bool)"
                raise ValueError(f"Lookup field {name!r} must be {kind}; got {value!r}")
            try:
                location.append(index[value])
            except KeyError:
                raise ValueError(
                    f"Lookup field {name!r} value {value!r} is not on the model grid"
                ) from None

        location = tuple(location)
        std = self._std[()] if self._std.ndim == 0 else self._std[location]
        return float(self._mean[location]), float(std)

    def batch_predict(self, candidates) -> list[tuple[float, float]]:
        return [self.predict(candidate) for candidate in candidates]
