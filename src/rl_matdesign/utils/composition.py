"""Canonical candidate identities shared by prediction caches and accounting."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np

from ..action_encoding import typed_identity


def composition_key(candidate: Any, *, decimals: int | None = None) -> tuple:
    """Freeze structured candidates with exact types and meaningful positions.

    Mapping insertion order never matters; list/tuple positions always matter.
    Boolean, numeric and string identities never coerce into one another.
    Equal finite numbers such as 1 and 1.0 denote the same physical value.
    Composition environments already emit canonical float amounts; integer-ratio
    environments retain their own GCD normalization before this function.

    ``decimals`` is an explicit caller-selected rounding policy for floating
    values only, never string IDs. Prediction caches should use the exact default.
    """
    if isinstance(candidate, Mapping):
        pairs = [(composition_key(key), composition_key(value, decimals=decimals))
                 for key, value in candidate.items()]
        return ("mapping", tuple(sorted(pairs, key=lambda pair: repr(pair[0]))))
    if isinstance(candidate, (list, tuple)):
        return ("list" if isinstance(candidate, list) else "tuple",
                tuple(composition_key(value, decimals=decimals) for value in candidate))
    if candidate is None:
        return ("none",)
    if isinstance(candidate, (bool, np.bool_)):
        return ("bool", bool(candidate))
    if isinstance(candidate, str):
        return ("str", candidate)
    if isinstance(candidate, Integral):
        return ("number", int(candidate))
    if isinstance(candidate, Real):
        value = float(candidate)
        if not np.isfinite(value):
            raise ValueError("Candidate identity must contain finite numbers")
        return ("number", value if decimals is None else round(value, decimals))
    # Lossy repr conversion cannot be a reproducible scientific identity.
    typed_identity(candidate)
    raise TypeError(f"Unsupported candidate identity type: {type(candidate).__name__}")
