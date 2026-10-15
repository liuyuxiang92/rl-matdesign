"""Canonical keys shared by composition caches and cost accounting."""
from __future__ import annotations

from typing import Any


def composition_key(candidate: Any, *, decimals: int | None = None) -> Any:
    """Freeze flat/nested composition mappings, retaining categorical labels.

    ``decimals=None`` preserves exact float values for structure predictions.
    Baseline caches and cost counters explicitly use six decimal places.
    """
    if isinstance(candidate, dict):
        return tuple(sorted(
            (str(key), composition_key(value, decimals=decimals))
            for key, value in candidate.items()
        ))
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return str(candidate)
    return value if decimals is None else round(value, decimals)
