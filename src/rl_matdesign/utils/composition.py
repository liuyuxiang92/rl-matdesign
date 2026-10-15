"""Canonical keys shared by composition caches and cost accounting."""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any


def _contains_sequence(candidate: Any) -> bool:
    if isinstance(candidate, (list, tuple)):
        return True
    if isinstance(candidate, dict):
        return any(_contains_sequence(value) for value in candidate.values())
    return False


def _ordered_candidate_key(candidate: Any) -> Any:
    """Freeze a structured candidate without conflating identifiers and numbers."""
    if isinstance(candidate, dict):
        return ("mapping", frozenset(
            (_ordered_candidate_key(key), _ordered_candidate_key(value))
            for key, value in candidate.items()
        ))
    if isinstance(candidate, (list, tuple)):
        return ("list" if isinstance(candidate, list) else "tuple",
                tuple(_ordered_candidate_key(value) for value in candidate))
    if candidate is None:
        return ("none",)
    if isinstance(candidate, str):
        return ("str", candidate)
    if isinstance(candidate, bool):
        return ("bool", candidate)
    if isinstance(candidate, Integral):
        return ("int", int(candidate))
    if isinstance(candidate, Real):
        return ("float", float(candidate))
    return (type(candidate).__module__, type(candidate).__qualname__, str(candidate))


def composition_key(candidate: Any, *, decimals: int | None = None) -> Any:
    """Freeze a composition mapping or an ordered, structured candidate.

    ``decimals=None`` preserves exact float values for structure predictions.
    Baseline caches and cost counters explicitly use six decimal places.

    Payloads containing a list/tuple use typed keys for the *entire* payload,
    preserving sequence order, token identifiers (including ``"01"`` versus
    ``"1"``), and categorical metadata. These values are never rounded. The
    explicit marker separates them from legacy composition keys. Mappings
    without sequences keep their historical numeric coercion and rounding,
    preserving existing composition caches and checkpoint compatibility.
    """
    if _contains_sequence(candidate):
        return ("ordered_candidate", _ordered_candidate_key(candidate))
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
