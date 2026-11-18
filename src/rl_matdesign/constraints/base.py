"""Shared contract for optional scientific action filters.

Groups store filter instances in a list. Applying that list retains only actions
accepted by every filter, after intrinsic group feasibility has been checked.
"""
from __future__ import annotations


def apply_filters(filters, *, actions, **context):
    """Intersect action filters, rejecting malformed or expanding plugin results."""
    current = list(actions)
    for index, filter_ in enumerate(filters):
        if not current:
            return []
        # Keep an independent snapshot: plugins may prune their input in place.
        available = set(current)
        result = filter_.filter_actions(actions=list(current), **context)
        prefix = f"Constraint filter {index} ({type(filter_).__name__})"
        if not isinstance(result, (list, tuple)):
            raise ValueError(f"{prefix} must return a list or tuple of actions.")
        try:
            selected = set(result)
        except TypeError as exc:
            raise ValueError(f"{prefix} returned an invalid action encoding.") from exc
        if not selected.issubset(available):
            raise ValueError(f"{prefix} may only remove actions (subset required).")
        if len(selected) != len(result):
            raise ValueError(f"{prefix} must not duplicate actions.")
        current = list(result)
    return current


class ConstraintFilter:
    """Pass-through base for custom filters implementing ``filter_actions``.

    Chemical groups supply raw ``chemical_amounts``, integer lattice metadata,
    remaining steps, species and action values. ``prior_groups`` contains already
    completed groups. Categorical groups instead supply ``current_fields`` and
    ``code_to_value`` so plugins can interpret the selected field's value without
    treating assignments as elemental amounts. Context is copied by the group;
    filters must return a subset of their input actions, including an empty list
    when no valid continuation exists. Accept extra keyword arguments to ignore
    context that a particular scientific check does not need.
    """

    def filter_actions(self, *, actions, **context):
        return actions
