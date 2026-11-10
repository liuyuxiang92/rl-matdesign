"""ChainConstraintFilter — apply multiple constraint filters in sequence.

Composes N child constraint filters so each step's action list is pruned by
every filter in turn. Useful when a system has stacked rules — e.g. an oxide
that needs both charge neutrality (``SMACTChargeFilter``) **and** "step N = O"
(``LastStepElementFilter``).

YAML configuration
------------------
.. code-block:: yaml

    filters:
      - type: last_step_element        # cheap rule first
        required_elements: ["O"]
        nonzero_ratio_at_last: true
        reserve_for_last: true
      - type: smact_charge             # expensive rule second
        smact_anions:
          - {symbol: O, charge: -2, stoich: 3.0}
      - pauling_en                     # bare string == {type: pauling_en}

Each entry under ``filters:`` is a self-contained sub-config carrying its
own ``type:`` key plus whatever kwargs that child filter needs, or a bare
string as shorthand for a parameter-free filter (``{type: <string>}``). The
chain runs filters in list order; each filter's output feeds the next
filter's input. You never write ``type: chain`` yourself — a non-empty
``filters:`` list is auto-detected and auto-chained (see
:func:`~rl_matdesign.registry.build_constraints`).

Children are resolved through the same :func:`resolve_constraint` used at
the top level, so any combination of built-in short names and user FQN
plug-ins (``pkg.module:ClassName``) works inside a chain — including nesting
another ``chain`` if you really want to.

Filter contract
---------------
Each child may only remove actions from its immediate input. An empty output
means no action satisfies the constraints; it is propagated unchanged to the
environment. The rollout layer handles infeasible episodes. A child that adds
or duplicates actions violates this contract and raises ``ValueError``.

Performance note: put cheap, identity-based filters (e.g.
``last_step_element``) before expensive ones (e.g. ``smact_charge`` whose
SMACT screen iterates over oxidation-state combinations). Ordering does
not affect the final set, only the work done.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import ConstraintFilter


class ChainConstraintFilter(ConstraintFilter):
    """Apply a sequence of :class:`ConstraintFilter` instances in order."""

    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        from ..registry import resolve_constraint

        children_cfgs = cfg.get("filters")
        if not children_cfgs:
            raise ValueError(
                "ChainConstraintFilter requires a non-empty 'filters' list. "
                "Example: filters: [{type: smact_charge, ...}, "
                "{type: last_step_element, ...}]"
            )
        if not isinstance(children_cfgs, list):
            raise ValueError(
                f"'filters' must be a list of dicts (or bare strings), got "
                f"{type(children_cfgs).__name__}."
            )

        # Context shared with every child: the enclosing group/config's own
        # keys (e.g. a categorical group's `choices`), minus `filters` itself.
        # A handful of filters (e.g. sse_doping) auto-discover their setup from
        # the *enclosing group*, not from params repeated on every entry — under
        # the old singular `constraint_filter: <name>` spelling they received
        # the group spec directly, so chaining must not silently take that
        # context away. Each child's own keys still win on conflict.
        parent_context = {k: v for k, v in cfg.items() if k != "filters"}

        self.children: List[ConstraintFilter] = []
        for i, sub in enumerate(children_cfgs):
            if isinstance(sub, str):
                sub_kind, sub_cfg = sub, dict(parent_context)
            elif isinstance(sub, dict):
                sub_cfg = {**parent_context, **sub}
                sub_kind = sub_cfg.pop("type", None)
                if not sub_kind:
                    raise ValueError(
                        f"filters[{i}] is missing 'type' "
                        "(short name or 'pkg.module:ClassName' FQN)."
                    )
            else:
                raise ValueError(
                    f"filters[{i}] must be a dict or a bare string, got "
                    f"{type(sub).__name__}."
                )
            child = resolve_constraint(sub_kind, sub_cfg, env=env)
            if child is None:
                raise ValueError(
                    f"filters[{i}].type={sub_kind!r} resolved "
                    "to None — expected a ConstraintFilter instance."
                )
            self.children.append(child)

    def filter_actions(self, *, actions, **kw):
        current = list(actions)
        for index, child in enumerate(self.children):
            if not current:
                return []
            # A plug-in may prune its list in place. Retain an independent
            # snapshot so mutation cannot hide an added action or affect the
            # caller's action list.
            available = set(current)
            result = child.filter_actions(actions=list(current), **kw)
            prefix = f"Constraint filter {index} ({type(child).__name__})"
            if not isinstance(result, (list, tuple)):
                raise ValueError(f"{prefix} must return a list or tuple of actions.")
            try:
                selected = set(result)
            except TypeError as exc:
                raise ValueError(f"{prefix} returned an invalid action encoding.") from exc
            if not selected.issubset(available):
                raise ValueError(f"{prefix} must return a subset of its input actions.")
            if len(selected) != len(result):
                raise ValueError(f"{prefix} must not duplicate actions.")
            current = list(result)
        return current
