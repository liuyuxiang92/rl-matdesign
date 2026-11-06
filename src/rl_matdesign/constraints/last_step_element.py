"""LastStepElementFilter — force the final action to a required element.

Matches the reference DQN's "step 5 = O" rule from
``RL_materials_generation/env_constrained.py:299-344``.  The required
element is reserved for the last slot: earlier-step actions that pick it
are pruned, and at the last step only ``(required_element, digit)`` pairs
(optionally with nonzero digit) survive.

Plugs into the ``phase_filter`` slot of either :class:`CompositionEnv` or
:class:`IntegerRatioEnv`; the ``filter_actions`` signature matches
:class:`SMACTChargeFilter`.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import ConstraintFilter


class LastStepElementFilter(ConstraintFilter):
    """Restrict the final-step action to one of ``required_elements``.

    Parameters
    ----------
    required_elements:
        Element symbols allowed at the final step (e.g. ``["O"]``).
    nonzero_ratio:
        If True (default), disallow numerically zero ratio/fraction values at
        the final step, regardless of their string representation.
    reserve_for_last:
        If True (default), prune earlier-step actions that pick any
        element in ``required_elements`` so it is reserved for the last
        slot.  Set False to allow the required element anywhere.
    """

    def __init__(
        self,
        required_elements: List[str],
        *,
        nonzero_ratio: bool = True,
        reserve_for_last: bool = True,
    ) -> None:
        if not required_elements:
            raise ValueError("required_elements must be a non-empty list.")
        self.required_elements = set(required_elements)
        self.nonzero_ratio = nonzero_ratio
        self.reserve_for_last = reserve_for_last

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        units_map: Dict[str, int],
        steps_left: int,
        allowed_units: Sequence[int],
        possible_sums_by_k: List[Any],
        species_set: List[str],
        fraction_set: List[str],
        code_to_value: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        is_last_step = steps_left == 0

        filtered = []
        for elem_oh, comp_oh in actions:
            elem = decode_one_hot(elem_oh, species_set)
            if is_last_step:
                if elem not in self.required_elements:
                    continue
                if self.nonzero_ratio:
                    code = decode_one_hot(comp_oh, fraction_set)
                    value = code_to_value[code] if code_to_value is not None else code
                    try:
                        amount = Decimal(str(value))
                    except InvalidOperation as exc:
                        raise ValueError(
                            f"LastStepElementFilter requires numeric amounts, got {value!r}."
                        ) from exc
                    if not amount.is_finite():
                        raise ValueError(
                            f"LastStepElementFilter requires finite amounts, got {value!r}."
                        )
                    if amount == 0:
                        continue
            else:
                if self.reserve_for_last and elem in self.required_elements:
                    continue

            filtered.append((elem_oh, comp_oh))

        return filtered
