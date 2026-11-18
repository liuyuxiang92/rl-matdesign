"""Charge neutrality and Pauling filters for accumulated chemical amounts."""
from __future__ import annotations

from .base import ConstraintFilter


class SMACTChargeFilter(ConstraintFilter):
    """Check the full final formula, including earlier groups and fixed scaffold.

    Raw species amounts are expanded to elemental counts. Fixed scaffold counts
    are divided by scaffold_n_sites to put them on the same formula-unit basis.
    """

    def __init__(self, *, scaffold_fixed=None, scaffold_n_sites=1,
                 allow_alloys=True, tol=0.5, use_pauling=False):
        if scaffold_n_sites < 1:
            raise ValueError("scaffold_n_sites must be positive")
        self.allow_alloys = bool(allow_alloys)
        self.tol = float(tol)
        self.use_pauling = bool(use_pauling)
        self.scaffold_per_fu = {str(e): float(c) / scaffold_n_sites
                                for e, c in (scaffold_fixed or {}).items()}

    def filter_actions(self, *, actions, steps_left, species_set, action_values,
                       chemical_amounts, action_amounts, prior_groups=None, **_):
        if steps_left > 0:
            return actions
        from .charge import charge_neutral
        # New chemical domains retain raw amounts. Earlier sublattices are
        # part of the same formula (ABO3 needs both A and B), not additional
        # normalized fractions of the current group's action lattice.
        from pymatgen.core import Composition
        from ..encoding import decode_one_hot
        import numpy as np
        scaffold = dict(self.scaffold_per_fu)
        for previous in prior_groups or []:
            for label, amount in previous.items():
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    for element, count in Composition(label).as_dict().items():
                        scaffold[element] = scaffold.get(element, 0.0) + float(amount) * count
        values = action_amounts if action_amounts is not None else [float(value) for value in action_values]
        result = []
        for action in actions:
            amounts = dict(chemical_amounts)
            label = decode_one_hot(action[0], species_set)
            amounts[label] = amounts.get(label, 0.0) + values[int(np.asarray(action[1]).argmax())]
            full = dict(scaffold)
            for choice, amount in amounts.items():
                for element, count in Composition(choice).as_dict().items():
                    full[element] = full.get(element, 0.0) + amount * count
            full = {element: amount for element, amount in full.items() if amount}
            if charge_neutral(full, tol=self.tol, allow_alloys=self.allow_alloys, use_pauling=self.use_pauling):
                result.append(action)
        return result


class ElectronegativityFilter(SMACTChargeFilter):
    """Require both charge neutrality and Pauling electronegativity."""

    def __init__(self, *, use_pauling=True, **kwargs):
        super().__init__(use_pauling=use_pauling, **kwargs)
