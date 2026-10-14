from __future__ import annotations

from typing import Dict, Tuple, Optional


# Helper element sets for different primary-phase definitions.
# All elements here are cations; missing keys in a composition dict are treated as 0.

# Common dopant set used in multiple rules (excluding Ni, Fe, Co explicitly).
DOPANTS_BASE = {
    "Mg",
    "Ca",
    "Cu",
    "Zn",
    "Sr",
    "Al",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Ga",
    "In",
    "Zr",
    "La",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Ce",
    "Pr",
    "Tb",
    "Bi",
}


def _sum_except(comp: Dict[str, float], primary_elems) -> float:
    """Sum every cation outside a phase's primary set.

    Deriving dopants from the composition keeps the phase cap correct when a
    configured species is added before the compatibility set above is updated.
    """
    primary = set(primary_elems)
    return float(sum(v for e, v in comp.items() if e not in primary))


def _fe_not_dominant(comp: Dict[str, float], include_ni: bool, include_co: bool) -> bool:
    fe = comp.get("Fe", 0.0)
    others = []
    if include_ni:
        others.append(comp.get("Ni", 0.0))
    if include_co:
        others.append(comp.get("Co", 0.0))
    return fe <= (max(others) if others else 0.0)


def _ni_fe_ratio_ok(comp: Dict[str, float]) -> bool:
    """Approximate Ni:Fe ~ 2:1 to 3:1 where possible on 0.05 grid.

    Implemented as a constraint on Ni's share of the (Ni+Fe) total:
    Ni_share = Ni / (Ni + Fe) in [2/3, 3/4].
    """

    ni = comp.get("Ni", 0.0)
    fe = comp.get("Fe", 0.0)
    total = ni + fe
    if total <= 0.0:
        return False
    share = ni / total
    return (2.0 / 3.0) <= share <= (3.0 / 4.0)


def check_primary_phase(comp: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    """Check if a composition satisfies any primary-phase rule.

    Returns (ok, label) where label is one of
    {"NiFeCo", "NiFe", "CoFe", "Ni", "Co"} when ok is True,
    otherwise (False, None).
    """

    ni = comp.get("Ni", 0.0)
    fe = comp.get("Fe", 0.0)
    co = comp.get("Co", 0.0)

    # --- NiFeCo primary phase ---
    dop_sum = _sum_except(comp, {"Ni", "Fe", "Co"})
    tri_sum = ni + fe + co
    if dop_sum < 0.25 and tri_sum >= 0.75 and _fe_not_dominant(comp, include_ni=True, include_co=True):
        return True, "NiFeCo"

    # --- NiFe primary phase ---
    dop_sum = _sum_except(comp, {"Ni", "Fe"})
    bi_sum = ni + fe
    if dop_sum < 0.25 and bi_sum >= 0.75 and _fe_not_dominant(comp, include_ni=True, include_co=False) and _ni_fe_ratio_ok(comp):
        return True, "NiFe"

    # --- CoFe primary phase ---
    dop_sum = _sum_except(comp, {"Co", "Fe"})
    bi_sum = co + fe
    if dop_sum < 0.25 and bi_sum >= 0.75 and _fe_not_dominant(comp, include_ni=False, include_co=True):
        return True, "CoFe"

    # --- Ni primary phase ---
    dop_sum = _sum_except(comp, {"Ni"})
    if dop_sum < 0.25 and ni >= 0.75:
        return True, "Ni"

    # --- Co primary phase ---
    dop_sum = _sum_except(comp, {"Co"})
    if dop_sum < 0.25 and co >= 0.75:
        return True, "Co"

    return False, None
