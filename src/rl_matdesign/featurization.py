from __future__ import annotations

from decimal import Decimal
import re
from typing import Sequence

import warnings

import numpy as np

# Matminer may emit a verbose warning from `matminer.utils.data.PymatgenData`
# about `impute_nan=False`, sometimes repeatedly during featurization.
# Suppress it to keep stdout/stderr and tqdm output clean.
warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
    category=UserWarning,
    module=r"matminer\.utils\.data",
    append=False,
)

try:
    from pymatgen.core.composition import Composition
except ImportError:  # pragma: no cover - minimal installations
    Composition = None

try:
    from matminer.featurizers.base import MultipleFeaturizer
    import matminer.featurizers.composition as cf

    feature_calculators = MultipleFeaturizer(
        [
            cf.element.Stoichiometry(),
            cf.composite.ElementProperty.from_preset("magpie"),
            cf.orbital.ValenceOrbital(props=["avg"]),
            cf.ion.IonProperty(fast=True),
        ]
    )
    _NUM_FEATURES = len(feature_calculators.feature_labels())
except Exception:  # pragma: no cover
    # Allow a minimal install to run the pipeline (useful for smoke tests / CI).
    feature_calculators = None
    _NUM_FEATURES = 8


_featurize_cache: dict[tuple, np.ndarray] = {}

_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt "
    "Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBER = {symbol: index for index, symbol in enumerate(_ELEMENTS, 1)}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d*)?|\.\d+)?")


def _composition_items(formula: str) -> tuple:
    """Canonical chemical identity, never a hash of construction-order text.

    The flat formula subset used by composition environments is parsed exactly
    with decimal amounts. Pymatgen handles richer external formula syntax when
    installed. Invalid inputs have the same explicit zero-descriptor behavior
    in both the scientific and lightweight backends.
    """
    amounts: dict[str, Decimal] = {}
    tokens = list(_FORMULA_TOKEN.finditer(formula))
    if tokens and "".join(match.group() for match in tokens) == formula:
        for match in tokens:
            symbol, amount = match.groups()
            if symbol not in _ATOMIC_NUMBER:
                return ()
            amounts[symbol] = amounts.get(symbol, Decimal(0)) + Decimal(amount or "1")
    elif formula and Composition is not None:
        try:
            amounts = {str(symbol): Decimal(str(amount))
                       for symbol, amount in Composition(formula).get_el_amt_dict().items()}
        except Exception:
            return ()
    if any(symbol not in _ATOMIC_NUMBER or not amount.is_finite() or amount < 0
           for symbol, amount in amounts.items()):
        return ()
    return tuple(sorted((symbol, amount) for symbol, amount in amounts.items() if amount > 0))


def _lightweight_features(items: tuple) -> np.ndarray:
    """Eight normalized composition descriptors using only atomic numbers."""
    result = np.zeros(_NUM_FEATURES, dtype=float)
    if not items:
        return result
    amounts = np.array([float(amount) for _, amount in items], dtype=float)
    fractions = amounts / amounts.sum()
    atomic_numbers = np.array([_ATOMIC_NUMBER[symbol] for symbol, _ in items], dtype=float)
    mean = float(np.dot(fractions, atomic_numbers))
    std = float(np.sqrt(np.dot(fractions, (atomic_numbers - mean) ** 2)))
    values = np.array([
        len(items) / 118, mean / 118, std / 118,
        atomic_numbers.min() / 118, atomic_numbers.max() / 118,
        np.dot(fractions, fractions), fractions.max(),
        -np.dot(fractions, np.log(fractions)) / np.log(118),
    ])
    result[:min(result.size, values.size)] = values[:result.size]
    return result


def featurize_formula(formula: str) -> np.ndarray:
    """Featurize a (possibly partial) formula into a fixed-length float vector.

    Descriptors describe relative chemistry; absolute design amounts belong in
    the environment's explicit state encoding. Returns zeros for empty/invalid
    formulas. Both descriptor backends and the cache use canonical compositions,
    so formula token order is never a feature.
    """

    items = _composition_items(formula)
    if items in _featurize_cache:
        return _featurize_cache[items]

    if not items:
        result = np.zeros(_NUM_FEATURES, dtype=float)
        _featurize_cache[items] = result
        return result

    # Preferred: Magpie/matminer features.
    if feature_calculators is not None:
        try:
            chemical = Composition({symbol: float(amount) for symbol, amount in items})
            feats: Sequence[float] = feature_calculators.featurize(chemical)
            result = np.asarray(feats, dtype=float)
        except Exception:
            result = np.zeros(_NUM_FEATURES, dtype=float)
        _featurize_cache[items] = result
        return result

    result = _lightweight_features(items)
    _featurize_cache[items] = result
    return result
