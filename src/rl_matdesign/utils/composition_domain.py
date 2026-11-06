"""Exact discrete fraction domains, independent of models and environments.

The authored fraction menu defines the available choices. Bounds restrict that
menu; they never add values or round a physical limit outward. Completion uses
the actual menu of each remaining species, including sparse menus and optional
species, rather than assuming all integer values between two limits exist.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from math import lcm


def _decimal(value, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite decimal number, got {value!r}")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite decimal number, got {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _fraction_decimal(value, label: str) -> Decimal:
    result = _decimal(value, label)
    if not 0 <= result <= 1:
        raise ValueError(f"{label} must be within [0, 1], got {value!r}")
    return result


def _places(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _render(value: Fraction, places: int) -> str:
    scale = 10 ** places
    scaled = value * scale
    if scaled.denominator != 1:
        raise ValueError("Fraction cannot be represented at the requested decimal precision")
    whole, remainder = divmod(scaled.numerator, scale)
    return f"{whole}.{remainder:0{places}d}"


def expand_fraction_codes(spec) -> list[str]:
    """Expand a fraction range or explicit menu without moving its endpoints.

    Regular ranges contain ``min + i * step <= max``. A maximum that is not
    on this grid is not inserted. Explicit lists retain their authored order
    and gaps. All codes use the same exact decimal precision, at least two
    places, preserving the historical codes for existing aligned grids.
    """
    if isinstance(spec, Mapping):
        expected = {"min", "max", "step"}
        if set(spec) != expected:
            raise ValueError("fraction_set range must contain exactly min, max, and step")
        lower = _fraction_decimal(spec["min"], "fraction_set.min")
        upper = _fraction_decimal(spec["max"], "fraction_set.max")
        step = _decimal(spec["step"], "fraction_set.step")
        if step <= 0:
            raise ValueError("fraction_set.step must be positive")
        if lower > upper:
            raise ValueError("fraction_set.min must not exceed fraction_set.max")
        lo, hi, increment = Fraction(lower), Fraction(upper), Fraction(step)
        count = (hi - lo) // increment
        precision = max(2, _places(lower), _places(step))
        return [_render(lo + i * increment, precision) for i in range(count + 1)]

    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)) or not spec:
        raise ValueError("fraction_set must be a non-empty list or a min/max/step range")
    values = [_fraction_decimal(value, f"fraction_set[{index}]")
              for index, value in enumerate(spec)]
    if len(set(values)) != len(values):
        raise ValueError("fraction_set must not contain duplicate numerical values")
    precision = max(2, *(_places(value) for value in values))
    return [_render(Fraction(value), precision) for value in values]


def _positive_integer(value, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _nonnegative_integer(value, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def derive_total_units(codes: Sequence[str]) -> int:
    """Return the smallest integer denominator representing every menu value."""
    if not codes:
        raise ValueError("fraction_set must not be empty")
    denominators = [Fraction(_fraction_decimal(code, "fraction code")).denominator
                    for code in codes]
    return lcm(*denominators)


def fraction_units(codes: Sequence[str], total_units: int) -> list[int]:
    """Convert exact decimal codes to units, rejecting an incompatible lattice."""
    _positive_integer(total_units, "total_units")
    result = []
    for code in codes:
        value = Fraction(_fraction_decimal(code, "fraction code")) * total_units
        if value.denominator != 1:
            raise ValueError(
                f"Fraction {code!r} is not representable in units of 1/{total_units}"
            )
        result.append(value.numerator)
    return result


def normalize_unit_bounds(bounds: Sequence, total_units: int) -> tuple[int, int]:
    """Restrict a closed physical interval to integer units, without widening it.

    A positive lower bound remains positive after conversion. The returned
    interval still needs intersection with the actual menu, which may be sparse.
    An empty interval is reported explicitly instead of admitting a nearby value.
    """
    _positive_integer(total_units, "total_units")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise ValueError("element bounds must be a [lower, upper] pair")
    lower = Fraction(_fraction_decimal(bounds[0], "lower bound"))
    upper = Fraction(_fraction_decimal(bounds[1], "upper bound"))
    if lower > upper:
        raise ValueError("element bounds have lower > upper")
    lo, hi = lower * total_units, upper * total_units
    lower_units = -(-lo.numerator // lo.denominator)
    upper_units = hi.numerator // hi.denominator
    if lower_units > upper_units:
        raise ValueError("element bounds contain no representable fraction on the unit lattice")
    return lower_units, upper_units


def allowed_unit_menu(allowed_units: Sequence[int], bounds: tuple[int, int]) -> tuple[int, ...]:
    """Intersect an existing discrete menu with its closed integer bounds."""
    lower, upper = bounds
    _nonnegative_integer(lower, "lower unit bound")
    _nonnegative_integer(upper, "upper unit bound")
    if lower > upper:
        raise ValueError("unit bounds have lower > upper")
    values = tuple(_nonnegative_integer(value, "allowed unit value") for value in allowed_units)
    return tuple(value for value in values if lower <= value <= upper)


# Each pool entry is (actual legal units for one species, must_select_species).
# A zero-valued selection consumes one slot; skipping an optional species does
# not. Mandatory comes from the declared lower bound, not min(menu): a species
# with lower=0 can be absent even when its menu only contains positive amounts.
CompletionPool = Sequence[tuple[Sequence[int], bool]]


def _canonical_pool(pool: CompletionPool, max_total: int) -> tuple:
    canonical = []
    for menu, mandatory in pool:
        if not isinstance(mandatory, bool):
            raise ValueError("completion mandatory flags must be booleans")
        values = {_nonnegative_integer(value, "completion menu value") for value in menu}
        canonical.append((tuple(sorted(value for value in values if value <= max_total)), mandatory))
    # Species identities do not affect this purely arithmetic reachability
    # query; repeated identical entries remain distinct selectable species.
    return tuple(sorted(canonical))


@lru_cache(maxsize=4096)
def _completion_sums_cached(pool: tuple, slots: int, max_total: int) -> int:
    if slots > len(pool) or sum(mandatory for _, mandatory in pool) > slots:
        return 0
    mask = (1 << (max_total + 1)) - 1
    reachable = [0] * (slots + 1)
    reachable[0] = 1  # Sum zero is initially possible with zero selections.
    for index, (menu, mandatory) in enumerate(pool):
        updated = [0] * (slots + 1) if mandatory else reachable.copy()
        for count in range(1, min(slots, index + 1) + 1):
            previous = reachable[count - 1]
            if previous:
                for value in menu:
                    updated[count] |= previous << value
                updated[count] &= mask
        reachable = updated
    return reachable[slots]


def completion_sums(pool: CompletionPool, slots: int, max_total: int) -> int:
    """Return a bitset of sums reachable with exactly ``slots`` selected species.

    Bit ``s`` is one iff sum ``s`` is reachable while selecting every mandatory
    species. Each pool entry can be selected at most once, from its own menu.
    The cache is bounded and ignores species order, without merging species.
    """
    _nonnegative_integer(slots, "slots")
    _nonnegative_integer(max_total, "max_total")
    return _completion_sums_cached(_canonical_pool(pool, max_total), slots, max_total)


def can_complete(target: int, slots: int, pool: CompletionPool, max_total: int) -> bool:
    """Whether the exact remaining target is reachable from the supplied menus."""
    _nonnegative_integer(target, "target")
    if target > max_total:
        return False
    return bool(completion_sums(pool, slots, max_total) & (1 << target))
