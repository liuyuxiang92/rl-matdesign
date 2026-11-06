"""Exact counts of finite integer composition domains, without model imports.

Species have their own allowed integer menus. Counts describe terminal
assignments or unique compositions, never permutations of action order. Missing
optional species and explicitly selected zero amounts have the same composition
identity, but selecting zero still consumes an assignment slot.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from numbers import Integral


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _domain(
    menus: Mapping[str, Sequence[int]],
    total_units: int,
    required: Collection[str] | None,
    slots: int | None,
) -> tuple[list[tuple[tuple[int, ...], bool]], int, int | None]:
    total = _integer(total_units, "total_units")
    if slots is not None:
        slots = _integer(slots, "slots")
    if not isinstance(menus, Mapping):
        raise ValueError("menus must map species labels to integer menus.")
    if any(not isinstance(label, str) or not label.strip() for label in menus):
        raise ValueError("Species labels must be nonempty strings.")
    if required is None:
        required_set = set(menus)
    else:
        if isinstance(required, (str, bytes)) or not isinstance(required, Collection):
            raise ValueError("required must be a collection of species labels or None.")
        if any(not isinstance(label, str) for label in required):
            raise ValueError("required must contain species labels.")
        required_set = set(required)
        if len(required_set) != len(required):
            raise ValueError("required contains duplicate species labels.")
        if not required_set <= menus.keys():
            raise ValueError("required contains labels absent from menus.")
    pool = []
    for label in sorted(menus):
        menu = menus[label]
        if isinstance(menu, (str, bytes)) or not isinstance(menu, Sequence):
            raise ValueError(f"Menu for {label} must be a finite sequence of integers.")
        values = tuple(_integer(value, f"Menu value for {label}") for value in menu)
        if len(set(values)) != len(values):
            raise ValueError(f"Menu for {label} contains duplicate values.")
        # Values beyond the total can never occur in a nonnegative sum. An
        # empty effective menu makes a required species impossible, but an
        # optional species can still be omitted.
        pool.append((tuple(sorted(value for value in values if value <= total)), label in required_set))
    return pool, total, slots


def _sum_count(pool: list[tuple[tuple[int, ...], bool]], total: int) -> int:
    counts = {0: 1}
    for menu, mandatory in pool:
        values = menu if mandatory else tuple(sorted(set(menu) | {0}))
        updated: dict[int, int] = defaultdict(int)
        for subtotal, count in counts.items():
            for value in values:
                if subtotal + value <= total:
                    updated[subtotal + value] += count
        counts = updated
    return counts.get(total, 0)


def count_compositions(
    menus: Mapping[str, Sequence[int]],
    total_units: int,
    *,
    required: Collection[str] | None = None,
    slots: int | None = None,
) -> int:
    """Count unique compositions reachable under sum and assignment constraints.

    Parameters
    ----------
    menus
        Species to allowed nonnegative integer amounts. Menus may be sparse;
        missing integers are never filled in. Values must be unique per menu.
    total_units
        Required exact sum of amounts, in the same integer unit as every menu.
    required
        Species that must receive an explicit assignment. ``None`` means all
        species; an empty collection makes every species optional. Required
        does not mean nonzero: a required species can receive zero only if zero
        belongs to its menu. Express positive lower bounds with positive menus.
    slots
        Exact number of explicitly assigned species, including selected zeros.
        ``None`` imposes no slot-count restriction. Optional omission consumes
        no slot and is possible even if zero is absent from that species' menu.

    Returns
    -------
    int
        Number of distinct terminal species-to-positive-amount maps. Each map
        counts once if *any* permitted assignment of its zero/absent species
        fills the requested slots. Counts use arbitrary-precision Python ints.

    Notes
    -----
    The sparse dynamic program stores partial sums, not material candidates.
    With a slot restriction, it also tracks the minimum and maximum possible
    number of assigned species for each partial composition. Every integer in
    that interval is reachable by selecting additional optional zero amounts.
    Labels remain distinct; swapping values between two species generally
    changes the material. Reordering the input mapping does not change counts.
    """
    pool, total, slots = _domain(menus, total_units, required, slots)
    n_required = sum(mandatory for _, mandatory in pool)
    if slots is not None and not n_required <= slots <= len(pool):
        return 0
    if slots is None or n_required == len(pool):
        return _sum_count(pool, total)

    counts = {(0, 0, 0): 1}
    for menu, mandatory in pool:
        # One physical zero state per optional species. Its assignment may be
        # omitted, or selected (consuming a slot) only when 0 is actually legal.
        choices = [(value, 1, 1) for value in menu if value > 0 or mandatory]
        if not mandatory:
            choices.append((0, 0, int(0 in menu)))
        updated: dict[tuple[int, int, int], int] = defaultdict(int)
        for (subtotal, minimum, maximum), count in counts.items():
            for value, min_increment, max_increment in choices:
                new_sum = subtotal + value
                new_minimum = minimum + min_increment
                if new_sum <= total and new_minimum <= slots:
                    key = (new_sum, new_minimum, maximum + max_increment)
                    updated[key] += count
        counts = updated
    return sum(
        count for (subtotal, minimum, maximum), count in counts.items()
        if subtotal == total and minimum <= slots <= maximum
    )


def count_assignments(
    menus: Mapping[str, Sequence[int]],
    total_units: int,
    *,
    required: Collection[str] | None = None,
    slots: int | None = None,
) -> int:
    """Count terminal assignment states, distinguishing omitted/selected zero.

    Arguments have the same meaning as :func:`count_compositions`. This is a
    diagnostic count, not the unique scientific search-space cardinality when
    optional zero assignments are present. Each species is considered once;
    permutations of the order in which assignments were made are never counted.
    """
    pool, total, slots = _domain(menus, total_units, required, slots)
    if slots is not None and not sum(flag for _, flag in pool) <= slots <= len(pool):
        return 0
    counts = {(0, 0): 1}
    for menu, mandatory in pool:
        choices = [(value, 1) for value in menu]
        if not mandatory:
            choices.append((0, 0))
        updated: dict[tuple[int, int], int] = defaultdict(int)
        for (subtotal, selected), count in counts.items():
            for value, increment in choices:
                if subtotal + value <= total and (slots is None or selected + increment <= slots):
                    updated[(subtotal + value, selected + increment)] += count
        counts = updated
    return sum(
        count for (subtotal, selected), count in counts.items()
        if subtotal == total and (slots is None or selected == slots)
    )


class UniformCompositionSampler:
    """Draw uniformly over unique legal terminal composition maps.

    Completion counts weight each next *labelled value*. This differs from
    random action rollouts, whose many construction paths need not have equal
    multiplicity. Optional zero assignments are represented by one physical
    zero branch, with an interval of possible occupied slots; they never give
    a composition extra probability. No list of terminal candidates is built.
    Pass a dedicated ``random.Random`` instance to keep baseline randomness
    separate from oracle and learning streams. Sampling is with replacement.
    """

    def __init__(self, menus, total_units, *, required=None, slots=None):
        from functools import lru_cache

        pool, total, slots = _domain(menus, total_units, required, slots)
        self.labels = tuple(sorted(menus))
        self.total_units = total
        self.slots = slots
        self._choices = tuple(
            tuple([(value, 1, 1) for value in menu if value > 0 or mandatory]
                  + ([] if mandatory else [(0, 0, int(0 in menu))]))
            for menu, mandatory in pool
        )

        @lru_cache(maxsize=None)
        def completions(index, remaining, minimum, maximum):
            if remaining < 0 or (slots is not None and minimum > slots):
                return 0
            if index == len(self.labels):
                return int(remaining == 0 and
                           (slots is None or minimum <= slots <= maximum))
            return sum(completions(index + 1, remaining - value,
                                   minimum + lo, maximum + hi)
                       for value, lo, hi in self._choices[index])

        self._completions = completions
        self.size = completions(0, total, 0, 0)

    def sample(self, rng):
        """Return one positive-amount integer map using exact integer weights."""
        if not self.size:
            raise ValueError("Cannot sample an empty composition domain")
        rank = rng.randrange(self.size)
        return self.unrank(rank)

    def unrank(self, rank):
        """Return the unique composition at an exact integer lexicographic rank.

        This is also useful for independently enumerating a *small* domain;
        callers must not enumerate production benchmark spaces for sampling.
        """
        rank = _integer(rank, "rank")
        if rank >= self.size:
            raise ValueError(f"rank must be below domain size {self.size}")
        remaining, minimum, maximum = self.total_units, 0, 0
        result = {}
        for index, label in enumerate(self.labels):
            for value, lo, hi in self._choices[index]:
                count = self._completions(index + 1, remaining - value,
                                          minimum + lo, maximum + hi)
                if rank < count:
                    if value:
                        result[label] = value
                    remaining -= value
                    minimum += lo
                    maximum += hi
                    break
                rank -= count
            else:  # Only reachable if the internal count table is corrupted.
                raise RuntimeError("Composition completion counts are inconsistent")
        return result


def sampler_from_env(env) -> UniformCompositionSampler:
    """Exact uniform sampler for a single unfiltered fraction domain.

    General chemical filters need their own counted-domain implementation;
    silently calling a random-action rollout would not be uniform on materials.
    """
    schema = env.state_feature_schema()
    if schema.get("kind") != "fractional_composition":
        raise ValueError("Exact composition counting requires one fraction-composition group")
    if getattr(env, "phase_filter", None) is not None:
        raise ValueError("Exact counting/sampling does not include plugin filters; use an explicitly counted domain")
    return UniformCompositionSampler(schema["species_allowed_units"], schema["total_units"],
                                     required=schema["mandatory_species"], slots=schema["n_components"])
