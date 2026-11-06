from __future__ import annotations

import re
from .utils.reward_payload import call_terminal_reward

import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Mapping
from fractions import Fraction
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .featurization import featurize_formula
from .utils.composition_domain import (
    allowed_unit_menu,
    can_complete,
    completion_sums,
    derive_total_units as _derive_domain_units,
    expand_fraction_codes,
    fraction_units,
    normalize_unit_bounds,
)


EpisodeStyle = Literal["element_then_amount", "fixed_order_amount"]


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float
    allowed_actions: List = field(default_factory=list)


def _decimals_of(fraction_set: Sequence[str], minimum: int = 2) -> int:
    """Decimal places this grid's codes use — at least *minimum* (2).

    The precision is read off the ``fraction_set`` tokens themselves rather than
    configured, because those tokens are exactly what ``encode_choice`` must match.
    Returns 2 for every grid whose entries have <= 2 decimals (0.05, 0.01, …), so
    existing scenarios keep byte-identical codes; a 0.125 grid returns 3.
    """
    d = minimum
    for token in fraction_set:
        text = str(token)
        if "." in text:
            d = max(d, len(text.split(".", 1)[1]))
    return d


def _format_fraction(units: int, total_units: int = 20, decimals: int = 2) -> str:
    """Render an exactly representable fraction without a binary-float roundtrip.

    ``decimals`` must preserve the authored menu's precision. Reject insufficient
    precision instead of silently rounding to another composition's code.
    """
    scale = 10 ** decimals
    scaled = Fraction(units, total_units) * scale
    if scaled.denominator != 1:
        raise ValueError("Fraction is not exactly representable at the requested precision")
    whole, remainder = divmod(scaled.numerator, scale)
    return f"{whole}.{remainder:0{decimals}d}"


def _fractions_to_units(fractions: Sequence[str], total_units: int = 20) -> List[int]:
    """Convert codes exactly; an incompatible explicit lattice is an error."""
    return fraction_units(fractions, total_units)


def expand_fraction_set(spec):
    """Normalize a shared menu/range or a species-to-menu mapping.

    Per-species mappings retain their keys and sparse choices. The environment
    builds one common action alphabet while enforcing each species' own menu.
    """
    if isinstance(spec, Mapping) and spec and all(
        isinstance(value, (Mapping, list, tuple)) for value in spec.values()
    ):
        return {str(label): expand_fraction_codes(menu) for label, menu in spec.items()}
    return expand_fraction_codes(spec)


def derive_total_units(fraction_set) -> int:
    """Derive a shared exact lattice from all supplied fraction menus."""
    codes = ([code for menu in fraction_set.values() for code in menu]
             if isinstance(fraction_set, Mapping) else fraction_set)
    return _derive_domain_units(codes)


def _fraction_domains(spec, species_set):
    """Return canonical shared codes and matching per-species code menus."""
    expanded = expand_fraction_set(spec)
    if isinstance(expanded, Mapping):
        missing = sorted(set(species_set) - set(expanded))
        unknown = sorted(set(expanded) - set(species_set))
        if missing or unknown:
            raise ValueError(
                f"fraction_set species keys must match species_set; missing={missing}, unknown={unknown}"
            )
        seen = set()
        union = []
        for label in species_set:
            for code in expanded[label]:
                value = Fraction(code)
                if value not in seen:
                    seen.add(value)
                    union.append(code)
        codes = expand_fraction_codes(union)
        by_value = {Fraction(code): code for code in codes}
        menus = {label: [by_value[Fraction(code)] for code in expanded[label]]
                 for label in species_set}
    else:
        codes = expanded
        menus = {label: list(codes) for label in species_set}
    return codes, menus


def _possible_sums(units: Sequence[int], k: int, max_total: int) -> set[int]:
    sums = {0}
    for _ in range(k):
        next_sums: set[int] = set()
        for s in sums:
            for u in units:
                t = s + u
                if t <= max_total:
                    next_sums.add(t)
        sums = next_sums
    return sums


def _step_one_hot(step: int, max_steps: int) -> np.ndarray:
    if not (1 <= step <= max_steps):
        raise ValueError("Step out of range")
    v = np.zeros(max_steps, dtype=float)
    v[step - 1] = 1.0
    return v


DEFAULT_FRACTION_SET: List[str] = [
    "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35",
    "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80",
]


class CompositionEnv:
    """General N-step environment for constrained composition design.

    At each step the agent picks one element from ``species_set`` (no repeats)
    and one fraction from its exact menu. All fractions sum to exactly 1.0.
    Arithmetic feasibility uses the legal menu of every remaining species,
    preserving sparse choices and declared physical bounds.

    Parameters
    ----------
    species_set:
        Ordered list of candidate element symbols.
    fraction_set:
        A shared explicit menu or min/max/step range, or a mapping from every
        species to its own menu/range. A common exact integer lattice is derived
        from all choices; the action alphabet never broadens a species' menu.
    anion_formula:
        Appended to the terminal formula string.  Pass ``""`` for metallic alloys,
        ``"O2H1"`` for oxyhydroxides, ``"O3"`` for perovskite oxides, etc.
    n_components:
        Number of distinct cations in the final composition (episode length).
    reward_fn:
        Callable ``(formula: str) -> float`` evaluated only at the terminal step.
        Defaults to a zero reward (useful for testing).
    state_featurizer:
        Callable ``(partial_formula: str) -> np.ndarray``.
        Defaults to Magpie composite features via matminer.
    phase_filter:
        Optional :class:`~rl_matdesign.constraints.base.ConstraintFilter` instance
        that prunes ``allowed_actions`` at each step.  Pass ``None`` (default) for
        no domain constraint beyond fraction feasibility.
    total_units:
        Internal compatibility override; omitted values derive the exact common
        lattice. An override must represent every menu value exactly.
    element_bounds:
        Optional ``{element: (min_fraction, max_fraction)}`` overriding the
        global ``fraction_set`` range on a per-element basis.  Bounds are in
        the same fractional units as ``fraction_set`` (e.g. ``(0.45, 0.90)``).
        Elements absent from the dict inherit ``[0, 1]``.  Supported by BOTH
        episode styles. Under ``"element_then_amount"`` a lower bound > 0 makes
        that element **mandatory** (an absent element has fraction 0, which would
        violate ``lo>0``), and the legal amounts at each step are pruned so the
        remaining steps can always complete to a valid bounded composition.
    episode_style:
        ``"element_then_amount"`` chooses an unassigned species and amount.
        The deprecated ``"fixed_order_amount"`` alias assigns all species once
        (deriving n_components) but now allows their assignment order to vary.
        Terminal composition membership is preserved; trajectories change.
    """

    def __init__(
        self,
        *,
        species_set: Sequence[str],
        fraction_set: Sequence[str] | Mapping = DEFAULT_FRACTION_SET,
        anion_formula: str = "",
        n_components: int = 5,
        reward_fn: Callable[[str], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        phase_filter=None,
        total_units: Optional[int] = None,
        element_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        episode_style: EpisodeStyle = "element_then_amount",
    ) -> None:
        self.species_set = list(species_set)
        if not self.species_set or len(set(self.species_set)) != len(self.species_set):
            raise ValueError("species_set must contain distinct labels and cannot be empty.")
        # A coordinate identifies a species, never the order of assignment. The
        # action alphabet keeps its declared order and is recorded separately.
        self.state_labels = tuple(sorted(self.species_set))
        self.fraction_set, self._species_fraction_codes = _fraction_domains(fraction_set, self.species_set)
        self.anion_formula = anion_formula
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter
        # See IntegerRatioEnv: skip phase_filter when disabled (constrain_training).
        self.constraints_enabled = True
        if episode_style not in ("element_then_amount", "fixed_order_amount"):
            raise ValueError(
                f"episode_style must be 'element_then_amount' or "
                f"'fixed_order_amount', got {episode_style!r}."
            )
        if episode_style == "fixed_order_amount":
            warnings.warn(
                "episode_style='fixed_order_amount' is deprecated: all species are "
                "still assigned once, but their assignment order is now free.",
                DeprecationWarning, stacklevel=2,
            )
            self.n_components = len(self.species_set)
        else:
            self.n_components = n_components
        self.episode_style = "element_then_amount"
        if (not isinstance(self.n_components, int) or isinstance(self.n_components, bool)
                or not 1 <= self.n_components <= len(self.species_set)):
            raise ValueError("n_components must be an integer from 1 to len(species_set)")
        self.max_steps = self.n_components

        derived_units = derive_total_units(self.fraction_set)
        # Explicit total_units remains an internal constructor compatibility
        # argument. It must represent every code exactly; YAML derives it.
        self._total_units = derived_units if total_units is None else total_units
        self._frac_decimals = _decimals_of(self.fraction_set)
        self._allowed_units = _fractions_to_units(self.fraction_set, self._total_units)
        self._code_units = dict(zip(self.fraction_set, self._allowed_units))
        self._unit_codes = dict(zip(self._allowed_units, self.fraction_set))
        self._possible_sums_by_k: List[set[int]] = [
            _possible_sums(self._allowed_units, k, self._total_units)
            for k in range(self.n_components + 1)
        ]

        if element_bounds is not None and not isinstance(element_bounds, Mapping):
            raise ValueError("element_bounds must be a mapping of species to [lower, upper]")
        bounds_spec = element_bounds or {}
        unknown = sorted(set(bounds_spec) - set(self.species_set))
        if unknown:
            raise ValueError(f"element_bounds keys not in species_set: {unknown}")
        self._bounds_by_symbol = {}
        self._species_allowed_units = {}
        self._mandatory_species = set()
        for label in self.species_set:
            try:
                bounds = normalize_unit_bounds(bounds_spec.get(label, (0, 1)), self._total_units)
            except ValueError as exc:
                raise ValueError(f"element_bounds[{label!r}]: {exc}") from exc
            self._bounds_by_symbol[label] = bounds
            menu = _fractions_to_units(self._species_fraction_codes[label], self._total_units)
            self._species_allowed_units[label] = allowed_unit_menu(menu, bounds)
            if bounds[0] > 0:
                self._mandatory_species.add(label)
        # Retain compatibility introspection while the true legal domain is the
        # per-species sparse menu above, not a shared interval approximation.
        self._element_unit_bounds = ([self._bounds_by_symbol[label] for label in self.species_set]
                                     if element_bounds is not None else None)
        self._element_unit_bounds_by_sym = (dict(self._bounds_by_symbol)
                                            if element_bounds is not None else None)
        if len(self._mandatory_species) > self.n_components:
            raise ValueError(
                f"element_bounds requires {len(self._mandatory_species)} mandatory species "
                f"but n_components={self.n_components}"
            )
        if not self._species_completable(self._total_units, self.n_components, self.species_set):
            raise ValueError(
                "Composition domain is infeasible: no selection of "
                f"{self.n_components} distinct species can sum to 1 within "
                "their fraction menus and element_bounds."
            )

        self.path: List[EpisodeStep] = []
        # A present key with value zero is an assignment. The mapping therefore
        # carries both values and assignment status without a second mutable mask.
        self._units_map: Dict[str, int] = {}
        self._descriptor_dim = int(self._descriptor_features().size)

    def initialize(self) -> None:
        """Reset the environment to the start of a new episode."""
        self.path = []
        self._units_map = {}

    @property
    def counter(self) -> int:
        """Number of assigned slots, including explicit zero assignments."""
        return len(self._units_map)

    @property
    def _selected(self) -> set[str]:
        """Compatibility view; the assignment map is the authoritative record."""
        return set(self._units_map)

    @property
    def _used_units(self) -> int:
        return sum(self._units_map.values())

    @property
    def state(self) -> str:
        """Canonical partial formula for display and descriptor adapters.

        Assignment history and zero-valued choices are represented by the map,
        not by concatenated formula tokens. They remain present in the explicit
        state encoding even when this chemical formula is empty.
        """
        items = sorted(self._units_map.items(), key=lambda item: (-item[1], item[0]))
        return "".join(
            f"{element if re.fullmatch(r'[A-Z][a-z]?', element) else '(' + element + ')'}{_format_fraction(units, self._total_units, self._frac_decimals)}"
            for element, units in items if units > 0
        )

    @property
    def remaining_units(self) -> int:
        return self._total_units - self._used_units

    @property
    def terminal_formula(self) -> str:
        """Canonical formula string at the terminal step (empty if not terminal)."""
        if self.counter != self.n_components:
            return ""

        return f"{self.state}{self.anion_formula}"

    def _descriptor_features(self) -> np.ndarray:
        features = np.asarray(self.state_featurizer(self.state), dtype=float)
        if features.ndim != 1:
            raise ValueError("state_featurizer must return a one-dimensional descriptor vector.")
        if hasattr(self, "_descriptor_dim") and features.size != self._descriptor_dim:
            raise ValueError("state_featurizer dimension must remain fixed across episode states.")
        return features

    def _explicit_state_features(self) -> np.ndarray:
        fractions = [self._units_map.get(label, 0) / self._total_units
                     for label in self.state_labels]
        # Positive-only choices encode assignment status in their value. A
        # separate bit is needed only when assigning zero is a legal operation.
        assigned = ([float(label in self._units_map) for label in self.state_labels]
                    if 0 in self._allowed_units else [])
        resources = [self.remaining_units / self._total_units,
                     self.n_components - self.counter]
        return np.asarray([*fractions, *assigned, *resources], dtype=float)

    def current_state_features(self) -> np.ndarray:
        """Descriptor plus explicit assignments and remaining design resources.

        Material descriptors often normalize total amount. The appended absolute
        values retain the distinction between, for example, Ni=0.2 and Ni=0.6.
        Fixed labels make equivalent assignment orders exactly the same state.
        """
        return np.concatenate((self._descriptor_features(), self._explicit_state_features()))

    def state_feature_schema(self) -> dict:
        """Serializable contract for checkpoint/replay compatibility checks."""
        descriptor_dim = self._descriptor_dim
        fn = self.state_featurizer
        identity = f"{getattr(fn, '__module__', type(fn).__module__)}:{getattr(fn, '__qualname__', type(fn).__qualname__)}"
        has_mask = 0 in self._allowed_units
        explicit_labels = [f"fraction:{label}" for label in self.state_labels]
        if has_mask:
            explicit_labels.extend(f"assigned:{label}" for label in self.state_labels)
        explicit_labels.extend(("remaining_fraction", "remaining_slots"))
        return {
            "schema_version": 3,
            "kind": "fractional_composition",
            "labels": list(self.state_labels),
            "descriptor_dim": descriptor_dim,
            "descriptor_identity": identity,
            "descriptor_schema": ("canonical_composition_v2" if fn is featurize_formula else "custom"),
            "layout": ["descriptor", "absolute_fractions", *(["assigned_mask"] if has_mask else []), "resources"],
            "explicit_feature_labels": explicit_labels,
            "assignment_mask": has_mask,
            "resources": {"remaining_fraction": "remaining_units / total_units",
                          "remaining_slots": "n_components - assigned_slots"},
            "feature_dim": descriptor_dim + len(explicit_labels),
            "total_units": self._total_units,
            "n_components": self.n_components,
            "action_species_order": list(self.species_set),
            "action_value_order": list(self.fraction_set),
            "species_fraction_codes": {label: list(self._species_fraction_codes[label])
                                       for label in self.state_labels},
            "species_allowed_units": {label: list(self._species_allowed_units[label])
                                      for label in self.state_labels},
            "element_unit_bounds": {label: list(self._bounds_by_symbol[label])
                                    for label in self.state_labels},
            "mandatory_species": sorted(self._mandatory_species),
            "episode_style": self.episode_style,
        }

    def cation_fractions(self) -> Dict[str, float]:
        return {label: self._units_map[label] / self._total_units
                for label in self.state_labels if label in self._units_map}

    def terminal_cation_fractions(self) -> Dict[str, float]:
        if self.counter != self.n_components:
            raise RuntimeError("terminal_cation_fractions called before terminal step")
        return {label: value for label, value in self.cation_fractions().items() if value > 0}

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key for dedup/visit-tracking.

        Quantises fractions to ``self._total_units`` and sorts alphabetically.
        Used by ``training.py`` as the cross-env dedup interface.
        """
        if self.counter != self.n_components:
            raise RuntimeError("terminal_comp_key called before terminal step")
        return tuple((label, self._units_map[label]) for label in self.state_labels
                     if self._units_map.get(label, 0) > 0)

    def _species_completable(self, target: int, k_slots: int, labels: Sequence[str]) -> bool:
        """Exact completion using each remaining species' own legal menu."""
        if target < 0 or k_slots < 0:
            return False
        pool = [(self._species_allowed_units[label], label in self._mandatory_species)
                for label in labels]
        return can_complete(target, k_slots, pool, self._total_units)

    def _completable(self, target: int, k_slots: int, pool: Sequence[Tuple[int, int]]) -> bool:
        """Compatibility helper for bounds on the shared menu; exact on gaps."""
        if target < 0 or k_slots < 0:
            return False
        menus = [(allowed_unit_menu(self._allowed_units, bounds), bounds[0] > 0)
                 for bounds in pool]
        return can_complete(target, k_slots, menus, self._total_units)

    def _allowed_units_for_symbol(self, elem: str) -> List[int]:
        if elem in self._units_map or self.counter >= self.n_components:
            return []
        remaining_labels = [label for label in self.species_set
                            if label not in self._units_map and label != elem]
        pool = [(self._species_allowed_units[label], label in self._mandatory_species)
                for label in remaining_labels]
        reachable = completion_sums(pool, self.n_components - self.counter - 1, self._total_units)
        return [value for value in self._species_allowed_units[elem]
                if value <= self.remaining_units
                and reachable & (1 << (self.remaining_units - value))]

    def _allowed_fraction_units_now(self, *, for_element_idx: Optional[int] = None) -> List[int]:
        """Compatibility view of values legal for a species or any next species."""
        if for_element_idx is not None:
            return self._allowed_units_for_symbol(self.species_set[for_element_idx])
        available = {value for label in self.species_set if label not in self._units_map
                     for value in self._allowed_units_for_symbol(label)}
        return [value for value in self._allowed_units if value in available]

    def allowed_actions(
        self, *, prior_groups: Optional[List[Dict[str, float]]] = None
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return all feasibility-guaranteed (elem_onehot, comp_onehot) pairs.

        ``prior_groups`` is forwarded to the constraint filter for cross-group
        coupling under :class:`MultiGroupEnv`; it is ``None`` (and ignored) for
        standalone single-group use, so existing callers are unaffected.
        """
        if self.counter >= self.n_components:
            return []

        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        elem_units = {label: self._allowed_units_for_symbol(label)
                      for label in self.species_set if label not in self._units_map}

        for elem, units in elem_units.items():
            elem_oh = tuple(encode_choice(elem, self.species_set).tolist())
            for u in units:
                comp = self._unit_codes[u]
                comp_oh = tuple(encode_choice(comp, self.fraction_set).tolist())
                actions.append((elem_oh, comp_oh))

        if self.phase_filter is not None and self.constraints_enabled:
            # Retain the legal domain independently: a plug-in may prune its
            # copy in place, but cannot add actions or mutate environment state.
            available = set(actions)
            kw = dict(
                actions=list(actions),
                units_map=dict(self._units_map),
                steps_left=self.n_components - self.counter - 1,
                allowed_units=list(self._allowed_units),
                possible_sums_by_k=[set(sums) for sums in self._possible_sums_by_k],
                species_set=list(self.species_set),
                fraction_set=list(self.fraction_set),
            )
            # Only forward prior_groups when it is actually set (i.e. from
            # MultiGroupEnv). Standalone single-group use passes None, so filters
            # are called exactly as before — external/user filters that predate
            # the cross-group hook keep working unchanged.
            if prior_groups is not None:
                kw["prior_groups"] = deepcopy(prior_groups)
            filtered = self.phase_filter.filter_actions(**kw)
            prefix = f"Composition constraint filter ({type(self.phase_filter).__name__})"
            if not isinstance(filtered, (list, tuple)):
                raise ValueError(f"{prefix} must return a list or tuple of actions.")
            try:
                selected = set(filtered)
            except TypeError as exc:
                raise ValueError(f"{prefix} returned an invalid action encoding.") from exc
            if not selected.issubset(available):
                raise ValueError(f"{prefix} must return a subset of its input actions.")
            if len(selected) != len(filtered):
                raise ValueError(f"{prefix} must not duplicate actions.")
            actions = list(filtered)

        return actions

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        if self.counter >= self.n_components:
            raise RuntimeError("Cannot sample an action after the episode terminated.")
        actions = self.allowed_actions()
        if not actions:
            from .utils.rollout_errors import InfeasibleEpisode
            raise InfeasibleEpisode("No valid composition actions remain under the configured constraints.")
        return random.choice(actions)

    def step(
        self,
        action: Tuple[Tuple[float, ...], Tuple[float, ...]],
        *,
        prior_groups: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        if self.counter >= self.n_components:
            raise RuntimeError("step() called after the episode terminated.")
        elem_oh, comp_oh = action
        elem = decode_one_hot(elem_oh, self.species_set)
        comp_str = decode_one_hot(comp_oh, self.fraction_set)

        if elem in self._units_map:
            raise ValueError(f"Repeated element '{elem}' is not allowed.")
        comp_units = self._code_units[comp_str]
        if comp_units not in self._species_allowed_units[elem]:
            raise ValueError(f"Amount {comp_str} for {elem!r} is outside its fraction menu or bounds.")
        if comp_units not in self._allowed_units_for_symbol(elem):
            raise ValueError("Action makes a valid terminal composition impossible.")

        current_allowed = self.allowed_actions(prior_groups=prior_groups)
        action_key = tuple(tuple(np.asarray(part).tolist()) for part in action)
        if action_key not in current_allowed:
            raise ValueError("Action is not allowed by the current composition constraints.")
        s_material = self.current_state_features()

        self._units_map[elem] = comp_units

        reward = 0.0
        if self.counter == self.n_components:
            reward = float(call_terminal_reward(self.reward_fn, self.terminal_cation_fractions(), self.terminal_formula))

        s_step = _step_one_hot(self.counter, self.n_components)

        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=elem_oh,
                action_comp_onehot=comp_oh,
                reward=reward,
                allowed_actions=current_allowed,
            )
        )
