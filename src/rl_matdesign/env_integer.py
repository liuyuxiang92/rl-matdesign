"""Integer-token composition design with hard budgets and canonical materials.

A complete action adds a nonnegative integer amount to a labelled species.
Repeated species add; zero still consumes a token. Hard token requirements are
independent of optional chemical screening and do not prescribe action order.
State retains absolute counts and selection counts, while terminal material
identity and oracle payloads use a positive GCD-reduced empirical composition.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from math import gcd
from numbers import Integral
import random
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .action_encoding import ValueSpec
from .env import EpisodeStep, _step_one_hot
from .featurization import featurize_formula
from .utils.reward_payload import call_terminal_reward


DEFAULT_RATIO_SET = [str(value) for value in range(10)]
Action = Tuple[Tuple[float, ...], Tuple[float, ...]]


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def normalize_token_requirements(raw) -> dict:
    """Validate the concise public spelling; count is exact, not a second bound."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("token_requirements must be a mapping keyed by species")
    result = {}
    for symbol, spec in raw.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("token requirement species must be non-empty strings")
        if not isinstance(spec, Mapping) or set(spec) - {"count", "positive"} or "count" not in spec:
            raise ValueError(f"token_requirements[{symbol!r}] requires count and accepts only count/positive")
        count = _integer(spec["count"], f"token_requirements[{symbol!r}].count")
        positive = spec.get("positive", False)
        if not isinstance(positive, bool):
            raise ValueError(f"token_requirements[{symbol!r}].positive must be a boolean")
        result[symbol] = {"count": count, "positive": positive}
    return dict(sorted(result.items()))


def normalize_integer_group_spec(group: dict) -> dict:
    """Migrate only the known reserved-positive singleton O rule to its domain.

    Arbitrary multi-element, nonreserved, nonpositive or external construction
    filters retain their declared behavior. This conversion preserves the
    terminal token domain while deliberately freeing the execution order.
    """
    # Keep runtime filter instances intact for direct Python callers; only the
    # declarative filter list and normalized requirements are replaced below.
    result = dict(group)
    requirements = normalize_token_requirements(result.get("token_requirements"))
    filters, migrated = [], False
    for entry in result.get("filters") or []:
        known = (isinstance(entry, Mapping) and entry.get("type") == "last_step_element"
                 and entry.get("required_elements") == ["O"]
                 and entry.get("nonzero_ratio_at_last", True) is True
                 and entry.get("reserve_for_last", True) is True)
        if not known:
            filters.append(entry)
            continue
        expected = {"count": 1, "positive": True}
        if "O" in requirements and requirements["O"] != expected:
            raise ValueError("Legacy reserved positive O rule conflicts with token_requirements['O']")
        requirements["O"] = expected
        migrated = True
    if requirements or "token_requirements" in result:
        result["token_requirements"] = dict(sorted(requirements.items()))
    if migrated:
        result["filters"] = filters
        warnings.warn("Integer O-last construction is migrated to one positive O token at any step; "
                      "use token_requirements: {O: {count: 1, positive: true}}", DeprecationWarning,
                      stacklevel=2)
    return result


@dataclass(frozen=True)
class _TokenRequirement:
    symbol: str
    minimum_picks: int
    maximum_picks: int
    positive: bool


class IntegerRatioEnv:
    """Accumulate integer tokens, with a fixed maximum and exact token domain.

    ``token_requirements={"O": {"count": 1, "positive": True}}`` means exactly
    one positive O token anywhere in the episode. Other species may repeat;
    their zero tokens still consume slots. ``constrain_training=False`` only
    disables optional filters, never these domain rules or nonempty material.
    """

    candidate_order_sensitive = False
    assignment_order_sensitive = False

    def __init__(self, *, species_set: Sequence[str], ratio_set: Sequence[str] = DEFAULT_RATIO_SET,
                 n_components: int = 5, reward_fn: Callable[[str], float] | None = None,
                 state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
                 phase_filter=None, token_requirements=None) -> None:
        if isinstance(species_set, (str, bytes)) or not species_set:
            raise ValueError("species_set must be a non-empty sequence of species")
        self.species_set = list(species_set)
        if any(not isinstance(symbol, str) or not symbol for symbol in self.species_set):
            raise ValueError("species_set entries must be non-empty strings")
        if len(set(self.species_set)) != len(self.species_set):
            raise ValueError("species_set entries must be unique")
        if isinstance(ratio_set, (str, bytes)) or not ratio_set:
            raise ValueError("ratio_set must be a non-empty sequence of nonnegative integers")
        digits = []
        for value in ratio_set:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("ratio_set values must be nonnegative integers, not booleans")
            try:
                digit = int(value)
                if digit < 0 or float(value) != digit or not np.isfinite(float(digit)):
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("ratio_set values must be finite nonnegative integers") from exc
            digits.append(digit)
        if len(set(digits)) != len(digits):
            raise ValueError("ratio_set values must be unique physical integers")
        self._digits = tuple(digits)
        self.fraction_set = self.ratio_set = [str(value) for value in digits]
        self.value_specs = [ValueSpec(value, "numeric") for value in digits]
        self.n_components = self.max_steps = _integer(n_components, "n_components", minimum=1)
        self.anion_formula = ""
        self.episode_style = "integer_accumulation"
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter
        self.constraints_enabled = True
        self._total_units = None
        self.state_labels = sorted(self.species_set)
        normalized = normalize_token_requirements(token_requirements)
        unknown = set(normalized) - set(self.species_set)
        if unknown:
            raise ValueError(f"token_requirements references unknown species: {sorted(unknown)}")
        self._requirements = tuple(_TokenRequirement(symbol, spec["count"], spec["count"], spec["positive"])
                                   for symbol, spec in normalized.items())
        self._required_by_symbol = {requirement.symbol: requirement for requirement in self._requirements}
        self._unrestricted = set(self.species_set) - set(normalized)
        if any(requirement.positive and requirement.minimum_picks and not any(value > 0 for value in digits)
               for requirement in self._requirements):
            raise ValueError("A positive token requirement has no positive value in ratio_set")
        self._element_onehots = [tuple(row) for row in np.eye(len(self.species_set))]
        self._value_onehots = [tuple(row) for row in np.eye(len(self.ratio_set))]
        self.initialize()
        self._descriptor_dim = self._descriptor_features().size
        if not self._domain_completable(self._digits_map, self._pick_counts, self.n_components):
            raise ValueError("Integer token domain admits no positive complete material within its token budget")

    @property
    def token_requirements(self) -> dict:
        return {requirement.symbol: {"count": requirement.minimum_picks, "positive": requirement.positive}
                for requirement in self._requirements}

    def initialize(self) -> None:
        self.path: List[EpisodeStep] = []
        self._digits_map: Dict[str, int] = {}
        self._pick_counts: Dict[str, int] = {}

    @property
    def counter(self) -> int:
        return sum(self._pick_counts.values())

    @property
    def is_terminal(self) -> bool:
        return self.counter == self.n_components

    @property
    def is_done(self) -> bool:
        return self.is_terminal

    def _is_terminal(self) -> bool:
        return self.is_terminal

    @staticmethod
    def _formula(counts: Mapping[str, int]) -> str:
        return "".join(f"{symbol}{count}" for symbol, count in sorted(counts.items()) if count > 0)

    @property
    def state(self) -> str:
        """Canonical partial empirical formula; explicit features retain zeros."""
        return self._formula(self._digits_map)

    @property
    def terminal_formula(self) -> str:
        return self._formula(self.reduced_cation_counts()) if self.is_terminal else ""

    def cation_digits(self) -> Dict[str, int]:
        """Raw accumulated amounts, including selected zero-token species."""
        return dict(sorted(self._digits_map.items()))

    def cation_fractions(self) -> Dict[str, int]:
        """Compatibility view of raw partial counts; use the terminal accessor for an oracle."""
        return self.cation_digits()

    def reduced_cation_counts(self) -> Dict[str, int]:
        """Canonical terminal empirical counts for a structured group payload."""
        if not self.is_terminal:
            raise RuntimeError("reduced_cation_counts requires a complete material")
        items = [(symbol, value) for symbol, value in sorted(self._digits_map.items()) if value > 0]
        if not items:
            raise RuntimeError("An empty integer composition is not a material")
        factor = reduce(gcd, (value for _, value in items))
        return {symbol: value // factor for symbol, value in items}

    def terminal_cation_fractions(self) -> Dict[str, float]:
        counts = self.reduced_cation_counts()
        total = sum(counts.values())
        return {symbol: count / total for symbol, count in counts.items()}

    def terminal_comp_key(self) -> tuple:
        return tuple(self.reduced_cation_counts().items())

    def _descriptor_features(self) -> np.ndarray:
        features = np.asarray(self.state_featurizer(self.state), dtype=float)
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("state_featurizer must return a finite one-dimensional descriptor")
        if hasattr(self, "_descriptor_dim") and features.size != self._descriptor_dim:
            raise ValueError("state_featurizer dimension must remain fixed")
        return features

    def current_state_features(self) -> np.ndarray:
        return np.concatenate((self._descriptor_features(),
                               [self._digits_map.get(symbol, 0) for symbol in self.state_labels],
                               [self._pick_counts.get(symbol, 0) for symbol in self.state_labels],
                               [self.n_components - self.counter]))

    def state_feature_schema(self) -> dict:
        fn = self.state_featurizer
        identity = f"{getattr(fn, '__module__', type(fn).__module__)}:{getattr(fn, '__qualname__', type(fn).__qualname__)}"
        return {"schema_version": 2, "kind": "integer_token_composition", "labels": list(self.state_labels),
                "descriptor_dim": int(self._descriptor_dim), "descriptor_identity": identity,
                "descriptor_schema": "canonical_composition_v2" if fn is featurize_formula else "custom",
                "layout": ["descriptor", "absolute_counts", "pick_counts", "remaining_tokens"],
                "feature_dim": int(self._descriptor_dim + 2 * len(self.state_labels) + 1),
                "max_tokens": self.n_components, "token_requirements": self.token_requirements,
                "action_species_order": list(self.species_set), "action_value_order": list(self.ratio_set)}

    def design_schema(self) -> dict:
        return {"version": 2, "kind": "integer", "action_protocol": "add_integer_token",
                "token_requirements": self.token_requirements, "positive_terminal": True,
                "terminal_identity": "positive_gcd_reduced_counts", "state": self.state_feature_schema()}

    def _domain_completable(self, counts: Mapping[str, int], picks: Mapping[str, int], remaining: int) -> bool:
        if remaining < 0:
            return False
        needed = 0
        positive_future = bool(self._unrestricted) and any(value > 0 for value in self._digits)
        for requirement in self._requirements:
            selected = picks.get(requirement.symbol, 0)
            if selected > requirement.maximum_picks:
                return False
            deficit = max(0, requirement.minimum_picks - selected)
            needed += deficit
            if deficit and any(value > 0 for value in self._digits):
                positive_future = True
        if needed > remaining or not self._unrestricted and needed != remaining:
            return False
        return sum(counts.values()) > 0 or remaining > 0 and positive_future

    def _domain_allows(self, element: str, amount: int) -> bool:
        requirement = self._required_by_symbol.get(element)
        if requirement is not None:
            if self._pick_counts.get(element, 0) >= requirement.maximum_picks or requirement.positive and amount == 0:
                return False
        counts, picks = dict(self._digits_map), dict(self._pick_counts)
        counts[element] = counts.get(element, 0) + amount
        picks[element] = picks.get(element, 0) + 1
        return self._domain_completable(counts, picks, self.n_components - self.counter - 1)

    def allowed_actions(self, *, prior_groups: Optional[List[dict]] = None) -> List[Action]:
        if self.is_terminal:
            return []
        actions = [(self._element_onehots[i], self._value_onehots[j])
                   for i, element in enumerate(self.species_set) for j, amount in enumerate(self._digits)
                   if self._domain_allows(element, amount)]
        if self.phase_filter is not None and self.constraints_enabled:
            available = set(actions)
            filtered = self.phase_filter.filter_actions(
                actions=list(actions), units_map=dict(self._digits_map),
                steps_left=self.n_components - self.counter - 1,
                allowed_units=list(self._digits), possible_sums_by_k=[],
                species_set=list(self.species_set), fraction_set=list(self.ratio_set),
                prior_groups=deepcopy(prior_groups),
            )
            prefix = f"Integer constraint filter ({type(self.phase_filter).__name__})"
            if not isinstance(filtered, (list, tuple)):
                raise ValueError(f"{prefix} must return a list or tuple of actions")
            try:
                selected = set(filtered)
            except TypeError as exc:
                raise ValueError(f"{prefix} returned an invalid action encoding") from exc
            if not selected.issubset(available):
                raise ValueError(f"{prefix} may only remove legal actions (subset required)")
            if len(selected) != len(filtered):
                raise ValueError(f"{prefix} must not duplicate actions")
            actions = list(filtered)
        return actions

    def sample_random_action(self) -> Action:
        if self.is_terminal:
            raise RuntimeError("sample_random_action() called after integer episode termination")
        actions = self.allowed_actions()
        if not actions:
            from .utils.rollout_errors import InfeasibleEpisode
            raise InfeasibleEpisode("No valid integer-token action completes this prefix")
        return random.choice(actions)

    @staticmethod
    def _one_hot_index(vector, size: int) -> int:
        values = np.asarray(vector, dtype=float)
        if (values.ndim != 1 or values.size != size or not np.all((values == 0) | (values == 1))
                or np.count_nonzero(values == 1) != 1):
            raise ValueError("Integer action must contain valid one-hot vectors")
        return int(np.argmax(values))

    def step(self, action: Action, *, prior_groups=None) -> None:
        if self.is_terminal:
            raise RuntimeError("step() called after integer episode termination")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Integer action must be a pair of one-hot vectors")
        element_index = self._one_hot_index(action[0], len(self.species_set))
        value_index = self._one_hot_index(action[1], len(self.ratio_set))
        normalized = (self._element_onehots[element_index], self._value_onehots[value_index])
        current_allowed = self.allowed_actions(prior_groups=prior_groups)
        if normalized not in current_allowed:
            raise ValueError("Integer action violates the token domain or active chemical constraints")
        previous_features = self.current_state_features()
        element, amount = self.species_set[element_index], self._digits[value_index]
        self._digits_map[element] = self._digits_map.get(element, 0) + amount
        self._pick_counts[element] = self._pick_counts.get(element, 0) + 1
        reward = (float(call_terminal_reward(self.reward_fn, self.terminal_cation_fractions(), self.terminal_formula))
                  if self.is_terminal else 0.0)
        self.path.append(EpisodeStep(state_material_features=previous_features,
                                     state_step_onehot=_step_one_hot(self.counter, self.n_components),
                                     action_elem_onehot=normalized[0], action_comp_onehot=normalized[1],
                                     reward=reward, allowed_actions=current_allowed))
