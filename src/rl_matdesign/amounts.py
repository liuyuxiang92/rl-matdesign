"""Exact additive chemical-amount domains, independent of a scenario builder.

Actions add a menu amount to a labelled species. Final bounds apply to accumulated
amounts, including zero for absent species. Completion masks use bitset dynamic
programming over species and the remaining action budget, not action histories.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd, isfinite, lcm
import random
import re

import numpy as np

from .action_encoding import ValueSpec
from .constraints.base import apply_filters
from .domain_runtime import EpisodeStep, one_hot_index
from .featurization import featurize_formula


def _number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        decimal = Decimal(str(value))
        if not decimal.is_finite() or not isfinite(float(decimal)):
            raise ValueError
        return Fraction(decimal)
    except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc


def _code(value: Fraction) -> str:
    # All accepted inputs are finite decimals, so this division terminates.
    numerator, denominator = value.numerator, value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("Amount cannot be represented as a finite decimal")
    places = max(twos, fives)
    scaled = abs(numerator) * (10 ** places // value.denominator)
    sign = "-" if numerator < 0 else ""
    if places == 0:
        return sign + str(scaled)
    whole, remainder = divmod(scaled, 10 ** places)
    return (sign + f"{whole}.{remainder:0{places}d}").rstrip("0").rstrip(".")


def _menu(spec):
    if isinstance(spec, Mapping):
        if set(spec) != {"min", "max", "step"}:
            raise ValueError("An amount range requires exactly min, max and step")
        low, high, step = (_number(spec[key], f"amount_set.{key}") for key in ("min", "max", "step"))
        if step <= 0 or low > high:
            raise ValueError("Amount range needs min <= max and step > 0")
        values = [low + index * step for index in range(int((high - low) // step) + 1)]
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        values = [_number(value, "amount_set entry") for value in spec]
    else:
        raise ValueError("amount_set must be a nonempty list or a min/max/step range")
    if not values or len(set(values)) != len(values):
        raise ValueError("Amount menus must be nonempty and contain distinct numerical values")
    return tuple(values)


def _formula(amounts):
    return "".join(
        f"{label if re.fullmatch(r'[A-Z][a-z]?', label) else '(' + label + ')'}{_code(value)}"
        for label, value in sorted(amounts.items()) if value
    )


class ChemicalAmountsGroup:
    """One exact additive amount group with optional total and uniqueness.

    ``composition_resolver`` is a Python hook taking the raw amount mapping and
    returning a formula or elemental mapping for descriptor calculation. For a
    signed edit domain without that hook, the chemical block explicitly describes
    absolute edit amounts; the signed amounts remain in the state vector.
    """

    candidate_order_sensitive = False
    assignment_order_sensitive = False

    def __init__(self, spec=None, *, species_set=None, amount_set=None, n_steps=None,
                 total=None, unique=None, element_bounds=None, name=None,
                 state_featurizer=featurize_formula, filters=(),
                 composition_resolver=None, candidate_validator=None):
        if spec is not None and not isinstance(spec, Mapping):
            raise ValueError("Chemical amount group specification must be a mapping")
        cfg = dict(spec or {})
        allowed = {"name", "kind", "species_set", "amount_set", "n_steps", "total", "unique", "element_bounds", "filters"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"Unsupported composition group fields: {sorted(unknown)}")
        if cfg.get("kind", "composition") != "composition":
            raise ValueError("ChemicalAmountsGroup requires kind: composition")
        for key, value in (("species_set", species_set), ("amount_set", amount_set),
                           ("n_steps", n_steps), ("total", total), ("unique", unique),
                           ("element_bounds", element_bounds), ("name", name)):
            if value is not None:
                cfg[key] = value
        labels = cfg.get("species_set")
        if not isinstance(labels, (list, tuple)) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ) or len(set(labels)) != len(labels):
            raise ValueError("species_set must contain distinct nonempty string labels")
        self.species_set = list(labels)
        self.state_labels = tuple(sorted(labels))
        self.name = str(cfg.get("name", "composition"))
        horizon = cfg.get("n_steps")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("n_steps must be a positive integer")
        self.n_steps = self.max_steps = horizon
        self.unique = cfg.get("unique", False)
        if not isinstance(self.unique, bool):
            raise ValueError("unique must be a boolean")
        if self.unique and horizon > len(labels):
            raise ValueError("A unique group cannot have more steps than species")
        raw_menus = cfg.get("amount_set")
        if isinstance(raw_menus, Mapping) and set(raw_menus) != {"min", "max", "step"}:
            if set(raw_menus) != set(labels):
                raise ValueError("Per-species amount_set keys must match species_set")
            menus = tuple(_menu(raw_menus[label]) for label in labels)
        else:
            menus = (_menu(raw_menus),) * len(labels)
        values = tuple(dict.fromkeys(value for menu in menus for value in menu))
        denominator = lcm(*(value.denominator for value in values))
        numerators = [int(value * denominator) for value in values if value]
        numerator = reduce(gcd, (abs(n) for n in numerators), 0) or denominator
        self._quantum = Fraction(numerator, denominator)
        self.action_values = [_code(value) for value in values]
        self.value_specs = [ValueSpec(float(value), "numeric", (self.name, "amount")) for value in values]
        self._values = values
        self._digits = tuple(int(value / self._quantum) for value in values)
        self._menus = tuple(tuple(int(value / self._quantum) for value in menu) for menu in menus)
        self._menu_indices = tuple(tuple(values.index(value) for value in menu) for menu in menus)
        self._allowed_units = sorted(set(self._digits))
        self._species_amount_codes = {label: [_code(v) for v in menu] for label, menu in zip(labels, menus)}
        # Physical amount per lattice unit is quantum; its reciprocal converts
        # normalized OOH amounts into the same exact lattice.
        reciprocal = 1 / self._quantum
        self.amount_scale = reciprocal.numerator if reciprocal.denominator == 1 else float(reciprocal)
        self.total = None if cfg.get("total") is None else _number(cfg["total"], "total")
        self._target = None
        if self.total is not None:
            target = self.total / self._quantum
            if target.denominator != 1:
                raise ValueError("The requested total is infeasible on the amount lattice")
            self._target = int(target)
        self._span = horizon * max(abs(value) for value in self._digits)
        self._offset = self._span
        self._all_bits = (1 << (2 * self._span + 1)) - 1
        self._zero = 1 << self._offset
        self._base_cache = {}
        union_reachable = self._reachable(tuple(self._allowed_units))
        self._possible_sums_by_k = [set(self._integers(bits)) for bits in union_reachable]
        bounds = cfg.get("element_bounds") or {}
        if not isinstance(bounds, Mapping) or set(bounds) - set(labels):
            raise ValueError("element_bounds keys must be species_set labels")
        self.element_bounds = {}
        self._bounds = []
        for label, menu in zip(labels, self._menus):
            default = (min(0, horizon * min(menu)), max(0, horizon * max(menu)))
            if label not in bounds:
                self._bounds.append(default)
                continue
            pair = bounds[label]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each element_bounds entry must be [minimum, maximum]")
            lo, hi = (_number(value, f"element_bounds.{label}") for value in pair)
            if lo > hi:
                raise ValueError("Element bound minimum must not exceed maximum")
            low = -((-lo / self._quantum).numerator // (-lo / self._quantum).denominator)
            high = (hi / self._quantum).numerator // (hi / self._quantum).denominator
            self._bounds.append((low, high))
            self.element_bounds[label] = [float(lo), float(hi)]
        self.state_featurizer = state_featurizer
        self.composition_resolver = composition_resolver
        self.candidate_validator = candidate_validator
        if self.candidate_validator is not None and not callable(self.candidate_validator):
            raise ValueError("candidate_validator must be callable")
        self._signed_domain = any(value < 0 for value in self._digits)
        self.filters = list(filters)
        self._legal = lru_cache(maxsize=256)(self._compute_legal)
        self.initialize()
        self._descriptor_dim = int(self._descriptor_features().size)
        if not self._legal(tuple(self._units), tuple(self._picks), horizon):
            raise ValueError("Chemical amount domain is infeasible: no complete candidate satisfies the constraints")

    def _shift(self, bits, amount):
        return ((bits << amount) if amount >= 0 else (bits >> -amount)) & self._all_bits

    def _integers(self, bits):
        while bits:
            low = bits & -bits
            yield low.bit_length() - 1 - self._offset
            bits ^= low

    def _range_mask(self, low, high):
        low, high = max(low, -self._span), min(high, self._span)
        if low > high:
            return 0
        return ((1 << (high - low + 1)) - 1) << (low + self._offset)

    def _reachable(self, menu):
        if menu not in self._base_cache:
            rows = [self._zero]
            for _ in range(self.n_steps):
                bits = 0
                for amount in menu:
                    bits |= self._shift(rows[-1], amount)
                rows.append(bits)
            self._base_cache[menu] = tuple(rows)
        return self._base_cache[menu]

    def _convolution(self, first, second):
        if first.bit_count() > second.bit_count():
            first, second = second, first
        result = 0
        for amount in self._integers(first):
            result |= self._shift(second, amount)
        return result

    def _combine(self, first, second, remaining):
        result = [0] * (remaining + 1)
        for left, left_bits in enumerate(first):
            if not left_bits:
                continue
            for right in range(min(len(second) - 1, remaining - left) + 1):
                if second[right]:
                    result[left + right] |= self._convolution(left_bits, second[right])
        return result

    @staticmethod
    def _combine_counts(first, second, remaining):
        result = 0
        for count in range(remaining + 1):
            if first & (1 << count):
                result |= second << count
        return result & ((1 << (remaining + 1)) - 1)

    def _options(self, index, units, picks, remaining):
        low, high = self._bounds[index]
        mask = self._range_mask(low - units, high - units)
        base = self._reachable(self._menus[index])
        cap = min(remaining, 1 - picks) if self.unique else remaining
        return [base[count] & mask if count <= cap else 0 for count in range(remaining + 1)]

    def _compute_legal(self, units, picks, remaining):
        if remaining <= 0:
            return ()
        options = [self._options(index, units[index], picks[index], remaining)
                   for index in range(len(self.species_set))]
        legal = []
        if self._target is None:
            count_options = [sum(1 << count for count, bits in enumerate(rows) if bits) for rows in options]
            prefix = [1]
            for counts in count_options:
                prefix.append(self._combine_counts(prefix[-1], counts, remaining))
            suffix = [0] * (len(options) + 1)
            suffix[-1] = 1
            for index in range(len(options) - 1, -1, -1):
                suffix[index] = self._combine_counts(count_options[index], suffix[index + 1], remaining)
            for index, menu in enumerate(self._menus):
                if self.unique and picks[index]:
                    continue
                other = self._combine_counts(prefix[index], suffix[index + 1], remaining)
                low, high = self._bounds[index]
                base = self._reachable(menu)
                cap = 0 if self.unique else remaining - 1
                for value_index in self._menu_indices[index]:
                    amount = self._digits[value_index]
                    mask = self._range_mask(low - units[index] - amount, high - units[index] - amount)
                    if any((other & (1 << (remaining - 1 - count))) and (base[count] & mask)
                           for count in range(cap + 1)):
                        legal.append((index, value_index))
        else:
            identity = [self._zero] + [0] * remaining
            prefix = [identity]
            for rows in options:
                prefix.append(self._combine(prefix[-1], rows, remaining))
            suffix = [None] * (len(options) + 1)
            suffix[-1] = identity
            for index in range(len(options) - 1, -1, -1):
                suffix[index] = self._combine(options[index], suffix[index + 1], remaining)
            target = self._target - sum(units)
            for index, menu in enumerate(self._menus):
                if self.unique and picks[index]:
                    continue
                other = self._combine(prefix[index], suffix[index + 1], remaining)
                low, high = self._bounds[index]
                mask = self._range_mask(low - units[index], high - units[index])
                cap = 0 if self.unique else remaining - 1
                wanted = []
                for count in range(cap + 1):
                    bits = 0
                    for rest in self._integers(other[remaining - 1 - count]):
                        own = target - rest
                        if -self._span <= own <= self._span:
                            bits |= 1 << (own + self._offset)
                    wanted.append(bits & mask)
                base = self._reachable(menu)
                for value_index in self._menu_indices[index]:
                    amount = self._digits[value_index]
                    if any(self._shift(base[count], amount) & wanted[count] for count in range(cap + 1)):
                        legal.append((index, value_index))
        return tuple(legal)

    def initialize(self):
        self.counter = 0
        self._units = [0] * len(self.species_set)
        self._picks = [0] * len(self.species_set)
        self._units_map = {}
        self.path = []

    @property
    def is_terminal(self):
        return self.counter == self.n_steps

    @property
    def is_done(self):
        return self.is_terminal

    @property
    def remaining_units(self):
        return None if self._target is None else self._target - sum(self._units)

    @property
    def state(self):
        return _formula({label: units * self._quantum for label, units in zip(self.species_set, self._units)})

    @property
    def terminal_formula(self):
        return self.state if self.is_terminal else ""

    def cation_fractions(self):
        return {label: float(units * self._quantum)
                for label, units, picks in zip(self.species_set, self._units, self._picks) if picks}

    def terminal_cation_fractions(self):
        if not self.is_terminal:
            raise RuntimeError("A terminal amount mapping requires a complete candidate")
        # Chemical absence is zero regardless of which zero-valued labels were
        # selected. Pick counts stay in the state; material payloads and caches
        # must not distinguish these equivalent terminal recipes.
        return {label: amount for label, amount in self.cation_fractions().items() if amount}

    def terminal_comp_key(self):
        if not self.is_terminal:
            raise RuntimeError("A terminal identity requires a complete candidate")
        return tuple(sorted((label, float(units * self._quantum))
                            for label, units in zip(self.species_set, self._units) if units))

    def _descriptor_features(self):
        amounts = self.cation_fractions()
        if self.composition_resolver is not None:
            resolved = self.composition_resolver(dict(amounts))
            if isinstance(resolved, str):
                formula = resolved
            elif isinstance(resolved, Mapping):
                elemental = {str(label): _number(value, "resolved composition")
                             for label, value in resolved.items()}
                if any(value < 0 for value in elemental.values()):
                    raise ValueError("A resolved material composition cannot contain negative amounts")
                formula = _formula(elemental)
            else:
                raise ValueError("composition_resolver must return a formula or amount mapping")
        else:
            formula = _formula({label: abs(units) * self._quantum
                                for label, units in zip(self.species_set, self._units)})
        features = np.asarray(self.state_featurizer(formula), dtype=float)
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("Chemical descriptor must be a finite one-dimensional vector")
        if hasattr(self, "_descriptor_dim") and features.size != self._descriptor_dim:
            raise ValueError("Chemical descriptor dimension must stay fixed")
        return features

    def current_state_features(self):
        rows = {label: (float(units * self._quantum), picks)
                for label, units, picks in zip(self.species_set, self._units, self._picks)}
        resources = ([] if self._target is None else [float(self.remaining_units * self._quantum)])
        resources.append(self.n_steps - self.counter)
        return np.concatenate((self._descriptor_features(), [rows[label][0] for label in self.state_labels],
                               [rows[label][1] for label in self.state_labels], resources))

    def state_feature_schema(self):
        resources = (["remaining_total"] if self._target is not None else []) + ["remaining_steps"]
        mode = "resolved_composition" if self.composition_resolver is not None else (
            "absolute_edit_amounts" if self._signed_domain else "formula_unit_composition")
        return {"schema_version": 1, "kind": "chemical_amounts", "labels": list(self.state_labels),
                "descriptor_dim": self._descriptor_dim, "descriptor_mode": mode,
                "descriptor_identity": f"{getattr(self.state_featurizer, '__module__', '')}:{getattr(self.state_featurizer, '__qualname__', type(self.state_featurizer).__qualname__)}",
                "layout": ["descriptor", "raw_amounts", "pick_counts", *resources],
                "feature_dim": self._descriptor_dim + 2 * len(self.species_set) + len(resources),
                "quantum": _code(self._quantum), "unique": self.unique,
                "action_species_order": list(self.species_set), "action_value_order": list(self.action_values)}

    def design_schema(self):
        return {"version": 1, "kind": "composition", "action_protocol": "add_amount",
                "n_steps": self.n_steps, "unique": self.unique,
                "total": None if self.total is None else _code(self.total),
                "element_bounds": deepcopy(self.element_bounds),
                "amount_menus": {label: list(codes) for label, codes in self._species_amount_codes.items()},
                "state": self.state_feature_schema(), "terminal_identity": "raw_accumulated_amounts"}

    def _action(self, label, value):
        return (tuple(float(index == label) for index in range(len(self.species_set))),
                tuple(float(index == value) for index in range(len(self.action_values))))

    def allowed_actions(self, *, prior_groups=None):
        indices = self._legal(tuple(self._units), tuple(self._picks), self.n_steps - self.counter)
        if self.candidate_validator is not None:
            def valid(pair):
                label_index, value_index = pair
                amounts = self.cation_fractions()
                amounts[self.species_set[label_index]] = float(
                    (self._units[label_index] + self._digits[value_index]) * self._quantum)
                return bool(self.candidate_validator(amounts))
            indices = tuple(pair for pair in indices if valid(pair))
        actions = [self._action(*pair) for pair in indices]
        if self.filters and actions:
            kwargs = dict(actions=list(actions), units_map=dict(self._units_map),
                          steps_left=self.n_steps - self.counter - 1, allowed_units=list(self._digits),
                          possible_sums_by_k=[set(row) for row in self._possible_sums_by_k],
                          species_set=list(self.species_set), action_values=list(self.action_values),
                          chemical_amounts=dict(self.cation_fractions()),
                          amount_quantum=float(self._quantum), action_amounts=[float(v) for v in self._values])
            if prior_groups is not None:
                kwargs["prior_groups"] = deepcopy(prior_groups)
            actions = apply_filters(self.filters, **kwargs)
        return actions

    def step(self, action, *, prior_groups=None):
        if self.is_terminal:
            raise RuntimeError("Cannot step a completed amount group")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("An amount action must be a pair of one-hot vectors")
        label = one_hot_index(action[0], len(self.species_set))
        value = one_hot_index(action[1], len(self.action_values))
        normalized = self._action(label, value)
        allowed = self.allowed_actions(prior_groups=prior_groups)
        if normalized not in allowed:
            raise ValueError("Amount action is infeasible or excluded by constraints")
        before = self.current_state_features()
        step_one_hot = tuple(float(index == self.counter) for index in range(self.n_steps))
        self._units[label] += self._digits[value]
        self._picks[label] += 1
        self._units_map[self.species_set[label]] = self._units[label]
        self.counter += 1
        self.path.append(EpisodeStep(state_material_features=before, state_step_onehot=step_one_hot,
                                     action_elem_onehot=normalized[0], action_comp_onehot=normalized[1],
                                     reward=0.0, allowed_actions=allowed))

    def sample_random_action(self):
        if self.is_terminal:
            raise RuntimeError("Cannot sample an action from a completed amount group")
        actions = self.allowed_actions()
        if not actions:
            from .utils.rollout_errors import InfeasibleEpisode
            raise InfeasibleEpisode("No valid chemical amount actions remain")
        return random.choice(actions)
