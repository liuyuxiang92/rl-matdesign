"""Position-labelled sequence assignments with exact completion constraints.

Blank construction and parent mutation use the same complete action: assign a
monomer identity to an explicit position. The order of assignments is irrelevant;
the final positions retain their meaning. Molecular chemistry belongs to the
external predictor, which may supply intrinsic token-count constraints.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import json
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from .action_encoding import ActionEncoding, LabelEncoding, ValueEncoding, ValueSpec


Action = tuple[tuple[float, ...], tuple[float, ...]]


def _token_list(values: Any, *, label: str, unique: bool = True) -> list[str]:
    if label == "sequence tokens" and isinstance(values, dict):
        raise ValueError("CSV sequence tokens are no longer supported; supply an explicit list of token IDs")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list of string IDs")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty string IDs (quote numeric IDs)")
    # IDs belong to the predictor's catalog. Do not coerce numerical strings,
    # trim them, or interpret their spelling as an element symbol.
    result = list(values)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate token IDs")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _code(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class SequenceGroup:
    """Fill all blank positions or edit between one and K parent positions.

    Every position can be assigned at most once. A parent edit must change its
    original token; parent-only tokens may be retained but are not replacements.
    ``num_mutations`` remains the maximum number of edits, as in the original
    mutation API. A valid parent candidate may STOP after at least one edit.
    Reaching the maximum edits, or filling every blank position, ends the group
    automatically. STOP itself consumes one step, so the horizon is still K/L.

    Count bounds and optional uniqueness apply to the final candidate. An
    intermediate parent may temporarily violate them if a valid completion
    exists (for example, swapping two tokens through a duplicate intermediate).
    """

    candidate_order_sensitive = True
    assignment_order_sensitive = False

    def __init__(self, spec: dict) -> None:
        if spec.get("filters") or spec.get("constraint_filter") is not None:
            raise ValueError("Sequence groups use count_constraints, not composition filters")
        self.name = str(spec.get("name", "sequence"))
        self.tokens = _token_list(spec.get("tokens"), label="sequence tokens")
        self.mutation_mode = "parent" in spec or "num_mutations" in spec
        if self.mutation_mode:
            if "parent" not in spec or "num_mutations" not in spec:
                raise ValueError("parent and num_mutations must be provided together")
            self.parent = _token_list(spec["parent"], label="sequence parent", unique=False)
            self.length = len(self.parent)
            if "length" in spec and _integer(spec["length"], "sequence length") != self.length:
                raise ValueError("sequence length must equal the parent length")
            self.num_mutations = _integer(spec["num_mutations"], "num_mutations")
            if not 1 <= self.num_mutations <= self.length:
                raise ValueError("num_mutations must be between 1 and the parent length")
            self.max_edits = self.num_mutations
            self._initial = list(self.parent)
        else:
            self.length = _integer(spec.get("length"), "sequence length")
            if self.length <= 0:
                raise ValueError("sequence length must be positive")
            self.parent = []
            self.num_mutations = 0
            self.max_edits = self.length
            self._initial = [None] * self.length
        self.max_steps = self.n_components = self.max_edits
        self.allow_repeats = spec.get("allow_repeats", True)
        if not isinstance(self.allow_repeats, bool):
            raise ValueError("allow_repeats must be a boolean")
        self.constraints_enabled = True  # Optional-filter toggle cannot disable intrinsic validity.
        self.phase_filter = None
        self._state_tokens = list(dict.fromkeys(self.tokens + self.parent))
        self._token_index = {token: index for index, token in enumerate(self._state_tokens)}
        self._read_constraints(spec.get("count_constraints", []))

        # Feasibility depends on constraint membership, not all 100+ individual
        # monomer identities, except when final uniqueness must be enforced.
        signatures = []
        self._token_signature = []
        for token in self._state_tokens:
            signature = tuple(int(token in members) for members in self._constraint_tokens)
            if signature not in signatures:
                signatures.append(signature)
            self._token_signature.append(signatures.index(signature))
        self._signatures = tuple(signatures)
        self._group_tokens = tuple(
            frozenset(token for token in self.tokens if self._group_of(token) == group)
            for group in range(len(signatures))
        )
        self._feasible = lru_cache(maxsize=100_000)(self._can_complete)

        # Opaque compatibility codes retain the existing pair-of-one-hots API.
        # Their actual categorical meaning is carried by the feature specs.
        label_values = [("position", position + 1) for position in range(self.length)]
        self.species_set = [_code(label) for label in label_values]
        self.fraction_set = [_code(["token", token]) for token in self.tokens]
        value_specs = [ValueSpec(token, "categorical", (self.name, "monomer")) for token in self.tokens]
        self._stop_label = self._stop_value = None
        if self.mutation_mode:
            self._stop_label = len(label_values)
            self._stop_value = len(value_specs)
            label_values.append(("control", "stop"))
            self.species_set.append(_code(["control", "stop"]))
            self.fraction_set.append(_code(["control", "stop"]))
            value_specs.append(ValueSpec("stop", "categorical", (self.name, "control")))
        self.encoding = ActionEncoding(LabelEncoding(label_values), ValueEncoding.from_specs(value_specs))
        self._label_onehots = [tuple(row) for row in np.eye(len(self.species_set))]
        self._value_onehots = [tuple(row) for row in np.eye(len(self.fraction_set))]
        self.initialize()
        if not self._feasible(()):
            raise ValueError("Sequence constraints admit no complete candidate within the assignment budget")

    def _read_constraints(self, constraints: Any) -> None:
        if not isinstance(constraints, list):
            raise ValueError("count_constraints must be a list")
        members, lower, upper = [], [], []
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise ValueError(f"count_constraints[{index}] must be a mapping")
            if "where" in constraint:
                raise ValueError("CSV 'where' selectors are no longer supported; supply explicit tokens")
            if set(constraint) - {"tokens", "min", "max"}:
                raise ValueError(f"count_constraints[{index}] has unsupported keys")
            if "tokens" not in constraint:
                raise ValueError("Each count constraint needs an explicit tokens list")
            selected = set(_token_list(constraint["tokens"], label="constraint tokens"))
            unknown = selected - self._token_index.keys()
            if unknown:
                raise ValueError(f"Count constraint references unknown tokens: {sorted(unknown)}")
            lo = _integer(constraint.get("min", 0), "count min")
            hi = _integer(constraint.get("max", self.length), "count max")
            if not 0 <= lo <= hi <= self.length:
                raise ValueError("Count bounds must satisfy 0 <= min <= max <= length")
            members.append(tuple(token for token in self._state_tokens if token in selected))
            lower.append(lo)
            upper.append(hi)
        self._constraint_tokens = tuple(members)
        self._lower, self._upper = tuple(lower), tuple(upper)

    @property
    def value_specs(self) -> list[ValueSpec]:
        return list(self.encoding.values.specs)

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""  # Generic material formula parsing must not encode this group.
        self._picked = list(self._initial)
        self._edited: set[int] = set()
        self._done = False

    @property
    def is_terminal(self) -> bool:
        return self._done

    @property
    def is_done(self) -> bool:
        return self.is_terminal

    def _is_terminal(self) -> bool:
        return self.is_terminal

    def _group_of(self, token: str) -> int:
        return self._token_signature[self._token_index[token]]

    def _fixed_assignments(self, position: int | None = None, token: str | None = None) -> tuple:
        fixed = {index: self._picked[index] for index in self._edited}
        if position is not None:
            fixed[position] = token
        return tuple((index, self._group_of(value) if self.allow_repeats else value)
                     for index, value in sorted(fixed.items()))

    def _can_complete(self, fixed: tuple) -> bool:
        finalized = dict(fixed)
        free = [position for position in range(self.length) if position not in finalized]
        remaining_budget = self.max_edits - len(finalized)
        if remaining_budget < 0:
            return False
        # A blank is never a valid retained monomer. Parent mode may retain
        # untouched positions, but its candidate domain excludes zero edits.
        minimum = max(1 - len(finalized), 0) if self.mutation_mode else len(free)
        for remaining in range(minimum, remaining_budget + 1):
            for changing in combinations(free, remaining):
                changing_set = set(changing)
                retained = [self._initial[position] for position in free if position not in changing_set]
                if any(token is None for token in retained):
                    continue
                if self.allow_repeats:
                    groups = [self._group_of(token) for token in retained] + list(finalized.values())
                    available = self._group_tokens
                else:
                    fixed_tokens = retained + list(finalized.values())
                    if len(set(fixed_tokens)) != len(fixed_tokens):
                        continue
                    groups = [self._group_of(token) for token in fixed_tokens]
                    available = tuple(tokens.difference(fixed_tokens) for tokens in self._group_tokens)
                counts = tuple(sum(self._signatures[group][i] for group in groups)
                               for i in range(len(self._lower)))
                if any(count > hi or count + remaining < lo
                       for count, lo, hi in zip(counts, self._lower, self._upper)):
                    continue
                options = tuple(tuple(group for group, tokens in enumerate(available)
                                      if tokens - {self._initial[position]}) for position in changing)
                if any(not groups for groups in options):
                    continue
                if self._assign_signatures(changing, options, counts, available):
                    return True
        return False

    def _assign_signatures(self, positions: tuple, options: tuple, counts: tuple,
                           available: tuple) -> bool:
        capacities = tuple(len(tokens) for tokens in available)
        if not self.allow_repeats and sum(capacities) < len(positions):
            return False
        parent_ids = tuple(self._token_index.get(self._initial[position], -1) for position in positions)

        @lru_cache(maxsize=None)
        def visit(offset: int, current: tuple, left: tuple, forbidden: tuple) -> bool:
            if offset == len(positions):
                return all(lo <= count <= hi for lo, count, hi in zip(self._lower, current, self._upper))
            for index, (lo, count, hi) in enumerate(zip(self._lower, current, self._upper)):
                bits = [[self._signatures[group][index] for group in choices] for choices in options[offset:]]
                if count + sum(max(row) for row in bits) < lo or count + sum(min(row) for row in bits) > hi:
                    return False
            for group in options[offset]:
                new_left, new_forbidden = left, forbidden
                if not self.allow_repeats:
                    if left[group] == 0:
                        continue
                    excluded = parent_ids[offset] if self._initial[positions[offset]] in available[group] else -1
                    old = forbidden[group]
                    combined = excluded if old == -3 else old if old == excluded else -2
                    # Exact matching for pools where each position excludes at
                    # most its parent token: exhaustion cannot leave a shared
                    # forbidden token as the only available replacement.
                    if left[group] == 1 and combined >= 0:
                        continue
                    new_left = tuple(value - int(i == group) for i, value in enumerate(left))
                    new_forbidden = tuple(combined if i == group else value for i, value in enumerate(forbidden))
                updated = tuple(value + bit for value, bit in zip(current, self._signatures[group]))
                if visit(offset + 1, updated, new_left, new_forbidden):
                    return True
            return False

        return visit(0, counts, () if self.allow_repeats else capacities,
                     () if self.allow_repeats else (-3,) * len(capacities))

    def _final_valid(self) -> bool:
        if any(token is None for token in self._picked):
            return False
        if not self.allow_repeats and len(set(self._picked)) != self.length:
            return False
        return all(lo <= sum(token in members for token in self._picked) <= hi
                   for members, lo, hi in zip(self._constraint_tokens, self._lower, self._upper))

    def _can_stop(self) -> bool:
        return (self.mutation_mode and not self.is_terminal and bool(self._edited)
                and len(self._edited) < self.max_edits and self._final_valid())

    def _assignment_allowed(self, position: int, token_index: int) -> bool:
        if (self.is_terminal or position in self._edited or not 0 <= position < self.length
                or not 0 <= token_index < len(self.tokens)):
            return False
        token = self.tokens[token_index]
        return token != self._initial[position] and self._feasible(self._fixed_assignments(position, token))

    def allowed_actions(self, *, prior_groups=None) -> list[Action]:
        if self.is_terminal:
            return []
        actions = [(self._label_onehots[position], self._value_onehots[token])
                   for position in range(self.length) for token in range(len(self.tokens))
                   if self._assignment_allowed(position, token)]
        if self._can_stop():
            actions.append((self._label_onehots[self._stop_label], self._value_onehots[self._stop_value]))
        return actions

    @staticmethod
    def _one_hot_index(vector: Sequence[float], size: int) -> int:
        try:
            values = np.asarray(vector, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sequence action must contain valid one-hot vectors") from exc
        if (values.ndim != 1 or values.size != size
                or not np.all((values == 0.0) | (values == 1.0))
                or np.count_nonzero(values == 1.0) != 1):
            raise ValueError("Sequence action must contain valid one-hot vectors")
        return int(np.argmax(values))

    def step(self, action: Action) -> None:
        if self.is_terminal:
            raise RuntimeError("Sequence step() called after termination")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Sequence action must be a pair of one-hot vectors")
        position = self._one_hot_index(action[0], len(self.species_set))
        token_index = self._one_hot_index(action[1], len(self.fraction_set))
        if position == self._stop_label and token_index == self._stop_value:
            if not self._can_stop():
                raise ValueError("STOP requires a valid parent candidate after at least one completed edit")
            self._done = True
        else:
            if not self._assignment_allowed(position, token_index):
                raise ValueError("Assignment is not allowed at this position and value")
            self._picked[position] = self.tokens[token_index]
            self._edited.add(position)
            self._done = len(self._edited) == self.max_edits
        self.counter += 1

    def current_state_features(self) -> np.ndarray:
        positions = np.zeros((self.length, len(self._state_tokens)), dtype=float)
        for position, token in enumerate(self._picked):
            if token is not None:
                positions[position, self._token_index[token]] = 1.0
        edited = np.asarray([float(position in self._edited) for position in range(self.length)])
        return np.concatenate((positions.reshape(-1), edited,
                               [float(self.max_edits - len(self._edited)), float(self.is_terminal)]))

    def state_feature_schema(self) -> dict:
        return {"version": 2, "encoding": "position_assignments",
                "positions": list(range(1, self.length + 1)), "tokens": list(self._state_tokens),
                "blocks": [{"name": "position_token_one_hot", "shape": [self.length, len(self._state_tokens)]},
                           {"name": "edited_positions", "shape": [self.length]},
                           {"name": "remaining_edits", "shape": [1]},
                           {"name": "terminal", "shape": [1]}]}

    def design_schema(self) -> dict:
        return {"version": 2, "kind": "sequence", "action_protocol": "position_value_assignment",
                "initial": list(self._initial), "replacement_tokens": list(self.tokens),
                "min_edits": 1 if self.mutation_mode else self.length, "max_edits": self.max_edits,
                "max_steps": self.max_steps, "allow_repeats": self.allow_repeats,
                "count_constraints": [{"tokens": list(tokens), "min": lo, "max": hi}
                                      for tokens, lo, hi in zip(self._constraint_tokens, self._lower, self._upper)],
                "state": self.state_feature_schema(), "actions": self.encoding.to_schema()}

    def action_features(self) -> np.ndarray:
        return self.encoding.labels.features

    def cation_fractions(self) -> list[str | None]:
        """Compatibility accessor; values retain explicit sequence positions."""
        return list(self._picked)

    def terminal_comp_key(self) -> tuple:
        if not self.is_terminal:
            raise RuntimeError("terminal_comp_key requires a completed sequence")
        return tuple(self._picked)
