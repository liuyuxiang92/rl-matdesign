"""Ordered, categorical sequences with exact completion-aware count bounds.

This module deliberately knows nothing about peptides or molecular chemistry.
Tokens may be monomer IDs, process operations, or any other catalog entries.
Membership in count constraints is supplied by the application; neither token
spelling nor molecular structures are used to infer it.
"""
from __future__ import annotations

import json
from itertools import combinations
from functools import lru_cache
from typing import Any, List, Sequence, Tuple

import numpy as np


Action = Tuple[Tuple[float, ...], Tuple[float, ...]]


def _token_list(values: Any, *, label: str, unique: bool = True) -> List[str]:
    if label == "sequence tokens" and isinstance(values, dict):
        raise ValueError(
            "CSV sequence tokens are no longer supported; replace tokens: "
            "{csv: ..., id_column: ...} with an explicit list of token names. "
            "Keep chemical definitions in the predictor's configuration."
        )
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list of string IDs.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty string IDs (quote numeric IDs).")
    tokens = [value.strip() for value in values]
    if unique and len(set(tokens)) != len(tokens):
        raise ValueError(f"{label} contains duplicate token IDs.")
    return tokens


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


class SequenceGroup:
    """Build an ordered sequence, or replace distinct positions of a parent.

    ``tokens`` is a non-empty list of replacement/building token names. Without
    ``parent`` and ``num_mutations``, ``length`` consecutive token actions build
    a sequence. With both fields, each mutation has two decisions: select an
    untouched position, then select a token different from the parent there.
    Parent tokens can be retained without being available as replacements.

    Count constraints select explicit token sets and give inclusive min/max
    bounds. All bounds and optional token uniqueness apply to the final
    sequence; intermediate edits can violate them if a valid completion exists.
    """

    order_sensitive = True

    def __init__(self, spec: dict) -> None:
        if spec.get("filters") or spec.get("constraint_filter") is not None:
            raise ValueError(
                "Sequence groups do not support composition filters; "
                "use count_constraints for token-count bounds."
            )
        self.tokens = _token_list(spec.get("tokens"), label="sequence tokens")
        self.mutation_mode = "parent" in spec or "num_mutations" in spec
        if self.mutation_mode:
            if "parent" not in spec or "num_mutations" not in spec:
                raise ValueError("parent and num_mutations must be provided together.")
            self.parent = _token_list(spec["parent"], label="sequence parent", unique=False)
            self.length = len(self.parent)
            if "length" in spec and _integer(spec["length"], "sequence length") != self.length:
                raise ValueError("sequence length must equal the parent length.")
            self.num_mutations = _integer(spec["num_mutations"], "num_mutations")
            if not 1 <= self.num_mutations <= self.length:
                raise ValueError("num_mutations must be between 1 and the parent length.")
            self.n_components = 2 * self.num_mutations
            # Action identity is separate from residue identity: an arbitrary
            # user token cannot collide with a position action.
            self.species_set = [
                json.dumps(["position", i + 1], separators=(",", ":"))
                for i in range(self.length)
            ] + [json.dumps(["token", token], separators=(",", ":")) for token in self.tokens]
        else:
            self.parent = []
            self.num_mutations = 0
            self.length = _integer(spec.get("length"), "sequence length")
            if self.length <= 0:
                raise ValueError("sequence length must be positive.")
            self.n_components = self.length
            self.species_set = list(self.tokens)
        self.allow_repeats = spec.get("allow_repeats", True)
        if not isinstance(self.allow_repeats, bool):
            raise ValueError("allow_repeats must be a boolean.")
        self.fraction_set = ["1.00"]
        self.constraints_enabled = True
        self.phase_filter = None
        self._state_tokens = list(dict.fromkeys(self.tokens + self.parent))
        self._token_index = {token: i for i, token in enumerate(self._state_tokens)}

        constraints = spec.get("count_constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("count_constraints must be a list.")
        selected, lower, upper = [], [], []
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise ValueError(f"count_constraints[{index}] must be a mapping.")
            if "where" in constraint:
                raise ValueError(
                    "CSV 'where' selectors are no longer supported; use an explicit "
                    "tokens list in each count constraint."
                )
            if set(constraint) - {"tokens", "min", "max"}:
                raise ValueError(f"count_constraints[{index}] has unsupported keys.")
            if "tokens" not in constraint:
                raise ValueError("Each count constraint needs an explicit tokens list.")
            members = set(_token_list(constraint["tokens"], label="constraint tokens"))
            unknown = members - self._token_index.keys()
            if unknown:
                raise ValueError(f"Count constraint references unknown tokens: {sorted(unknown)}.")
            lo = _integer(constraint.get("min", 0), "count min")
            hi = _integer(constraint.get("max", self.length), "count max")
            if not 0 <= lo <= hi <= self.length:
                raise ValueError("Count bounds must satisfy 0 <= min <= max <= length.")
            selected.append(members)
            lower.append(lo)
            upper.append(hi)
        self._lower, self._upper = tuple(lower), tuple(upper)
        self._constraint_tokens = tuple(
            tuple(token for token in self._state_tokens if token in members)
            for members in selected
        )

        # Tokens with identical constraint memberships are interchangeable for
        # feasibility. Grouping them keeps a 100+ token catalog cheap to mask.
        signatures: List[Tuple[int, ...]] = []
        self._token_signature: List[int] = []
        capacities: List[int] = []
        for token in self._state_tokens:
            signature = tuple(int(token in members) for members in selected)
            if signature not in signatures:
                signatures.append(signature)
                capacities.append(0)
            group = signatures.index(signature)
            self._token_signature.append(group)
            capacities[group] += int(token in self.tokens)
        self._signatures = tuple(signatures)
        self._initial_capacity = tuple(capacities) if not self.allow_repeats else ()
        self._feasible = lru_cache(maxsize=100_000)(self._can_complete)
        self._edit_feasible = lru_cache(maxsize=100_000)(self._can_complete_edits)
        self._group_tokens = tuple(
            frozenset(token for token in self.tokens if self._group_of(token) == group)
            for group in range(len(self._signatures))
        )
        self._actions: List[Action] = [
            (tuple(float(i == j) for j in range(len(self.species_set))), (1.0,))
            for i in range(len(self.species_set))
        ]
        self.initialize()
        feasible = (
            self._edit_feasible((), -1) if self.mutation_mode
            else self._feasible(self.n_components, self._counts, self._capacity)
        )
        if not feasible:
            raise ValueError("Sequence count constraints admit no complete sequence.")

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""  # sequence features are supplied by current_state_features
        self._picked: List[str] = list(self.parent)
        self._edited: set[int] = set()
        self._pending_position: int | None = None
        self._used: set[int] = set()
        self._counts = (0,) * len(self._lower)
        self._capacity = self._initial_capacity

    def _advance_counts(self, counts: tuple, group: int) -> tuple:
        return tuple(value + bit for value, bit in zip(counts, self._signatures[group]))

    def _consume_capacity(self, capacity: tuple, group: int) -> tuple:
        if self.allow_repeats:
            return ()
        return tuple(value - int(i == group) for i, value in enumerate(capacity))

    def _can_complete(self, remaining: int, counts: tuple, capacity: tuple) -> bool:
        if remaining == 0:
            return all(lo <= n <= hi for lo, n, hi in zip(self._lower, counts, self._upper))
        if not self.allow_repeats and sum(capacity) < remaining:
            return False
        active = [
            group for group in range(len(self._signatures))
            if self.allow_repeats or capacity[group] > 0
        ]
        for index, (lo, n, hi) in enumerate(zip(self._lower, counts, self._upper)):
            if self.allow_repeats:
                bits = [self._signatures[group][index] for group in active]
                most = remaining if any(bits) else 0
                least = remaining if all(bits) else 0
            else:
                inside = sum(capacity[group] for group in active if self._signatures[group][index])
                outside = sum(capacity[group] for group in active if not self._signatures[group][index])
                most, least = min(remaining, inside), max(0, remaining - outside)
            if n + most < lo or n + least > hi:
                return False
        if not self._lower:
            return True
        return any(
            self._feasible(
                remaining - 1,
                self._advance_counts(counts, group),
                self._consume_capacity(capacity, group),
            )
            for group in active
        )

    def _token_allowed(self, index: int) -> bool:
        if self.counter >= self.n_components or (not self.allow_repeats and index in self._used):
            return False
        group = self._token_signature[index]
        return self._feasible(
            self.n_components - self.counter - 1,
            self._advance_counts(self._counts, group),
            self._consume_capacity(self._capacity, group),
        )

    def _group_of(self, token: str) -> int:
        return self._token_signature[self._token_index[token]]

    def _fixed_edits(self, extra_position: int = -1, extra_token: str = "") -> tuple:
        edits = {position: self._picked[position] for position in self._edited}
        if extra_position >= 0:
            edits[extra_position] = extra_token
        return tuple(
            (position, self._group_of(token) if self.allow_repeats else token)
            for position, token in sorted(edits.items())
        )

    def _can_complete_edits(self, fixed: tuple, pending: int) -> bool:
        """Exactly complete the edit budget without enumerating token products.

        Enumerate subsets of the remaining positions, then assign constraint
        signatures. For unique-token sequences, an exact matching condition
        handles token identity within each signature (see _assign_edit_groups).
        """
        finalized = dict(fixed)
        remaining = self.num_mutations - len(finalized)
        free = [position for position in range(self.length) if position not in finalized]
        if remaining < 0 or remaining > len(free) or (pending >= 0 and pending not in free):
            return False
        if pending >= 0:
            if remaining == 0:
                return False
            choices = (
                (pending,) + other
                for other in combinations([p for p in free if p != pending], remaining - 1)
            )
        else:
            choices = combinations(free, remaining)
        for changing in choices:
            changing_set = set(changing)
            retained = [self.parent[p] for p in free if p not in changing_set]
            if self.allow_repeats:
                groups = [self._group_of(token) for token in retained] + list(finalized.values())
                available = self._group_tokens
            else:
                fixed_tokens = retained + list(finalized.values())
                if len(set(fixed_tokens)) != len(fixed_tokens):
                    continue
                groups = [self._group_of(token) for token in fixed_tokens]
                available = tuple(tokens.difference(fixed_tokens) for tokens in self._group_tokens)
            counts = tuple(sum(self._signatures[g][i] for g in groups) for i in range(len(self._lower)))
            if any(n > hi or n + remaining < lo for n, lo, hi in zip(counts, self._lower, self._upper)):
                continue
            options = [
                tuple(group for group, tokens in enumerate(available) if tokens - {self.parent[p]})
                for p in changing
            ]
            if any(not groups for groups in options):
                continue
            if self._assign_edit_groups(tuple(changing), tuple(options), counts, available):
                return True
        return False

    def _assign_edit_groups(self, positions: tuple, options: tuple, counts: tuple, available: tuple) -> bool:
        capacities = tuple(len(tokens) for tokens in available)
        if not self.allow_repeats and sum(capacities) < len(positions):
            return False
        # A replacement can use any token in its signature except its original
        # token. Hall's matching condition therefore reduces to two checks:
        # assigned <= available; at equality, assignments cannot all forbid the
        # same available token. -1 means no exclusion, -2 mixed, -3 unassigned.
        parent_ids = tuple(self._token_index[self.parent[p]] for p in positions)

        @lru_cache(maxsize=None)
        def visit(offset: int, current: tuple, left: tuple, forbidden: tuple) -> bool:
            if offset == len(positions):
                return all(lo <= n <= hi for lo, n, hi in zip(self._lower, current, self._upper))
            for i, (lo, n, hi) in enumerate(zip(self._lower, current, self._upper)):
                future_bits = [[self._signatures[g][i] for g in opts] for opts in options[offset:]]
                if n + sum(max(bits) for bits in future_bits) < lo:
                    return False
                if n + sum(min(bits) for bits in future_bits) > hi:
                    return False
            for group in options[offset]:
                new_left, new_forbidden = left, forbidden
                if not self.allow_repeats:
                    if left[group] == 0:
                        continue
                    excluded = parent_ids[offset] if self.parent[positions[offset]] in available[group] else -1
                    old = forbidden[group]
                    combined = excluded if old == -3 else old if old == excluded else -2
                    # Filling this signature would leave the common forbidden
                    # token unused, so no injective assignment can exist.
                    if left[group] == 1 and combined >= 0:
                        continue
                    new_left = tuple(value - int(i == group) for i, value in enumerate(left))
                    new_forbidden = tuple(combined if i == group else value for i, value in enumerate(forbidden))
                if visit(offset + 1, self._advance_counts(current, group), new_left, new_forbidden):
                    return True
            return False

        return visit(0, counts, () if self.allow_repeats else capacities,
                     () if self.allow_repeats else (-3,) * len(capacities))

    def _mutation_action_allowed(self, index: int) -> bool:
        if self.counter >= self.n_components:
            return False
        if self._pending_position is None:
            return (
                index < self.length and index not in self._edited
                and self._edit_feasible(self._fixed_edits(), index)
            )
        if index < self.length:
            return False
        token = self.tokens[index - self.length]
        position = self._pending_position
        return token != self.parent[position] and self._edit_feasible(
            self._fixed_edits(position, token), -1,
        )

    def allowed_actions(self, *, prior_groups=None) -> List[Action]:
        # prior_groups is accepted for the MultiGroupEnv inner-group interface.
        allowed = self._mutation_action_allowed if self.mutation_mode else self._token_allowed
        return [action for index, action in enumerate(self._actions) if allowed(index)]

    @staticmethod
    def _one_hot_index(vector: Sequence[float], size: int) -> int:
        try:
            values = np.asarray(vector, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sequence action must contain valid one-hot vectors.") from exc
        if (
            values.ndim != 1 or values.size != size
            or not np.all((values == 0.0) | (values == 1.0))
            or np.count_nonzero(values == 1.0) != 1
        ):
            raise ValueError("Sequence action must contain valid one-hot vectors.")
        return int(np.argmax(values))

    def step(self, action: Action) -> None:
        if self.counter >= self.n_components:
            raise RuntimeError("Sequence step() called after termination.")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Sequence action must be a pair of one-hot vectors.")
        index = self._one_hot_index(action[0], len(self.species_set))
        self._one_hot_index(action[1], 1)
        allowed = self._mutation_action_allowed if self.mutation_mode else self._token_allowed
        if not allowed(index):
            raise ValueError(f"Action {self.species_set[index]!r} is not allowed at this step.")
        if self.mutation_mode:
            if self._pending_position is None:
                self._pending_position = index
            else:
                self._picked[self._pending_position] = self.tokens[index - self.length]
                self._edited.add(self._pending_position)
                self._pending_position = None
            self.counter += 1
            return
        group = self._token_signature[index]
        self._counts = self._advance_counts(self._counts, group)
        self._capacity = self._consume_capacity(self._capacity, group)
        self._used.add(index)
        self._picked.append(self.species_set[index])
        self.counter += 1

    def current_state_features(self) -> np.ndarray:
        features = np.zeros((self.length, len(self._state_tokens)), dtype=float)
        for position, token in enumerate(self._picked):
            features[position, self._token_index[token]] = 1.0
        if not self.mutation_mode:
            return features.reshape(-1)
        edited = np.array([float(p in self._edited) for p in range(self.length)])
        pending = np.array([float(p == self._pending_position) for p in range(self.length)])
        return np.concatenate((features.reshape(-1), edited, pending))

    def cation_fractions(self) -> List[str]:
        """Compatibility accessor: this group's candidate is an ordered list."""
        return list(self._picked)

    def terminal_comp_key(self) -> tuple:
        return tuple(self._picked)

    def design_schema(self) -> dict:
        """Resolved model-input semantics, independent of catalog file paths."""
        schema = {
            "kind": "sequence",
            "tokens": list(self.tokens),
            "length": self.length,
            "allow_repeats": self.allow_repeats,
            "count_constraints": [
                {"tokens": list(tokens), "min": lo, "max": hi}
                for tokens, lo, hi in zip(self._constraint_tokens, self._lower, self._upper)
            ],
        }
        if self.mutation_mode:
            schema.update({
                "parent": list(self.parent),
                "num_mutations": self.num_mutations,
                "action_order": ["position", "replacement"],
                "state_tokens": list(self._state_tokens),
            })
        return schema
