"""Ordered, categorical sequences with exact completion-aware count bounds.

This module deliberately knows nothing about peptides or molecular chemistry.
Tokens may be monomer IDs, process operations, or any other catalog entries.
Catalog annotations and membership in count constraints are supplied by the
application; neither token spelling nor SMILES is used to infer them.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


Action = Tuple[Tuple[float, ...], Tuple[float, ...]]


def _token_list(values: Any, *, label: str) -> List[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be a non-empty list of string IDs.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain non-empty string IDs (quote numeric IDs).")
    tokens = [value.strip() for value in values]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{label} contains duplicate token IDs.")
    return tokens


def _read_catalog(spec: Any) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not isinstance(spec, dict):
        return _token_list(spec, label="sequence tokens"), {}
    if set(spec) - {"csv", "id_column"}:
        raise ValueError("CSV tokens accept only 'csv' and 'id_column'.")
    path = spec.get("csv")
    column = spec.get("id_column")
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("CSV tokens require a 'csv' file path.")
    if not isinstance(column, str) or not column:
        raise ValueError("CSV tokens require an explicit 'id_column'.")
    metadata: Dict[str, Dict[str, str]] = {}
    # utf-8-sig also accepts a normal UTF-8 file, and strips spreadsheet BOMs.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or any(not name for name in fields) or len(set(fields)) != len(fields):
            raise ValueError("Token CSV must have unique, non-empty column names.")
        if column not in fields:
            raise ValueError(f"Token CSV is missing ID column {column!r}.")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed token CSV row {line}: wrong number of fields.")
            token = row[column].strip()
            if not token:
                raise ValueError(f"Token CSV row {line} has a missing token ID.")
            if token in metadata:
                raise ValueError(f"Token CSV row {line} duplicates token ID {token!r}.")
            metadata[token] = dict(row)
    if not metadata:
        raise ValueError("Token CSV contains no catalog entries.")
    return list(metadata), metadata


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


class SequenceGroup:
    """Fixed-length token choices, retaining order and optionally repetitions.

    ``tokens`` is either a list of string IDs or ``{csv, id_column}``.
    ``length`` is required. Repetition is allowed unless ``allow_repeats`` is
    false. Each ``count_constraints`` entry selects tokens using either
    ``tokens: [...]`` or ``where: {column: ..., equals: 'text'}`` and supplies
    inclusive ``min`` / ``max`` bounds (defaults: 0 / length).

    Count constraints are hard search-space invariants. Setting
    ``constraints_enabled=False`` does not disable them. Action masking tests
    *joint* feasibility of all bounds, including overlapping token sets.
    """

    order_sensitive = True

    def __init__(self, spec: dict) -> None:
        if spec.get("filters") or spec.get("constraint_filter") is not None:
            raise ValueError(
                "Sequence groups do not support composition filters; "
                "use count_constraints for token-count bounds."
            )
        self.species_set, self.token_metadata = _read_catalog(spec.get("tokens"))
        self.n_components = _integer(spec.get("length"), "sequence length")
        if self.n_components <= 0:
            raise ValueError("sequence length must be positive.")
        self.allow_repeats = spec.get("allow_repeats", True)
        if not isinstance(self.allow_repeats, bool):
            raise ValueError("allow_repeats must be a boolean.")
        self.fraction_set = ["1.00"]
        self.constraints_enabled = True
        self.phase_filter = None
        self._token_index = {token: i for i, token in enumerate(self.species_set)}

        constraints = spec.get("count_constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("count_constraints must be a list.")
        selected, lower, upper = [], [], []
        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise ValueError(f"count_constraints[{index}] must be a mapping.")
            if set(constraint) - {"tokens", "where", "min", "max"}:
                raise ValueError(f"count_constraints[{index}] has unsupported keys.")
            if ("tokens" in constraint) == ("where" in constraint):
                raise ValueError("Each count constraint needs exactly one of tokens or where.")
            if "tokens" in constraint:
                members = set(_token_list(constraint["tokens"], label="constraint tokens"))
                unknown = members - self._token_index.keys()
                if unknown:
                    raise ValueError(f"Count constraint references unknown tokens: {sorted(unknown)}.")
            else:
                where = constraint["where"]
                if not isinstance(where, dict) or set(where) != {"column", "equals"}:
                    raise ValueError("where must contain exactly 'column' and 'equals'.")
                column, expected = where["column"], where["equals"]
                if not self.token_metadata:
                    raise ValueError("where selectors require a CSV token catalog.")
                if not isinstance(column, str) or column not in next(iter(self.token_metadata.values())):
                    raise ValueError(f"Token CSV is missing selector column {column!r}.")
                if not isinstance(expected, str):
                    raise ValueError("where.equals must be a string matching CSV text; quote it in YAML.")
                members = {
                    token for token, row in self.token_metadata.items()
                    if row[column] == expected
                }
                if not members:
                    raise ValueError(f"Count selector {where!r} matches no catalog tokens.")
            lo = _integer(constraint.get("min", 0), "count min")
            hi = _integer(constraint.get("max", self.n_components), "count max")
            if not 0 <= lo <= hi <= self.n_components:
                raise ValueError("Count bounds must satisfy 0 <= min <= max <= length.")
            selected.append(members)
            lower.append(lo)
            upper.append(hi)
        self._lower, self._upper = tuple(lower), tuple(upper)
        self._constraint_tokens = tuple(
            tuple(token for token in self.species_set if token in members)
            for members in selected
        )

        # Tokens with identical constraint memberships are interchangeable for
        # feasibility. Grouping them keeps a 100+ token catalog cheap to mask.
        signatures: List[Tuple[int, ...]] = []
        self._token_signature: List[int] = []
        capacities: List[int] = []
        for token in self.species_set:
            signature = tuple(int(token in members) for members in selected)
            if signature not in signatures:
                signatures.append(signature)
                capacities.append(0)
            group = signatures.index(signature)
            self._token_signature.append(group)
            capacities[group] += 1
        self._signatures = tuple(signatures)
        self._initial_capacity = tuple(capacities) if not self.allow_repeats else ()
        self._feasible = lru_cache(maxsize=100_000)(self._can_complete)
        self._actions: List[Action] = [
            (tuple(float(i == j) for j in range(len(self.species_set))), (1.0,))
            for i in range(len(self.species_set))
        ]
        self.initialize()
        if not self._feasible(self.n_components, self._counts, self._capacity):
            raise ValueError("Sequence count constraints admit no complete sequence.")

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""  # sequence features are supplied by current_state_features
        self._picked: List[str] = []
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

    def allowed_actions(self, *, prior_groups=None) -> List[Action]:
        # Count constraints are local to this sequence; prior groups are accepted
        # to match the MultiGroupEnv inner-group interface.
        return [action for index, action in enumerate(self._actions) if self._token_allowed(index)]

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
        if not self._token_allowed(index):
            raise ValueError(f"Token {self.species_set[index]!r} is not allowed at this step.")
        group = self._token_signature[index]
        self._counts = self._advance_counts(self._counts, group)
        self._capacity = self._consume_capacity(self._capacity, group)
        self._used.add(index)
        self._picked.append(self.species_set[index])
        self.counter += 1

    def current_state_features(self) -> np.ndarray:
        features = np.zeros((self.n_components, len(self.species_set)), dtype=float)
        for position, token in enumerate(self._picked):
            features[position, self._token_index[token]] = 1.0
        return features.reshape(-1)

    def cation_fractions(self) -> List[str]:
        """Compatibility accessor: this group's candidate is an ordered list."""
        return list(self._picked)

    def terminal_comp_key(self) -> tuple:
        return tuple(self._picked)

    def design_schema(self) -> dict:
        """Resolved model-input semantics, independent of catalog file paths."""
        return {
            "kind": "sequence",
            "tokens": list(self.species_set),
            "length": self.n_components,
            "allow_repeats": self.allow_repeats,
            "count_constraints": [
                {"tokens": list(tokens), "min": lo, "max": hi}
                for tokens, lo, hi in zip(self._constraint_tokens, self._lower, self._upper)
            ],
        }
