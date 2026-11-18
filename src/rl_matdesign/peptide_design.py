"""Compile peptide target selectors and serialize categorical stages for an oracle.

The design parent belongs here. It is independent of an external predictor's
training/reference parent. Chemistry and molecular assembly remain plugin-owned.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Integral

import numpy as np

from .categorical import _field_values


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _monomers(values, label):
    if not isinstance(values, (list, tuple)) or not values or any(
            not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{label} must be a non-empty list of monomer IDs")
    return list(values)


def _resolve_labels(labels, parent, length):
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("peptide labels must be a non-empty list of selectors")
    if all(isinstance(label, Integral) and not isinstance(label, bool) for label in labels):
        resolved = [int(label) for label in labels]
    elif all(isinstance(label, str) and label.strip() for label in labels):
        if parent is None:
            raise ValueError("residue-name labels need a structure.parent; novel labels are positions")
        occurrences = Counter()
        resolved = []
        for token in labels:
            positions = [index + 1 for index, original in enumerate(parent) if original == token]
            occurrence = occurrences[token]
            if occurrence >= len(positions):
                raise ValueError(f"selector {token!r} exceeds its occurrences in the original parent")
            resolved.append(positions[occurrence])
            occurrences[token] += 1
    else:
        raise ValueError("peptide labels must use either integer positions or residue names, not mixed selectors")
    if any(not 1 <= position <= length for position in resolved):
        raise ValueError("peptide position selector is outside the sequence length")
    if len(set(resolved)) != len(resolved):
        raise ValueError("peptide selectors must resolve to distinct positions within a group")
    return resolved


class PeptideDesign:
    """Immutable compiled design plus full positional state/serialization."""

    candidate_order_sensitive = True

    def __init__(self, structure, groups):
        unknown = set(structure) - {"builder", "parent", "length", "require_change"}
        if unknown:
            raise ValueError(f"Unsupported peptide structure fields: {sorted(unknown)}")
        self.parent = (_monomers(structure["parent"], "structure.parent")
                       if "parent" in structure else None)
        if self.parent is not None:
            if "length" in structure:
                raise ValueError("peptide structure uses parent or length, not both")
            self.length = len(self.parent)
        else:
            self.length = _positive_int(structure.get("length"), "structure.length")
        require_change = structure.get("require_change", [])
        if not isinstance(require_change, list) or any(not isinstance(n, str) for n in require_change):
            raise ValueError("structure.require_change must be a list of group names")
        if len(set(require_change)) != len(require_change):
            raise ValueError("structure.require_change must not repeat a group name")
        self.require_change = list(require_change)
        if require_change and self.parent is None:
            raise ValueError("require_change needs an initialized parent")
        names = [g.get("name") for g in groups]
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("peptide groups require unique non-empty names")
        if "n_terminal" not in names or "peptide" not in names:
            raise ValueError("peptide design needs n_terminal and peptide groups")
        self.position_groups = [name for name in names if name != "n_terminal"]
        if set(require_change) - set(self.position_groups):
            raise ValueError("require_change references an unknown positional group")
        self.groups = []
        self.constraints = []
        compiled_by_name, occupied = {}, set()
        self.tokens = list(dict.fromkeys(self.parent or []))
        total_steps = 0
        for original in groups:
            g = deepcopy(original)
            name = g["name"]
            if g.get("kind") != "categorical":
                raise ValueError("peptide design groups must be categorical")
            if name == "n_terminal":
                choices = g.get("choices")
                if (not isinstance(choices, list) or len(choices) != 1
                        or choices[0].get("name") != "treatment"
                        or any(v not in ("H", "Ac") for v in _field_values(choices[0].get("values")))):
                    raise ValueError("n_terminal needs one treatment choice with H/Ac values")
                if g.get("n_steps", 1) != 1:
                    raise ValueError("n_terminal requires one assignment")
                self.groups.append(g)
                compiled_by_name[name] = g
                continue
            labels = g.get("labels")
            if "choices" in g:
                if "labels" in g or "values" in g:
                    raise ValueError("peptide group uses choices or labels/values, not both")
                if not isinstance(g["choices"], list) or not g["choices"]:
                    raise ValueError("peptide choices must be a non-empty list")
                labels = [choice.get("name") for choice in g["choices"]]
            dynamic = isinstance(labels, dict)
            if dynamic:
                if set(labels) != {"unassigned_from"} or not isinstance(labels["unassigned_from"], str):
                    raise ValueError("dynamic labels require exactly unassigned_from: earlier_group")
                if self.parent is not None:
                    raise ValueError("unassigned_from is for novel construction, not initialized parent editing")
                source_name = labels["unassigned_from"]
                source = compiled_by_name.get(source_name)
                if source is None or source_name == "n_terminal" or "_unassigned_from" in source:
                    raise ValueError("unassigned_from must name an earlier ordinary positional group")
                resolved = list(source["_resolved_labels"])
                remaining = len(resolved) - source.get("n_steps", len(resolved))
                if remaining < 1 or g.get("n_steps", remaining) != remaining:
                    raise ValueError("unassigned_from n_steps must fill exactly the source's remaining positions")
                g["n_steps"] = remaining
                g["_unassigned_from"] = source_name
            else:
                resolved = _resolve_labels(labels, self.parent, self.length)
                if occupied.intersection(resolved):
                    raise ValueError("peptide groups select overlapping original positions")
                occupied.update(resolved)
            g["_display_labels"] = deepcopy(labels)
            g["_resolved_labels"] = list(resolved)
            if "choices" in g:
                for choice, position in zip(g["choices"], resolved):
                    choice["name"] = position
                menus = [choice.get("values") for choice in g["choices"]]
            else:
                g["labels"] = list(resolved)
                menus = [g.get("values")]
            for menu in menus:
                for token in _monomers(_field_values(menu), f"{name}.values"):
                    if token not in self.tokens:
                        self.tokens.append(token)
            if self.parent is not None:
                g["_initial_values"] = {position: self.parent[position - 1] for position in resolved}
                g["_require_change"] = name in require_change
            total_steps += _positive_int(g.get("n_steps", len(resolved)), f"{name}.n_steps")
            self.groups.append(g)
            compiled_by_name[name] = g
        if self.parent is None and (occupied != set(range(1, self.length + 1)) or total_steps != self.length):
            raise ValueError("novel peptide stages must cover and assign every position exactly once")
        if self.parent is None:
            consumers = Counter(g["_unassigned_from"] for g in self.groups if "_unassigned_from" in g)
            for g in self.groups:
                if g["name"] == "n_terminal" or "_unassigned_from" in g:
                    continue
                remaining = len(g["_resolved_labels"]) - g.get("n_steps", len(g["_resolved_labels"]))
                if remaining < 0 or consumers[g["name"]] != int(remaining > 0):
                    raise ValueError("each partial novel stage needs exactly one stage filling its remaining positions")
        self._token_index = {token: index for index, token in enumerate(self.tokens)}

    @staticmethod
    def _catalogs(group):
        """Effective per-target menus, including position-specific retention."""
        if "choices" in group:
            menus = {choice["name"]: _field_values(choice["values"]) for choice in group["choices"]}
        else:
            menus = {position: _field_values(group["values"]) for position in group["_resolved_labels"]}
        initial = group.get("_initial_values", {})
        for position, values in menus.items():
            if position not in initial:
                continue
            original = initial[position]
            if group.get("_require_change", False):
                menus[position] = [value for value in values if value != original]
            elif original not in values:
                values.append(original)
        return menus

    def _count_range(self, members):
        groups = {g["name"]: g for g in self.groups if g["name"] != "n_terminal"}
        if self.parent is not None:
            minimum = maximum = sum(token in members for token in self.parent)
            for group in groups.values():
                lows, highs = [], []
                for position, values in self._catalogs(group).items():
                    if not values:
                        continue  # This target cannot be one of the selected edits.
                    baseline = int(self.parent[position - 1] in members)
                    deltas = [int(value in members) - baseline for value in values]
                    lows.append(min(deltas))
                    highs.append(max(deltas))
                count = group.get("n_steps", len(group["_resolved_labels"]))
                if count > len(lows):
                    raise ValueError("peptide assignment budget exceeds targets with a valid replacement")
                minimum += sum(sorted(lows)[:count])
                maximum += sum(sorted(highs, reverse=True)[:count])
            return minimum, maximum

        # A novel source and its remaining-position stage partition the same
        # targets. Choose exactly the source quota; each target's final value
        # is independent once its stage has been selected.
        complements = {g["_unassigned_from"]: g for g in groups.values() if "_unassigned_from" in g}
        minimum = maximum = 0
        for name, group in groups.items():
            if "_unassigned_from" in group:
                continue
            menus = self._catalogs(group)
            lows = {p: min(int(v in members) for v in values) for p, values in menus.items()}
            highs = {p: max(int(v in members) for v in values) for p, values in menus.items()}
            complement = complements.get(name)
            if complement is None:
                minimum += sum(lows.values())
                maximum += sum(highs.values())
                continue
            other = self._catalogs(complement)
            other_lows = {p: min(int(v in members) for v in values) for p, values in other.items()}
            other_highs = {p: max(int(v in members) for v in values) for p, values in other.items()}
            count = group.get("n_steps", len(group["_resolved_labels"]))
            minimum += sum(other_lows.values()) + sum(sorted(lows[p] - other_lows[p] for p in menus)[:count])
            maximum += sum(other_highs.values()) + sum(sorted((highs[p] - other_highs[p] for p in menus), reverse=True)[:count])
        return minimum, maximum

    def validate_constraints(self, bounds):
        """Verify every design trajectory satisfies plugin-owned token counts.

        This optional startup hook never identifies chemical groups from token
        names. It consumes the plugin's explicit membership sets and rejects a
        menu that could generate a terminal molecule outside those bounds.
        """
        if not isinstance(bounds, list):
            raise ValueError("peptide sequence_constraints hook must return a list of count mappings")
        normalized = []
        for index, bound in enumerate(bounds):
            if not isinstance(bound, dict) or "tokens" not in bound or set(bound) - {"tokens", "min", "max"}:
                raise ValueError(f"peptide count constraint {index} needs tokens and optional min/max only")
            tokens = _monomers(bound["tokens"], f"peptide count constraint {index}.tokens")
            if len(set(tokens)) != len(tokens):
                raise ValueError(f"peptide count constraint {index} repeats token IDs")
            unknown = set(tokens) - self._token_index.keys()
            if unknown:
                raise ValueError(f"peptide count constraint {index} references unknown tokens: {sorted(unknown)}")
            lo, hi = bound.get("min", 0), bound.get("max", self.length)
            if (isinstance(lo, bool) or isinstance(hi, bool)
                    or not isinstance(lo, Integral) or not isinstance(hi, Integral)
                    or not 0 <= lo <= hi <= self.length):
                raise ValueError("peptide count bounds must be integers with 0 <= min <= max <= length")
            minimum, maximum = self._count_range(set(tokens))
            if minimum < lo or maximum > hi:
                raise ValueError(
                    f"peptide catalogs can produce {minimum}..{maximum} residues in count constraint {index}, "
                    f"but the predictor requires {lo}..{hi}; adjust stage targets or replacement values")
            normalized.append({"tokens": tokens, "min": int(lo), "max": int(hi)})
        self.constraints = normalized

    def _sequence(self, payload, *, complete):
        sequence = list(self.parent) if self.parent is not None else [None] * self.length
        if set(payload) - set(self.position_groups) - {"n_terminal"}:
            raise ValueError("peptide payload contains unknown design groups")
        assigned = set()
        for name in self.position_groups:
            values = payload.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"peptide stage {name!r} must contain position assignments")
            for position, token in values.items():
                if isinstance(position, bool) or not isinstance(position, Integral) or not 1 <= position <= self.length:
                    raise ValueError("peptide payload contains an invalid position")
                if position in assigned:
                    raise ValueError("peptide stages assign the same position twice")
                if token not in self._token_index:
                    raise ValueError(f"peptide payload contains unknown monomer {token!r}")
                assigned.add(position)
                sequence[position - 1] = token
        if complete and any(token is None for token in sequence):
            raise ValueError("terminal peptide has unassigned positions")
        return sequence

    def assemble_candidate(self, payload):
        if set(payload) != set(self.position_groups) | {"n_terminal"}:
            raise ValueError("terminal peptide payload must contain every compiled design group")
        sequence = self._sequence(payload, complete=True)
        terminal = payload.get("n_terminal")
        if not isinstance(terminal, dict) or set(terminal) != {"treatment"} or terminal["treatment"] not in ("H", "Ac"):
            raise ValueError("peptide n_terminal must contain treatment H or Ac")
        return {"n_terminal": deepcopy(terminal), "peptide": sequence}

    def current_state_features(self, payload):
        sequence = self._sequence(payload, complete=False)
        values = np.zeros((self.length, len(self.tokens)), dtype=float)
        for position, token in enumerate(sequence):
            if token is not None:
                values[position, self._token_index[token]] = 1.0
        initialized = np.full(self.length, self.parent is not None, dtype=float)
        filled = np.asarray([token is not None for token in sequence], dtype=float)
        edited = np.asarray([token is not None and (self.parent is None or token != self.parent[position])
                             for position, token in enumerate(sequence)], dtype=float)
        return np.concatenate((values.ravel(), initialized, filled, edited))

    def state_feature_schema(self):
        return {"schema_version": 3, "kind": "peptide_positional_state", "length": self.length,
                "tokens": list(self.tokens), "position_major": True,
                "initialized_mask": True, "filled_mask": True, "edited_mask": True,
                "initial_sequence": deepcopy(self.parent)}

    def design_schema(self):
        return {"version": 1, "builder": "peptide", "length": self.length,
                "parent": deepcopy(self.parent), "require_change": list(self.require_change),
                "position_groups": list(self.position_groups),
                "intrinsic_count_constraints": deepcopy(self.constraints), "state": self.state_feature_schema()}


def compile_groups(cfg):
    """Return independent group specs and an optional candidate adapter."""
    groups = deepcopy(cfg.get("groups", []))
    structure = cfg.get("structure")
    if not isinstance(structure, dict) or structure.get("builder") != "peptide":
        return groups, None
    design = PeptideDesign(structure, groups)
    return design.groups, design
