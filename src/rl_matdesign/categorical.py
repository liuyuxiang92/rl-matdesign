"""Typed assignments to immutable labelled targets, with exact stage budgets.

The public action remains one (target, value) pair. Initial values, retention
and target eligibility are compiled by an application builder, not inferred from
residue spelling by this generic domain.
"""
from __future__ import annotations
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, List, Sequence
import numpy as np
from .action_encoding import ValueEncoding, ValueSpec, typed_identity
from .encoding import encode_choice
from .constraints.base import apply_filters
from .domain_runtime import one_hot_index
from .utils.composition import composition_key

def _field_values(spec: Any) -> List[Any]:
    """Expand physical grids without rounding or string coercion of IDs."""
    if isinstance(spec, dict):
        if set(spec) != {"min", "max", "step"}:
            raise ValueError("field values grid requires exactly min, max and step")
        if any(isinstance(value, bool) for value in spec.values()):
            raise ValueError("field grid endpoints must be finite numbers")
        try:
            lo, hi, step = (Decimal(str(spec[key])) for key in ("min", "max", "step"))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("field grid endpoints must be finite numbers") from exc
        if not all(value.is_finite() for value in (lo, hi, step)) or step <= 0 or hi < lo:
            raise ValueError("field grid requires finite endpoints, step > 0 and max >= min")
        intervals = (hi - lo) / step
        if intervals != intervals.to_integral_value():
            raise ValueError("field grid max must lie exactly on the min/step grid")
        exact = [lo + index * step for index in range(int(intervals) + 1)]
        integer_grid = all(value == value.to_integral_value() for value in (lo, hi, step))
        values = [int(value) if integer_grid else float(value) for value in exact]
    elif isinstance(spec, (list, tuple)):
        values = list(spec)
    else:
        raise ValueError("field values must be a non-empty list or min/max/step grid")
    if not values:
        raise ValueError("field values must be non-empty")
    for value in values:
        if not isinstance(value, (str, bool, Real)) or isinstance(value, Real) and not np.isfinite(value):
            raise ValueError("field values must be scalar strings, booleans or finite numbers")
    if len({typed_identity(value) for value in values}) != len(values):
        raise ValueError("field values must have unique typed identities")
    numbers = [value for value in values if isinstance(value, Real) and not isinstance(value, bool)]
    if len(set(numbers)) != len(numbers):
        raise ValueError("field values must not repeat an equal numeric value such as 1 and 1.0")
    return deepcopy(values)


class CategoricalGroup:
    """Assign each target at most once; repeated values at different targets are legal.

    A group mapping accepts choices or labels/values and an exact stage budget.
    Private compiled fields carry application-owned initialization and selection.
    """
    candidate_order_sensitive = False
    assignment_order_sensitive = False

    def __init__(self, spec, *, filters=()):
        if not isinstance(spec, dict):
            raise ValueError("Categorical group specification must be a mapping")
        spec = deepcopy(spec)
        allowed = {"name", "kind", "choices", "labels", "values", "n_steps", "filters",
                   "_initial_values", "_require_change", "_unassigned_from", "_display_labels", "_resolved_labels"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"Unsupported categorical group fields: {sorted(unknown)}")
        if spec.get("kind", "categorical") != "categorical":
            raise ValueError("CategoricalGroup requires kind: categorical")
        self.name = str(spec.get("name", "fields"))
        self.filters = list(filters)
        self._initial = deepcopy(spec.get("_initial_values", {}))
        self.require_change = spec.get("_require_change", False)
        self.unassigned_from = spec.get("_unassigned_from")
        self.display_labels = deepcopy(spec.get("_display_labels"))
        if not isinstance(self.require_change, bool):
            raise ValueError("compiled require_change must be a boolean")
        if "choices" in spec:
            if "labels" in spec or "values" in spec:
                raise ValueError("categorical uses either choices or labels/values, not both")
            choices = spec["choices"]
        else:
            labels = spec.get("labels")
            if not isinstance(labels, (list, tuple)) or not labels:
                raise ValueError("categorical labels must be a non-empty list; dynamic selectors need compilation")
            choices = [{"name": label, "values": spec.get("values")} for label in labels]
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError("categorical group needs a non-empty choices list")
        self.slots, self._slot_specs = [], []
        self._slot_codes, self._code_to_value, specs_by_code = [], [], {}
        self._state_values = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("categorical choices must be mappings")
            unknown = set(choice) - {"name", "values"}
            if unknown:
                raise ValueError(f"Unsupported categorical choice fields: {sorted(unknown)}")
            label = choice.get("name")
            if (isinstance(label, bool) or not isinstance(label, (str, Integral))
                    or isinstance(label, str) and not label.strip()):
                raise ValueError("categorical choices need a non-empty string or integer name")
            if isinstance(label, Integral):
                label = int(label)
            values = _field_values(choice.get("values"))
            # A retained initial token belongs only to its original target.
            if label in self._initial and not self.require_change:
                initial = self._initial[label]
                if not any(composition_key(v) == composition_key(initial) for v in values):
                    values.append(deepcopy(initial))
            numeric = all(isinstance(v, Real) and not isinstance(v, bool) for v in values)
            kind = "numeric" if numeric else "categorical"
            codes = [repr(float(v)) if numeric else typed_identity(v) for v in values]
            specs = [ValueSpec(v, kind, (self.name, "value")) for v in values]
            self.slots.append((label, deepcopy(values)))
            self._slot_codes.append(list(zip(values, codes)))
            self._code_to_value.append(dict(zip(codes, values)))
            for code, value_spec in zip(codes, specs):
                specs_by_code.setdefault(code, value_spec)
            state_values = list(values)
            if label in self._initial and not any(composition_key(v) == composition_key(self._initial[label]) for v in state_values):
                state_values.append(deepcopy(self._initial[label]))
            self._state_values.append(state_values)
            state_numeric = all(isinstance(v, Real) and not isinstance(v, bool) for v in state_values)
            self._slot_specs.append([ValueSpec(v, "numeric" if state_numeric else "categorical", (self.name, "value")) for v in state_values])
        self.species_set = [label for label, _ in self.slots]
        if len(set(self.species_set)) != len(self.species_set):
            raise ValueError("categorical slot names must be unique after selector compilation")
        if set(self._initial) - set(self.species_set):
            raise ValueError("initialized categorical values reference unknown targets")
        self.action_values = list(specs_by_code)
        self.value_specs = list(specs_by_code.values())
        self._slot_encodings = [ValueEncoding.from_specs(s) for s in self._slot_specs]
        budget = spec.get("n_steps", len(self.slots))
        if isinstance(budget, bool) or not isinstance(budget, Integral) or not 1 <= budget <= len(self.slots):
            raise ValueError("categorical n_steps must be an integer between 1 and the number of targets")
        self.max_steps = int(budget)
        # Static impossibility must be reported before rollout.
        usable = sum(any(not self.require_change or label not in self._initial
                         or composition_key(v) != composition_key(self._initial[label])
                         for v, _ in self._slot_codes[i])
                     for i, label in enumerate(self.species_set))
        if usable < self.max_steps:
            raise ValueError("categorical assignment budget exceeds targets with a valid replacement")
        self.initialize()

    def initialize(self):
        self.counter = 0
        self.state = ""
        self._picked = {}
        self._eligible = set(self.species_set)

    @property
    def assigned_labels(self):
        return frozenset(self._picked)

    def set_eligible_labels(self, labels):
        labels = set(labels)
        if labels - set(self.species_set):
            raise ValueError("categorical eligibility references unknown targets")
        self._eligible = labels

    @property
    def is_terminal(self):
        return self.counter == self.max_steps

    @property
    def is_done(self):
        return self.is_terminal

    def _is_terminal(self):
        return self.is_terminal

    def allowed_actions(self, *, prior_groups=None):
        if self.is_terminal:
            return []
        result = []
        for index, (label, _) in enumerate(self.slots):
            if label in self._picked or label not in self._eligible:
                continue
            actions = [(tuple(encode_choice(label, self.species_set)), tuple(encode_choice(code, self.action_values)))
                       for value, code in self._slot_codes[index]
                       if not self.require_change or label not in self._initial
                       or composition_key(value) != composition_key(self._initial[label])]
            if self.filters:
                previous = deepcopy(prior_groups or [])
                if self._picked:
                    previous.append(deepcopy(self._picked))
                actions = apply_filters(self.filters,
                    actions=list(actions), units_map={}, steps_left=self.max_steps - self.counter - 1,
                    allowed_units=[], possible_sums_by_k=[], species_set=list(self.species_set),
                    action_values=list(self.action_values), code_to_value=deepcopy(self._code_to_value[index]),
                    prior_groups=previous, current_fields=deepcopy(self._picked))
            result.extend(actions)
        return result

    def step(self, action, *, prior_groups=None):
        if self.is_terminal:
            raise RuntimeError("categorical step() called after termination")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("categorical action must be a pair of one-hot vectors")
        li = one_hot_index(action[0], len(self.species_set))
        vi = one_hot_index(action[1], len(self.action_values))
        label, code = self.species_set[li], self.action_values[vi]
        if label in self._picked or code not in self._code_to_value[li]:
            raise ValueError("Assignment is not allowed for this field")
        normalized = (tuple(encode_choice(label, self.species_set)), tuple(encode_choice(code, self.action_values)))
        if normalized not in self.allowed_actions(prior_groups=prior_groups):
            raise ValueError("Assignment is excluded by the field constraints")
        value = deepcopy(self._code_to_value[li][code])
        self._picked[label] = value
        self.counter += 1

    def cation_fractions(self):
        return deepcopy({**self._initial, **self._picked})

    def terminal_cation_fractions(self):
        if not self.is_terminal:
            raise RuntimeError("Terminal fields require a complete assignment stage")
        return self.cation_fractions()

    def current_state_features(self):
        current = self.cation_fractions()
        features = []
        for (label, _), values, encoding in zip(self.slots, self._state_values, self._slot_encodings):
            if label in current:
                index = next(i for i, v in enumerate(values) if composition_key(v) == composition_key(current[label]))
                features.extend(encoding.encode(index))
            else:
                features.extend(np.zeros(encoding.feature_dim))
        features.extend(float(label in current) for label in self.species_set)
        features.extend(float(label in self._initial) for label in self.species_set)
        features.extend(float(label in self._picked) for label in self.species_set)
        features.extend(float(label in self._picked and (label not in self._initial or composition_key(current[label]) != composition_key(self._initial[label]))) for label in self.species_set)
        features.extend(float(label in self._eligible) for label in self.species_set)
        features.append(float(self.max_steps - self.counter))
        return np.asarray(features, dtype=float)

    def state_feature_schema(self):
        schema = {"schema_version": 3, "kind": "labelled_field_assignments",
                  "fields": [{"name": label, "values": encoding.to_schema()} for (label, _), encoding in zip(self.slots, self._slot_encodings)],
                  "filled_mask": list(self.species_set)}
        schema.update(initial_values=[[label, value] for label, value in self._initial.items()],
                      initialized_mask=list(self.species_set), assigned_mask=list(self.species_set),
                      edited_mask=list(self.species_set), eligible_mask=list(self.species_set),
                      remaining_assignments=True)
        return schema

    def design_schema(self):
        return {"version": 3, "kind": "categorical", "action_protocol": "label_value_assignment",
                "slots": deepcopy(self.slots),
                "display_labels": deepcopy(self.display_labels), "n_steps": self.max_steps,
                "require_change": self.require_change, "unassigned_from": self.unassigned_from,
                "state": self.state_feature_schema()}

    def terminal_comp_key(self):
        if not self.is_terminal:
            raise RuntimeError("terminal_comp_key requires complete fields")
        return composition_key(self.cation_fractions())
