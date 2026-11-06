"""Compose named design groups while preserving their scientific identities.

Groups run in declared order so downstream sublattice constraints can inspect
completed groups. Within a sequence or categorical group, an action assigns a
value to an explicit position or field; the execution order does not define
that label. Composition/integer groups retain their intrinsic domain rules.

Each group contributes a separate, namespaced action-label and value alphabet.
Values expose physical numerical magnitudes or categorical identity features via
``value_specs``. State features concatenate the complete state of each group.
Terminal candidates retain group names, mapping field identities and sequence
positions; predictors receive these structured values without chemical coercion.

Group termination is explicit: a parent mutation can STOP before its maximum
horizon and advance immediately to the next group. ``n_components`` remains the
maximum global horizon used to size the step feature vector.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from numbers import Real
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .action_encoding import ValueEncoding, ValueSpec, typed_identity
from .encoding import encode_choice
from .sequence import SequenceGroup
from .utils.composition import composition_key
from .env import CompositionEnv, EpisodeStep, _step_one_hot
from .env_integer import IntegerRatioEnv, normalize_integer_group_spec
from .featurization import featurize_formula


def normalize_group_spec(g: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a composition group's ``fraction_set`` into its canonical explicit-list
    form and derive ``total_units`` from it.

    ``fraction_set`` accepts two public forms (see
    :func:`~rl_matdesign.env.expand_fraction_set`):

    * a regular grid ``{min, max, step}``;
    * an explicit list ``[v1, v2, ...]``.

    ``total_units`` is never read from the config — it is always derived from
    the (expanded) ``fraction_set`` (see
    :func:`~rl_matdesign.env.derive_total_units`), so there is exactly one
    public knob for grid resolution, not two that could silently disagree.

    A mandatory-presence / "host absorbs the rest" requirement is expressed via
    ``element_bounds`` on the relevant species (e.g. ``{P: [0.92, 0.98]}``), not
    a dedicated ``host:`` knob — the host is just another entry in
    ``species_set`` like any other.

    Returns the spec unchanged for ``kind != "composition"`` — ``categorical``
    and ``integer`` groups have their own, unrelated value-grid conventions
    (``choices``/``values`` and ``ratio_set`` respectively) and are not touched
    here.
    """
    g = dict(g)
    g.setdefault("sites", 1)
    kind = g.get("kind", "composition")

    if kind == "integer":
        return normalize_integer_group_spec(g)
    if kind != "composition":
        return g

    fraction_set = g.get("fraction_set")
    if fraction_set is None:
        raise ValueError(
            f"group {g.get('name', '?')!r} (kind=composition) needs 'fraction_set' "
            "— either an explicit list or a {min, max, step} regular grid."
        )
    from .env import derive_total_units, expand_fraction_set

    expanded = expand_fraction_set(fraction_set)
    g["fraction_set"] = expanded
    g["total_units"] = derive_total_units(expanded)
    return g


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
    """Assign a typed value to any unfilled named field in one complete action.

    A choice with ``element`` retains its chemistry for structure builders; a
    choice with just ``name`` is a generic field and is never a composition.
    Pure numerical catalogs use physical magnitude features. Mixed or labelled
    catalogs use typed one-hot identities without ordinal distances. Equal numeric
    spellings such as 1 and 1.0 are one physical value and cannot both be listed;
    True, 1 and "1" remain distinct choices.
    """

    candidate_order_sensitive = False
    assignment_order_sensitive = False

    def __init__(self, choices: Sequence[Dict[str, Any]], *, phase_filter=None,
                 name: str = "fields") -> None:
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError("categorical group needs a non-empty choices list")
        self.name = name
        self.slots = []
        self.field_slots = []
        self._slot_specs = []
        self._slot_codes = []
        self._code_to_value = []
        specs_by_code = {}
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("categorical choices must be mappings")
            label = choice.get("name", choice.get("element"))
            if not isinstance(label, str) or not label.strip():
                raise ValueError("categorical choices need a non-empty string name or element")
            values = _field_values(choice.get("values"))
            numeric = (all(isinstance(value, Real) and not isinstance(value, bool) for value in values)
                       and len({float(value) for value in values}) == len(values))
            kind = "numeric" if numeric else "categorical"
            specs = [ValueSpec(value, kind, (name, "value")) for value in values]
            # Numeric local codes keep the existing chemistry filter protocol;
            # the outer environment namespaces all local action codes.
            codes = [repr(float(value)) if numeric else typed_identity(value) for value in values]
            self.slots.append((label, str(choice.get("element", label)), values))
            self.field_slots.append("element" not in choice)
            self._slot_specs.append(specs)
            self._slot_codes.append(list(zip(values, codes)))
            self._code_to_value.append(dict(zip(codes, values)))
            for code, value_spec in zip(codes, specs):
                specs_by_code.setdefault(code, value_spec)
        self.species_set = [label for label, _, _ in self.slots]
        if len(set(self.species_set)) != len(self.species_set):
            raise ValueError("categorical slot names must be unique")
        self.slot_element = {label: element for label, element, _ in self.slots}
        self.has_fields = any(self.field_slots)
        self.fraction_set = list(specs_by_code)
        self.value_specs = list(specs_by_code.values())
        self._slot_encodings = [ValueEncoding.from_specs(specs) for specs in self._slot_specs]
        self.n_components = self.max_steps = len(self.slots)
        self.phase_filter = phase_filter
        self.constraints_enabled = True
        self.initialize()

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""
        self._picked = {}
        self._picked_indices = {}

    @property
    def is_terminal(self) -> bool:
        return len(self._picked) == self.n_components

    @property
    def is_done(self) -> bool:
        return self.is_terminal

    def _is_terminal(self) -> bool:
        return self.is_terminal

    def allowed_actions(self, *, prior_groups=None):
        result = []
        for index, (name, _, _) in enumerate(self.slots):
            if name in self._picked:
                continue
            actions = [(tuple(encode_choice(name, self.species_set)),
                        tuple(encode_choice(code, self.fraction_set)))
                       for _, code in self._slot_codes[index]]
            if self.phase_filter is not None and self.constraints_enabled:
                # Filters receive one field's choices at a time, just as before.
                # Already assigned local values are supplied separately; at the
                # final chemical choice they also contribute to whole-formula checks.
                previous = deepcopy(prior_groups or [])
                if self._picked:
                    previous.append(deepcopy(self._picked))
                available = set(actions)
                filtered = self.phase_filter.filter_actions(
                    actions=list(actions), units_map={}, steps_left=self.n_components - self.counter - 1,
                    allowed_units=[], possible_sums_by_k=[], species_set=list(self.species_set),
                    fraction_set=list(self.fraction_set), code_to_value=deepcopy(self._code_to_value[index]),
                    prior_groups=previous, current_fields=deepcopy(self._picked),
                )
                prefix = f"Categorical constraint filter ({type(self.phase_filter).__name__})"
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
            result.extend(actions)
        return result

    def step(self, action, *, prior_groups=None) -> None:
        if self.is_terminal:
            raise RuntimeError("categorical step() called after termination")
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("categorical action must be a pair of one-hot vectors")
        label_index = SequenceGroup._one_hot_index(action[0], len(self.species_set))
        value_index = SequenceGroup._one_hot_index(action[1], len(self.fraction_set))
        name, code = self.species_set[label_index], self.fraction_set[value_index]
        if name in self._picked or code not in self._code_to_value[label_index]:
            raise ValueError("Assignment is not allowed for this field")
        normalized = (tuple(encode_choice(name, self.species_set)),
                      tuple(encode_choice(code, self.fraction_set)))
        if normalized not in self.allowed_actions(prior_groups=prior_groups):
            raise ValueError("Assignment is excluded by the field constraints")
        self._picked[name] = deepcopy(self._code_to_value[label_index][code])
        self._picked_indices[name] = next(i for i, (_, c) in enumerate(self._slot_codes[label_index]) if c == code)
        self.counter += 1

    def cation_fractions(self) -> Dict[str, Any]:
        return deepcopy(self._picked)

    def current_state_features(self) -> np.ndarray:
        features = []
        for (name, _, _), encoding in zip(self.slots, self._slot_encodings):
            features.extend(encoding.encode(self._picked_indices[name]) if name in self._picked
                            else np.zeros(encoding.feature_dim))
        features.extend(float(name in self._picked) for name in self.species_set)
        return np.asarray(features, dtype=float)

    def state_feature_schema(self) -> dict:
        return {"schema_version": 2, "kind": "labelled_field_assignments",
                "fields": [{"name": name, "values": encoding.to_schema()}
                           for (name, _, _), encoding in zip(self.slots, self._slot_encodings)],
                "filled_mask": list(self.species_set)}

    def design_schema(self) -> dict:
        return {"version": 2, "kind": "categorical", "action_protocol": "label_value_assignment",
                "slots": deepcopy(self.slots), "field_slots": list(self.field_slots),
                "state": self.state_feature_schema()}

    def terminal_comp_key(self) -> tuple:
        if not self.is_terminal:
            raise RuntimeError("terminal_comp_key requires complete fields")
        return composition_key(self._picked)


class MultiGroupEnv:
    """Fixed-order sequence of groups (composition: sum-to-1, or categorical: pick-list)."""

    def __init__(
        self,
        *,
        groups: Sequence[Dict[str, Any]],
        reward_fn: Callable[[Dict[str, Dict[str, float]]], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
    ) -> None:
        if not groups:
            raise ValueError("MultiGroupEnv requires a non-empty 'groups' list.")

        self.reward_fn = reward_fn or (lambda _groups: 0.0)
        self.state_featurizer = state_featurizer
        self.episode_style = "multi_group"
        self._constraints_enabled = True
        self.has_sequences = any(group.get("kind") == "sequence" for group in groups)
        self.has_fields = any(group.get("kind") == "categorical" and
                              any("element" not in choice for choice in group.get("choices", []))
                              for group in groups)
        self.has_explicit_features = self.has_sequences or self.has_fields
        # Aliased chemical slots can be builder control flags (O_a/O_b), not
        # atom counts. Their physical formula is known only to the builder.
        self.requires_builder_formula = self.has_explicit_features or any(
            group.get("kind") == "categorical" and any(
                choice.get("name", choice.get("element")) != choice.get("element")
                for choice in group.get("choices", [])
            ) for group in groups
        )
        self.candidate_order_sensitive = self.has_sequences
        self.assignment_order_sensitive = False

        # Build one inner CompositionEnv per group. Inner reward is a no-op — this
        # env computes the single terminal reward over all groups.
        self.group_names: List[str] = []
        self._sites: List[int] = []
        self._kinds: List[str] = []
        self._inners: List[Any] = []
        for i, g in enumerate(groups):
            name = str(g.get("name", f"group{i}"))
            kind = g.get("kind", "composition")
            if name in self.group_names:
                raise ValueError(f"Duplicate group name {name!r}")
            self.group_names.append(name)
            self._sites.append(int(g.get("sites", 1)))
            self._kinds.append(kind)
            if kind == "categorical":
                inner = CategoricalGroup(
                    g["choices"], phase_filter=g.get("constraint_filter"), name=name,
                )
            elif kind == "sequence":
                inner = SequenceGroup({**g, "name": name})
            elif kind == "integer":
                inner = IntegerRatioEnv(
                    species_set=g["species_set"],
                    ratio_set=g["ratio_set"],
                    n_components=g.get("n_components", 5),
                    state_featurizer=state_featurizer,
                    phase_filter=g.get("constraint_filter"),
                    token_requirements=g.get("token_requirements"),
                )
            elif kind == "composition":
                inner = CompositionEnv(
                    species_set=g["species_set"],
                    fraction_set=g.get("fraction_set") or None,
                    anion_formula="",  # the predictor/recipe assembles the real structure
                    n_components=g.get("n_components", 5),
                    reward_fn=lambda _f: 0.0,
                    state_featurizer=state_featurizer,
                    phase_filter=g.get("constraint_filter"),
                    total_units=g.get("total_units"),
                    element_bounds=g.get("element_bounds"),
                    episode_style=g.get("episode_style", "element_then_amount"),
                )
            else:
                raise ValueError(f"Unknown group kind {kind!r}")
            if not hasattr(inner, "value_specs"):
                inner.value_specs = [ValueSpec(float(value), "numeric", (name, "amount"))
                                     for value in inner.fraction_set]
            self._inners.append(inner)

        # Explicit group namespaces prevent equal position/field labels and
        # equal-looking values from becoming the same action in different groups.
        self.species_set, self.fraction_set, self.value_specs = [], [], []
        self._label_maps, self._value_maps = [], []
        for name, inner in zip(self.group_names, self._inners):
            self._label_maps.append(list(range(len(self.species_set), len(self.species_set) + len(inner.species_set))))
            self._value_maps.append(list(range(len(self.fraction_set), len(self.fraction_set) + len(inner.fraction_set))))
            self.species_set.extend(typed_identity((name, "label", label)) for label in inner.species_set)
            self.fraction_set.extend(typed_identity((name, "value", value)) for value in inner.fraction_set)
            self.value_specs.extend(ValueSpec(spec.value, spec.kind, (name, spec.namespace))
                                    for spec in inner.value_specs)
        self._union_label_onehots = [tuple(row) for row in np.eye(len(self.species_set))]
        self._union_value_onehots = [tuple(row) for row in np.eye(len(self.fraction_set))]
        self._chemical_labels = []
        for kind, inner in zip(self._kinds, self._inners):
            if kind in ("composition", "integer"):
                self._chemical_labels.extend(inner.species_set)
            elif kind == "categorical":
                self._chemical_labels.extend(None if is_field else element
                                             for (_, element, _), is_field in zip(inner.slots, inner.field_slots))
            else:
                self._chemical_labels.extend([None] * len(inner.species_set))
        self._label_feature_table = None

        # Total episode length across all groups (what training.py reads).
        self._group_lengths = [inner.n_components for inner in self._inners]
        self.n_components = int(sum(self._group_lengths))
        self.max_steps = self.n_components

        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._gi: int = 0  # active group index
        self._completed: List[Dict[str, float]] = []  # finished groups' comps

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    @property
    def constraints_enabled(self) -> bool:
        return self._constraints_enabled

    @constraints_enabled.setter
    def constraints_enabled(self, value: bool) -> None:
        # Propagate to every inner group env so the constrain_training toggle on
        # the wrapper disables/enables all sub-filters at once.
        self._constraints_enabled = bool(value)
        for inner in self._inners:
            inner.constraints_enabled = bool(value)

    def initialize(self) -> None:
        for inner in self._inners:
            inner.initialize()
        self.counter = 0
        self.path = []
        self._gi = 0
        self._completed = []

    def _active(self):
        return self._inners[self._gi]

    def _concat_features(self) -> np.ndarray:
        """Keep each labelled group's full state, including remaining resources."""
        return np.concatenate(
            [np.asarray(inner.current_state_features(), dtype=float)
             if hasattr(inner, "current_state_features")
             else np.asarray(self.state_featurizer(inner.state), dtype=float)
             for inner in self._inners]
        )

    def state_feature_schema(self) -> dict:
        return {
            "schema_version": 2, "kind": "labelled_groups",
            "groups": [
                {"name": name, "sites": sites,
                 "state": inner.state_feature_schema()
                 if hasattr(inner, "state_feature_schema") else {"kind": "descriptor-v1"}}
                for name, sites, inner in zip(self.group_names, self._sites, self._inners)
            ],
        }

    def current_state_features(self) -> np.ndarray:
        """Material features of the current state (uniform accessor; see CompositionEnv)."""
        return self._concat_features()

    def action_features(self) -> np.ndarray:
        """Label identity plus descriptors for explicitly chemical labels only.

        The chemical block uses the same scaling as the flat composition path.
        Unique element labels enter its fit once even across multiple groups;
        positions and named fields have zero entries in that block.
        """
        if self._label_feature_table is None:
            labels = list(dict.fromkeys(label for label in self._chemical_labels if label is not None))
            identity = np.eye(len(self.species_set), dtype=float)
            if labels:
                from sklearn.preprocessing import StandardScaler
                raw = np.asarray([self.state_featurizer(label) for label in labels], dtype=float)
                if raw.ndim != 2 or not np.isfinite(raw).all():
                    raise ValueError("Chemical action descriptors must be a finite 2-D table")
                standardized = StandardScaler().fit_transform(raw)
                row_for_label = {label: index for index, label in enumerate(labels)}
                chemical = np.zeros((len(self.species_set), raw.shape[1]), dtype=float)
                for row, label in enumerate(self._chemical_labels):
                    if label is not None:
                        chemical[row] = standardized[row_for_label[label]]
                identity = np.concatenate((identity, chemical), axis=1)
            identity.flags.writeable = False
            self._label_feature_table = identity
        return self._label_feature_table.view()

    def action_feature_schema(self) -> dict:
        table = self.action_features()
        # Canonical little-endian bytes make the digest platform independent.
        digest = hashlib.sha256(np.asarray(table, dtype="<f8").tobytes()).hexdigest()
        return {"version": 2, "encoding": "label_identity_and_standardized_chemical_descriptor",
                "chemical_labels": list(self._chemical_labels), "shape": list(table.shape),
                "fit_labels": list(dict.fromkeys(label for label in self._chemical_labels if label is not None)),
                "feature_sha256": digest}

    def design_schema(self) -> dict:
        return {"version": 2, "kind": "labelled_groups", "group_order": list(self.group_names),
                "groups": [{"name": name, "kind": kind, "sites": sites,
                            "design": inner.design_schema() if hasattr(inner, "design_schema")
                            else inner.state_feature_schema() if hasattr(inner, "state_feature_schema")
                            else {"kind": "descriptor-v1", "species": list(inner.species_set),
                                  "values": list(inner.fraction_set), "max_steps": inner.n_components}}
                           for name, kind, sites, inner in zip(self.group_names, self._kinds, self._sites, self._inners)],
                "labels": list(self.species_set), "value_codes": list(self.fraction_set),
                "values": ValueEncoding.from_specs(self.value_specs).to_schema(),
                "label_features": self.action_feature_schema()}

    def _to_union(self, group_action, inner):
        gi = next(index for index, candidate in enumerate(self._inners) if candidate is inner)
        label = SequenceGroup._one_hot_index(group_action[0], len(inner.species_set))
        value = SequenceGroup._one_hot_index(group_action[1], len(inner.fraction_set))
        return (self._union_label_onehots[self._label_maps[gi][label]],
                self._union_value_onehots[self._value_maps[gi][value]])

    def _from_union(self, action):
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Action must be a pair of one-hot vectors")
        label = SequenceGroup._one_hot_index(action[0], len(self.species_set))
        value = SequenceGroup._one_hot_index(action[1], len(self.fraction_set))
        try:
            li, vi = self._label_maps[self._gi].index(label), self._value_maps[self._gi].index(value)
        except ValueError as exc:
            raise ValueError("Action belongs to another group") from exc
        inner = self._active()
        return (tuple(np.eye(len(inner.species_set))[li]), tuple(np.eye(len(inner.fraction_set))[vi]))

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Union-encoded allowed actions for the active group's current step."""
        if self.is_terminal:
            return []
        inner = self._active()
        raw = inner.allowed_actions(prior_groups=self._completed)
        return [self._to_union(a, inner) for a in raw]

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        if self.is_terminal:
            raise RuntimeError("step() called after the episode terminated.")

        inner = self._active()
        # State features (all groups, BEFORE applying this action) and the
        # union-encoded allowed actions are captured pre-mutation, mirroring
        # CompositionEnv.step's ordering.
        s_material = self._concat_features()
        current_allowed = self.allowed_actions()

        group_action = self._from_union(action)
        normalized_action = self._to_union(group_action, inner)
        if normalized_action not in current_allowed:
            raise ValueError("Action is not allowed in the current design state")
        if isinstance(inner, (CategoricalGroup, IntegerRatioEnv, CompositionEnv)):
            inner.step(group_action, prior_groups=self._completed)
        else:
            inner.step(group_action)

        self.counter += 1
        s_step = _step_one_hot(self.counter, self.n_components)

        # If the active group just finished, record its composition for
        # downstream groups' prior_groups, and advance to the next group.
        if self._inner_terminal(inner):
            self._completed = self._completed + [self._terminal_group_payload(inner)]
            self._gi = min(self._gi + 1, len(self._inners) - 1)

        reward = 0.0
        if self.is_terminal:
            reward = float(self.reward_fn(self.terminal_cation_fractions()))

        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=action[0],
                action_comp_onehot=action[1],
                reward=reward,
                allowed_actions=current_allowed,
            )
        )

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        import random

        if self.is_terminal:
            raise RuntimeError("sample_random_action() called after group episode termination")
        actions = self.allowed_actions()
        if not actions:
            from .utils.rollout_errors import InfeasibleEpisode
            raise InfeasibleEpisode("No valid group action completes this prefix")
        return random.choice(actions)

    # ------------------------------------------------------------------
    # Terminal accessors (the predictor/dedup interface)
    # ------------------------------------------------------------------

    @staticmethod
    def _inner_terminal(inner) -> bool:
        state = getattr(inner, "is_terminal", None)
        if state is not None:
            return bool(state() if callable(state) else state)
        check = getattr(inner, "_is_terminal", None)
        return bool(check()) if callable(check) else inner.counter >= inner.n_components

    @property
    def is_terminal(self) -> bool:
        return len(self._completed) == len(self._inners)

    @property
    def is_done(self) -> bool:
        return self.is_terminal

    def _is_terminal(self) -> bool:
        return self.is_terminal

    @staticmethod
    def _terminal_group_payload(inner):
        if isinstance(inner, IntegerRatioEnv):
            return inner.reduced_cation_counts()
        if isinstance(inner, CompositionEnv):
            return inner.terminal_cation_fractions()
        # Sequence positions and field zeros are meaningful identities.
        return inner.cation_fractions()

    def terminal_cation_fractions(self) -> Dict[str, Dict[str, float]]:
        """Structured ``{group_name: {element: fraction}}`` for the predictor."""
        if not self._is_terminal():
            raise RuntimeError("terminal_cation_fractions called before terminal step")
        return {
            name: self._terminal_group_payload(inner)
            for name, inner in zip(self.group_names, self._inners)
        }

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key over all groups for dedup/visit-tracking."""
        if not self.is_terminal:
            raise RuntimeError("terminal_comp_key requires completed groups")
        # Each domain owns its material equivalence: integer ratios use GCD
        # reduction, sequences keep positions, fields keep scalar identities.
        # Freezing raw payloads here would lose integer-ratio equivalence.
        return tuple(sorted((name, inner.terminal_comp_key())
                            for name, inner in zip(self.group_names, self._inners)))

    def assembled_composition(self) -> Dict[str, float]:
        """Per-formula-unit composition ``{element: count}`` across all groups.

        Each group contributes ``fraction × sites`` per element, so the result is
        the real chemical composition (counts per formula unit). For a single group
        with ``sites=1`` this is just the fractions — consistent with a plain
        ``fraction`` env.
        """
        if self.has_explicit_features:
            raise TypeError("Sequence tokens and named fields are not an elemental composition")
        if self.requires_builder_formula:
            raise TypeError("Aliased categorical slots require a builder to determine the physical composition")
        out: Dict[str, float] = {}
        for inner, sites, kind in zip(self._inners, self._sites, self._kinds):
            payload = self._terminal_group_payload(inner) if self._inner_terminal(inner) else inner.cation_fractions()
            for el, val in payload.items():
                if kind == "categorical":
                    # categorical values are already per-f.u. counts; non-numeric
                    # labels (e.g. an O "form") are not atoms — the builder resolves them.
                    if isinstance(val, (int, float)):
                        out[el] = out.get(el, 0.0) + float(val)
                else:
                    out[el] = out.get(el, 0.0) + val * sites
        return out

    @property
    def terminal_formula(self) -> str:
        """Readable assembled chemical formula (per formula unit), for generated.csv."""
        if not self._is_terminal():
            return ""
        if self.requires_builder_formula:
            return json.dumps(self.terminal_cation_fractions(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        items = sorted(
            ((el, n) for el, n in self.assembled_composition().items() if n > 1e-9),
            key=lambda t: (-t[1], t[0]),
        )
        return "".join(f"{el}{n:.3g}" for el, n in items)
