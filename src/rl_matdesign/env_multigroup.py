"""Compose named design groups while preserving their scientific identities.

Groups run in declared order so downstream constraints can inspect completed
groups. Categorical actions assign values to immutable targets. Chemical amount
actions accumulate quantities under the group's exact domain constraints.

Each group contributes a separate, namespaced action-label and value alphabet.
Values expose physical numerical magnitudes or categorical identity features via
``value_specs``. State features concatenate the complete state of each group.
Terminal candidates retain group names, mapping field identities and sequence
positions; predictors receive these structured values without chemical coercion.

Both domains use exact stage budgets. ``max_steps`` is the total horizon used
to size the step feature vector.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .action_encoding import ValueEncoding, ValueSpec, typed_identity
from .categorical import CategoricalGroup
from .amounts import ChemicalAmountsGroup
from .utils.composition import composition_key
from .domain_runtime import EpisodeStep, one_hot_index, step_one_hot
from .featurization import featurize_formula


class MultiGroupEnv:
    """Ordered chemical amount and categorical assignment stages."""

    def __init__(
        self,
        *,
        groups: Sequence[Dict[str, Any]],
        reward_fn: Callable[[Dict[str, Dict[str, float]]], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        candidate_adapter=None,
    ) -> None:
        if not groups:
            raise ValueError("MultiGroupEnv requires a non-empty 'groups' list.")

        self.candidate_adapter = candidate_adapter
        self.reward_fn = reward_fn or (lambda _groups: 0.0)
        self.state_featurizer = state_featurizer
        self.requires_builder_formula = any(group.get("kind") == "categorical" for group in groups)
        self.candidate_order_sensitive = bool(getattr(candidate_adapter, "candidate_order_sensitive", False))
        self.assignment_order_sensitive = False

        self.group_names: List[str] = []
        self._kinds: List[str] = []
        self._inners: List[Any] = []
        for group in groups:
            name = group.get("name")
            kind = group.get("kind", "composition")
            if not isinstance(name, str) or not name.strip() or name in self.group_names:
                raise ValueError("Groups require unique nonempty names")
            self.group_names.append(name)
            self._kinds.append(kind)
            if kind == "categorical":
                inner = CategoricalGroup(group)
            elif kind == "composition":
                inner = ChemicalAmountsGroup(group, state_featurizer=state_featurizer)
            else:
                raise ValueError(f"Unknown group kind {kind!r}; use composition or categorical")
            self._inners.append(inner)

        # Explicit group namespaces prevent equal position/field labels and
        # equal-looking values from becoming the same action in different groups.
        self.species_set, self.action_values, self.value_specs = [], [], []
        self._label_maps, self._value_maps = [], []
        for name, inner in zip(self.group_names, self._inners):
            self._label_maps.append(list(range(len(self.species_set), len(self.species_set) + len(inner.species_set))))
            self._value_maps.append(list(range(len(self.action_values), len(self.action_values) + len(inner.action_values))))
            self.species_set.extend(typed_identity((name, "label", label)) for label in inner.species_set)
            self.action_values.extend(typed_identity((name, "value", value)) for value in inner.action_values)
            self.value_specs.extend(ValueSpec(spec.value, spec.kind, (name, spec.namespace))
                                    for spec in inner.value_specs)
        self._union_label_onehots = [tuple(row) for row in np.eye(len(self.species_set))]
        self._union_value_onehots = [tuple(row) for row in np.eye(len(self.action_values))]
        self._chemical_labels = []
        for inner in self._inners:
            self._chemical_labels.extend(inner.species_set if isinstance(inner, ChemicalAmountsGroup)
                                         else [None] * len(inner.species_set))
        self._label_feature_table = None
        self.max_steps = sum(inner.max_steps for inner in self._inners)

        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._gi: int = 0  # active group index
        self._completed: List[Dict[str, float]] = []  # finished groups' comps
        self._refresh_dependencies()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        for inner in self._inners:
            inner.initialize()
        self.counter = 0
        self.path = []
        self._gi = 0
        self._completed = []
        self._refresh_dependencies()

    def _refresh_dependencies(self):
        for index, inner in enumerate(self._inners):
            source_name = getattr(inner, "unassigned_from", None)
            if source_name is None:
                continue
            if source_name not in self.group_names[:index]:
                raise ValueError("unassigned_from must refer to an earlier group")
            source = self._inners[self.group_names.index(source_name)]
            if not isinstance(source, CategoricalGroup):
                raise ValueError("unassigned_from source must be categorical")
            inner.set_eligible_labels(set(inner.species_set) - source.assigned_labels)

    def _raw_group_payload(self):
        return {name: inner.cation_fractions() for name, inner in zip(self.group_names, self._inners)}

    def _active(self):
        return self._inners[self._gi]

    def _concat_features(self) -> np.ndarray:
        """Keep full state, including initialized values and remaining resources."""
        self._refresh_dependencies()
        blocks = [np.asarray(inner.current_state_features(), dtype=float) for inner in self._inners]
        hook = getattr(self.candidate_adapter, "current_state_features", None)
        if callable(hook):
            blocks.append(np.asarray(hook(self._raw_group_payload()), dtype=float))
        return np.concatenate(blocks)

    def state_feature_schema(self) -> dict:
        schema = {"schema_version": 3, "kind": "labelled_groups",
                  "groups": [{"name": name, "state": inner.state_feature_schema()}
                             for name, inner in zip(self.group_names, self._inners)]}
        hook = getattr(self.candidate_adapter, "state_feature_schema", None)
        if callable(hook):
            schema["candidate"] = hook()
        return schema

    def current_state_features(self) -> np.ndarray:
        """Complete material and resource features of the current state."""
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
        schema = {"version": 3, "kind": "labelled_groups", "group_order": list(self.group_names),
                  "groups": [{"name": name, "kind": kind, "design": inner.design_schema()}
                             for name, kind, inner in zip(self.group_names, self._kinds, self._inners)],
                  "labels": list(self.species_set), "value_codes": list(self.action_values),
                  "values": ValueEncoding.from_specs(self.value_specs).to_schema(),
                  "label_features": self.action_feature_schema()}
        hook = getattr(self.candidate_adapter, "design_schema", None)
        if callable(hook):
            schema["candidate"] = hook()
        return schema

    def _to_union(self, group_action, inner):
        gi = next(index for index, candidate in enumerate(self._inners) if candidate is inner)
        label = one_hot_index(group_action[0], len(inner.species_set))
        value = one_hot_index(group_action[1], len(inner.action_values))
        return (self._union_label_onehots[self._label_maps[gi][label]],
                self._union_value_onehots[self._value_maps[gi][value]])

    def _from_union(self, action):
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise ValueError("Action must be a pair of one-hot vectors")
        label = one_hot_index(action[0], len(self.species_set))
        value = one_hot_index(action[1], len(self.action_values))
        try:
            li, vi = self._label_maps[self._gi].index(label), self._value_maps[self._gi].index(value)
        except ValueError as exc:
            raise ValueError("Action belongs to another group") from exc
        inner = self._active()
        return (tuple(np.eye(len(inner.species_set))[li]), tuple(np.eye(len(inner.action_values))[vi]))

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Union-encoded allowed actions for the active group's current step."""
        if self.is_terminal:
            return []
        self._refresh_dependencies()
        inner = self._active()
        raw = inner.allowed_actions(prior_groups=self._completed)
        return [self._to_union(a, inner) for a in raw]

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        if self.is_terminal:
            raise RuntimeError("step() called after the episode terminated.")

        inner = self._active()
        # Record the complete pre-action state and the legal action set.
        s_material = self._concat_features()
        current_allowed = self.allowed_actions()

        group_action = self._from_union(action)
        normalized_action = self._to_union(group_action, inner)
        if normalized_action not in current_allowed:
            raise ValueError("Action is not allowed in the current design state")
        inner.step(group_action, prior_groups=self._completed)

        self.counter += 1
        s_step = step_one_hot(self.counter, self.max_steps)

        # If the active group just finished, record its composition for
        # downstream groups' prior_groups, and advance to the next group.
        if inner.is_terminal:
            self._completed = self._completed + [self._terminal_group_payload(inner)]
            self._gi = min(self._gi + 1, len(self._inners) - 1)
        self._refresh_dependencies()

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
        return inner.terminal_cation_fractions()

    def terminal_cation_fractions(self) -> Dict[str, Dict[str, float]]:
        """Structured ``{group_name: {element: fraction}}`` for the predictor."""
        if not self._is_terminal():
            raise RuntimeError("terminal_cation_fractions called before terminal step")
        payload = {name: self._terminal_group_payload(inner)
                   for name, inner in zip(self.group_names, self._inners)}
        hook = getattr(self.candidate_adapter, "assemble_candidate", None)
        return hook(payload) if callable(hook) else payload

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key over all groups for dedup/visit-tracking."""
        if not self.is_terminal:
            raise RuntimeError("terminal_comp_key requires completed groups")
        if self.candidate_adapter is not None:
            return composition_key(self.terminal_cation_fractions())
        # Each group retains its raw quantities or labelled assignment identity.
        return tuple(sorted((name, inner.terminal_comp_key())
                            for name, inner in zip(self.group_names, self._inners)))

    def assembled_composition(self) -> Dict[str, float]:
        """Expand compound recipe labels into their elemental formula-unit amounts."""
        if self.requires_builder_formula:
            raise TypeError("This design needs its builder to determine physical composition")
        from pymatgen.core import Composition
        out: Dict[str, float] = {}
        for inner in self._inners:
            for label, amount in inner.cation_fractions().items():
                for element, count in Composition(label).as_dict().items():
                    out[element] = out.get(element, 0.0) + float(amount) * float(count)
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
