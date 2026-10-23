"""Named experimental fields share RL plumbing without chemical semantics."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import random

import numpy as np
import pytest

from rl_matdesign.configuration import build_env
from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.env_multigroup import CategoricalGroup, MultiGroupEnv
from rl_matdesign.utils.composition import FieldCandidate, composition_key
from rl_matdesign.utils.design_schema import save_or_check_design_schema
from rl_matdesign.utils.timing import PredictorTimer


class RecordingPredictor:
    def __init__(self):
        self.candidates = []

    def predict(self, candidate):
        self.candidates.append(candidate)
        return float(len(candidate)), 0.0


class DeterministicFieldPredictor:
    """Small external-predictor fixture used through the real YAML registry."""

    def __init__(self, cfg=None, seed=None):
        self._cache = {}

    def predict(self, candidate):
        key = composition_key(candidate)
        if key not in self._cache:
            fields = candidate["conditions"]
            self._cache[key] = (fields["temperature"] / 100 + fields["water"], 0.0)
        return self._cache[key]


def field_config():
    return {"groups": [{
        "name": "conditions", "kind": "categorical", "choices": [
            {"name": "temperature", "values": {"min": 100, "max": 200, "step": 50}},
            {"name": "water", "values": {"min": 0.001, "max": 0.003, "step": 0.001}},
            {"name": "vessel", "values": ["01", "1", 1, 1.0, True]},
        ],
    }]}


def make_env(cfg=None):
    predictor = RecordingPredictor()
    env, label, _ = build_env(cfg or field_config(), predictor)
    assert label == "multi_group"
    env.initialize()
    return env, predictor


def complete(env, picks):
    env.initialize()
    for pick in picks:
        env.step(env.allowed_actions()[pick])
    return env.terminal_cation_fractions()


def test_named_fields_expose_selected_values_to_later_policy_decisions():
    low, predictor = make_env()
    high, _ = make_env()
    assert low.has_fields and low.has_explicit_features
    assert not low.has_sequences and not low.order_sensitive
    assert low.n_components == 3
    assert low.current_state_features().shape == (11,)
    assert not low.current_state_features().any()
    low.step(low.allowed_actions()[0])
    high.step(high.allowed_actions()[2])
    assert not np.array_equal(low.current_state_features(), high.current_state_features())
    assert np.unique(low.action_features(), axis=0).shape[0] == 3
    assert not predictor.candidates, "Partial candidates must not call the external oracle"


def test_field_values_are_normalized_for_actions_but_original_at_predictor():
    env, predictor = make_env()
    for slot in env._inners[0]._slot_codes:
        assert len({code for _, code in slot}) == len(slot)
        assert all(0 <= float(code) <= 1 for _, code in slot)
    candidate = complete(env, [1, 2, 0])
    assert isinstance(candidate, FieldCandidate)
    assert candidate == {"conditions": {"temperature": 150, "water": 0.003, "vessel": "01"}}
    assert predictor.candidates[-1] == candidate
    assert json.loads(env.terminal_formula) == candidate
    assert env.path[-1].reward == 1.0
    with pytest.raises(TypeError, match="named fields"):
        env.assembled_composition()


def test_irregular_numeric_field_codes_keep_relative_physical_distance():
    group = CategoricalGroup([{"name": "temperature", "values": [100, 110, 200]}])
    assert [float(code) for _, code in group._slot_codes[0]] == [0.0, 0.1, 1.0]


def test_small_grid_values_and_typed_tokens_never_merge():
    env, _ = make_env()
    keys = set()
    for water in range(3):
        for token in range(5):
            candidate = complete(env, [0, water, token])
            keys.add(env.terminal_comp_key())
            assert composition_key(candidate, decimals=6) == env.terminal_comp_key()
    assert len(keys) == 15
    tiny = [FieldCandidate({"x": value}) for value in [1e-8, 2e-8]]
    assert composition_key(tiny[0], decimals=6) != composition_key(tiny[1], decimals=6)


def test_candidate_mapping_order_is_irrelevant_but_checkpoint_input_order_is_not(tmp_path):
    original_cfg = field_config()
    original_cfg["groups"].append({"name": "other", "kind": "categorical", "choices": [
        {"name": "duration", "values": [5, 10]},
    ]})
    original, _ = make_env(original_cfg)
    candidate = complete(original, [1, 2, 0, 1])
    reordered_cfg = deepcopy(original_cfg)
    reordered_cfg["groups"].reverse()
    reordered_cfg["groups"][1]["choices"].reverse()
    reordered, _ = make_env(reordered_cfg)
    assert complete(reordered, [1, 0, 2, 1]) == candidate
    assert reordered.terminal_comp_key() == original.terminal_comp_key()
    assert original.design_schema() != reordered.design_schema()
    out = tmp_path / "trained"
    save_or_check_design_schema(original.design_schema(), str(out))
    assert (out / "design_schema.json").is_file()
    with pytest.raises(ValueError, match="Field design differs"):
        save_or_check_design_schema(reordered.design_schema(), str(tmp_path / "other"),
                                    load_paths=[str(out / "qnet.pt")])


@pytest.mark.parametrize("values", [
    [], [1, 1], [float("nan")], [float("inf")], [[1]], [None], "abc",
    {"min": 0, "max": 1, "step": 0},
    {"min": 0, "max": 1, "step": -1},
    {"min": 2, "max": 1, "step": 1},
    {"min": 0, "max": 1, "step": 0.3},
    {"min": 0, "max": float("inf"), "step": 1},
    {"min": 0, "max": 1},
])
def test_invalid_field_value_spaces_are_rejected(values):
    with pytest.raises(ValueError):
        CategoricalGroup([{"name": "x", "values": values}])


def test_fixed_field_filling_order_rejects_permutation_augmentation():
    from rl_matdesign.training import _augment_episode_in_buffer

    env, predictor = make_env()
    expected = complete(env, [1, 1, 1])
    buffer = deque()
    with pytest.raises(ValueError, match="fixed filling order"):
        _augment_episode_in_buffer(env, env.path, buffer, env.action_features(), env.fraction_set, K=1)
    assert not buffer
    assert env.terminal_cation_fractions() == expected
    assert len(predictor.candidates) == 1


def test_timer_and_baselines_preserve_typed_field_identity():
    from baselines._common import decode_choices, random_choices, score_composition

    env, predictor = make_env()
    env.reward_fn = lambda _: 0.0
    timed = PredictorTimer(predictor)
    cache = {}
    for token in range(5):
        decoded = decode_choices(env, [0, 0, token])
        assert isinstance(decoded["comp"], FieldCandidate)
        score_composition(timed, decoded["comp"], cache)
        # A deep copy preserves the marker; field mapping order does not matter.
        copied = deepcopy(decoded["comp"])
        copied["conditions"] = dict(reversed(list(copied["conditions"].items())))
        score_composition(timed, copied, cache)
    assert timed.n_calls == timed.n_unique == len(cache) == 5
    _, random_decoded = random_choices(env, random.Random(3))
    assert isinstance(random_decoded["comp"], FieldCandidate)


def test_legacy_element_categorical_features_codes_and_terminal_stay_unchanged():
    env = MultiGroupEnv(groups=[{"name": "site", "kind": "categorical", "choices": [
        {"element": "Cl", "values": [0.6, 1.4]},
        {"element": "O", "values": ["none", "oxide"]},
    ]}], state_featurizer=lambda _: np.array([7.0, 8.0]))
    env.initialize()
    assert not env.has_fields and not env.has_explicit_features
    assert env.action_features() is None
    assert np.array_equal(env.current_state_features(), [7.0, 8.0])
    assert [decode_one_hot(action[1], env.fraction_set) for action in env.allowed_actions()] == ["0.60", "1.40"]
    complete(env, [1, 1])
    assert env.terminal_cation_fractions() == {"site": {"Cl": 1.4, "O": "oxide"}}
    assert type(env.terminal_cation_fractions()) is dict
    assert env.terminal_formula == "Cl1.4"
    assert "field_encoding" not in env.design_schema()
