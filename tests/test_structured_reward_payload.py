"""Oxide components must reach property adapters without formula reparsing."""
from __future__ import annotations

import numpy as np
import pytest

from rl_matdesign.configuration import _TerminalReward, build_env
from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.featurization import featurize_formula
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator


def assign(env, label, amount):
    active = env._active() if hasattr(env, "_active") else env
    action = next(action for action in env.allowed_actions()
                  for local in [env._from_union(action) if hasattr(env, "_from_union") else action]
                  if decode_one_hot(local[0], active.species_set) == label
                  and float(decode_one_hot(local[1], active.action_values)) == amount)
    env.step(action)


@pytest.mark.parametrize("size", [1, 2])
def test_scalar_and_deferred_rewards_keep_component_labels_and_amounts(size):
    class Oracle:
        def __init__(self):
            self.seen = []
        def evaluation_batch_size(self):
            return size
        def predict(self, candidate):
            self.seen.append(candidate.copy())
            return candidate["SiO2"], 0.0
    oracle = Oracle()
    env, _, _ = build_env({"groups": [{"name": "glass", "kind": "composition",
        "species_set": ["SiO2", "B2O3", "Na2O"], "amount_set": [.25, .5, .75],
        "total": 1, "unique": True, "n_steps": 2}]}, oracle)
    paths = iter([[('SiO2', .25), ('B2O3', .75)], [('Na2O', .5), ('SiO2', .5)]])
    def rollout():
        env.initialize()
        for label, amount in next(paths):
            assign(env, label, amount)
    evaluator = EpisodeBatchEvaluator(env)
    first = evaluator.next(rollout, limit=2)
    second = evaluator.next(rollout, limit=1)
    expected = [{"SiO2": .25, "B2O3": .75}, {"Na2O": .5, "SiO2": .5}]
    assert oracle.seen == expected
    assert [first.composition, second.composition] == expected
    assert [first.path[-1].reward, second.path[-1].reward] == [.25, .5]
    assert first.prediction == (.25, 0.) and second.prediction == (.5, 0.)


def test_compound_descriptor_uses_formula_units_not_suffix_concatenation():
    env = ChemicalAmountsGroup(species_set=["SiO2", "B2O3"], amount_set=[.25, .75],
                         n_steps=2, total=1, unique=True)
    assign(env, 'SiO2', .25)
    assign(env, 'B2O3', .75)
    assert "(SiO2)0.25" in env.terminal_formula
    np.testing.assert_allclose(featurize_formula(env.state), featurize_formula("Si0.25B1.5O2.75"))


def test_terminal_reward_requires_explicit_structured_mapping():
    class Oracle:
        def predict(self, candidate):
            return 7, 0
    reward = _TerminalReward(Oracle())
    with pytest.raises(TypeError, match="mapping"):
        reward("Ni0.25Fe0.75")
    assert reward({"Ni": .25, "Fe": .75}) == 7
