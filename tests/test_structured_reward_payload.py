"""Oxide components must reach property adapters without formula reparsing."""
from __future__ import annotations

import numpy as np
import pytest

from rl_matdesign.configuration import _TerminalReward
from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.env import CompositionEnv
from rl_matdesign.featurization import featurize_formula
from rl_matdesign.utils.episode_batch import EpisodeBatchEvaluator


def assign(env, label, amount):
    action = next(action for action in env.allowed_actions()
                  if decode_one_hot(action[0], env.species_set) == label
                  and float(decode_one_hot(action[1], env.fraction_set)) == amount)
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
    env = CompositionEnv(species_set=["SiO2", "B2O3", "Na2O"],
                         fraction_set=["0.25", "0.50", "0.75"], n_components=2,
                         total_units=4, reward_fn=_TerminalReward(oracle, "fraction"))
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
    env = CompositionEnv(species_set=["SiO2", "B2O3"], fraction_set=["0.25", "0.75"],
                         n_components=2, total_units=4)
    assign(env, 'SiO2', .25)
    assign(env, 'B2O3', .75)
    assert "(SiO2)0.25" in env.terminal_formula
    np.testing.assert_allclose(featurize_formula(env.state), featurize_formula("Si0.25B1.5O2.75"))


def test_plain_formula_callbacks_remain_supported():
    calls = []
    env = CompositionEnv(species_set=["Ni", "Fe"], fraction_set=["0.25", "0.75"],
                         n_components=2, total_units=4, reward_fn=lambda formula: calls.append(formula) or 7)
    assign(env, 'Ni', .25)
    assign(env, 'Fe', .75)
    assert calls == [env.terminal_formula]
    assert env.path[-1].reward == 7
