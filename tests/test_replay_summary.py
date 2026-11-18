"""Replay summaries preserve current structured candidates and discard fragments."""
import collections
import numpy as np
import pytest

from rl_matdesign.action_features import value_alphabet
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.training import add_episode_to_buffer
from scripts.summarize_replay_buffer import _decode_episodes


@pytest.mark.parametrize("groups", [
    [dict(name="glass", kind="composition", species_set=["SiO2", "Na2O"],
          amount_set=[.25, .75], total=1, unique=True, n_steps=2)],
    [dict(name="positions", kind="categorical", labels=[1, 2], values=["A", "G"], n_steps=2)],
])
def test_replay_summary_uses_typed_indices_and_no_predictor(groups):
    calls = []
    env = MultiGroupEnv(groups=groups, state_featurizer=lambda _: np.zeros(2),
                        reward_fn=lambda candidate: calls.append(candidate) or 4.0)
    while not env.is_terminal:
        env.step(env.allowed_actions()[0])
    original = env.terminal_cation_fractions()
    buffer = collections.deque()
    add_episode_to_buffer(env.path, buffer, env.action_features(), value_alphabet(env))
    reward = env.reward_fn
    episodes = _decode_episodes(buffer, env=env)
    assert episodes[0].candidate == original
    assert episodes[0].terminal_reward == 4.0
    assert len(calls) == 1 and env.reward_fn is reward
    assert _decode_episodes(list(buffer)[1:], env=env) == []
    assert _decode_episodes(list(buffer)[:-1], env=env) == []


def test_invalid_action_indices_are_not_reported_as_real_candidates():
    env = MultiGroupEnv(groups=[dict(name="x", kind="categorical", labels=[1], values=["A"], n_steps=1)])
    with pytest.raises(ValueError):
        _decode_episodes([dict(s_step=[1], a_elem_idx=8, a_comp_idx=0, reward=1, done=True)], env=env)
