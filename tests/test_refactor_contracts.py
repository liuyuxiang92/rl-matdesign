"""Scientific contracts that the general refactor must preserve.

These are small black-box environment checks, not snapshots of network weights
or the historical OOH random stream. Expensive predictors are not loaded. The
OOH check exercises its real sampling entry point with only frame preparation
and model inference replaced; it does not validate DeepMD numerical results.
"""
from __future__ import annotations

from collections import Counter
import random
import re

import numpy as np
import pytest

from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.env import CompositionEnv
from rl_matdesign.env_integer import IntegerRatioEnv
from rl_matdesign.env_multigroup import MultiGroupEnv


def _composition_probe(formula):
    """A small, absolute-amount probe independent of matminer's availability."""
    amounts = Counter()
    for symbol, amount in re.findall(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)", formula):
        amounts[symbol] += float(amount)
    return np.array([amounts[symbol] for symbol in ("Ni", "Fe", "Co", "Ti", "O")])


def _decoded_actions(env):
    return [
        (decode_one_hot(element, env.species_set), decode_one_hot(value, env.fraction_set))
        for element, value in env.allowed_actions()
    ]


def _assign(env, element, amount):
    for action in env.allowed_actions():
        local = env._from_union(action) if isinstance(env, MultiGroupEnv) else action
        active = env._active() if isinstance(env, MultiGroupEnv) else env
        if (decode_one_hot(local[0], active.species_set) == element
                and float(decode_one_hot(local[1], active.fraction_set)) == amount):
            env.step(action)
            return
    raise AssertionError(f"No legal action for {element}={amount}; {_decoded_actions(env)}")


def _composition_env(reward_fn=None):
    return CompositionEnv(
        species_set=["Ni", "Fe", "Co"],
        fraction_set=[f"{units / 10:.2f}" for units in range(1, 9)],
        total_units=10,
        n_components=3,
        state_featurizer=_composition_probe,
        reward_fn=reward_fn,
    )


def test_equal_partial_assignments_have_equal_continuations_and_reward_input():
    # An order-sensitive spy ensures the environment canonicalizes the predictor
    # input. A reward that itself parses a formula would hide such a regression.
    predictor_inputs = []

    def predictor(formula):
        predictor_inputs.append(formula)
        return float(sum((i + 1) * ord(char) for i, char in enumerate(formula)))

    a, b = _composition_env(predictor), _composition_env(predictor)
    for element, amount in [("Ni", 0.2), ("Fe", 0.4)]:
        _assign(a, element, amount)
    for element, amount in [("Fe", 0.4), ("Ni", 0.2)]:
        _assign(b, element, amount)

    assert a.cation_fractions() == b.cation_fractions() == {"Ni": 0.2, "Fe": 0.4}
    assert a.remaining_units == b.remaining_units == 4
    assert a.counter == b.counter == 2
    np.testing.assert_array_equal(a.current_state_features(), b.current_state_features())
    assert _decoded_actions(a) == _decoded_actions(b) == [("Co", "0.40")]
    for env in (a, b):
        _assign(env, "Co", 0.4)
        assert [step.reward for step in env.path[:-1]] == [0.0, 0.0]
    assert predictor_inputs == [a.terminal_formula, a.terminal_formula]
    assert a.path[-1].reward == b.path[-1].reward
    assert a.terminal_comp_key() == b.terminal_comp_key()


def test_absolute_budget_changes_legal_continuations():
    # This asserts the physical budget, not the current default descriptor. The
    # new encoder must ALSO expose that distinction; that is a later acceptance
    # criterion, not a test freezing the current loss of absolute state.
    a, b = _composition_env(), _composition_env()
    _assign(a, "Ni", 0.2)
    _assign(b, "Ni", 0.6)
    assert a.remaining_units == 8
    assert b.remaining_units == 4
    assert ("Fe", "0.60") in _decoded_actions(a)
    assert ("Fe", "0.60") not in _decoded_actions(b)
    assert all(element != "Ni" for element, _ in _decoded_actions(a))
    assert all(element != "Ni" for element, _ in _decoded_actions(b))


def test_fractional_zero_assignment_consumes_a_slot_without_reusing_element():
    env = CompositionEnv(
        species_set=["Ni", "Fe", "Co"],
        fraction_set=["0.00", "0.40", "0.60", "1.00"],
        total_units=10,
        n_components=3,
        state_featurizer=_composition_probe,
    )
    _assign(env, "Ni", 0.0)
    assert env.counter == 1
    assert env.remaining_units == 10
    assert all(element != "Ni" for element, _ in _decoded_actions(env))
    _assign(env, "Fe", 0.4)
    _assign(env, "Co", 0.6)
    assert env.counter == 3
    assert "Ni" not in env.terminal_formula
    assert env.terminal_comp_key() == (("Co", 6), ("Fe", 4))


def test_integer_tokens_accumulate_repeats_and_zero_still_consumes_a_slot():
    env = IntegerRatioEnv(
        species_set=["Fe", "Ti", "O"], n_components=5,
        state_featurizer=_composition_probe,
    )
    for element, amount in [("Fe", 3), ("Fe", 4), ("Ti", 0), ("Fe", 0), ("O", 2)]:
        _assign(env, element, amount)
    assert env.counter == 5
    assert env.cation_digits() == {"Fe": 7, "Ti": 0, "O": 2}
    assert env.terminal_cation_fractions() == {"Fe": 7 / 9, "O": 2 / 9}
    assert env.terminal_comp_key() == (("Fe", 7), ("O", 2))
    assert "Ti" not in env.terminal_formula


def test_integer_terminal_identity_reduces_common_factor():
    keys = []
    for amounts in [(2, 4, 6), (1, 2, 3)]:
        env = IntegerRatioEnv(
            species_set=["Fe", "Ti", "O"], n_components=3,
            state_featurizer=_composition_probe,
        )
        for element, amount in zip(("Fe", "Ti", "O"), amounts):
            _assign(env, element, amount)
        keys.append(env.terminal_comp_key())
    assert keys[0] == keys[1] == (("Fe", 1), ("O", 3), ("Ti", 2))


def test_sublattice_identity_is_not_lost_when_bulk_composition_is_equal():
    groups = [dict(name=name, species_set=["Ni", "Fe"],
                   fraction_set=["0.20", "0.80"], total_units=5,
                   n_components=2, sites=1)
              for name in ("A_site", "B_site")]
    envs = [MultiGroupEnv(groups=groups, state_featurizer=_composition_probe)
            for _ in range(2)]
    for env, ni_amounts in zip(envs, [(0.2, 0.8), (0.8, 0.2)]):
        for amount in ni_amounts:
            _assign(env, "Ni", amount)
            _assign(env, "Fe", round(1.0 - amount, 1))
    a, b = envs
    assert a.assembled_composition() == b.assembled_composition() == {"Ni": 1.0, "Fe": 1.0}
    assert a.terminal_comp_key() != b.terminal_comp_key()
    assert a.terminal_cation_fractions()["A_site"] != b.terminal_cation_fractions()["A_site"]
    assert not np.array_equal(a.current_state_features(), b.current_state_features())


class _SymbolSites:
    """Minimal atom-site container; no ASE, file I/O, or model dependency."""

    def __init__(self, symbols):
        self.symbols = list(symbols)

    def copy(self):
        return _SymbolSites(self.symbols)

    def get_chemical_symbols(self):
        return list(self.symbols)

    def set_chemical_symbols(self, symbols):
        self.symbols = list(symbols)


def _sampling_probe(seed):
    from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor

    predictor = DeepMDOverpotentialPredictor.__new__(DeepMDOverpotentialPredictor)
    predictor.cfg = DPConfig(seed=seed, n_random_configs=4, adsorbates=())
    predictor.base_slab = _SymbolSites(["Ni"] * 20 + ["O", "H"])
    predictor.dp_models = [object()]
    predictor._infer_metal_elem_from_slab = lambda *_: "Ni"
    observed_sites = []

    def prepare(doped, **_):
        sites = tuple(doped.get_chemical_symbols())
        observed_sites.append(sites)
        return sites, None, None, [], []

    predictor._build_dp_inputs_for_one_doped_slab = prepare
    # Output depends on occupied sites so a sampling error also changes score.
    predictor._eval_models_on_prepared_inputs = lambda coords, *_: [
        float(sum(i for i, element in enumerate(coords) if element == "Fe"))
    ]
    return predictor, observed_sites


@pytest.mark.parametrize("seed", [0, 20261105])
def test_ooh_sampling_repeats_independently_of_query_order_and_global_rng(seed):
    predictor, sites = _sampling_probe(seed)
    composition = {"Ni": 0.6, "Fe": 0.4}
    first = predictor.predict_overpotential(composition)
    expected_sites = tuple(sites)
    assert len(expected_sites) == 4
    assert all(Counter(sample) == {"Ni": 12, "Fe": 8, "O": 1, "H": 1}
               for sample in expected_sites)

    # An intervening candidate and an unrelated global RNG must not move the
    # candidate's local structure stream. This uses exactly representable atom
    # counts: tie-breaking when rounding counts is deliberately not certified.
    predictor.predict_overpotential({"Ni": 0.5, "Fe": 0.5})
    previous_state = random.getstate()
    try:
        random.seed(seed + 123)
        for _ in range(17):
            random.random()
        sites.clear()
        repeated = predictor.predict_overpotential({"Fe": 0.4, "Ni": 0.6})
    finally:
        random.setstate(previous_state)
    assert tuple(sites) == expected_sites
    assert repeated == first

    fresh, fresh_sites = _sampling_probe(seed)
    assert fresh.predict_overpotential(composition) == first
    assert tuple(fresh_sites) == expected_sites


def test_ooh_evaluation_seed_controls_sampled_atom_positions():
    a, a_sites = _sampling_probe(1)
    b, b_sites = _sampling_probe(2)
    for predictor in (a, b):
        predictor.predict_overpotential({"Ni": 0.6, "Fe": 0.4})
    assert a_sites != b_sites
