"""Composition and categorical stages share one explicit group environment."""
import random

import numpy as np
import pytest

from rl_matdesign.amounts import ChemicalAmountsGroup
from rl_matdesign.constraints.base import ConstraintFilter
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.encoding import decode_one_hot


def chemical(name, species, values, steps, **kwargs):
    return {"name": name, "kind": "composition", "species_set": species,
            "amount_set": values, "n_steps": steps, "total": 1, "unique": True, **kwargs}


def drive(env):
    env.initialize()
    while env.counter < env.max_steps:
        env.step(env.allowed_actions()[0])
    return env


def pick(env, label, value):
    inner = env._active()
    for action in env.allowed_actions():
        local = env._from_union(action)
        if (decode_one_hot(local[0], inner.species_set) == label
                and float(decode_one_hot(local[1], inner.action_values)) == value):
            env.step(action)
            return
    raise AssertionError((label, value))


def test_single_group_preserves_native_state_actions_and_payload():
    spec = chemical("alloy", ["Fe", "Ni", "Co", "Mn"], [.25, .5, .75], 4)
    direct = drive(ChemicalAmountsGroup(spec))
    grouped = drive(MultiGroupEnv(groups=[spec]))
    assert grouped.max_steps == direct.max_steps
    assert grouped._inners[0].action_values == direct.action_values
    assert grouped.species_set != direct.species_set
    for a, b in zip(direct.path, grouped.path):
        np.testing.assert_array_equal(a.state_material_features, b.state_material_features)
        np.testing.assert_array_equal(a.state_step_onehot, b.state_step_onehot)
        np.testing.assert_array_equal(a.action_elem_onehot, b.action_elem_onehot)
        np.testing.assert_array_equal(a.action_comp_onehot, b.action_comp_onehot)
    assert grouped.terminal_cation_fractions() == {"alloy": direct.terminal_cation_fractions()}


def test_two_groups_preserve_separate_totals_and_label_identity():
    env = drive(MultiGroupEnv(groups=[
        chemical("A", ["Mn", "Ni", "P"], [.05, .95], 2),
        chemical("B", ["Ni", "O", "Cl"], [.1, .2, .7], 3)]))
    assert env.max_steps == 5
    assert len(set(env.species_set)) == 6
    assert set(env.terminal_cation_fractions()) == {"A", "B"}
    for payload in env.terminal_cation_fractions().values():
        assert sum(payload.values()) == pytest.approx(1)
    assert hash(env.terminal_comp_key())


@pytest.mark.parametrize("categorical", [False, True])
def test_later_filter_receives_completed_prior_group(categorical):
    captured = []
    class Record(ConstraintFilter):
        def filter_actions(self, *, actions, prior_groups=None, **kwargs):
            captured.append(prior_groups)
            return actions
    later = ({"name": "settings", "kind": "categorical", "choices": [{"name": "Cl", "values": [.6, 1]}]}
             if categorical else chemical("second", ["S", "O", "Cl"], [.1, .2, .7], 3))
    env = MultiGroupEnv(groups=[chemical("first", ["Mn", "P"], [.05, .95], 2), later])
    env._inners[1].filters = [Record()]
    drive(env)
    real = [value for value in captured if value is not None]
    assert real and all(value == [{"Mn": .05, "P": .95}] for value in real)


def test_host_final_bound_absorbs_remaining_total():
    values = [i / 100 for i in range(2, 9)] + [i / 100 for i in range(92, 99)]
    env = MultiGroupEnv(groups=[chemical("host", ["Mn", "Ni", "Ru", "P"], values, 2,
                                       element_bounds={"P": [.92, .98]})])
    rng = random.Random(0)
    for _ in range(50):
        env.initialize()
        while env.allowed_actions():
            env.step(rng.choice(env.allowed_actions()))
        candidate = env.terminal_cation_fractions()["host"]
        metals = set(candidate) - {"P"}
        assert len(metals) == 1
        assert candidate[next(iter(metals))] in values[:7]
        assert sum(candidate.values()) == pytest.approx(1)


def test_retired_sites_multiplier_is_rejected():
    with pytest.raises(ValueError, match="sites"):
        MultiGroupEnv(groups=[chemical("host", ["Ni", "Fe"], [.25, .75], 2, sites=4)])


def test_categorical_values_remain_fields_even_with_element_names():
    env = MultiGroupEnv(groups=[chemical("chemistry", ["Mn", "P"], [.05, .95], 2),
        {"name": "settings", "kind": "categorical", "choices": [
            {"name": "O", "values": ["none", "oxide"]}, {"name": "Cl", "values": [.6, 1.4]}]}])
    drive(env)
    assert env.terminal_cation_fractions()["settings"] == {"O": "none", "Cl": .6}
    assert env.requires_builder_formula
    with pytest.raises(TypeError, match="builder"):
        env.assembled_composition()
    assert '"settings"' in env.terminal_formula


def test_four_group_episode_keeps_eighth_grid_precision():
    specs = [chemical(name, species, {"min": 0, "max": 1, "step": .125}, len(species))
             for name, species in [("A", ["Cs", "Rb", "K"]), ("B1", ["Ag", "Cu", "Na"]),
                                   ("B3", ["Bi", "Sb", "In", "Ga"]), ("X", ["Cl", "Br", "I"])]]
    env = MultiGroupEnv(groups=specs)
    rng = random.Random(0)
    for _ in range(5):
        env.initialize()
        while env.allowed_actions():
            env.step(rng.choice(env.allowed_actions()))
        assert env.counter == 13
        for candidate in env.terminal_cation_fractions().values():
            assert sum(candidate.values()) == 1
            assert all(value * 8 == round(value * 8) for value in candidate.values())


def test_grid_origin_and_authored_value_order_are_preserved():
    env = ChemicalAmountsGroup(species_set=["Ti", "Al"], n_steps=2,
                               amount_set={"min": .005, "max": .05, "step": .02})
    assert env.action_values == ["0.005", "0.025", "0.045"]
    assert env.amount_scale == 200
