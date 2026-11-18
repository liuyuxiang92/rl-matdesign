"""Physical bounds and unique amount assignments for OOH and Ti alloys."""
from __future__ import annotations

import random
import numpy as np
import pytest


def _zero_feats(_state: str) -> np.ndarray:
    """Featurizer stub so tests don't pull in matminer at import time."""
    return np.zeros(4)


def test_existing_ooh_path_unchanged():
    """Default constructor produces the same env behavior as before the refactor."""
    from rl_matdesign.amounts import ChemicalAmountsGroup
    env = ChemicalAmountsGroup(total=1, unique=True,
        species_set=["Ni", "Fe", "Co", "Mn", "Cu", "Al"],
        amount_set=["0.05", "0.10", "0.15", "0.20", "0.25", "0.30",
                      "0.35", "0.40", "0.45", "0.50"],

        n_steps=5,
        state_featurizer=_zero_feats,
    )
    assert env.unique
    assert env.element_bounds == {}
    assert env.amount_scale == 20
    env.initialize()
    for _ in range(5):
        env.step(env.allowed_actions()[0])
    assert env.counter == 5
    assert len(env.terminal_cation_fractions()) == 5
    assert sum(env.terminal_cation_fractions().values()) == pytest.approx(1)


def _ti_alloy_env(state_featurizer=_zero_feats):
    """Helper: build the Ti-alloy-style env used across several tests."""
    from rl_matdesign.amounts import ChemicalAmountsGroup
    return ChemicalAmountsGroup(total=1, unique=True,
        species_set=["Ti", "Al", "V", "Cr", "Zr"],
        amount_set=[f"{i/100:.2f}" for i in range(0, 101)],

        n_steps=5,

        element_bounds={
            "Ti": (0.45, 0.90), "Al": (0.00, 0.25),
            "V": (0.00, 0.15), "Cr": (0.00, 0.10), "Zr": (0.00, 0.10),
        },

        state_featurizer=state_featurizer,
    )


def test_ti_alloy_env_construction():
    env = _ti_alloy_env()
    assert env.max_steps == 5
    assert env.unique
    assert env.element_bounds["Ti"] == [.45, .90]
    assert env.amount_scale == 100


def test_ti_alloy_100_random_episodes_respect_bounds():
    """The strict invariant: any random rollout must satisfy bounds AND sum=1.0."""
    env = _ti_alloy_env()
    bounds = {"Ti": (0.45, 0.90), "Al": (0, 0.25), "V": (0, 0.15),
              "Cr": (0, 0.10), "Zr": (0, 0.10)}
    random.seed(0)
    for _ in range(100):
        env.initialize()
        for _ in range(env.max_steps):
            actions = env.allowed_actions()
            assert actions, "no actions should always be available in a feasible Ti env"
            env.step(random.choice(actions))
        comp = env.terminal_cation_fractions()
        for el, (lo, hi) in bounds.items():
            v = comp.get(el, 0.0)
            assert lo - 1e-9 <= v <= hi + 1e-9
        assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_unique_stage_allows_free_order_without_repeated_assignments():
    """Each remaining label can be selected when it has a feasible amount."""
    from rl_matdesign.encoding import decode_one_hot
    env = _ti_alloy_env()
    env.initialize()
    assert {decode_one_hot(action[0], env.species_set) for action in env.allowed_actions()} == set(env.species_set)
    selected = set()
    for _ in range(env.max_steps):
        actions = env.allowed_actions()
        assert all(decode_one_hot(action[0], env.species_set) not in selected for action in actions)
        action = actions[-1]
        selected.add(decode_one_hot(action[0], env.species_set))
        env.step(action)
    assert selected == set(env.species_set)


def test_infeasible_bounds_raise():
    from rl_matdesign.amounts import ChemicalAmountsGroup
    with pytest.raises(ValueError) as info:
        ChemicalAmountsGroup(total=1, unique=True,
            species_set=["A", "B"],
            amount_set=["0.1", "0.2"],
            n_steps=2,

            element_bounds={"A": (0.0, 0.2), "B": (0.0, 0.2)},  # max sum = 0.4 < 1.0

        )
    assert "infeasible" in str(info.value).lower()


def _ti_alloy_eta_env(n_steps=4, state_featurizer=_zero_feats):
    """Ti-alloy-style env in unique_amounts mode with per-element bounds."""
    from rl_matdesign.amounts import ChemicalAmountsGroup
    return ChemicalAmountsGroup(total=1, unique=True,
        species_set=["Ti", "Al", "V", "Cr", "Zr"],
        amount_set=[f"{i/100:.2f}" for i in range(0, 101)],

        n_steps=n_steps,

        element_bounds={
            "Ti": (0.45, 0.90), "Al": (0.00, 0.25),
            "V": (0.00, 0.15), "Cr": (0.00, 0.10), "Zr": (0.05, 0.45),
        },

        state_featurizer=state_featurizer,
    )


def test_unique_amounts_with_bounds_constructs():
    env = _ti_alloy_eta_env()
    assert env.unique
    assert env.element_bounds["Ti"] == [.45, .90]
    assert env.element_bounds["Zr"] == [.05, .45]


def test_unique_amounts_random_episodes_respect_bounds():
    """Free element choice + per-element bounds: every rollout obeys bounds,
    sums to 1.0, and includes the mandatory (lo>0) cations Ti and Zr."""
    env = _ti_alloy_eta_env(n_steps=4)
    bounds = {"Ti": (0.45, 0.90), "Al": (0, 0.25), "V": (0, 0.15),
              "Cr": (0, 0.10), "Zr": (0.05, 0.45)}
    random.seed(0)
    for _ in range(200):
        env.initialize()
        for _ in range(env.max_steps):
            actions = env.allowed_actions()
            assert actions, "feasible env must always offer a legal action"
            env.step(random.choice(actions))
        comp = env.terminal_cation_fractions()
        assert abs(sum(comp.values()) - 1.0) < 1e-9
        for el, f in comp.items():
            lo, hi = bounds[el]
            assert lo - 1e-9 <= f <= hi + 1e-9
        assert comp.get("Ti", 0.0) > 0 and comp.get("Zr", 0.0) > 0   # mandatory


def test_unique_amounts_too_many_mandatory_raises():
    from rl_matdesign.amounts import ChemicalAmountsGroup
    with pytest.raises(ValueError, match="infeasible"):
        ChemicalAmountsGroup(total=1, unique=True,
            species_set=["A", "B", "C"],
            amount_set=[f"{i/100:.2f}" for i in range(0, 101)],
            n_steps=2,

            element_bounds={"A": (0.2, 0.5), "B": (0.2, 0.5), "C": (0.2, 0.5)},  # 3 mandatory > 2 slots

        )


def test_unique_amounts_infeasible_total_raises():
    from rl_matdesign.amounts import ChemicalAmountsGroup
    with pytest.raises(ValueError, match="infeasible"):
        ChemicalAmountsGroup(total=1, unique=True,
            species_set=["A", "B", "C"],
            amount_set=[f"{i/100:.2f}" for i in range(0, 101)],
            n_steps=2,

            element_bounds={"A": (0.0, 0.2), "B": (0.0, 0.2), "C": (0.0, 0.2)},  # max 2-subset = 0.4 < 1.0

        )


def test_unique_stage_accepts_al_before_ti_but_rejects_repeat():
    """The host template does not constrain assignment order."""
    from rl_matdesign.encoding import encode_choice
    env = _ti_alloy_env()
    env.initialize()
    elem_oh = tuple(encode_choice("Al", env.species_set).tolist())
    comp_oh = tuple(encode_choice("0.1", env.action_values).tolist())
    env.step((elem_oh, comp_oh))
    assert env.cation_fractions() == {"Al": 0.10}
    with pytest.raises(ValueError, match="infeasible"):
        env.step((elem_oh, comp_oh))


def test_lo_greater_than_hi_raises():
    from rl_matdesign.amounts import ChemicalAmountsGroup
    with pytest.raises(ValueError):
        ChemicalAmountsGroup(total=1, unique=True,
            species_set=["A", "B"],
            amount_set=["0.5"],
            n_steps=2,
            element_bounds={"A": (0.6, 0.4), "B": (0.0, 1.0)},

        )
