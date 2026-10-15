"""Core OOH constraints, independent of analysis scripts and private configs."""
import numpy as np
import pytest

from abcde_ooh.constraints.primary_phase import check_primary_phase
from rl_matdesign.env import CompositionEnv
from rl_matdesign.registry import build_constraints


SPECIES = ["Ni", "Fe", "Co", "Mg", "Ca", "Al", "Ga", "Hf"]
FRACTIONS = [f"{i / 20:.2f}" for i in range(1, 17)]


def make_env(phase="any"):
    env = CompositionEnv(
        species_set=SPECIES,
        fraction_set=FRACTIONS,
        n_components=5,
        state_featurizer=lambda _: np.zeros(1),
    )
    config = {"type": "ooh_phase", "target_phases": [phase]}
    env.phase_filter = build_constraints({"filters": [config]}, env=env)
    return env, env.phase_filter.children[0]


@pytest.mark.parametrize("phase,composition", [
    ("NiFeCo", {"Ni": 8, "Fe": 8, "Mg": 1, "Al": 1, "Ga": 2}),
    ("NiFeCo", {"Ni": 8, "Co": 8, "Mg": 1, "Al": 1, "Hf": 2}),
    ("CoFe", {"Co": 16, "Mg": 1, "Al": 1, "Ga": 1, "Hf": 1}),
    ("NiFe", {"Ni": 16, "Mg": 1, "Al": 1, "Ga": 1, "Hf": 1}),
])
def test_named_primary_elements_must_be_present(phase, composition):
    _, constraint = make_env(phase)
    assert not constraint._is_feasible(composition, 0, 0)


@pytest.mark.parametrize("phase,composition", [
    ("any", {"Ni": 8, "Fe": 7, "Mg": 1, "Al": 2, "Hf": 2}),
    ("any", {"Ni": 7, "Fe": 9, "Mg": 1, "Al": 1, "Hf": 2}),
    ("NiFe", {"Ni": 8, "Fe": 8, "Mg": 1, "Al": 1, "Hf": 2}),
])
def test_dopant_fe_and_ratio_limits(phase, composition):
    _, constraint = make_env(phase)
    assert not constraint._is_feasible(composition, 0, 0)


def test_valid_nife_ratio_is_still_accepted():
    _, constraint = make_env("NiFe")
    assert constraint._is_feasible(
        {"Ni": 12, "Fe": 4, "Mg": 1, "Al": 1, "Hf": 2}, 0, 0
    )


@pytest.mark.parametrize("phase", ["any", "NiFeCo", "NiFe", "CoFe", "Ni", "Co"])
def test_public_rollouts_finish_with_five_valid_cations(phase):
    env, _ = make_env(phase)
    rng = np.random.default_rng(20260924)
    for _ in range(8):
        env.initialize()
        for step in range(5):
            actions = env.allowed_actions()
            assert actions, (phase, step, env._units_map)
            env.step(actions[int(rng.integers(len(actions)))])
        assert len(env._units_map) == 5
        assert all(amount > 0 for amount in env._units_map.values())
        assert sum(env._units_map.values()) == 20
        assert check_primary_phase({e: u / 20 for e, u in env._units_map.items()})[0]
        # Inspect the requested phase as well as the union accepted by the checker.
        ni, fe, co = (env._units_map.get(e, 0) for e in ("Ni", "Fe", "Co"))
        if phase == "NiFeCo":
            assert ni + fe + co > 15 and fe <= max(ni, co)
            assert min(ni, fe, co) > 0
        elif phase == "NiFe":
            assert ni + fe > 15 and 2 / 3 <= ni / (ni + fe) <= 3 / 4
        elif phase == "CoFe":
            assert co + fe > 15 and fe <= co
            assert fe > 0
        elif phase in {"Ni", "Co"}:
            assert env._units_map[phase] > 15
