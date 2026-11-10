"""GlassNet wiring on synthetic domains, without research configs or assets."""
from fractions import Fraction
from itertools import product
from math import comb

import pytest

from rl_matdesign.configuration import build_env
from rl_matdesign.config_schema import resolve_config
from rl_matdesign.predictors.glassnet import GlassNetPredictor
from rl_matdesign.predictors.outputs import OutputSchema, ScientificOutput
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.registry import resolve_predictor
from rl_matdesign.utils.design_count import count_compositions


def test_glassnet_registry_owns_only_input_basis_and_cache_batch_settings():
    predictor = resolve_predictor("glassnet", {"oxide_basis": ["SiO2", "B2O3"],
        "batch_size": 16, "cache_size": 32, "output": "Tg", "direction": "min"})
    assert isinstance(predictor, GlassNetPredictor)
    assert predictor.oxide_basis == ("B2O3", "SiO2")
    assert predictor.batch_size == 16 and predictor.cache_size == 32
    assert predictor._model is None


def toy_domain(denominator):
    # The full three-coordinate simplex is an arithmetic fixture, not a
    # calibrated or recommended glass-design domain.
    return {"groups": [{"name": "mixture", "kind": "composition",
        "species_set": ["SiO2", "Na2O", "CaO"], "n_components": 3,
        "fraction_set": [str(index / denominator) for index in range(denominator + 1)],
    }]}


@pytest.mark.parametrize("denominator", [2, 4, 8])
def test_toy_oxide_simplex_count_matches_enumeration_and_closed_form(denominator):
    env, kind, _ = build_env(toy_domain(denominator), object())
    assert kind == "fraction" and env.n_components == 3
    menus = env._species_allowed_units
    exhaustive = sum(sum(values) == env._total_units for values in product(*menus.values()))
    assert count_compositions(menus, env._total_units, slots=3) == exhaustive == comb(denominator + 2, 2)


def test_toy_oxide_refinement_preserves_coarser_points():
    menus = []
    for denominator in [2, 4, 8]:
        env, _, _ = build_env(toy_domain(denominator), object())
        menus.append({label: {Fraction(amount, env._total_units) for amount in values}
                      for label, values in env._species_allowed_units.items()})
    for label in menus[0]:
        assert menus[0][label] < menus[1][label] < menus[2][label]


def test_one_authored_budget_derives_both_algorithm_schedules():
    cfg = {**toy_domain(2), "train_budget": 7, "dqn_warmup_eps": 2, "pg_batch_eps": 3}
    effective = resolve_config(cfg).config
    assert effective["dqn_warmup_eps"] + effective["dqn_num_train_eps"] == 7
    assert effective["pg_num_iters"] == 3
    assert "dqn_num_train_eps" not in cfg and "pg_num_iters" not in cfg


def test_shared_glass_model_evaluates_once_with_synthetic_physical_gate(monkeypatch):
    from rl_matdesign.registry import PREDICTORS
    calls, instances = [], []

    class FakeGlass:
        output_schema = {"Tg": OutputSchema("Tg", "K"),
                         "Density": OutputSchema("Density", "g/cm^3")}

        def predict_outputs(self, candidate):
            calls.append(dict(candidate))
            # Arbitrary toy outputs test unit-preserving wiring only.
            return {
                "Tg": ScientificOutput(self.output_schema["Tg"], 10 + 40 * candidate.get("SiO2", 0)),
                "Density": ScientificOutput(self.output_schema["Density"], 1 + 2 * candidate.get("CaO", 0)),
            }

    def factory(cfg, **_):
        leaf = FakeGlass()
        instances.append(leaf)
        return leaf

    monkeypatch.setitem(PREDICTORS, "glassnet", factory)
    cfg = {
        "models": {"toy": {"predictor": "glassnet", "oxide_basis": ["SiO2", "Na2O", "CaO"]}},
        "properties": [
            {"name": "toy_temperature", "model_ref": "toy", "output": "Tg", "objective": "mean", "direction": "max"},
            {"name": "toy_density", "model_ref": "toy", "output": "Density", "objective": "mean", "direction": "max"},
        ],
        "reward": {"type": "gated", "gates": [{"name": "toy_gate",
            "score": {"terms": {"toy_temperature": 1}}, "min": 25}],
            "objective": {"terms": {"toy_density": 1}}, "failure": {"reward": -7}},
    }
    predictor = StructureScorePredictor(cfg, seed=0)
    passing, failing = {"SiO2": .5, "CaO": .5}, {"Na2O": 1.0}
    assert predictor.predict(passing) == (2.0, None)
    # Gate failure subtracts the raw violation (25 - 10) from the base reward.
    assert predictor.predict(failing) == (-22.0, None)
    assert len(instances) == 1 and len(calls) == 2
    assert predictor.per_objective_stats(passing)["toy_temperature"][0] == 30
    assert len(calls) == 2  # Diagnostics reuse the successful evaluation.
