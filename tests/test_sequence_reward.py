"""Ordered candidates reuse the existing predictor and uncertainty contracts.

The leaf below is a deterministic test fixture, not an adapter to a real GP.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.composition import composition_key
from rl_matdesign.utils.timing import candidate_key


class _DistributionLeaf:
    def __init__(self, cfg, *, seed=None):
        self.cfg = cfg
        self.seed = seed
        self.calls = 0
        self.last_candidate = None

    def predict(self, candidate):
        self.calls += 1
        self.last_candidate = candidate
        return self.cfg.get("reported_distribution", (8.0, 0.75))


def _config(*, direction="max", objective="mean", predictor=None, **kwargs):
    return {
        "k": 2.0,
        "properties": [{
            "name": "response",
            "predictor": predictor or f"{__name__}:_DistributionLeaf",
            "direction": direction,
            "objective": objective,
            "adapter_option": "forwarded",
        }],
        **kwargs,
    }


def _candidate(sequence=None, cap="01"):
    return {
        "sequence": list(sequence if sequence is not None else ["01", "X", "1"]),
        "n_terminal": cap,
        "marked_positions": {"bridge": [1, 3]},
    }


@pytest.mark.parametrize("left,right", [
    (["A", "B"], ["B", "A"]),
    (["01"], ["1"]),
    (["1"], [1]),
    ([True], [1]),
    ([None], ["None"]),
    (["A"], ("A",)),
    ([1.00000001], [1.00000002]),
    ([], {}),
    ([{"id": "01"}], [{"id": "1"}]),
])
def test_ordered_keys_preserve_order_types_and_exact_identifiers(left, right):
    assert composition_key(left, decimals=6) != composition_key(right, decimals=6)
    # Keys can be stored in dicts and sets, including nested mappings.
    assert len({composition_key(left), composition_key(right)}) == 2


def test_structured_keys_preserve_numeric_cap_ids_but_ignore_mapping_order():
    first, second = _candidate(cap="01"), _candidate(cap="1")
    reordered = dict(reversed(list(first.items())))
    assert composition_key(first) == composition_key(reordered)
    assert composition_key(first) != composition_key(second)
    # Timing and model caches agree on structured identity, including metadata.
    assert candidate_key(first) == StructureScorePredictor._key(first)
    assert candidate_key(first) != candidate_key(second)


def test_legacy_composition_numeric_coercion_and_rounding_are_unchanged():
    assert composition_key({"Ni": "0.75000001", "Fe": 0.25}, decimals=6) == (
        ("Fe", 0.25), ("Ni", 0.75),
    )
    assert composition_key({"group": {"amount": "01"}}) == (
        ("group", (("amount", 1.0),)),
    )


@pytest.mark.parametrize("direction,sign", [("max", 1.0), ("min", -1.0)])
@pytest.mark.parametrize("objective,uncertainty_term", [
    ("mean", 0.0),
    ("mean_minus_kstd", -1.5),
    ("mean_plus_kstd", 1.5),
])
def test_existing_reward_engine_scores_distributions_on_sequence_candidates(
    direction, sign, objective, uncertainty_term,
):
    predictor = StructureScorePredictor(_config(direction=direction, objective=objective))
    candidate = _candidate()
    assert predictor.predict(candidate) == pytest.approx((sign * 8.0 + uncertainty_term, 0.75))
    # Raw physical values are exposed by per_objective_stats; predict_raw is the
    # existing signed/weighted aggregate, not the original model's raw mean.
    assert predictor.per_objective_stats(candidate) == {"response": (8.0, 0.75)}
    assert predictor.predict_raw(candidate) == (sign * 8.0, 0.0)
    assert predictor.properties[0]["instance"][0].calls == 1
    assert predictor._shared_builder is None


def test_fqn_leaf_receives_full_payload_config_and_seed_without_repeated_scoring():
    predictor = StructureScorePredictor(_config(), seed=17)
    candidate = _candidate()
    predictor.predict(candidate)
    predictor.predict_raw(candidate)
    predictor.per_objective_stats(candidate)
    leaf = predictor.properties[0]["instance"][0]
    assert isinstance(leaf, _DistributionLeaf)
    assert leaf.cfg["adapter_option"] == "forwarded"
    assert leaf.seed == 17
    assert leaf.last_candidate is candidate
    assert leaf.calls == 1


def test_registered_leaf_cache_distinguishes_sequence_order_and_n_terminal(monkeypatch):
    from rl_matdesign.registry import PREDICTORS

    monkeypatch.setitem(PREDICTORS, "test_distribution", _DistributionLeaf)
    predictor = StructureScorePredictor(_config(predictor="test_distribution"))
    candidates = [
        _candidate(["01", "1"], cap="01"),
        _candidate(["1", "01"], cap="01"),
        _candidate(["01", "1"], cap="1"),
    ]
    for candidate in candidates:
        predictor.predict(candidate)
        assert predictor.per_objective_stats(candidate) == {"response": (8.0, 0.75)}
    leaf = predictor.properties[0]["instance"][0]
    assert leaf.calls == 3
    predictor.batch_predict(candidates)
    assert leaf.calls == 3


def test_structured_candidate_cache_keeps_existing_lru_limit():
    predictor = StructureScorePredictor(_config(predict_cache_size=2))
    first, second, third = [_candidate(cap=cap) for cap in ["01", "1", "unmodified"]]
    predictor.predict(first)
    predictor.predict(second)
    predictor.predict(first)  # Keep first, evict second on the following call.
    predictor.predict(third)
    assert len(predictor._stats_cache) == 2
    predictor.predict(first)
    leaf = predictor.properties[0]["instance"][0]
    assert leaf.calls == 3
    predictor.predict(second)
    assert leaf.calls == 4


@pytest.mark.parametrize("distribution", [
    (float("nan"), 0.1),
    (float("inf"), 0.1),
    (-float("inf"), 0.1),
    (1.0, float("nan")),
    (1.0, float("inf")),
    (1.0, -float("inf")),
    (1.0, -0.1),
    (np.array([1.0]), 0.1),
    (1.0, np.array([0.1])),
    (np.array([1.0, 2.0]), 0.1),
    (1.0, np.array([0.1, 0.2])),
])
def test_invalid_leaf_distribution_is_rejected_without_caching(distribution):
    cfg = _config()
    cfg["properties"][0]["reported_distribution"] = distribution
    predictor = StructureScorePredictor(cfg)
    with pytest.raises(ValueError):
        predictor.predict(_candidate())
    assert not predictor._stats_cache


def test_numpy_scalar_leaf_distribution_is_accepted():
    cfg = _config()
    cfg["properties"][0]["reported_distribution"] = (np.float64(8.0), np.float32(0.75))
    predictor = StructureScorePredictor(cfg)
    assert predictor.predict(_candidate()) == (8.0, 0.75)


def _typed_cap_env(values):
    from rl_matdesign.env_multigroup import MultiGroupEnv

    return MultiGroupEnv(
        groups=[
            {"name": "cap", "kind": "categorical", "choices": [
                {"name": "id", "element": "id", "values": values},
            ]},
            {"name": "sequence", "kind": "sequence", "length": 1, "tokens": ["A"]},
        ],
        reward_fn=lambda candidate: 0.0,
        state_featurizer=lambda state: np.zeros(1),
    )


@pytest.mark.parametrize("values", [[1, "1"], [True, 1], [1, 1.0]])
def test_environment_dedup_key_preserves_categorical_value_types(values):
    env = _typed_cap_env(values)
    keys = []
    for choice in (0, 1):
        env.initialize()
        env.step(env.allowed_actions()[choice])
        env.step(env.allowed_actions()[0])
        candidate = env.terminal_cation_fractions()
        assert type(candidate["cap"]["id"]) is type(values[choice])
        keys.append(env.terminal_comp_key())
        assert keys[-1] == composition_key(candidate)
    assert keys[0] != keys[1]


def test_fresh_run_cannot_relabel_old_weights_with_new_vocabulary(tmp_path):
    from rl_matdesign.utils.design_schema import save_or_check_design_schema

    original = _typed_cap_env(["first", "second"]).design_schema()
    changed = _typed_cap_env(["second", "first"]).design_schema()
    save_or_check_design_schema(original, str(tmp_path))
    manifest = tmp_path / "sequence_schema.json"
    original_bytes = manifest.read_bytes()
    weights = tmp_path / "qnet.pt"
    weights.write_bytes(b"original checkpoint")

    # A new training run writes its manifest before training. If training then
    # fails, it must not leave the old weights falsely certified for new IDs.
    with pytest.raises(ValueError, match="new output directory"):
        save_or_check_design_schema(changed, str(tmp_path))
    assert manifest.read_bytes() == original_bytes
    assert weights.read_bytes() == b"original checkpoint"
    assert not (tmp_path / "sequence_schema.json.tmp").exists()
    with pytest.raises(ValueError, match="Sequence design differs"):
        save_or_check_design_schema(changed, str(tmp_path), load_paths=[str(weights)])


def test_existing_weights_without_schema_cannot_be_certified_retroactively(tmp_path):
    from rl_matdesign.utils.design_schema import save_or_check_design_schema

    (tmp_path / "qnet.pt").write_bytes(b"unknown checkpoint")
    with pytest.raises(ValueError, match="lack"):
        save_or_check_design_schema(_typed_cap_env(["first"]).design_schema(), str(tmp_path))
    assert not (tmp_path / "sequence_schema.json").exists()


@pytest.mark.parametrize("values", [[True, 1], [1, 1.0]])
def test_checkpoint_schema_rejects_reordered_equal_but_differently_typed_caps(tmp_path, values):
    from rl_matdesign.utils.design_schema import save_or_check_design_schema

    original = _typed_cap_env(values).design_schema()
    reordered = _typed_cap_env(list(reversed(values))).design_schema()
    save_or_check_design_schema(original, str(tmp_path))
    weights = tmp_path / "policy.pt"
    weights.write_bytes(b"original checkpoint")
    with pytest.raises(ValueError, match="Sequence design differs"):
        save_or_check_design_schema(reordered, str(tmp_path), load_paths=[str(weights)])
    with pytest.raises(ValueError, match="Sequence design differs"):
        save_or_check_design_schema(reordered, str(tmp_path))
