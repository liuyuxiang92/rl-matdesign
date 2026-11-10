"""Independent scientific contracts for the reward engine's named output path."""
from copy import deepcopy
import math

import pytest
import numpy as np

from rl_matdesign.predictors.outputs import OutputSchema, ModelSamples, ScientificOutput, aggregate_samples
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError


class MultiOutput:
    output_schema = {"temperature": OutputSchema("temperature", "K"),
                     "expansion": OutputSchema("expansion", "K^-1")}
    def __init__(self, cfg, *, seed=None):
        self.calls = []
        self.closes = 0
    def predict(self, candidate):
        raise AssertionError("Named scientific outputs must win over reward-oriented legacy predict")
    def predict_outputs(self, candidate, **conditions):
        self.calls.append((deepcopy(candidate), deepcopy(conditions)))
        return {name: aggregate_samples(schema, [ModelSamples("m", ("candidate",), (value,))])
                for (name, schema), value in zip(self.output_schema.items(), (300., 2e-6))}
    def close(self):
        self.closes += 1


@pytest.fixture
def registered(monkeypatch):
    import rl_matdesign.registry as registry
    instances = []
    def factory(kind, cfg, *, seed=None):
        obj = MultiOutput(cfg, seed=seed)
        instances.append(obj)
        return obj
    monkeypatch.setattr(registry, "resolve_predictor", factory)
    return registry, instances


def shared_config():
    return {"models": {"glass": {"predictor": "test:Multi", "model": "external.ckpt"}},
            "properties": [{"name": "T", "model_ref": "glass", "output": "temperature", "direction": "max"},
                           {"name": "alpha", "model_ref": "glass", "output": "expansion", "direction": "min"}]}


def test_model_reference_loads_once_predicts_once_and_closes_once(registered):
    _, instances = registered
    p = StructureScorePredictor(shared_config(), seed=12)
    candidate = {"SiO2": .8, "Na2O": .2}
    assert p.cached_outputs(candidate) is None
    assert instances[0].calls == []
    assert p.predict(candidate) == (300 - 2e-6, None)
    outputs = p.predict_outputs(candidate)
    assert outputs["T"].schema == OutputSchema("T", "K")
    assert outputs["alpha"].schema == OutputSchema("alpha", "K^-1")
    assert outputs["T"].model_spread is None
    assert p.per_objective_stats(candidate) == {"T": (300., None), "alpha": (2e-6, None)}
    assert len(instances) == len(instances[0].calls) == 1
    assert p.cached_outputs(candidate) == outputs
    p.close(); p.close()
    assert instances[0].closes == 1


def test_named_batch_preserves_records_after_lru_eviction(registered):
    cfg = shared_config(); cfg["predict_cache_size"] = 1
    p = StructureScorePredictor(cfg, seed=1)
    candidates = [{"SiO2": value} for value in (1, 2, 3)]
    results = p.batch_predict_outputs(candidates)
    assert [record["T"].mean for record in results] == [300] * 3
    assert len(registered[1][0].calls) == 3
    assert all(p.cached_outputs(candidate) is not None for candidate in candidates)


@pytest.mark.parametrize("edit", [
    lambda c: c["properties"][0].update(predictor="test:Multi"),
    lambda c: c["properties"][0].update(model="again"),
    lambda c: c["properties"][0].update(model_ref="missing"),
    lambda c: c["models"].update(unused={"predictor": "test:Multi"}),
    lambda c: c["models"]["glass"].update(direction="min"),
])
def test_ambiguous_model_configuration_is_rejected(registered, edit):
    cfg = shared_config(); edit(cfg)
    with pytest.raises(EvaluationProtocolError):
        StructureScorePredictor(cfg)
    assert all(obj.closes == 1 for obj in registered[1])


def test_shared_inference_separates_operating_conditions(registered):
    cfg = shared_config()
    cfg["properties"][0]["fparam"] = [300]
    cfg["properties"][1]["fparam"] = [500]
    p = StructureScorePredictor(cfg)
    p.predict({"SiO2": 1})
    assert [call[1]["fparam"] for call in registered[1][0].calls] == [[300], [500]]


def scalar_predictor(monkeypatch, *, rows=None, folded=None, transform="none", **settings):
    import rl_matdesign.registry as registry
    class Leaf:
        def score_structures(self, cells):
            return rows
    class Folded:
        def predict(self, candidate):
            return folded
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: Folded() if folded else Leaf())
    cfg = {"properties": [{"name": "p", "predictor": "test:Leaf", "direction": "max",
                            "transform": transform, **settings}]}
    if rows is not None:
        cfg["structure_sources"] = {"s": {"builder": "substitute", "base_poscar": "fake"}}
        class Builder:
            def build(self, candidate, **kwargs):
                return [object()] * len(rows)
        monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    return StructureScorePredictor(cfg, seed=0)


@pytest.mark.parametrize("transform,values", [("exp", [0., math.log(9)]), ("pow10", [0., math.log10(9)])])
def test_nonlinear_transform_is_before_mean_and_scatter(monkeypatch, transform, values):
    p = scalar_predictor(monkeypatch, rows=values, transform=transform, unit="K", uncertainty="config_spread")
    assert p.predict({"Fe": 1}) == (pytest.approx(5.), None)
    output = p.predict_outputs({"Fe": 1})["p"]
    assert output.mean == pytest.approx(5)
    assert output.config_spread == pytest.approx(4)
    assert output.model_spread is None
    assert output.schema == OutputSchema("p", "K")
    assert output.samples[0].values == pytest.approx((1, 9))
    assert output.mean != pytest.approx(3)  # exp(mean(log(values))) is the wrong physical statistic.


def test_legacy_native_uncertainty_is_not_model_spread(monkeypatch):
    p = scalar_predictor(monkeypatch, folded=(7., 2.), objective="mean_minus_kstd")
    assert p.predict({"Fe": 1}) == (5., None)
    output = p.predict_outputs({"Fe": 1})["p"]
    assert output.native_uncertainty == 2
    assert output.model_spread is output.config_spread is output.samples is None


def test_self_folded_nonlinear_transform_is_rejected_without_raw_distribution(monkeypatch):
    p = scalar_predictor(monkeypatch, folded=(math.log(3), .1), transform="exp")
    with pytest.raises(EvaluationProtocolError, match="raw samples"):
        p.predict({"Fe": 1})


def test_singleton_penalty_is_not_silently_filled_with_zero(monkeypatch):
    p = scalar_predictor(monkeypatch, rows=[1., 3.], objective="mean_minus_kstd")
    with pytest.raises(EvaluationProtocolError, match="unavailable"):
        p.predict({"Fe": 1})
    cfg = deepcopy(p.cfg); cfg["properties"][0]["uncertainty"] = "config_spread"
    alternate = StructureScorePredictor(cfg)
    assert alternate.predict({"Fe": 1}) == (1., None)


def test_named_output_unit_conflict_and_missing_selector_are_fatal(registered):
    cfg = shared_config(); cfg["properties"][0]["unit"] = "C"
    with pytest.raises(EvaluationProtocolError, match="unit conflicts"):
        StructureScorePredictor(cfg).predict({"SiO2": 1})
    cfg = shared_config(); cfg["properties"][0].pop("output")
    with pytest.raises(EvaluationProtocolError, match="select one"):
        StructureScorePredictor(cfg).predict({"SiO2": 1})


def test_named_sources_use_separate_inference_payloads(registered, monkeypatch):
    registry, instances = registered
    class StructMulti(MultiOutput):
        predict_structure_outputs = MultiOutput.predict_outputs
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: (instances.append(StructMulti({})) or instances[-1]))
    class Builder:
        def build(self, candidate, **kwargs): return [object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = shared_config(); cfg["structure_sources"] = {name: {"builder": "substitute", "base_poscar": "fake"} for name in ("a", "b")}
    for prop, source in zip(cfg["properties"], ("a", "b")): prop["source"] = source
    StructureScorePredictor(cfg).predict({"Fe": 1})
    assert len(instances[0].calls) == 2


def test_candidate_value_error_preserves_neighbor_but_invalid_contract_is_fatal(registered):
    _, instances = registered
    p = StructureScorePredictor(shared_config())
    ordinary = instances[0].predict_outputs
    def predict(candidate, **kwargs):
        if candidate["SiO2"] == 0: raise ValueError("unsupported candidate chemistry")
        return ordinary(candidate, **kwargs)
    instances[0].predict_outputs = predict
    results = p.batch_predict([{"SiO2": 0}, {"SiO2": 1}], return_exceptions=True)
    assert type(results[0]) is ValueError
    assert results[1] == (300-2e-6, None)
    instances[0].predict_outputs = lambda candidate: {"bad": 3.}
    with pytest.raises(EvaluationProtocolError):
        p.batch_predict([{"SiO2": 2}], return_exceptions=True)


def test_dp_rectangular_model_and_configuration_axes_and_raw_exp(monkeypatch):
    import rl_matdesign.utils.dp_eval as dp
    cfg = {"base_poscar": "fake", "properties": [{"name": "p", "predictor": "dp_property", "models": ["a", "b"],
                            "direction": "max", "transform": "exp"}]}
    p = StructureScorePredictor(cfg)
    p._get_prop_models = lambda prop: ([object(), object()], {})
    monkeypatch.setattr(dp, "eval_property_ensemble", lambda *args, **kwargs: np.log([[1., 3.], [5., 7.]]))
    output, _ = p._score_output(p.properties[0], [object(), object()])
    assert (output.mean, output.model_spread, output.config_spread) == pytest.approx((4, 2, 1))
    monkeypatch.setattr(dp, "eval_property_ensemble", lambda *args, **kwargs: [[1., 2.]])
    with pytest.raises(EvaluationProtocolError, match="complete model/configuration"):
        p._score_output(p.properties[0], [object(), object()])


def test_physical_oracle_identity_ignores_reward_and_records_model_assets(registered, tmp_path):
    model = tmp_path / "model.ckpt"; model.write_text("weights")
    cfg = shared_config(); cfg["models"]["glass"]["model"] = str(model)
    a = StructureScorePredictor(cfg, seed=1)
    cfg["properties"][0].update(weight=5, scale=3, direction="min", objective="mean_plus_kstd")
    b = StructureScorePredictor(cfg, seed=1)
    assert a.oracle_identity() == b.oracle_identity()
    assert a.oracle_identity()["model_assets"]["glass"][0]["status"] == "resolved"
    model.write_text("new weights")
    with pytest.raises(EvaluationProtocolError, match="asset changed"):
        a.predict({"SiO2": 1})


def test_shared_model_sweep_metadata_and_outputs_use_matching_conditions(registered, monkeypatch):
    registry, instances = registered
    class Swept(MultiOutput):
        def predict_outputs(self, candidate, *, fparam):
            self.current = fparam[0]
            self.calls.append(self.current)
            return {name: ScientificOutput(schema, (10 - abs(self.current-2)) if name == "temperature" else self.current)
                    for name, schema in self.output_schema.items()}
        def candidate_metadata(self, candidate):
            return {"condition": self.current}
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: (instances.append(Swept({})) or instances[-1]))
    cfg = shared_config(); cfg["sweep"] = {"name": "condition", "values": [1, 2, 3]}
    cfg["properties"][0]["fparam"] = [None]
    cfg["properties"][1].update(fparam=[5], weight=0)
    p = StructureScorePredictor(cfg)
    candidate = {"SiO2": 1}
    assert p.predict(candidate) == (10., None)
    assert p.per_objective_stats(candidate)["condition"] == (2., None)
    assert p.predict_outputs(candidate)["T"].mean == 10
    metadata = p.candidate_metadata(candidate)
    assert metadata == {"T": [{"condition": 2}], "alpha": [{"condition": 5}]}
    assert instances[0].calls == [1, 5, 2, 3]  # Fixed condition reused across the entire sweep.


def test_external_rng_is_candidate_model_local_and_preserves_training_streams(registered, monkeypatch):
    import random
    import torch
    from rl_matdesign.utils.seeding import capture_rng_state
    registry, _ = registered
    class RandomOutput(MultiOutput):
        def __init__(self, cfg, *, seed=None):
            super().__init__(cfg, seed=seed)
            self.offset = random.random() + np.random.random() + float(torch.rand(()))
        def predict_outputs(self, candidate):
            value = self.offset + random.random() + np.random.random() + float(torch.rand(()))
            return {name: ScientificOutput(schema, value) for name, schema in self.output_schema.items()}
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: RandomOutput(args[1], **kwargs))
    cfg = shared_config(); cfg["predict_cache_size"] = 1
    before = capture_rng_state()
    p = StructureScorePredictor(cfg, seed=23)
    first, second = {"Na2O": .2, "SiO2": .8}, {"Na2O": .3, "SiO2": .7}
    outputs = p.predict_outputs(first)
    p.predict(second)
    assert p.predict_outputs(dict(reversed(list(first.items())))) == outputs
    cfg["properties"].reverse()
    q = StructureScorePredictor(cfg, seed=23)
    q.batch_predict([second, first])
    assert q.predict_outputs(first) == outputs
    after = capture_rng_state()
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])


def test_dp_model_reference_shares_full_vector_inference(monkeypatch):
    import rl_matdesign.registry as registry
    import rl_matdesign.utils.dp_eval as dp
    class Builder:
        def build(self, candidate, **kwargs): return [object(), object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = {"base_poscar": "fake", "models": {"dpa": {"predictor": "dp_property", "models": ["a", "b"]}},
           "properties": [{"name": "one", "model_ref": "dpa", "output_index": 0, "direction": "max"},
                          {"name": "two", "model_ref": "dpa", "output_index": 1, "direction": "max"}]}
    p = StructureScorePredictor(cfg)
    p._get_prop_models = lambda prop: ([object(), object()], {})
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(1)
        return np.array([[[1., 10.], [3., 30.]], [[5., 50.], [7., 70.]]])
    monkeypatch.setattr(dp, "eval_property_outputs", evaluate)
    output = p.predict_outputs({"Fe": 1})
    assert (output["one"].mean, output["two"].mean) == (4, 40)
    assert (output["one"].model_spread, output["two"].config_spread) == (2, 10)
    assert len(calls) == 1
    assert p.cost_counters()["dp"] == {"ensemble_evaluation_calls": 1,
        "model_configuration_pairs_attempted": 4, "network_forwards": None}


def test_pooled_spread_is_explicitly_selected_without_relabeling(monkeypatch):
    p = scalar_predictor(monkeypatch, rows=[1., 3.], uncertainty="pooled_spread", objective="mean_minus_kstd")
    assert p.predict({"Fe": 1}) == (1., None)
    output = p.predict_outputs({"Fe": 1})["p"]
    assert output.pooled_spread == output.config_spread == 1.
    assert output.model_spread is None


def test_cost_counters_deduplicate_shared_instances_and_keep_unknown(registered):
    p = StructureScorePredictor(shared_config())
    info = p.cost_counters()
    assert len(info["leaf_backends"]) == 1
    assert info["leaf_backends"][0]["properties"] == ["T", "alpha"]
    assert info["leaf_backends"][0]["status"] == "unknown"
    assert all(value is None for value in info["leaf_backends"][0]["counters"].values())
    leaf = registered[1][0]
    leaf.cache_info = lambda: {"requests": len(leaf.calls), "model_batches": len(leaf.calls),
                               "model_candidates": len(leaf.calls), "cache_hits": 0, "batch_duplicates": 0}
    p.predict({"SiO2": 1}); p.predict({"SiO2": 1})
    info = p.cost_counters()
    assert info["leaf_backends"][0]["counters"]["model_batches"] == 1
    assert len(leaf.calls) == 1
    assert info["dp"]["network_forwards"] is None


def test_multiple_self_folded_models_require_raw_samples(monkeypatch):
    p = scalar_predictor(monkeypatch, folded=(7., .2), model=["a", "b"])
    with pytest.raises(EvaluationProtocolError, match="Multiple self-folding"):
        p.predict({"Fe": 1})


def test_classifier_gate_uses_model_spread_when_reward_selects_config_spread(monkeypatch):
    import rl_matdesign.registry as registry
    import rl_matdesign.utils.dp_eval as dp
    class Builder:
        def build(self, candidate, **kwargs): return [object(), object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = {"base_poscar": "fake", "properties": [{"name": "pure", "predictor": "dp_classification",
           "models": ["a", "b"], "output_index": 1, "direction": "max",
           "objective": "mean_minus_kstd", "uncertainty": "config_spread"}],
           "reward": {"type": "gated", "gates": [{"name": "purity", "score": {"terms": {"pure": 1}}, "min": .85}],
                      "objective": {"terms": {"pure": 1}}, "failure": {"reward": -10}}}
    p = StructureScorePredictor(cfg)
    p._get_prop_models = lambda prop: ([object(), object()], {})
    monkeypatch.setattr(dp, "eval_classification_ensemble", lambda *args, **kwargs: [[.8, 1.], [.8, 1.]])
    candidate = {"Fe": 1}
    stats = p.per_objective_stats(candidate)
    assert stats["pure"] == pytest.approx((.9, .1))
    assert p.reward_details(candidate)["gate_passed"] is True
    assert p.predict(candidate) == pytest.approx((.8, None))
    assert p.raw_combine(stats) == pytest.approx(.9)
