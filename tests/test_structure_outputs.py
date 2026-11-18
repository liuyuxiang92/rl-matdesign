"""Independent scientific contracts for the reward engine's named output path."""
from copy import deepcopy
import math

import pytest
import numpy as np

from rl_matdesign.predictors.outputs import OutputSchema, ModelSamples, ScientificOutput, aggregate_samples
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError


def structure_config(n_configs=1):
    """A current declaration; tests replace only the expensive construction."""
    return {"builder": "pipeline", "base_poscar": "fake", "n_random_configs": n_configs,
            "steps": [{"type": "substitute", "a_site_symbol": "Fe", "a_dopant_group": "alloy"}]}


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
    return {"properties": [{"name": "T", "predictor": "test:Multi", "models": ["external.ckpt"], "output": "temperature", "direction": "max"},
                           {"name": "alpha", "predictor": "test:Multi", "models": ["external.ckpt"], "output": "expansion", "direction": "min"}],
            "reward": {"objective": {"terms": {"T": 1, "alpha": 1}}}}


def test_identical_inline_models_load_once_predict_once_and_close_once(registered):
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


def test_named_batch_preserves_records_after_lru_eviction(registered, monkeypatch):
    monkeypatch.setattr("rl_matdesign.predictors.structure_score._PREDICTION_CACHE_SIZE", 1)
    cfg = shared_config()
    p = StructureScorePredictor(cfg, seed=1)
    candidates = [{"SiO2": value} for value in (1, 2, 3)]
    results = p.batch_predict_outputs(candidates)
    assert [record["T"].mean for record in results] == [300] * 3
    assert len(registered[1][0].calls) == 3
    assert all(p.cached_outputs(candidate) is not None for candidate in candidates)


@pytest.mark.parametrize("edit", [
    lambda c: c.update(models={"old": {"predictor": "test:Multi"}}),
    lambda c: c.update(predictor="structure_score"),
    lambda c: c.update(model="old.ckpt"),
    lambda c: c.update(objective="mean"),
    lambda c: c.update(uncertainty="model_spread"),
    lambda c: c["properties"][0].update(model="again"),
    lambda c: c["properties"][0].update(dp_models=["again"]),
    lambda c: c["properties"][0].update(model_ref="missing"),
])
def test_retired_model_formats_are_rejected(registered, edit):
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
        cfg["structure"] = structure_config(len(rows))
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


def test_shared_structure_reuses_matching_conditions_and_separates_distinct_conditions(registered, monkeypatch):
    registry, instances = registered
    class StructMulti(MultiOutput):
        predict_structure_outputs = MultiOutput.predict_outputs
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: (instances.append(StructMulti({})) or instances[-1]))
    builds = []
    class Builder:
        def build(self, candidate, **kwargs):
            builds.append(candidate)
            return [object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = shared_config()
    cfg["structure"] = structure_config()
    for prop in cfg["properties"]:
        prop["fparam"] = [300]
    cfg["properties"].append({"name": "T_other", "predictor": "test:Multi", "models": ["external.ckpt"], "output": "temperature",
                              "direction": "max", "fparam": [500]})
    predictor = StructureScorePredictor(cfg)
    candidate = {"Fe": 1}
    outputs = predictor.predict_outputs(candidate)
    assert set(outputs) == {"T", "alpha", "T_other"}
    assert len(builds) == 1
    assert len(instances) == 1
    assert [call[1]["fparam"] for call in instances[0].calls] == [[300], [500]]
    assert predictor.predict_outputs(candidate) == outputs
    assert len(builds) == 1 and len(instances[0].calls) == 2


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
    cfg = {"structure": structure_config(2), "properties": [{"name": "p", "predictor": "dp_property", "models": ["a", "b"],
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
    cfg = shared_config()
    for prop in cfg["properties"]:
        prop["models"] = [str(model)]
    a = StructureScorePredictor(cfg, seed=1)
    cfg["properties"][0].update(scale=3, direction="min", objective="mean_plus_kstd")
    cfg["reward"]["objective"]["terms"]["T"] = 5
    b = StructureScorePredictor(cfg, seed=1)
    assert a.oracle_identity() == b.oracle_identity()
    assets = list(a.oracle_identity()["model_assets"].values())
    assert len(assets) == 1 and assets[0][0]["status"] == "resolved"
    model.write_text("new weights")
    with pytest.raises(EvaluationProtocolError, match="asset changed"):
        a.predict({"SiO2": 1})


def test_shared_model_explicit_condition_metadata_and_outputs_stay_paired(registered, monkeypatch):
    registry, instances = registered
    class Conditioned(MultiOutput):
        def predict_outputs(self, candidate, *, fparam):
            self.current = fparam[0]
            self.calls.append(self.current)
            return {name: ScientificOutput(schema, (10 - abs(self.current-2)) if name == "temperature" else self.current)
                    for name, schema in self.output_schema.items()}
        def candidate_metadata(self, candidate):
            return {"condition": self.current}
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: (instances.append(Conditioned({})) or instances[-1]))
    cfg = shared_config()
    cfg["properties"] = [
        {"name": f"T{value}", "predictor": "test:Multi", "models": ["external.ckpt"], "output": "temperature", "direction": "max", "fparam": [value]}
        for value in (1, 2, 3)] + [
        {"name": "alpha", "predictor": "test:Multi", "models": ["external.ckpt"], "output": "expansion", "direction": "min", "fparam": [5]}]
    cfg["reward"] = {"type": "max", "alternatives": [
        {"name": f"condition{value}", "terms": {f"T{value}": 1.}} for value in (1, 2, 3)]}
    p = StructureScorePredictor(cfg)
    candidate = {"SiO2": 1}
    assert p.predict(candidate) == (10., None)
    assert p.reward_details(candidate)["selected_alternative"] == "condition2"
    assert p.per_objective_stats(candidate) == {"T1": (9., None), "T2": (10., None), "T3": (9., None), "alpha": (5., None)}
    assert {name: output.mean for name, output in p.predict_outputs(candidate).items()} == {
        "T1": 9., "T2": 10., "T3": 9., "alpha": 5.}
    assert p.candidate_metadata(candidate) == {
        "T1": [{"condition": 1}], "T2": [{"condition": 2}], "T3": [{"condition": 3}], "alpha": [{"condition": 5}]}
    assert instances[0].calls == [1, 2, 3, 5]


def test_external_rng_is_candidate_model_local_and_preserves_training_streams(registered, monkeypatch):
    monkeypatch.setattr("rl_matdesign.predictors.structure_score._PREDICTION_CACHE_SIZE", 1)
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
    cfg = shared_config()
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


def test_dp_identical_inline_models_share_full_vector_inference(monkeypatch):
    import rl_matdesign.registry as registry
    import rl_matdesign.utils.dp_eval as dp
    class Builder:
        def build(self, candidate, **kwargs): return [object(), object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = {"structure": structure_config(2),
           "properties": [{"name": "one", "predictor": "dp_property", "models": ["a", "b"], "output_index": 0, "direction": "max"},
                          {"name": "two", "predictor": "dp_property", "models": ["a", "b"], "output_index": 1, "direction": "max"}],
           "reward": {"objective": {"terms": {"one": 1, "two": 1}}}}
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
    p = scalar_predictor(monkeypatch, folded=(7., .2), models=["a", "b"])
    with pytest.raises(EvaluationProtocolError, match="Multiple self-folding"):
        p.predict({"Fe": 1})


def test_classifier_gate_uses_model_spread_when_reward_selects_config_spread(monkeypatch):
    import rl_matdesign.registry as registry
    import rl_matdesign.utils.dp_eval as dp
    class Builder:
        def build(self, candidate, **kwargs): return [object(), object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    cfg = {"structure": structure_config(2), "properties": [{"name": "pure", "predictor": "dp_classification",
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


@pytest.mark.parametrize("changed", [
    {"models": ["different.ckpt"]}, {"head": "different"},
    {"output_index": 1}, {"adapter_setting": True},
])
def test_inline_sharing_preserves_leaf_constructor_configuration(registered, changed):
    cfg = shared_config()
    cfg["properties"][1].update(changed)
    predictor = StructureScorePredictor(cfg)
    assert len(registered[1]) == 2
    assert predictor.properties[0]["model_key"] != predictor.properties[1]["model_key"]


def test_dp_inline_temperatures_share_loaded_ensemble_and_separate_all_conditions(monkeypatch):
    import sys
    import types
    import rl_matdesign.registry as registry
    import rl_matdesign.utils.dp_eval as dp

    class Builder:
        def build(self, candidate, **kwargs):
            return [object()]
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: Builder())
    loaded = []
    class Model:
        def __init__(self, model_file, **kwargs):
            loaded.append((model_file, kwargs.get("head")))
        def get_type_map(self):
            return ["Fe"]
    for name in ("deepmd", "deepmd.pt", "deepmd.pt.infer", "deepmd.pt.infer.deep_eval"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["deepmd.pt.infer.deep_eval"].DeepProperty = Model
    calls = []
    def evaluate(structures, models, types, *, fparam, aparam):
        calls.append((fparam[0], aparam[0]))
        value = fparam[0] + aparam[0]
        return np.array([[[value, 10 * value]], [[value + 2, 10 * value + 2]]])
    monkeypatch.setattr(dp, "eval_property_outputs", evaluate)
    cfg = {"structure": structure_config(), "properties": [
        {"name": name, "predictor": "dp_property", "models": ["one.pt", "two.pt"],
         "head": "experiment", "direction": "max", "fparam": [temperature],
         "aparam": [atomic], "output_index": output}
        for name, temperature, atomic, output in (
            ("low", 460, 1, 0), ("high", 470, 1, 0),
            ("other_atom", 460, 2, 0), ("other_output", 460, 1, 1))],
        "reward": {"objective": {"terms": {"low": 1, "high": 1, "other_atom": 1, "other_output": 1}}}}
    predictor = StructureScorePredictor(cfg)
    outputs = predictor.predict_outputs({"Fe": 1})
    assert loaded == [("one.pt", "experiment"), ("two.pt", "experiment")]
    assert calls == [(460, 1), (470, 1), (460, 2)]
    assert {name: value.mean for name, value in outputs.items()} == {
        "low": 462, "high": 472, "other_atom": 463, "other_output": 4611}
    predictor.predict_outputs({"Fe": 2})
    assert len(loaded) == 2 and len(calls) == 6


def test_inline_scalar_properties_reuse_physical_call_and_matching_metadata(monkeypatch):
    import rl_matdesign.registry as registry
    calls = []
    class Leaf:
        def score(self, candidate):
            calls.append(deepcopy(candidate))
            return len(calls)
        def candidate_metadata(self, candidate):
            return {"call": len(calls)}
    monkeypatch.setattr(registry, "resolve_predictor", lambda *args, **kwargs: Leaf())
    cfg = {"properties": [
        {"name": "min", "predictor": "test:Scalar", "direction": "min"},
        {"name": "max", "predictor": "test:Scalar", "direction": "max"}],
        "reward": {"objective": {"terms": {"min": 1, "max": 1}}}}
    predictor = StructureScorePredictor(cfg)
    first = predictor.predict_outputs({"Fe": 1})
    assert {name: output.mean for name, output in first.items()} == {"min": 1, "max": 1}
    assert predictor.candidate_metadata({"Fe": 1}) == {
        "min": [{"call": 1}], "max": [{"call": 1}]}
    second = predictor.predict_outputs({"Ni": 1})
    assert {name: output.mean for name, output in second.items()} == {"min": 2, "max": 2}
    assert calls == [{"Fe": 1}, {"Ni": 1}]


@pytest.mark.parametrize("predictor", ["rf_magpie", "mgtransformer", "lookup", "ooh_frames",
                                      "rl_matdesign.predictors.ooh_frames:OOHFramesPredictor"])
def test_checkpoint_leaf_requires_public_models_before_construction(registered, predictor):
    with pytest.raises(EvaluationProtocolError, match="needs models"):
        StructureScorePredictor({"properties": [{"name": "p", "predictor": predictor, "direction": "max"}]})
    assert registered[1] == []


@pytest.mark.parametrize("models", [None, (), ("one.pt",), [], "one.pt", [None]])
def test_public_models_is_one_strict_nonempty_list_format(registered, models):
    with pytest.raises(EvaluationProtocolError, match="models must"):
        StructureScorePredictor({"properties": [{"name": "p", "predictor": "test:Leaf", "direction": "max", "models": models}]})
    assert registered[1] == []
