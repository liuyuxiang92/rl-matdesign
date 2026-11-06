"""Generic metadata, cache, and CSV contracts for direct FQN predictors."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json

import pytest


class InProcessMetadataPredictor:
    """Small stateful leaf used to expose mismatched composite cache lifetimes."""

    def __init__(self, cfg, *, seed=None):
        self.cfg = dict(cfg)
        self.calls = 0
        self.cache_size = int(cfg.get("cache_size", 200000))
        self.cache = OrderedDict()

    def _evaluate(self, candidate):
        key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self.calls += 1
        if self.cfg.get("counter", False):
            mean, std = float(self.calls), 0.0
            metadata = {"raw_mean": self.calls, "candidate": deepcopy(candidate)}
        else:
            mean, std = float(self.cfg.get("mean", 2.5)), float(self.cfg.get("std", 0.25))
            metadata = {
                "candidate": deepcopy(candidate), "calls": self.calls,
                "label": "测试", "valid": True, "optional": None,
                "formula": "metadata formula", "reward": -999, "positions": [2, 4],
                **deepcopy(self.cfg.get("metadata", {})),
            }
        self.cache[key] = mean, std, metadata
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return mean, std, metadata

    def predict(self, candidate):
        mean, std, _ = self._evaluate(candidate)
        return mean, std

    def candidate_metadata(self, candidate):
        return deepcopy(self._evaluate(candidate)[2])


class SampledMetadataPredictor(InProcessMetadataPredictor):
    """One model's scalar measurement for metadata ensemble ownership tests."""

    def score(self, candidate):
        return self._evaluate(candidate)[0]


@pytest.fixture
def composite():
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    def make(*property_specs, predict_cache_size=200000):
        properties = [{
            "name": "response", "direction": "max", "objective": "mean",
            "predictor": ("tests.test_predictor_metadata:SampledMetadataPredictor"
                          if isinstance(spec.get("model"), list)
                          else "tests.test_predictor_metadata:InProcessMetadataPredictor"),
            **spec,
        } for spec in (property_specs or ({},))]
        return StructureScorePredictor({
            "properties": properties, "predict_cache_size": predict_cache_size,
        })

    return make


def test_composite_forwards_single_leaf_metadata_and_reuses_prediction(composite):
    predictor = composite()
    candidate = {"peptide": ["01", "A", "1"], "n_terminal": {"treatment": "free"}}
    assert predictor.predict(candidate) == (2.5, None)
    leaf = predictor.properties[0]["instance"][0]
    expected = leaf.candidate_metadata(candidate)
    assert predictor.candidate_metadata(candidate) == expected
    assert expected["candidate"] == candidate
    assert expected["calls"] == 1
    assert predictor.predict_raw(candidate) == (2.5, None)
    assert predictor.per_objective_stats(candidate) == {"response": (2.5, 0.25)}
    assert predictor.candidate_metadata(candidate)["calls"] == 1


@pytest.mark.parametrize("property_specs,expected_sizes", [
    (({"name": "binding"}, {"name": "stability"}), {"binding": 1, "stability": 1}),
    (({"name": "binding", "model": ["first", "second"]},),
     {"binding": 2}),
    (({"name": "binding", "model": ["first", "second"]}, {"name": "stability"}),
     {"binding": 2, "stability": 1}),
    (({"name": "binding", "model": ["first", "second"]}, {"name": "binding[0]"}),
     {"binding": 2, "binding[0]": 1}),
])
def test_composite_namespaces_multiple_objectives_and_models(
    composite, property_specs, expected_sizes,
):
    predictor = composite(*property_specs)
    candidate = {"peptide": ["C", "A", "C"]}
    # Request metadata first: subsequent scoring must reuse those same results.
    metadata = predictor.candidate_metadata(candidate)
    assert set(metadata) == set(expected_sizes)
    assert all(isinstance(models, list) for models in metadata.values())
    assert {name: len(models) for name, models in metadata.items()} == expected_sizes
    leaves = [fields for models in metadata.values() for fields in models]
    assert all(fields["candidate"] == candidate for fields in leaves)
    assert all(fields["calls"] == 1 for fields in leaves)
    predictor.predict(candidate)
    assert predictor.candidate_metadata(candidate) == metadata


def test_composite_keeps_matching_metadata_when_leaf_cache_evicts_and_evicts_together(composite):
    # Different evaluations deliberately return different distributions. A leaf
    # with a smaller cache must not replace metadata held with composite stats.
    predictor = composite({"cache_size": 1, "counter": True}, predict_cache_size=2)
    first, second, third = ({"peptide": [token]} for token in ("A", "B", "C"))
    assert predictor.predict(first) == (1.0, None)
    assert predictor.predict(second) == (2.0, None)
    # first is absent from the leaf cache, but its matching stats/metadata remain
    # in the composite cache. Reading metadata also refreshes the shared LRU.
    assert predictor.predict(first) == (1.0, None)
    assert predictor.candidate_metadata(first) == {"raw_mean": 1, "candidate": first}
    assert predictor.predict(third) == (3.0, None)
    assert predictor.candidate_metadata(first)["raw_mean"] == 1
    # second was evicted from both composite caches; its metadata must now refer
    # to the new fourth evaluation, never to the old second evaluation.
    assert predictor.predict(second) == (4.0, None)
    assert predictor.candidate_metadata(second) == {"raw_mean": 4, "candidate": second}
    assert predictor.per_objective_stats(second) == {"response": (4.0, 0.0)}
    assert predictor.candidate_metadata(first)["raw_mean"] == 1


@pytest.mark.parametrize("property_specs", [
    ({},),
    ({"name": "binding", "model": ["first", "second"]}, {"name": "stability"}),
])
def test_composite_metadata_defensive_copy_on_capture_and_return(
    composite, monkeypatch, property_specs,
):
    predictor = composite(*property_specs)
    shared_metadata = {"nested": {"values": ["original"]}}
    for prop in predictor.properties:
        for leaf in prop["instance"]:
            monkeypatch.setattr(leaf, "candidate_metadata", lambda _: shared_metadata)
    candidate = {"peptide": ["A"]}
    predictor.predict(candidate)
    captured = predictor.candidate_metadata(candidate)
    expected = json.loads(json.dumps(captured))
    shared_metadata["nested"]["values"].append("mutated by leaf")
    assert predictor.candidate_metadata(candidate) == expected
    if "nested" in captured:
        captured["nested"]["values"].append("mutated by caller")
    else:
        captured["binding"][0]["nested"]["values"].append("mutated by caller")
    assert predictor.candidate_metadata(candidate) == expected


def test_batch_metadata_survives_leaf_and_lru_eviction_and_duplicate_reads(composite):
    predictor = composite({"cache_size": 1, "counter": True}, predict_cache_size=1)
    first, second, third = ({"peptide": [token]} for token in ("A", "B", "C"))
    predictions = predictor.batch_predict([first, second, first.copy(), third])
    assert predictions == [(1., None), (2., None), (1., None), (3., None)]
    assert len(predictor._stats_cache) == 1
    leaf = predictor.properties[0]["instance"][0]
    for candidate, expected in [(second, 2), (first, 1), (third, 3), (first, 1)]:
        assert predictor.predict_raw(candidate) == (float(expected), None)
        assert predictor.per_objective_stats(candidate)["response"] == (float(expected), 0.)
        assert predictor.candidate_metadata(candidate) == {"raw_mean": expected, "candidate": candidate}
    assert leaf.calls == 3
    # Starting another wave can promote a previous-wave record without asking a
    # leaf whose private LRU has already forgotten it.
    assert predictor.batch_predict([second, first]) == [(2., None), (1., None)]
    assert predictor.candidate_metadata(first)["raw_mean"] == 1
    assert leaf.calls == 3


def test_failed_metadata_is_not_cached_and_good_batch_neighbors_survive(composite, monkeypatch):
    predictor = composite({"cache_size": 1, "counter": True}, predict_cache_size=1)
    leaf = predictor.properties[0]["instance"][0]
    original = leaf.candidate_metadata
    bad, good = {"peptide": ["A"]}, {"peptide": ["B"]}

    def metadata(candidate):
        if candidate == bad:
            raise RuntimeError("metadata unavailable")
        return original(candidate)

    monkeypatch.setattr(leaf, "candidate_metadata", metadata)
    results = predictor.batch_predict([bad, good, bad], return_exceptions=True)
    assert isinstance(results[0], RuntimeError) and results[0] is results[2]
    assert results[1] == (2., None)
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats
    assert predictor.candidate_metadata(good)["raw_mean"] == 2
    monkeypatch.setattr(leaf, "candidate_metadata", original)
    assert predictor.predict(bad) == (3., None)
    assert predictor.candidate_metadata(bad)["raw_mean"] == 3


def test_sweep_record_keeps_metadata_from_selected_trial(monkeypatch):
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    predictor = StructureScorePredictor({
        "sweep": {"name": "temperature", "values": [200, 300, 400]},
        "properties": [{"name": "activity", "direction": "max", "objective": "mean",
                        "predictor": "tests.test_predictor_metadata:InProcessMetadataPredictor",
                        "fparam": [None]}],
    })
    current = {}

    def score(prop, payload, *, fparam=None):
        current["temperature"] = fparam[0]
        return -abs(fparam[0] - 300), 0.

    monkeypatch.setattr(predictor, "_score", score)
    leaf = predictor.properties[0]["instance"][0]
    monkeypatch.setattr(leaf, "candidate_metadata", lambda _: current)
    candidate = {"peptide": ["A"]}
    assert predictor.predict(candidate) == (0., None)
    assert current["temperature"] == 400
    assert predictor.per_objective_stats(candidate)["temperature"] == (300., None)
    assert predictor.candidate_metadata(candidate) == {"temperature": 300}


def test_sequence_constraints_apply_all_leaves_and_copy_plugin_data(composite, monkeypatch):
    predictor = composite({"name": "a", "model": ["one", "two"]}, {"name": "b"})
    bounds = [{"tokens": ["C"], "min": 2, "max": 2}]
    calls = []

    def hook(name, tokens):
        calls.append((name, list(tokens)))
        tokens.append("plugin mutation")
        return bounds

    for prop in predictor.properties:
        for leaf in prop["instance"]:
            monkeypatch.setattr(leaf, "sequence_constraints", hook, raising=False)
    tokens = ["A", "C", "parent-only"]
    actual = predictor.sequence_constraints("peptide", tokens)
    assert actual == bounds
    assert calls == [("peptide", tokens)] * 3
    bounds[0]["tokens"].append("plugin mutation")
    assert actual == [{"tokens": ["C"], "min": 2, "max": 2}]


@pytest.mark.parametrize("invalid", [None, {}, ["C"]])
def test_invalid_sequence_constraint_hook_fails_explicitly(composite, monkeypatch, invalid):
    predictor = composite()
    leaf = predictor.properties[0]["instance"][0]
    monkeypatch.setattr(leaf, "sequence_constraints", lambda *_: invalid, raising=False)
    with pytest.raises(ValueError, match="count-constraint mappings"):
        predictor.sequence_constraints("peptide", ["A", "C"])
