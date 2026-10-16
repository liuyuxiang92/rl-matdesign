"""Generic metadata, cache, and CSV contracts for direct FQN predictors."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import csv
import json

import pytest

from tests.sequence_fixture import sequence_config


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


@pytest.fixture
def composite():
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    def make(*property_specs, predict_cache_size=200000):
        properties = [{
            "name": "response", "direction": "max", "objective": "mean",
            "predictor": "tests.test_predictor_metadata:InProcessMetadataPredictor",
            **spec,
        } for spec in (property_specs or ({},))]
        return StructureScorePredictor({
            "properties": properties, "predict_cache_size": predict_cache_size,
        })

    return make


def test_composite_forwards_single_leaf_metadata_and_reuses_prediction(composite):
    predictor = composite()
    candidate = {"peptide": ["01", "A", "1"], "n_terminal": {"treatment": "free"}}
    assert predictor.predict(candidate) == (2.5, 0.25)
    leaf = predictor.properties[0]["instance"][0]
    expected = leaf.candidate_metadata(candidate)
    assert predictor.candidate_metadata(candidate) == expected
    assert expected["candidate"] == candidate
    assert expected["calls"] == 1
    assert predictor.predict_raw(candidate) == (2.5, 0.0)
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
    assert predictor.predict(first) == (1.0, 0.0)
    assert predictor.predict(second) == (2.0, 0.0)
    # first is absent from the leaf cache, but its matching stats/metadata remain
    # in the composite cache. Reading metadata also refreshes the shared LRU.
    assert predictor.predict(first) == (1.0, 0.0)
    assert predictor.candidate_metadata(first) == {"raw_mean": 1, "candidate": first}
    assert predictor.predict(third) == (3.0, 0.0)
    assert set(predictor._metadata_cache) == set(predictor._stats_cache)
    assert len(predictor._metadata_cache) == 2
    assert predictor.candidate_metadata(first)["raw_mean"] == 1
    # second was evicted from both composite caches; its metadata must now refer
    # to the new fourth evaluation, never to the old second evaluation.
    assert predictor.predict(second) == (4.0, 0.0)
    assert predictor.candidate_metadata(second) == {"raw_mean": 4, "candidate": second}
    assert predictor.per_objective_stats(second) == {"response": (4.0, 0.0)}
    assert set(predictor._metadata_cache) == set(predictor._stats_cache)
    assert len(predictor._metadata_cache) == 2
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


def _generate_one(predictor, *, max_attempts=2):
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    from rl_matdesign.configuration import build_env
    from rl_matdesign.training import generate_candidates

    class UniformPolicy(torch.nn.Module):
        def forward(self, state, step, element, fraction):
            return torch.zeros((len(state), 1), device=state.device)

    env, _, _ = build_env(sequence_config(), predictor)
    env.initialize()
    scaler = StandardScaler().fit(np.asarray([env.current_state_features()]))
    return generate_candidates(
        env=env, predictor=predictor, scaler=scaler, device=torch.device("cpu"),
        elem_feats_scaled=env.action_features(), fraction_set=env.fraction_set,
        policy=UniformPolicy(), n_exploit=1, n_explore=0,
        gen_temperature=0, max_attempts=max_attempts,
    )


def test_generation_exports_metadata_columns_without_overwriting_reward_and_caches(
    composite, tmp_path,
):
    predictor = composite()
    rows = _generate_one(predictor)
    assert len(rows) == 1
    row = rows[0]
    candidate = json.loads(row["candidate_json"])
    metadata = json.loads(row["metadata_json"])
    assert metadata["candidate"] == candidate
    assert metadata["calls"] == 1
    assert row["meta_calls"] == 1
    assert row["meta_label"] == "测试"
    assert row["meta_valid"] is True
    assert row["meta_optional"] is None
    assert row["meta_formula"] == "metadata formula"
    assert row["formula"] != "metadata formula"
    assert row["meta_reward"] == -999
    assert row["reward"] == row["dp_mean"] == 2.5
    assert row["obj_response_mean"] == 2.5
    assert row["obj_response_std"] == 0.25
    assert "meta_positions" not in row and metadata["positions"] == [2, 4]
    assert "meta_candidate" not in row
    assert predictor.candidate_metadata(candidate)["calls"] == 1

    path = tmp_path / "generated.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    with path.open(newline="", encoding="utf-8") as handle:
        saved = next(csv.DictReader(handle))
    assert json.loads(saved["metadata_json"]) == metadata
    assert saved["meta_calls"] == "1"
    assert saved["meta_label"] == "测试"


@pytest.mark.parametrize("first_failure", ["exception", "invalid_return"])
def test_generation_retries_same_candidate_after_metadata_failure_before_dedup(
    composite, monkeypatch, first_failure,
):
    predictor = composite()
    original = predictor.candidate_metadata
    metadata_attempts = []

    def flaky_metadata(candidate):
        metadata_attempts.append(candidate)
        if len(metadata_attempts) == 1:
            if first_failure == "exception":
                raise RuntimeError("temporary metadata failure")
            return []
        return original(candidate)

    monkeypatch.setattr(predictor, "candidate_metadata", flaky_metadata)
    rows = _generate_one(predictor, max_attempts=2)
    assert len(metadata_attempts) == 2
    assert metadata_attempts[0] == metadata_attempts[1]
    assert len(rows) == 1
    assert json.loads(rows[0]["candidate_json"]) == metadata_attempts[0]
    assert json.loads(rows[0]["metadata_json"])["calls"] == 1
