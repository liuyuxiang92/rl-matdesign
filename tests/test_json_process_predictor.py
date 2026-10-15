"""The predictor subprocess protocol needs no chemistry/model dependencies."""
from __future__ import annotations

import csv
import json
import sys
import textwrap

import pytest

from rl_matdesign.predictors.json_process import JsonProcessPredictor
from tests.sequence_fixture import sequence_config


@pytest.fixture
def worker(tmp_path):
    instances = []

    def start(body, **cfg):
        path = tmp_path / f"worker_{len(instances)}.py"
        path.write_text("import json, sys, time, os\n" + textwrap.dedent(body))
        instance = JsonProcessPredictor({
            "command": [sys.executable, "-u", str(path)],
            "startup_timeout": 5, "request_timeout": 5, **cfg,
        })
        instances.append(instance)
        return instance

    yield start
    for instance in instances:
        instance.close()


ECHO_WORKER = '''
print(json.dumps({"status": "ready", "metadata": {"model": "toy", "cwd": os.getcwd()}}))
calls = 0
for line in sys.stdin:
    request = json.loads(line)
    assert request["op"] == "predict"
    calls += 1
    print(json.dumps({"mean": 2.5, "std": 0.25, "metadata": {
        "candidate": request["candidate"], "calls": calls}}))
'''


def test_nested_order_and_identifiers_roundtrip_with_one_cached_prediction(worker, tmp_path):
    predictor = worker(ECHO_WORKER, cwd=str(tmp_path))
    candidate = {
        "sequence": ["01", "B", "1"], "n_terminal": "01",
        "marks": {"bridge": [1, 3]}, "nested": [None, True, 0.125],
    }
    assert predictor.startup_metadata == {"model": "toy", "cwd": str(tmp_path)}
    assert predictor.predict(candidate) == (2.5, 0.25)
    metadata = predictor.candidate_metadata(candidate)
    assert metadata == {"candidate": candidate, "calls": 1}
    metadata["candidate"]["sequence"].reverse()
    assert predictor.candidate_metadata(candidate)["candidate"] == candidate
    reversed_candidate = {**candidate, "sequence": list(reversed(candidate["sequence"]))}
    assert predictor.candidate_metadata(reversed_candidate)["calls"] == 2
    assert predictor.batch_predict([candidate, reversed_candidate]) == [(2.5, 0.25)] * 2
    predictor.close()
    assert predictor._process.poll() == 0
    predictor.close()
    with pytest.raises(RuntimeError, match="closed"):
        predictor.predict(candidate)


def test_cache_is_typed_and_lru_bounded(worker):
    predictor = worker(ECHO_WORKER, cache_size=2)
    first, second, third = {"id": "01"}, {"id": "1"}, {"id": 1}
    assert predictor.candidate_metadata(first)["calls"] == 1
    assert predictor.candidate_metadata(second)["calls"] == 2
    assert predictor.candidate_metadata(first)["calls"] == 1
    assert predictor.candidate_metadata(third)["calls"] == 3
    assert len(predictor._cache) == 2
    assert predictor.candidate_metadata(second)["calls"] == 4


@pytest.mark.parametrize("body,exception,match", [
    ('print("not json")\ntime.sleep(30)', RuntimeError, "not valid JSON"),
    ('print("[]")\ntime.sleep(30)', RuntimeError, "JSON object"),
    ('print(json.dumps({"status": "failed"}))\ntime.sleep(30)', RuntimeError, "startup failed"),
    ('print(json.dumps({"status": "ready", "metadata": []}))\ntime.sleep(30)', RuntimeError, "metadata"),
    ('sys.exit(7)', RuntimeError, "exited"),
    ('time.sleep(30)', TimeoutError, "startup timed out"),
])
def test_startup_failures_reap_worker(worker, monkeypatch, body, exception, match):
    import rl_matdesign.predictors.json_process as module

    spawned = []
    popen = module.subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", capture)
    with pytest.raises(exception, match=match):
        worker(body, startup_timeout=0.15 if exception is TimeoutError else 5)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert spawned[0].stdin.closed
    assert spawned[0].stdout.closed


@pytest.mark.parametrize("response,match", [
    ("not json", "not valid JSON"),
    ("[]", "JSON object"),
    ('{"mean": NaN, "std": 0}', "not valid JSON"),
    ('{"mean": Infinity, "std": 0}', "not valid JSON"),
    ('{"mean": 1e999, "std": 0}', "finite"),
    ('{"mean": 1, "std": -1}', "std >= 0"),
    ('{"mean": true, "std": 0}', "numeric scalars"),
    ('{"mean": "1", "std": 0}', "numeric scalars"),
    ('{"mean": [1], "std": 0}', "numeric scalars"),
    ('{"mean": 1}', "numeric scalars"),
    ('{"mean": 1, "std": 0, "metadata": []}', "metadata"),
])
def test_invalid_prediction_response_closes_worker(worker, response, match):
    predictor = worker(f'''
        print('{{"status": "ready"}}')
        for line in sys.stdin:
            print({response!r})
    ''')
    with pytest.raises(RuntimeError, match=match):
        predictor.predict({"sequence": ["A"]})
    assert predictor._process.poll() is not None


def test_worker_error_is_reported_and_next_request_still_matches(worker):
    predictor = worker('''
        print('{"status": "ready"}')
        for line in sys.stdin:
            request = json.loads(line)
            if request["candidate"] == "bad":
                print('{"error": "unsupported candidate"}')
            else:
                print('{"mean": 1, "std": 0}')
    ''')
    with pytest.raises(RuntimeError, match="unsupported candidate"):
        predictor.predict("bad")
    assert predictor.predict("good") == (1.0, 0.0)


@pytest.mark.parametrize("behavior,exception,match", [
    ("sys.exit(4)", RuntimeError, "exited"),
    ("time.sleep(30)", TimeoutError, "prediction timed out"),
])
def test_request_exit_and_timeout_reap_worker(worker, behavior, exception, match):
    predictor = worker(f'''
        print('{{"status": "ready"}}')
        sys.stdin.readline()
        {behavior}
    ''', request_timeout=0.15)
    with pytest.raises(exception, match=match):
        predictor.predict({"sequence": ["A"]})
    assert predictor._process.poll() is not None


def test_stalled_stdin_write_has_deadline(worker):
    predictor = worker('''
        print('{"status": "ready"}')
        time.sleep(30)
    ''', request_timeout=0.15)
    with pytest.raises(TimeoutError, match="request write timed out"):
        predictor.predict({"sequence": ["A" * 2_000_000]})
    assert predictor._process.poll() is not None


def test_close_kills_worker_ignoring_stdin_eof(worker):
    predictor = worker('''
        print('{"status": "ready"}')
        sys.stdin.read()
        time.sleep(30)
    ''')
    predictor.close()
    assert predictor._process.poll() is not None


def test_context_manager_closes(worker):
    with worker(ECHO_WORKER) as predictor:
        assert predictor.predict(["A"]) == (2.5, 0.25)
    assert predictor._process.poll() == 0


@pytest.mark.parametrize("cfg", [
    {}, {"command": "python worker.py"}, {"command": []},
    {"command": [sys.executable, 1]}, {"command": [""]},
    {"command": [sys.executable], "startup_timeout": 0},
    {"command": [sys.executable], "request_timeout": float("inf")},
    {"command": [sys.executable], "request_timeout": True},
    {"command": [sys.executable], "cache_size": 0},
    {"command": [sys.executable], "cache_size": 1.5},
])
def test_config_validation(cfg):
    with pytest.raises(ValueError):
        JsonProcessPredictor(cfg)


METADATA_WORKER = '''
print('{"status": "ready"}')
calls = 0
for line in sys.stdin:
    candidate = json.loads(line)["candidate"]
    calls += 1
    print(json.dumps({"mean": 2.5, "std": 0.25, "metadata": {
        "candidate": candidate, "calls": calls, "label": "测试", "valid": True,
        "optional": None, "formula": "metadata formula", "reward": -999,
        "positions": [2, 4]}}))
'''


@pytest.fixture
def composite(tmp_path):
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    instances = []

    def make(*property_specs, predict_cache_size=200000, worker_body=METADATA_WORKER):
        path = tmp_path / f"metadata_worker_{len(instances)}.py"
        path.write_text("import json, sys\n" + worker_body, encoding="utf-8")
        properties = [{
            "name": "response", "direction": "max", "objective": "mean",
            "predictor": "rl_matdesign.predictors.json_process:JsonProcessPredictor",
            "command": [sys.executable, "-u", str(path)],
            "startup_timeout": 5, "request_timeout": 5,
            **spec,
        } for spec in (property_specs or ({},))]
        predictor = StructureScorePredictor({
            "properties": properties, "predict_cache_size": predict_cache_size,
        })
        instances.append(predictor)
        return predictor

    yield make
    for predictor in instances:
        predictor.close()


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
    # Each process evaluation deliberately changes the distribution, as a
    # stochastic predictor may do. Divergent cache lifetimes must not mix calls.
    counter_worker = '''
print('{"status": "ready"}')
calls = 0
for line in sys.stdin:
    candidate = json.loads(line)["candidate"]
    calls += 1
    print(json.dumps({"mean": calls, "std": 0, "metadata": {
        "raw_mean": calls, "candidate": candidate}}))
'''
    predictor = composite({"cache_size": 1}, predict_cache_size=2, worker_body=counter_worker)
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
