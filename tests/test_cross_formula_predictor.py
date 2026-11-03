"""Cross-formula structure evaluation, caching and cost accounting without GPUs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.timing import PredictorTimer


class RecordingBuilder:
    def __init__(self, *, failures=(), batch_size=1):
        self.failures = set(failures)
        self.batch_size = batch_size
        self.calls = []
        self.waves = []

    def build(self, candidate, *, n_configs, rng):
        self.calls.append((candidate, n_configs))
        if candidate["Fe"] in self.failures:
            raise RuntimeError(f"build failed for {candidate['Fe']}")
        return [SimpleNamespace(candidate=candidate, index=i) for i in range(n_configs)]

    def build_many(self, candidates, *, n_configs, rng):
        self.waves.append(list(candidates))
        results = []
        for candidate in candidates:
            try:
                results.append(self.build(candidate, n_configs=n_configs, rng=rng))
            except Exception as exc:
                results.append(exc)
        return results

    def recommended_candidate_batch_size(self, n_configs):
        return self.batch_size


def _config(**overrides):
    cfg = {
        "base_poscar": "not-read-by-fake-builder",
        "n_random_configs": 2,
        "k": 1.0,
        "properties": [
            {"name": "benefit", "predictor": "dp_property", "models": ["unused"],
             "direction": "max", "weight": 2.0, "objective": "mean_minus_kstd"},
            {"name": "cost", "predictor": "dp_property", "models": ["unused"],
             "direction": "min", "scale": 5.0, "objective": "mean"},
        ],
    }
    cfg.update(overrides)
    return cfg


def _predictor(**overrides):
    predictor = StructureScorePredictor(_config(**overrides), seed=17)
    builder = RecordingBuilder()
    if predictor.share_structure:
        predictor._shared_builder = builder
    else:
        predictor._builders = {prop["name"]: RecordingBuilder()
                               for prop in predictor.properties}
    predictor._score = lambda prop, structures: (
        float(structures[0].candidate["Fe"]) * (10 if prop["name"] == "benefit" else 5),
        0.5 if prop["name"] == "benefit" else 0.0,
    )
    return predictor


def _candidate(value):
    return {"Fe": value, "Ni": 1.0 - value}


def test_batch_coalesces_duplicates_and_cache_hits_preserving_input_order():
    predictor = _predictor()
    builder = predictor._shared_builder
    first, second, cached = map(_candidate, [0.1, 0.2, 0.3])
    cached_result = predictor.predict(cached)
    results = predictor.batch_predict([second, cached, first, second.copy()])

    assert results == pytest.approx([(2.8, 0.25), cached_result, (0.9, 0.25), (2.8, 0.25)])
    assert builder.waves == [[second, first]]
    assert builder.calls == [(cached, 2), (second, 2), (first, 2)]
    assert predictor.batch_predict([first, second, cached]) == [results[2], results[0], cached_result]
    assert len(builder.calls) == 3


def test_shared_batch_builds_once_and_passes_same_structures_to_each_property():
    predictor = _predictor()
    seen = []

    def score(prop, structures):
        seen.append((prop["name"], structures))
        return 1.0, 0.0

    predictor._score = score
    candidates = [_candidate(0.1), _candidate(0.2)]
    predictor.batch_predict(candidates)
    assert predictor._shared_builder.waves == [candidates]
    assert [name for name, _ in seen] == ["benefit", "cost", "benefit", "cost"]
    assert seen[0][1] is seen[1][1]
    assert seen[2][1] is seen[3][1]
    assert seen[0][1] is not seen[2][1]


def test_independent_properties_batch_build_with_their_own_config_counts():
    cfg = _config(share_structure=False)
    cfg["properties"][0]["n_random_configs"] = 3
    predictor = _predictor(**cfg)
    candidates = [_candidate(0.1), _candidate(0.2)]
    results = predictor.batch_predict(candidates)
    assert results == pytest.approx([(0.9, 0.25), (2.8, 0.25)])
    for name, count in [("benefit", 3), ("cost", 2)]:
        assert predictor._builders[name].waves == [candidates]
        assert predictor._builders[name].calls == [(c, count) for c in candidates]


@pytest.mark.parametrize("shared", [True, False])
def test_failed_formula_does_not_discard_neighbors_or_enter_cache(shared):
    predictor = _predictor(share_structure=shared)
    bad, good = _candidate(0.1), _candidate(0.2)
    builder = predictor._shared_builder if shared else predictor._builders["benefit"]
    builder.failures.add(0.1)
    results = predictor.batch_predict([bad, good, bad.copy()], return_exceptions=True)

    assert isinstance(results[0], RuntimeError)
    assert results[0] is results[2]
    assert results[1] == pytest.approx((2.8, 0.25))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats
    assert predictor._key(good) in predictor._stats_cache
    if not shared:
        assert predictor._builders["cost"].waves == [[good]]
    builder.failures.clear()
    assert predictor.batch_predict([bad, good]) == pytest.approx([(0.9, 0.25), (2.8, 0.25)])
    assert builder.waves == [[bad, good], [bad]]


def test_scalar_builder_fallback_also_isolates_formula_errors():
    predictor = _predictor()
    builder = RecordingBuilder(failures={0.1})
    predictor._shared_builder = SimpleNamespace(build=builder.build)
    bad, good = _candidate(0.1), _candidate(0.2)
    results = predictor.batch_predict([bad, good], return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert results[1] == pytest.approx((2.8, 0.25))
    assert builder.calls == [(bad, 2), (good, 2)]


def test_scoring_failure_does_not_cache_partial_property_results():
    predictor = _predictor()
    original_score = predictor._score
    bad, good = _candidate(0.1), _candidate(0.2)

    def failing_score(prop, structures):
        if structures[0].candidate == bad and prop["name"] == "cost":
            raise ValueError("property inference failed")
        return original_score(prop, structures)

    predictor._score = failing_score
    results = predictor.batch_predict([bad, good], return_exceptions=True)
    assert isinstance(results[0], ValueError)
    assert results[1] == pytest.approx((2.8, 0.25))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats
    predictor._score = original_score
    assert predictor.batch_predict([bad])[0] == pytest.approx((0.9, 0.25))
    assert predictor._shared_builder.waves == [[bad, good], [bad]]


def test_nonfinite_reward_is_rejected_without_caching_its_statistics():
    predictor = _predictor()
    original_score = predictor._score
    bad, good = _candidate(0.1), _candidate(0.2)
    predictor._score = lambda prop, structures: (
        (float("nan"), 0.0) if structures[0].candidate == bad
        else original_score(prop, structures)
    )
    results = predictor.batch_predict([bad, good], return_exceptions=True)
    assert isinstance(results[0], ValueError)
    assert results[1] == pytest.approx((2.8, 0.25))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats


def test_default_error_mode_raises_but_keeps_successful_neighbor_result():
    predictor = _predictor()
    bad, good = _candidate(0.1), _candidate(0.2)
    predictor._shared_builder.failures.add(0.1)
    with pytest.raises(RuntimeError, match="build failed"):
        predictor.batch_predict([bad, good])
    assert predictor.predict(good) == pytest.approx((2.8, 0.25))
    assert len(predictor._shared_builder.calls) == 2


def test_batch_diagnostics_do_not_rebuild_when_wave_exceeds_lru_capacity():
    predictor = _predictor(predict_cache_size=1)
    candidates = [_candidate(value) for value in [0.1, 0.2, 0.3]]
    results = predictor.batch_predict(candidates)
    assert len(predictor._stats_cache) == 1
    for candidate, result in zip(candidates, results):
        stats = predictor.per_objective_stats(candidate)
        assert predictor.predict(candidate) == result
        assert predictor.predict_raw(candidate)[0] == pytest.approx(predictor.raw_combine(stats))
        assert predictor.reward_details(candidate)["reward"] == pytest.approx(result[0])
    # The next wave can also reuse the previous wave's evicted diagnostics.
    assert predictor.batch_predict(candidates[::-1]) == results[::-1]
    assert len(predictor._shared_builder.calls) == 3


def test_batch_preserves_gated_sweep_choice_for_each_formula():
    cfg = _config(sweep={"name": "temperature", "values": [300, 500]})
    cfg["reward"] = {
        "type": "gated",
        "gates": [{"name": "safe", "score": {"terms": {"cost": 1.0}}, "max": 0.5}],
        "objective": {"terms": {"benefit": 1.0}},
        "failure": {"reward": -10.0},
    }
    for prop in cfg["properties"]:
        prop.update(fparam=[None], objective="mean")
    predictor = _predictor(**cfg)
    calls = []

    def score(prop, structures, *, fparam=None):
        fraction = structures[0].candidate["Fe"]
        temperature = fparam[0]
        calls.append((fraction, prop["name"], temperature))
        if prop["name"] == "benefit":
            return float(temperature) * fraction, 0.0
        return (0.9 if fraction == 0.1 and temperature == 500 else 0.2), 0.0

    predictor._score = score
    first, second = _candidate(0.1), _candidate(0.2)
    assert predictor.batch_predict([first, second]) == [(30.0, 0.0), (100.0, 0.0)]
    assert predictor.per_objective_stats(first)["temperature"] == (300.0, 0.0)
    assert predictor.per_objective_stats(second)["temperature"] == (500.0, 0.0)
    assert predictor.reward_details(first)["gate_passed"] is True
    assert len(calls) == 8
    assert len(predictor._shared_builder.calls) == 2


def test_recommended_wave_size_delegates_to_structure_builders():
    predictor = _predictor()
    assert predictor.evaluation_batch_size() == 1
    predictor._shared_builder.batch_size = 4
    assert predictor.evaluation_batch_size() == 4
    independent = _predictor(share_structure=False)
    independent._builders["benefit"].batch_size = 2
    independent._builders["cost"].batch_size = 3
    assert independent.evaluation_batch_size() == 3


def test_empty_predictor_batch_keeps_previous_wave_diagnostics():
    predictor = _predictor(predict_cache_size=1)
    candidates = [_candidate(0.1), _candidate(0.2)]
    predictor.batch_predict(candidates)
    assert predictor.batch_predict([]) == []
    predictor.per_objective_stats(candidates[0])
    assert predictor._shared_builder.waves == [candidates]
    assert len(predictor._shared_builder.calls) == 2


def test_empty_timer_batch_does_not_call_predictor_or_clock(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("empty batches must not start work")

    timer = PredictorTimer(SimpleNamespace(batch_predict=unexpected), t0=0)
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", unexpected)
    assert timer.batch_predict([]) == []
    assert timer.batch_predict([], return_exceptions=True) == []
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (0, 0, 0.0)


def test_timer_partial_batch_records_successes_and_all_elapsed_cost(monkeypatch):
    first, bad, second = map(_candidate, [0.1, 0.2, 0.3])
    error = RuntimeError("one formula failed")
    seen = []

    def batch(candidates, *, return_exceptions):
        seen.append((candidates, return_exceptions))
        return [(3.0, 0.0), error, (7.0, 0.1), (3.0, 0.0)]

    inner = SimpleNamespace(supports_partial_batch=True, batch_predict=batch)
    timer = PredictorTimer(inner, t0=0)
    ticks = iter([10.0, 22.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = [first, bad, second, first.copy()]
    results = timer.batch_predict(candidates, return_exceptions=True)
    assert seen == [(candidates, True)]
    assert results[1] is error
    assert timer.n_calls == 3
    assert timer.n_unique == 2
    assert timer.best_mean == 7.0
    assert timer.t_predict_s == 12.0


def test_timer_partial_mode_uses_scalar_fallback_for_legacy_batch(monkeypatch):
    calls = []

    def predict(candidate):
        calls.append(candidate)
        if candidate["Fe"] == 0.2:
            raise ValueError("bad formula")
        return candidate["Fe"], 0.0

    def legacy_batch(candidates):
        pytest.fail("legacy batch cannot isolate formula errors")

    timer = PredictorTimer(SimpleNamespace(predict=predict, batch_predict=legacy_batch), t0=0)
    ticks = iter([1.0, 7.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = list(map(_candidate, [0.1, 0.2, 0.3]))
    results = timer.batch_predict(candidates, return_exceptions=True)
    assert calls == candidates
    assert results[0] == (0.1, 0.0)
    assert isinstance(results[1], ValueError)
    assert results[2] == (0.3, 0.0)
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (2, 2, 6.0)


def test_timer_honors_partial_batch_signature_without_capability_marker(monkeypatch):
    calls = []
    failure = RuntimeError("one plugin candidate failed")

    class CustomPlugin:
        def evaluation_batch_size(self):
            return 3

        def batch_predict(self, candidates, *, return_exceptions=False):
            calls.append((candidates, return_exceptions))
            return [(2.0, 0.0), failure, (4.0, 0.1)]

        def predict(self, candidate):
            pytest.fail("partial-capable plugin must retain its batch execution")

    timer = PredictorTimer(CustomPlugin(), t0=0)
    ticks = iter([2.0, 11.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = list(map(_candidate, [0.1, 0.2, 0.3]))
    assert timer.evaluation_batch_size() == 3
    results = timer.batch_predict(candidates, return_exceptions=True)
    assert calls == [(candidates, True)]
    assert results == [(2.0, 0.0), failure, (4.0, 0.1)]
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (2, 2, 9.0)
    assert timer.best_mean == 4.0


def test_timer_batch_failure_records_elapsed_cost_without_observations(monkeypatch):
    def fail(candidates, *, return_exceptions):
        raise TimeoutError("worker pool failed")

    timer = PredictorTimer(SimpleNamespace(supports_partial_batch=True, batch_predict=fail), t0=0)
    ticks = iter([2.0, 7.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    results = timer.batch_predict([_candidate(0.1), _candidate(0.2)], return_exceptions=True)
    assert all(isinstance(result, TimeoutError) for result in results)
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (0, 0, 5.0)
    assert timer.best_mean is None


def test_timer_default_mode_preserves_legacy_batch_api(monkeypatch):
    calls = []

    def batch(candidates):
        calls.append(candidates)
        return [(1.0, 0.0)] * len(candidates)

    timer = PredictorTimer(SimpleNamespace(batch_predict=batch), t0=0)
    ticks = iter([1.0, 5.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = [_candidate(0.1), _candidate(0.2)]
    assert timer.batch_predict(candidates) == [(1.0, 0.0), (1.0, 0.0)]
    assert calls == [candidates]
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (2, 2, 4.0)
