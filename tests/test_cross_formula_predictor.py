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

    def build_many_seeded(self, candidates, *, n_configs, source_spec, eval_seed):
        import numpy as np
        self.waves.append(list(candidates))
        results = []
        for candidate in candidates:
            try:
                results.append(self.build(candidate, n_configs=n_configs, rng=np.random.default_rng(
                    source_spec.seed_for(candidate, eval_seed=eval_seed))))
            except Exception as exc:
                results.append(exc)
        return results

    def recommended_candidate_batch_size(self, n_configs):
        return self.batch_size


def _config(**overrides):
    cfg = {
        "structure": {"builder": "pipeline", "base_poscar": "not-read-by-fake-builder", "n_random_configs": 2,
                      "steps": [{"type": "substitute", "a_site_symbol": "Fe", "a_dopant_group": "alloy"}]},
        "k": 1.0,
        "properties": [
            {"name": "benefit", "predictor": "dp_property", "models": ["unused"],
             "direction": "max", "objective": "mean_minus_kstd"},
            {"name": "cost", "predictor": "dp_property", "models": ["unused"],
             "direction": "min", "scale": 5.0, "objective": "mean"},
        ],
        "reward": {"objective": {"terms": {"benefit": 2.0, "cost": 1.0}}},
    }
    cfg.update(overrides)
    return cfg


def _predictor(**overrides):
    cfg = _config(**overrides)
    predictor = StructureScorePredictor(cfg, seed=17)
    predictor._source_builders["structure"] = RecordingBuilder()
    predictor._score = lambda prop, structures: (
        float(structures[0].candidate["Fe"]) * (10 if prop["name"] == "benefit" else 5),
        0.5 if prop["name"] == "benefit" else 0.0,
    )
    return predictor


def _candidate(value):
    return {"Fe": value, "Ni": 1.0 - value}


def test_batch_coalesces_duplicates_and_cache_hits_preserving_input_order():
    predictor = _predictor()
    builder = predictor._source_builders["structure"]
    first, second, cached = map(_candidate, [0.1, 0.2, 0.3])
    cached_result = predictor.predict(cached)
    results = predictor.batch_predict([second, cached, first, second.copy()])

    assert results == pytest.approx([(2.8, None), cached_result, (0.9, None), (2.8, None)])
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
    assert predictor._source_builders["structure"].waves == [candidates]
    assert [name for name, _ in seen] == ["benefit", "cost", "benefit", "cost"]
    assert seen[0][1] is seen[1][1]
    assert seen[2][1] is seen[3][1]
    assert seen[0][1] is not seen[2][1]


def test_properties_cannot_override_the_shared_sample_count():
    cfg = _config()
    cfg["properties"][0]["n_random_configs"] = 3
    with pytest.raises(ValueError, match="structure-owned"):
        StructureScorePredictor(cfg, seed=17)


def test_failed_formula_does_not_discard_neighbors_or_enter_cache():
    predictor = _predictor()
    bad, good = _candidate(0.1), _candidate(0.2)
    builder = predictor._source_builders["structure"]
    builder.failures.add(0.1)
    results = predictor.batch_predict([bad, good, bad.copy()], return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert results[0] is results[2]
    assert results[1] == pytest.approx((2.8, None))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats
    assert predictor._key(good) in predictor._stats_cache
    builder.failures.clear()
    assert predictor.batch_predict([bad, good]) == pytest.approx([(0.9, None), (2.8, None)])
    assert builder.waves == [[bad, good], [bad]]


def test_scalar_builder_fallback_also_isolates_formula_errors():
    predictor = _predictor()
    builder = RecordingBuilder(failures={0.1})
    predictor._source_builders["structure"] = SimpleNamespace(build=builder.build)
    bad, good = _candidate(0.1), _candidate(0.2)
    results = predictor.batch_predict([bad, good], return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert results[1] == pytest.approx((2.8, None))
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
    assert results[1] == pytest.approx((2.8, None))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats
    predictor._score = original_score
    assert predictor.batch_predict([bad])[0] == pytest.approx((0.9, None))
    assert predictor._source_builders["structure"].waves == [[bad, good], [bad]]


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
    assert results[1] == pytest.approx((2.8, None))
    assert predictor._key(bad) not in predictor._stats_cache
    assert predictor._key(bad) not in predictor._batch_stats


def test_default_error_mode_raises_but_keeps_successful_neighbor_result():
    predictor = _predictor()
    bad, good = _candidate(0.1), _candidate(0.2)
    predictor._source_builders["structure"].failures.add(0.1)
    with pytest.raises(RuntimeError, match="build failed"):
        predictor.batch_predict([bad, good])
    assert predictor.predict(good) == pytest.approx((2.8, None))
    assert len(predictor._source_builders["structure"].calls) == 2


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
    assert len(predictor._source_builders["structure"].calls) == 3


def test_batch_preserves_explicit_condition_outputs_and_reward_choice():
    cfg = _config()
    cfg["properties"] = [
        {"name": f"{name}_T{temperature}", "predictor": "dp_property", "models": ["unused"],
         "direction": "max", "objective": "mean", "fparam": [temperature]}
        for temperature in (300, 500) for name in ("benefit", "cost")]
    cfg["reward"] = {"type": "max", "alternatives": [
        {"name": f"T{temperature}", "terms": {f"benefit_T{temperature}": 1., f"cost_T{temperature}": -1.}}
        for temperature in (300, 500)]}
    predictor = _predictor(**cfg)
    calls = []
    def score(prop, structures):
        fraction = structures[0].candidate["Fe"]
        temperature = prop["fparam"][0]
        calls.append((fraction, prop["name"], temperature))
        if prop["name"].startswith("benefit"):
            return float(temperature) * fraction, 0.0
        return (200. if fraction == 0.1 and temperature == 500 else 0.), 0.0
    predictor._score = score
    first, second = _candidate(0.1), _candidate(0.2)
    assert predictor.batch_predict([first, second]) == [(30.0, None), (100.0, None)]
    assert len(predictor.per_objective_stats(first)) == 4
    assert len(predictor.per_objective_stats(second)) == 4
    assert predictor.reward_details(first)["selected_alternative"] == "T300"
    assert predictor.reward_details(second)["selected_alternative"] == "T500"
    assert len(calls) == 8
    assert len(predictor._source_builders["structure"].calls) == 2


def test_recommended_wave_size_delegates_to_structure_builder():
    predictor = _predictor()
    assert predictor.evaluation_batch_size() == 1
    predictor._source_builders["structure"].batch_size = 4
    assert predictor.evaluation_batch_size() == 4


def test_empty_predictor_batch_keeps_previous_wave_diagnostics():
    predictor = _predictor(predict_cache_size=1)
    candidates = [_candidate(0.1), _candidate(0.2)]
    predictor.batch_predict(candidates)
    assert predictor.batch_predict([]) == []
    predictor.per_objective_stats(candidates[0])
    assert predictor._source_builders["structure"].waves == [candidates]
    assert len(predictor._source_builders["structure"].calls) == 2


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

    inner = SimpleNamespace(candidate_seeded_batch=True, supports_partial_batch=True, batch_predict=batch)
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
    ticks = iter([1.0, 3.0, 3.0, 5.0, 5.0, 7.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = list(map(_candidate, [0.1, 0.2, 0.3]))
    results = timer.batch_predict(candidates, return_exceptions=True)
    assert calls == candidates
    assert results[0] == (0.1, 0.0)
    assert isinstance(results[1], ValueError)
    assert results[2] == (0.3, 0.0)
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (2, 2, 6.0)


def test_seeded_timer_honors_partial_batch_signature_without_partial_marker(monkeypatch):
    calls = []
    failure = RuntimeError("one plugin candidate failed")

    class CustomPlugin:
        candidate_seeded_batch = True

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

    timer = PredictorTimer(SimpleNamespace(candidate_seeded_batch=True, supports_partial_batch=True, batch_predict=fail), t0=0)
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

    timer = PredictorTimer(SimpleNamespace(candidate_seeded_batch=True, batch_predict=batch), t0=0)
    ticks = iter([1.0, 5.0])
    monkeypatch.setattr("rl_matdesign.utils.timing.time.perf_counter", lambda: next(ticks))
    candidates = [_candidate(0.1), _candidate(0.2)]
    assert timer.batch_predict(candidates) == [(1.0, 0.0), (1.0, 0.0)]
    assert calls == [candidates]
    assert (timer.n_calls, timer.n_unique, timer.t_predict_s) == (2, 2, 4.0)
