"""Cross-formula queue ordering, utilization, failures and bounded batch sizing."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from tests.test_parallel_structures import (
    _initializations,
    _wait_stopped,
    pool_factory,
)


@pytest.mark.parametrize("device_count,n_configs", [(4, 2), (8, 4)])
def test_two_formulas_fill_all_gpus_with_exactly_one_run_per_config(
    pool_factory, monkeypatch, capsys, device_count, n_configs,
):
    physical = [str(index * 2 + 3) for index in range(device_count)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(physical))
    pool, root = pool_factory(
        cfg_extra={"test_track_execution": True},
        parallel_extra={"devices": list(range(device_count))},
    )
    assert pool.recommended_candidate_batch_size(n_configs) == 2
    candidates = [{"id": index, "mode": "barrier", "barrier_count": device_count}
                  for index in range(2)]
    result = pool.build_many(candidates, n_configs=n_configs)
    assert [len(atoms) for atoms in result] == [n_configs, n_configs]
    atoms = [cell for ensemble in result for cell in ensemble]
    assert len({cell.info["pid"] for cell in atoms}) == device_count
    assert [cell.info["candidate_id"] for cell in atoms] == [0] * n_configs + [1] * n_configs
    assert sorted(cell.info["device"] for cell in atoms) == sorted(physical)
    assert len(_initializations(root)) == device_count
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == device_count
    log = capsys.readouterr().out
    assert f"candidate 2/2 config {n_configs}" in log


def test_free_gpu_takes_next_formula_before_slow_configuration_finishes(pool_factory):
    pool, root = pool_factory(cfg_extra={"test_track_execution": True})
    result = pool.build_many([
        {"id": "first", "wait_for": "second-started", "wait_for_device": "3"},
        {"id": "second", "release": "second-started"},
    ], n_configs=2)
    assert [len(atoms) for atoms in result] == [2, 2]
    events = [json.loads(line) for line in (root / "execution_events.jsonl").read_text().splitlines()]
    second_start = next(i for i, row in enumerate(events)
                        if row["candidate_id"] == "second" and row["event"] == "start")
    slow_end = next(i for i, row in enumerate(events)
                   if row["candidate_id"] == "first" and row["device"] == "3" and row["event"] == "end")
    assert second_start < slow_end
    assert events[second_start]["device"] == "7"
    active = {"3": 0, "7": 0}
    for row in events:
        active[row["device"]] += 1 if row["event"] == "start" else -1
        assert active[row["device"]] in (0, 1)
    assert active == {"3": 0, "7": 0}
    assert len(_initializations(root)) == 2
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 4


def test_batch_seeds_and_output_order_are_candidate_major(pool_factory):
    first, _ = pool_factory(cfg_extra={"test_delays": {"3": .1}})
    second, _ = pool_factory(cfg_extra={"test_delays": {"7": .1}})
    candidates = [{"id": index} for index in range(3)]
    first_rng = np.random.default_rng(77)
    second_rng = np.random.default_rng(77)
    first_result = first.build_many(candidates, n_configs=3, rng=first_rng)
    second_result = [second.build(candidate, n_configs=3, rng=second_rng)
                     for candidate in candidates]
    expected_seeds = np.random.default_rng(77).integers(0, 2**32, size=9, dtype=np.uint64)
    expected_draws = [int(np.random.default_rng(int(seed)).integers(0, 2**31))
                      for seed in expected_seeds]
    for result in (first_result, second_result):
        atoms = [cell for ensemble in result for cell in ensemble]
        assert [cell.info["draw"] for cell in atoms] == expected_draws
        assert [cell.info["candidate_id"] for cell in atoms] == [0] * 3 + [1] * 3 + [2] * 3
    assert first_rng.integers(0, 2**32) == second_rng.integers(0, 2**32)


def test_any_successful_configuration_is_kept_per_formula_and_pool_remains_usable(pool_factory):
    pool, root = pool_factory()
    result = pool.build_many([{"mode": "one_fail"}, {}, {"mode": "all_fail"}], n_configs=2)
    assert len(result[0]) == 1
    assert result[0][0].info["device"] == "7"
    assert len(result[1]) == 2
    assert isinstance(result[2], RuntimeError)
    assert "candidate 3/3" in str(result[2])
    original_pids = {row["pid"] for row in _initializations(root)}
    later = pool.build_many([{}, {}], n_configs=2)
    assert [len(atoms) for atoms in later] == [2, 2]
    assert {row["pid"] for row in _initializations(root)} == original_pids


def test_four_gpu_single_formula_starts_only_needed_workers_then_expands(pool_factory, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11")
    pool, root = pool_factory(parallel_extra={
        "devices": [0, 1, 2, 3], "candidate_batch_size": "auto",
    })
    assert pool.recommended_candidate_batch_size(2) == 2
    assert len(pool.build({}, n_configs=2)) == 2
    assert len(_initializations(root)) == 2
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 2
    result = pool.build_many([
        {"mode": "barrier", "barrier_count": 4},
        {"mode": "barrier", "barrier_count": 4},
    ], n_configs=2)
    assert len({cell.info["pid"] for ensemble in result for cell in ensemble}) == 4
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 6


@pytest.mark.skipif(os.name != "posix", reason="worker process-group cleanup is POSIX")
def test_batch_timeout_stops_all_workers_and_nested_md_process(pool_factory):
    pool, root = pool_factory(parallel_extra={"timeout_s": 2})
    with pytest.raises(TimeoutError, match="candidate 1/2 config 1"):
        pool.build_many([{"mode": "hang"}, {}], n_configs=1)
    child_path = root / "grandchild.pid"
    assert child_path.exists()
    pids = [row["pid"] for row in _initializations(root)] + [int(child_path.read_text())]
    assert not pool._workers
    _wait_stopped(pids)
    heartbeat = (root / "heartbeat").read_text()
    time.sleep(.1)
    assert (root / "heartbeat").read_text() == heartbeat


def test_batch_worker_crash_aborts_pool_and_later_batch_recovers(pool_factory):
    pool, root = pool_factory()
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        pool.build_many([{"mode": "crash"}, {}, {}], n_configs=1)
    assert not pool._workers
    old_pids = {row["pid"] for row in _initializations(root)}
    _wait_stopped(old_pids)
    result = pool.build_many([{}, {}], n_configs=1)
    assert [len(atoms) for atoms in result] == [1, 1]
    assert not old_pids.intersection(cell.info["pid"] for ensemble in result for cell in ensemble)


@pytest.mark.parametrize("setting,expected", [(None, 2), (1, 1), (7, 7), ("auto", 2)])
def test_candidate_batch_size_is_lazy_and_explicit_values_are_preserved(pool_factory, setting, expected):
    options = {} if setting is None else {"candidate_batch_size": setting}
    pool, root = pool_factory(parallel_extra=options)
    assert pool.recommended_candidate_batch_size(1) == expected
    assert not pool._workers
    assert not root.exists()
    assert pool.build_many([], n_configs=1) == []
    assert not pool._workers


def test_auto_batch_size_rounds_up_to_fill_gpu_slots(pool_factory, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11")
    pool, _ = pool_factory(parallel_extra={"devices": [0, 1, 2, 3]})
    assert pool.recommended_candidate_batch_size(3) == 2
    assert pool.recommended_candidate_batch_size(5) == 1


def test_single_gpu_and_single_configuration_are_supported(pool_factory):
    pool, root = pool_factory(parallel_extra={"devices": [0]})
    assert pool.recommended_candidate_batch_size(1) == 1
    assert [len(atoms) for atoms in pool.build_many([{}, {}], n_configs=1)] == [1, 1]
    assert len(_initializations(root)) == 1
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None, [], {}])
def test_invalid_candidate_batch_size_rejected(pool_factory, value):
    with pytest.raises(ValueError, match="candidate_batch_size"):
        pool_factory(parallel_extra={"candidate_batch_size": value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
def test_batch_configuration_count_remains_strict(pool_factory, value):
    pool, _ = pool_factory()
    with pytest.raises(ValueError, match="n_configs"):
        pool.build_many([{}, {}], n_configs=value)
    with pytest.raises(ValueError, match="n_configs"):
        pool.recommended_candidate_batch_size(value)


def test_legacy_success_minimum_does_not_reject_small_ensembles(pool_factory):
    with pytest.warns(DeprecationWarning, match="min_successful_configs"):
        pool, _ = pool_factory(parallel_extra={"min_successful_configs": 5})
    assert [len(atoms) for atoms in pool.build_many([{}, {}, {}], n_configs=1)] == [1, 1, 1]
