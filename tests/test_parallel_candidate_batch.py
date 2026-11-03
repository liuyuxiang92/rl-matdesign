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


@pytest.mark.parametrize("workers_per_device,n_candidates", [(1, 2), (2, 4)])
def test_two_configs_per_formula_fill_four_gpus_concurrently(
    pool_factory, monkeypatch, capsys, workers_per_device, n_candidates,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11")
    slots = 4 * workers_per_device
    pool, root = pool_factory(parallel_extra={
        "devices": [0, 1, 2, 3], "workers_per_device": workers_per_device,
    })
    candidates = [{"id": index, "mode": "barrier", "barrier_count": slots}
                  for index in range(n_candidates)]
    result = pool.build_many(candidates, n_configs=2)
    assert [len(atoms) for atoms in result] == [2] * n_candidates
    atoms = [cell for ensemble in result for cell in ensemble]
    assert len({cell.info["pid"] for cell in atoms}) == slots
    assert [cell.info["candidate_id"] for cell in atoms] == [
        index for index in range(n_candidates) for _ in range(2)
    ]
    for device in ("3", "7", "9", "11"):
        assert sum(cell.info["device"] == device for cell in atoms) == workers_per_device
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == slots
    log = capsys.readouterr().out
    assert f"candidate {n_candidates}/{n_candidates} config 2" in log


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


@pytest.mark.parametrize("minimum", [1, 2])
def test_failure_threshold_is_per_formula_and_workers_remain_usable(pool_factory, minimum):
    pool, root = pool_factory(parallel_extra={"min_successful_configs": minimum})
    result = pool.build_many([{"mode": "one_fail"}, {}, {"mode": "all_fail"}], n_configs=2)
    if minimum == 1:
        assert len(result[0]) == 1
        assert result[0][0].info["device"] == "7"
    else:
        assert isinstance(result[0], RuntimeError)
        assert "min_successful_configs=2" in str(result[0])
    assert len(result[1]) == 2
    assert isinstance(result[2], RuntimeError)
    assert "candidate 3/3" in str(result[2])
    original_pids = {row["pid"] for row in _initializations(root)}
    later = pool.build_many([{}, {}], n_configs=2)
    assert [len(atoms) for atoms in later] == [2, 2]
    assert {row["pid"] for row in _initializations(root)} == original_pids


def test_batch_auto_calibration_only_probes_first_formula_and_refines_width(pool_factory):
    pool, root = pool_factory(
        cfg_extra={"test_delays": {"3": .15, "7": .15}},
        parallel_extra={"workers_per_device": "auto", "candidate_batch_size": "auto"},
    )
    assert pool.recommended_candidate_batch_size(2) == 2
    candidates = [{"id": "first"}, {"id": "second"}]
    result = pool.build_many(candidates, n_configs=2, rng=np.random.default_rng(19))
    assert [len(atoms) for atoms in result] == [2, 2]
    assert pool.autotune_report["counts"] == [2, 2]
    calls = [json.loads(line) for line in (root / "build_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 14  # Five probes per GPU plus four production configs.
    assert [row["candidate_id"] for row in calls[:10]] == ["first"] * 10
    assert sum(row["candidate_id"] == "second" for row in calls) == 2
    assert pool.recommended_candidate_batch_size(2) == 2
    expected_seeds = np.random.default_rng(19).integers(0, 2**32, size=4, dtype=np.uint64)
    assert [cell.info["draw"] for ensemble in result for cell in ensemble] == [
        int(np.random.default_rng(int(seed)).integers(0, 2**31)) for seed in expected_seeds
    ]
    pool.build_many(candidates, n_configs=2)
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 18


def test_batch_auto_pair_failure_refines_recommended_width(pool_factory):
    pool, _ = pool_factory(parallel_extra={
        "workers_per_device": "auto", "candidate_batch_size": "auto",
    })
    assert pool.recommended_candidate_batch_size(2) == 2
    result = pool.build_many([{"mode": "pair_fail"}, {}], n_configs=2)
    assert [len(atoms) for atoms in result] == [2, 2]
    assert pool.autotune_report["counts"] == [1, 1]
    assert pool.recommended_candidate_batch_size(2) == 1


def test_bad_calibration_formula_does_not_drop_healthy_formulas(pool_factory, capsys):
    pool, _ = pool_factory(
        cfg_extra={"test_delays": {"3": .15, "7": .15}},
        parallel_extra={"workers_per_device": "auto", "candidate_batch_size": "auto"},
    )
    result = pool.build_many([{"mode": "all_fail"}, {}], n_configs=2)
    assert isinstance(result[0], RuntimeError)
    assert len(result[1]) == 2
    assert pool.autotune_report is None
    assert not pool._tuned
    assert "one slot per GPU" in capsys.readouterr().out
    assert [len(atoms) for atoms in pool.build_many([{}, {}], n_configs=2)] == [2, 2]
    assert pool._tuned


def test_four_gpu_auto_single_formula_does_not_probe_or_start_idle_slots(pool_factory, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11")
    pool, root = pool_factory(parallel_extra={
        "devices": [0, 1, 2, 3], "workers_per_device": "auto", "candidate_batch_size": "auto",
    })
    assert pool.recommended_candidate_batch_size(2) == 4
    assert len(pool.build({}, n_configs=2)) == 2
    assert pool.autotune_report is None
    assert len(_initializations(root)) == 2
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 2
    result = pool.build_many([
        {"mode": "barrier", "barrier_count": 4},
        {"mode": "barrier", "barrier_count": 4},
    ], n_configs=2)
    assert len({cell.info["pid"] for ensemble in result for cell in ensemble}) == 4
    assert pool.autotune_report is None  # Four configs still fit one slot per GPU.
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


@pytest.mark.parametrize("automatic", [False, True])
def test_batch_worker_crash_aborts_pool_and_later_batch_recovers(pool_factory, automatic):
    pool, root = pool_factory(parallel_extra={"workers_per_device": "auto" if automatic else 1})
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        pool.build_many([{"mode": "crash"}, {}, {}], n_configs=1)
    assert not pool._workers
    old_pids = {row["pid"] for row in _initializations(root)}
    _wait_stopped(old_pids)
    result = pool.build_many([{}, {}], n_configs=1)
    assert [len(atoms) for atoms in result] == [1, 1]
    assert not old_pids.intersection(cell.info["pid"] for ensemble in result for cell in ensemble)


@pytest.mark.parametrize("setting,expected", [(None, 1), (1, 1), (7, 7), ("auto", 2)])
def test_candidate_batch_size_is_lazy_and_explicit_values_are_preserved(pool_factory, setting, expected):
    options = {} if setting is None else {"candidate_batch_size": setting}
    pool, root = pool_factory(parallel_extra=options)
    assert pool.recommended_candidate_batch_size(1) == expected
    assert not pool._workers
    assert not root.exists()
    assert pool.build_many([], n_configs=1) == []
    assert not pool._workers


def test_auto_batch_size_rounds_up_to_fill_worker_slots(pool_factory):
    pool, _ = pool_factory(parallel_extra={
        "workers_per_device": 2, "candidate_batch_size": "auto",
    })
    assert pool.recommended_candidate_batch_size(3) == 2
    assert pool.recommended_candidate_batch_size(5) == 1


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


def test_minimum_success_is_not_validated_against_total_batch_size(pool_factory):
    pool, _ = pool_factory(parallel_extra={"min_successful_configs": 2})
    with pytest.raises(ValueError, match="exceeds n_configs"):
        pool.build_many([{}, {}, {}], n_configs=1)
