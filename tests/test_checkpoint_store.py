"""Incremental HPO storage is exact, durable, portable and fail-closed."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zlib

import numpy as np
import pytest
import torch

from rl_matdesign import training
from rl_matdesign.utils import checkpoint_store as store
from rl_matdesign.utils.checkpoint_serialization import save_checkpoint_payload
from tests.test_checkpoint_serialization_performance import _payload, _assert_exact


@pytest.fixture(autouse=True)
def close_writers():
    for writer in store._WRITERS.values():
        writer.close()
    store._WRITERS.clear()
    yield
    for writer in store._WRITERS.values():
        writer.close()
    store._WRITERS.clear()


def payload(rows=12):
    value = _payload(rows)
    value["hpo_state"]["budget"] = {"target_unique": 100, "attempts_started": 12}
    value["training_budget_state"]["seen_keys"] = [("one", 1), ("two", 2)]
    value["dp_cache"] = {("one", 1): {"prediction": np.array([-.0, 3.]), "extra": (True, None)}}
    value["visit_counts"] = {("one", 1): 2, ("two", 2): 1}
    value["predictor_ledger"] = {"seen": ["a", "b"], "phase_costs": {
        "warmup": {"seen": ["a"], "seconds": .25},
        "train": {"seen": ["b"], "seconds": .5}}}
    return value


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    stats = save_checkpoint_payload(str(temporary), value)
    os.replace(temporary, path)
    store.checkpoint_published(path)
    return stats


def database_rows(directory):
    with sqlite3.connect(Path(directory) / store._DATABASE) as connection:
        return connection.execute("SELECT COUNT(*) FROM objects").fetchone()[0]


@pytest.mark.parametrize("kind", ["dqn", "pg"])
def test_complex_hpo_payload_roundtrips_exactly_without_mutation(tmp_path, kind):
    value = payload()
    value["type"] = kind
    original = copy.deepcopy(value)
    path = tmp_path / "checkpoint.pt"
    stats = publish(path, value)
    _assert_exact(original, store.load_checkpoint(path))
    _assert_exact(original, value)
    raw = torch.load(path, weights_only=False)
    assert raw["buffer"] is None and raw["hpo_state"]["seen"] is None
    assert raw[store._MARKER]["database"] == store._DATABASE
    assert stats["checkpoint_new_blob_count"] > 0
    assert stats["checkpoint_new_blob_bytes"] > 0
    assert stats["checkpoint_store_bytes"] == (tmp_path / store._DATABASE).stat().st_size


def test_repeated_save_writes_only_new_rows_and_detects_nested_mutation(tmp_path):
    path, value = tmp_path / "checkpoint.pt", payload()
    first = publish(path, value)
    before_count = database_rows(tmp_path)
    second = publish(path, value)
    assert second["checkpoint_new_blob_count"] == 0
    assert second["checkpoint_new_blob_bytes"] == 0
    assert database_rows(tmp_path) == before_count
    new_row = copy.deepcopy(value["buffer"][-1])
    new_row["reward"] = 91.125
    value["buffer"].append(new_row)
    third = publish(path, value)
    assert third["checkpoint_new_blob_count"] == 1
    assert 0 < third["checkpoint_new_blob_bytes"] < first["checkpoint_new_blob_bytes"]
    value["buffer"][0]["s_raw"][0] += 1.
    fourth = publish(path, value)
    assert fourth["checkpoint_new_blob_count"] == 1
    _assert_exact(value, store.load_checkpoint(path))


def test_atomic_replace_failure_preserves_previous_references_and_recovery(tmp_path, monkeypatch):
    path, old = tmp_path / "checkpoint.pt", payload()
    monkeypatch.setattr(training, "capture_rng_state", lambda: old["rng_state"])
    training._save_checkpoint(str(path), old)
    before_bytes = path.read_bytes()
    updated = copy.deepcopy(old)
    updated["buffer"][0]["s_raw"][0] += 17.
    writer = store._WRITERS[str(tmp_path.resolve())]
    writer.publications = 63  # The failed replacement must not trigger scheduled GC.
    before_publications = writer.publications
    with monkeypatch.context() as patch:
        def refuse_replace(*_args):
            raise OSError("simulated checkpoint publish failure")
        patch.setattr(os, "replace", refuse_replace)
        with pytest.raises(OSError, match="simulated"):
            training._save_checkpoint(str(path), updated)
    assert path.read_bytes() == before_bytes
    assert writer.publications == before_publications
    _assert_exact(old, store.load_checkpoint(path))
    training._save_checkpoint(str(path), updated)
    _assert_exact(updated, store.load_checkpoint(path))


def test_gc_removes_dead_replay_records_without_breaking_latest(tmp_path):
    value, path = payload(rows=1), tmp_path / "checkpoint.pt"
    initial = publish(path, value)
    for index in range(1, 64):
        value["buffer"][0]["reward"] = float(index)
        publish(path, value)
    _assert_exact(value, store.load_checkpoint(path))
    assert database_rows(tmp_path) == initial["checkpoint_new_blob_count"]
    # SQLite freed pages are reused; another replacement must remain readable.
    value["buffer"][0]["reward"] = 300.
    publish(path, value)
    _assert_exact(value, store.load_checkpoint(path))


def test_checkpoint_directory_copy_is_portable_and_relative_paths_work(tmp_path, monkeypatch):
    original, copied = tmp_path / "original", tmp_path / "copied"
    value = payload()
    publish(original / "checkpoint.pt", value)
    shutil.copytree(original, copied)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    _assert_exact(value, store.load_checkpoint(Path("copied/checkpoint.pt")))
    # A relocated checkpoint can continue writing its copied store.
    value["buffer"][0]["reward"] += 1.
    publish(copied / "checkpoint.pt", value)
    _assert_exact(value, store.load_checkpoint(copied / "checkpoint.pt"))


def test_read_only_load_does_not_change_files_or_open_writer(tmp_path):
    value, path = payload(), tmp_path / "checkpoint.pt"
    publish(path, value)
    for writer in store._WRITERS.values():
        writer.close()
    store._WRITERS.clear()
    def snapshot():
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns,
                         hashlib.sha256(p.read_bytes()).hexdigest()) for p in tmp_path.iterdir()}
    before = snapshot()
    _assert_exact(value, store.load_checkpoint(path))
    assert snapshot() == before
    assert not store._WRITERS


@pytest.mark.parametrize("kind,with_budget", [("dqn", False), ("hpo_generation", True), ("other", True)])
def test_old_and_non_hpo_checkpoints_remain_standalone(tmp_path, kind, with_budget):
    value = payload()
    value["type"] = kind
    if not with_budget:
        del value["hpo_state"]["budget"]
    path = tmp_path / "checkpoint.pt"
    stats = publish(path, value)
    assert not (tmp_path / store._DATABASE).exists()
    assert "checkpoint_store_bytes" not in stats
    _assert_exact(value, store.load_checkpoint(path))
    _assert_exact(value, torch.load(path, weights_only=False))


@pytest.mark.parametrize("damage", ["missing_database", "missing_record", "bad_record", "wrong_ids",
                                    "wrong_field", "duplicate_field", "wrong_count"])
def test_corruption_is_rejected(tmp_path, damage):
    path = tmp_path / "checkpoint.pt"
    publish(path, payload())
    raw = torch.load(path, weights_only=False)
    if damage == "missing_database":
        (tmp_path / store._DATABASE).unlink()
    elif damage in {"missing_record", "bad_record"}:
        with sqlite3.connect(tmp_path / store._DATABASE) as connection:
            if damage == "missing_record":
                connection.execute("DELETE FROM objects WHERE id=(SELECT MIN(id) FROM objects)")
            else:
                connection.execute("UPDATE objects SET data=? WHERE id=(SELECT MIN(id) FROM objects)",
                                   (zlib.compress(b"corrupted"),))
    else:
        references = raw[store._MARKER]["references"]
        if damage == "wrong_ids":
            references[0]["sha256"] = "0" * 64
        elif damage == "wrong_field":
            references[0]["path"] = ["qnet_state"]
        elif damage == "duplicate_field":
            references.append(copy.deepcopy(references[0]))
        elif damage == "wrong_count":
            references[0]["count"] += 1
        torch.save(raw, path)
    with pytest.raises(ValueError, match="Checkpoint|checkpoint"):
        store.load_checkpoint(path)


def test_directory_entry_is_synced_before_reclamation(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    publish(path, payload())
    writer = store._WRITERS[str(tmp_path.resolve())]
    events = []
    monkeypatch.setattr(store.os, "fsync", lambda _fd: events.append("fsync"))
    monkeypatch.setattr(writer, "published", lambda: events.append("published"))
    store.checkpoint_published(path)
    assert events == ["fsync", "published"]


def test_database_changes_outside_writer_are_rejected(tmp_path):
    path, value = tmp_path / "checkpoint.pt", payload()
    publish(path, value)
    with sqlite3.connect(tmp_path / store._DATABASE) as connection:
        connection.execute("DELETE FROM objects WHERE id=(SELECT MIN(id) FROM objects)")
    with pytest.raises(ValueError, match="changed outside"):
        publish(path, value)


def test_tensor_and_unknown_object_containers_keep_ordinary_torch_loading(tmp_path):
    value, path = payload(), tmp_path / "checkpoint.pt"
    value["buffer"][0]["tensor"] = torch.tensor([3., 4.])
    value["dp_cache"]["custom"] = {"scaler": value["scaler"]}
    publish(path, value)
    raw = torch.load(path, weights_only=False, map_location="cpu")
    assert isinstance(raw["buffer"], list)
    assert isinstance(raw["dp_cache"], dict)
    assert raw["buffer"][0]["tensor"].device.type == "cpu"
    _assert_exact(value, store.load_checkpoint(path))


def test_content_chunks_reuse_fifo_interior_and_restore_independent_rows(tmp_path):
    value, path = payload(rows=512), tmp_path / "checkpoint.pt"
    first = publish(path, value)
    # Move by seven episode transitions while keeping the large interior fixed.
    appended = copy.deepcopy(value["buffer"][:7])
    for row in appended:
        row["reward"] += 100.
    value["buffer"] = value["buffer"][7:] + appended
    second = publish(path, value)
    assert second["checkpoint_new_blob_count"] < first["checkpoint_new_blob_count"] // 2
    assert second["checkpoint_new_blob_bytes"] < first["checkpoint_new_blob_bytes"] // 2
    _assert_exact(value, store.load_checkpoint(path))
    # Shared mutable row references must not accidentally become shared replay
    # state after loading (the previous column-packed replay had independent rows).
    value["buffer"] = [value["buffer"][0]] * 2
    publish(path, value)
    loaded = store.load_checkpoint(path)
    loaded["buffer"][0]["s_raw"][0] += 100.
    assert loaded["buffer"][1]["s_raw"][0] == value["buffer"][1]["s_raw"][0]


@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_finalized_job_hashes_database_and_cached_preflight_rejects_damage_read_only(
        tmp_path, monkeypatch, damage):
    from rl_matdesign.hpo.artifacts import checkpoint_files, file_digest
    from rl_matdesign.hpo.job_runner import JobRunner
    from rl_matdesign.hpo.restart import inspect_restart
    from tests.test_hpo_restart_preflight import _job, _snapshot

    directory, checkpoint = _job(tmp_path, terminal=True)
    publish(directory / "checkpoint.pt", checkpoint)
    database = directory / store._DATABASE
    assert database.is_file()
    # Exercise the real result writer's artifact enumeration, rather than
    # hand-building a hash map that could hide an omitted sidecar integration.
    runner = JobRunner.__new__(JobRunner)
    runner.out = tmp_path
    runner.plan = {"method": "a2c"}
    key = {"parameters": {}, "budget": 1, "seed": 42, "phase": "train", "input_models": {}}
    execution = {"key": key, "job_id": directory.name, "directory": str(directory),
                 "resume": False, "previous": None, "elapsed": 1.,
                 "recover_terminal": False, "returncode": 0, "argv": ["test-worker"]}
    ledger = json.loads((directory / "candidate_budget.json").read_text())
    result = runner._save_result(execution,
                                {"status": "COMPLETED", "eligible_for_selection": True},
                                ledger, checkpoint_files(directory), include_quality=False)
    assert result["checkpoint_paths"][store._DATABASE] == str(database)
    assert result["artifact_hashes"][str(database)] == file_digest(database)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k:
                        pytest.fail("cached preflight must validate hashes without unpickling"))
    assert inspect_restart(tmp_path)["job_counts"]["CACHED"] == 1
    if damage == "delete":
        database.unlink()
    else:
        database.write_bytes(b"corrupted SQLite backing store")
    before = _snapshot(tmp_path)
    report = inspect_restart(tmp_path)
    assert not report["ready"]
    assert report["job_counts"]["BLOCKED"] == 1
    assert store._DATABASE in report["jobs"][0]["reason"]
    assert _snapshot(tmp_path) == before
