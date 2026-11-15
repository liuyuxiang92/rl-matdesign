"""Checkpoint accounting preserves legacy integrity without rereading history."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.candidate_records import CandidateRecorder
from rl_matdesign.utils.timing import PredictorTimer


def _success(recorder, name):
    recorder.start()
    return recorder.success(SimpleNamespace(
        terminal_key=((name, 1),), formula=name,
        prediction_record={"dp_mean": 2.0, "label": "玻璃"}))


def _forbid_archive_reads(monkeypatch, archive):
    original = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        if path == archive and "r" in mode:
            raise AssertionError("A live checkpoint must not reread candidate history")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)


def test_repeated_checkpoints_hash_only_durable_appends(tmp_path, monkeypatch):
    recorder = CandidateRecorder(tmp_path, 10, 20.0)
    _success(recorder, "SiO₂")
    expected = hashlib.sha256(recorder.records_path.read_bytes()).hexdigest()
    with monkeypatch.context() as patch:
        _forbid_archive_reads(patch, recorder.records_path)
        assert recorder.checkpoint_state()["records_sha256"] == expected
        assert not _success(recorder, "SiO₂")
        assert recorder.checkpoint_state()["records_sha256"] == expected
        assert _success(recorder, "B₂O₃")
        state = recorder.checkpoint_state()
        assert recorder.checkpoint_state() == state
    assert state["records_sha256"] == hashlib.sha256(recorder.records_path.read_bytes()).hexdigest()
    assert state["budget"]["attempts_started"] == 3
    assert state["budget"]["unique_evaluated"] == 2


def test_legacy_digest_verified_once_then_extended_after_resume(tmp_path, monkeypatch):
    recorder = CandidateRecorder(tmp_path, 10, 20.0)
    _success(recorder, "SiO₂")
    legacy = recorder.checkpoint_state()
    recorder.close()
    # Legacy archives may have native CRLF bytes. Keep them byte-for-byte,
    # while the resumed recorder appends according to its current platform.
    old_bytes = recorder.records_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    recorder.records_path.write_bytes(old_bytes)
    legacy["records_sha256"] = hashlib.sha256(old_bytes).hexdigest()
    original = Path.read_bytes
    archive_reads = []

    def read_bytes(path):
        if path == recorder.records_path:
            archive_reads.append(path)
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_bytes)
        resumed = CandidateRecorder(tmp_path, 10, 20.0, resume_state=legacy)
    assert archive_reads == [recorder.records_path]
    with monkeypatch.context() as patch:
        _forbid_archive_reads(patch, recorder.records_path)
        assert resumed.checkpoint_state() == legacy
        _success(resumed, "B₂O₃")
        checkpoint = resumed.checkpoint_state()
    new_bytes = resumed.records_path.read_bytes()
    assert new_bytes.startswith(old_bytes)
    assert checkpoint["records_sha256"] == hashlib.sha256(new_bytes).hexdigest()
    resumed.close()
    again = CandidateRecorder(tmp_path, 10, 20.0, resume_state=checkpoint)
    assert again.checkpoint_state() == checkpoint


@pytest.mark.parametrize("mutation", ["append", "truncate", "same_size_edit", "replace", "delete"])
def test_live_checkpoint_refuses_external_archive_changes(tmp_path, mutation):
    recorder = CandidateRecorder(tmp_path, 10, 20.0)
    _success(recorder, "A")
    path = recorder.records_path
    before = path.stat()
    content = path.read_bytes()
    if mutation == "append":
        with path.open("ab") as handle:
            handle.write(b"corrupt tail\n")
    elif mutation == "truncate":
        path.write_bytes(b"")
    elif mutation == "same_size_edit":
        path.write_bytes(content.replace(b'"formula": "A"', b'"formula": "B"'))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif mutation == "replace":
        other = path.with_suffix(".replacement")
        other.write_bytes(content)
        os.replace(other, path)
    else:
        path.unlink()
    with pytest.raises(ValueError, match="archive changed outside"):
        recorder.checkpoint_state()


def test_resume_rejects_corrupt_tail_without_changing_budget(tmp_path):
    recorder = CandidateRecorder(tmp_path, 10, 20.0)
    _success(recorder, "A")
    saved = recorder.checkpoint_state()
    recorder.close()
    before = recorder.budget_path.read_bytes()
    with recorder.records_path.open("ab") as handle:
        handle.write(b'{"unfinished":')
    with pytest.raises(ValueError, match="archive differs"):
        CandidateRecorder(tmp_path, 10, 20.0, resume_state=saved)
    assert recorder.budget_path.read_bytes() == before


def test_predictor_snapshot_keeps_legacy_values_without_copying_seen_sets():
    class NoDeepcopySet(set):
        def __deepcopy__(self, memo):
            raise AssertionError("Do not copy a large set only to discard the copy")

    timer = PredictorTimer(object())
    candidates = [{"Ni": 0.7, "Fe": 0.3}, {"Ni": 0.5, "Fe": 0.5}]
    for candidate in candidates:
        timer._record(candidate, 2.0, 0.25)
        timer._observation(candidate, (2.0, 0.0))
    timer._phase_costs["train"]["metadata"] = {"values": [1]}
    expected = {
        "schema_version": 1, "n_calls": 2, "seen": sorted(timer._seen, key=repr),
        "t_predict_s": 0.5, "best_mean": 2.0,
        "diagnostic_requests": 0, "diagnostic_failures": 0, "phase": "train",
        "phase_costs": {name: {**deepcopy(cost), "seen": sorted(cost["seen"], key=repr)}
                        for name, cost in timer._phase_costs.items()},
    }
    timer._phase_costs["train"]["seen"] = NoDeepcopySet(timer._phase_costs["train"]["seen"])
    state = timer.state_dict()
    assert state == expected
    timer._phase_costs["train"]["seen"].clear()
    timer._phase_costs["train"]["metadata"]["values"].append(2)
    timer._seen.clear()
    assert state == expected
    restored = PredictorTimer(object())
    restored.load_state_dict(state)
    assert restored.state_dict() == expected
