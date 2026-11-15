"""Unchanged HPO ledger checks do not add writes to durable paid accounting."""
import json
import os
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.candidate_budget import EfficiencyLimitReached
from rl_matdesign.utils.candidate_records import CandidateRecorder


def _count_fsync(monkeypatch):
    calls = []
    original = os.fsync

    def sync(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(os, "fsync", sync)
    return calls


def _disk(recorder):
    return json.loads(recorder.budget_path.read_text())


def test_unchanged_wave_checks_skip_write_but_paid_transitions_are_durable(tmp_path, monkeypatch):
    calls = _count_fsync(monkeypatch)
    recorder = CandidateRecorder(tmp_path, 1)
    assert len(calls) == 1
    initial = recorder.budget_path.stat()
    for _ in range(3):
        assert recorder.wave_limit(1) == 1
        recorder.flush()
    assert len(calls) == 1
    assert recorder.budget_path.stat() == initial

    recorder.start()
    assert len(calls) == 2
    assert (_disk(recorder)["attempts_started"], _disk(recorder)["pending"]) == (1, 1)
    recorder.success(SimpleNamespace(terminal_key="A", formula="A1",
                                     prediction_record={"dp_mean": 1.}))
    assert len(calls) == 4  # Candidate archive and settled budget both fsynced.
    assert (_disk(recorder)["unique_evaluated"], _disk(recorder)["pending"]) == (1, 0)
    recorder.check_efficiency()
    recorder.flush()
    assert len(calls) == 4
    recorder.close()
    assert len(calls) == 5
    assert _disk(recorder)["status"] == "COMPLETED"
    recorder.close()
    assert len(calls) == 5


def test_failed_attempts_and_efficiency_terminal_state_are_synced(tmp_path, monkeypatch):
    calls = _count_fsync(monkeypatch)
    recorder = CandidateRecorder(tmp_path, 4)
    for _ in range(2):
        recorder.start()
        recorder.failure()
    assert len(calls) == 5
    assert _disk(recorder)["attempts_started"] == 2
    assert _disk(recorder)["failed_attempts"] == 2
    with pytest.raises(EfficiencyLimitReached):
        recorder.check_efficiency()
    assert len(calls) == 6
    assert _disk(recorder)["status"] == "INFEASIBLE_EFFICIENCY"
    recorder.close()
    assert len(calls) == 6


@pytest.mark.parametrize("change", ["rewrite", "replace", "delete"])
def test_external_ledger_changes_are_not_hidden_by_flush_cache(tmp_path, monkeypatch, change):
    calls = _count_fsync(monkeypatch)
    recorder = CandidateRecorder(tmp_path, 4)
    payload = '{"external": "preserve this evidence"}'
    if change == "rewrite":
        recorder.budget_path.write_text(payload)
    elif change == "replace":
        replacement = tmp_path / "external.json"
        replacement.write_text(payload)
        os.replace(replacement, recorder.budget_path)
    else:
        recorder.budget_path.unlink()
    with pytest.raises(ValueError, match="budget changed outside"):
        recorder.flush()
    assert len(calls) == 1
    if change != "delete":
        assert recorder.budget_path.read_text() == payload
    else:
        assert not recorder.budget_path.exists()


def test_failed_atomic_write_does_not_advance_cached_snapshot(tmp_path, monkeypatch):
    recorder = CandidateRecorder(tmp_path, 4)
    recorder.budget.start()
    original = recorder.budget_path.read_bytes()
    with monkeypatch.context() as patch:
        def fail(*_args):
            raise OSError("simulated replace failure")
        patch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="replace failure"):
            recorder.flush()
    assert recorder.budget_path.read_bytes() == original
    recorder.flush()
    assert _disk(recorder)["attempts_started"] == 1
    assert _disk(recorder)["pending"] == 1
