"""Training samples preserve evaluation properties and budget provenance."""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.training_samples import TrainingSampleWriter


def _episode(formula="BaCoO3", **prediction):
    return SimpleNamespace(
        formula=formula, terminal_key=(("Ba", 1), ("Co", 1)), metadata={},
        path=[SimpleNamespace(reward=-99.0)],
        prediction=(0.2, None),
        prediction_record={"formula": formula, "reward": 0.2, "dp_mean": 0.1, **prediction},
    )


def _metrics(budget, **extra):
    return {
        "phase": "pg_episode", "episode": budget, "iteration": 1,
        "training_budget_used": budget, "training_attempts": budget,
        "budget_counted": True, "training_budget_history_complete": True, **extra,
    }


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_counted_duplicates_are_saved_in_sampling_order_and_immediately_visible(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    for budget, formula in enumerate(["ZrCoO3", "BaCoO3", "ZrCoO3"], 1):
        writer.record(_episode(formula), _metrics(budget))
        assert len(_read(path)[1]) == budget  # No close/finalize is needed.
    fields, rows = _read(path)
    assert fields[:3] == ["formula", "reward", "dp_mean"]
    assert [row["formula"] for row in rows] == ["ZrCoO3", "BaCoO3", "ZrCoO3"]
    assert [row["training_budget_used"] for row in rows] == ["1", "2", "3"]


@pytest.mark.parametrize("counted", [False, None, 0, "True"])
def test_only_explicitly_budget_counted_episodes_are_saved(tmp_path, counted):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    writer.record(_episode(), _metrics(0, budget_counted=counted))
    writer.record(_episode(), _metrics(1, training_attempts=2))
    rows = _read(path)[1]
    assert len(rows) == 1
    assert rows[0]["training_attempts"] == "2"


def test_prediction_snapshot_is_not_replaced_by_training_metrics(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    writer.record(
        _episode(obj_delta_T400_mean=0.031, candidate_json={"B": {"Co": 1.0}}),
        _metrics(1, reward=-10, dp_mean=-10, obj_delta_T400_mean=-10,
                 loss=10, terminal_reward=-10, phase="dqn_warmup"),
    )
    fields, rows = _read(path)
    row = rows[0]
    assert row["reward"] == "0.2"
    assert row["dp_mean"] == "0.1"
    assert row["obj_delta_T400_mean"] == "0.031"
    assert json.loads(row["candidate_json"]) == {"B": {"Co": 1.0}}
    assert row["phase"] == "dqn_warmup"
    assert "loss" not in fields and "terminal_reward" not in fields
    assert row["terminal_comp_key"] == "(('Ba', 1), ('Co', 1))"


def test_unavailable_scientific_uncertainty_is_blank_instead_of_zero(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    episode = _episode(dp_std=None, dp_mean_minus_std=None, obj_delta_mean=.07, obj_delta_std=None)
    writer.record(episode, _metrics(1))
    row = _read(path)[1][0]
    assert row["dp_std"] == row["dp_mean_minus_std"] == row["obj_delta_std"] == ""
    assert row["obj_delta_mean"] == "0.07"
    assert episode.prediction == (0.2, None), "the generation tuple has a separate role"


def test_fallback_reward_and_identity_come_from_episode_snapshot(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    episode = _episode()
    episode.prediction_record = {}
    episode.metadata = {"iteration": 17, "phase": "warmup"}
    writer.record(episode, {"budget_counted": True, "training_budget_used": 1})
    row = _read(path)[1][0]
    assert row["formula"] == "BaCoO3"
    assert row["reward"] == "-99.0"
    assert row["phase"] == "warmup" and row["iteration"] == "17"


def test_schema_expansion_preserves_all_rows_and_keeps_identity_last(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    writer.record(_episode(obj_first=1), _metrics(1))
    writer.record(_episode(obj_second=2), _metrics(2))
    fields, rows = _read(path)
    assert fields.index("obj_first") < fields.index("obj_second") < fields.index("phase")
    assert rows[0]["obj_first"] == "1" and rows[0]["obj_second"] == ""
    assert rows[1]["obj_first"] == "" and rows[1]["obj_second"] == "2"
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_retains_session_and_handles_new_columns(tmp_path):
    path = tmp_path / "training_samples.csv"
    initial = TrainingSampleWriter(path)
    initial.start(0)
    initial.record(_episode(obj_first=1), _metrics(1))
    resumed = TrainingSampleWriter(path, append=True)
    resumed.start(1)
    resumed.record(_episode(obj_second=2), _metrics(2))
    rows = _read(path)[1]
    assert len(rows) == 2
    assert rows[0]["training_session_id"] == rows[1]["training_session_id"]
    assert rows[1]["training_samples_history_complete"] == "True"
    assert rows[0]["obj_second"] == "" and rows[1]["obj_second"] == "2"


def test_resume_checkpoint_rollback_archives_original_before_trimming(tmp_path):
    path = tmp_path / "training_samples.csv"
    initial = TrainingSampleWriter(path)
    initial.start(0)
    for budget in range(1, 4):
        initial.record(_episode(f"original-{budget}"), _metrics(budget))
    original_bytes = path.read_bytes()
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="preserved previous CSV"):
        resumed.start(1)
    archives = list(tmp_path.glob("training_samples.superseded-*.csv"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original_bytes
    assert [row["formula"] for row in _read(path)[1]] == ["original-1"]
    resumed.record(_episode("replacement-2"), _metrics(2))
    assert [row["formula"] for row in _read(path)[1]] == ["original-1", "replacement-2"]


def test_missing_resume_history_is_explicitly_marked(tmp_path):
    path = tmp_path / "training_samples.csv"
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="no prior sample rows"):
        resumed.start(5)
    resumed.record(_episode(), _metrics(6))
    row = _read(path)[1][0]
    assert row["training_budget_used"] == "6"
    assert row["training_samples_history_complete"] == "False"


def test_checkpoint_ahead_of_csv_marks_missing_history_on_subsequent_resumes(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    writer.record(_episode(), _metrics(1))
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="does not contain complete history"):
        resumed.start(3)
    resumed.record(_episode(), _metrics(4))
    again = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="does not contain complete history"):
        again.start(4)
    again.record(_episode(), _metrics(5))
    assert _read(path)[1][-1]["training_samples_history_complete"] == "False"


def test_weights_only_resume_uses_separate_session_for_reset_budget(tmp_path):
    path = tmp_path / "training_samples.csv"
    initial = TrainingSampleWriter(path)
    initial.start(0)
    initial.record(_episode("old"), _metrics(1))
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="separate training_session_id"):
        resumed.start(0, history_complete=False)
    resumed.record(_episode("new"), _metrics(1))
    rows = _read(path)[1]
    assert [row["formula"] for row in rows] == ["old", "new"]
    assert rows[0]["training_session_id"] != rows[1]["training_session_id"]
    assert rows[1]["training_samples_history_complete"] == "False"


def test_explicit_checkpoint_session_rolls_back_only_that_session(tmp_path):
    path = tmp_path / "training_samples.csv"
    original = TrainingSampleWriter(path)
    original.start(0)
    for budget in range(1, 4):
        original.record(_episode(f"A-{budget}"), _metrics(budget))
    weights_only = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="separate training_session_id"):
        weights_only.start(0, history_complete=False)
    weights_only.record(_episode("B-1"), _metrics(1))
    before_rollback = path.read_bytes()

    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="preserved previous CSV"):
        resumed.start(1, session_id=original.session_id)
    resumed.record(_episode("A-replacement-2"), _metrics(2))
    rows = _read(path)[1]
    assert [row["formula"] for row in rows] == ["A-1", "B-1", "A-replacement-2"]
    assert rows[0]["training_session_id"] == rows[2]["training_session_id"] == original.session_id
    assert rows[1]["training_session_id"] == weights_only.session_id
    assert rows[2]["training_samples_history_complete"] == "True"
    assert list(tmp_path.glob("training_samples.superseded-*.csv"))[0].read_bytes() == before_rollback

    # Interleaved sessions remain resumable by their own checkpoint identity.
    again = TrainingSampleWriter(path, append=True)
    again.start(2, session_id=original.session_id)
    again.record(_episode("A-3"), _metrics(3))
    assert _read(path)[1][-1]["training_session_id"] == original.session_id


def test_multiple_sessions_without_checkpoint_identity_never_guess_latest(tmp_path):
    path = tmp_path / "training_samples.csv"
    first = TrainingSampleWriter(path)
    first.start(0)
    first.record(_episode("first"), _metrics(1))
    second = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="separate training_session_id"):
        second.start(0, history_complete=False)
    second.record(_episode("second"), _metrics(1))
    previous = _read(path)[1]
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="multiple sessions"):
        resumed.start(1)
    resumed.record(_episode("unknown-origin"), _metrics(2))
    rows = _read(path)[1]
    assert rows[:2] == previous
    assert rows[-1]["training_samples_history_complete"] == "False"
    assert rows[-1]["training_session_id"] not in {first.session_id, second.session_id}
    assert not list(tmp_path.glob("training_samples.superseded-*.csv"))


@pytest.mark.parametrize("budget", [0, 4])
def test_missing_explicit_session_retains_identity_but_marks_history_incomplete(tmp_path, budget):
    path = tmp_path / "training_samples.csv"
    initial = TrainingSampleWriter(path)
    initial.start(0)
    initial.record(_episode("other-session"), _metrics(1))
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning, match="checkpoint session 'missing-session'"):
        resumed.start(budget, session_id="missing-session")
    resumed.record(_episode("continued"), _metrics(budget + 1))
    rows = _read(path)[1]
    assert rows[0]["formula"] == "other-session"
    assert rows[1]["training_session_id"] == "missing-session"
    assert rows[1]["training_samples_history_complete"] == "False"


def test_new_run_preserves_existing_file_in_archive(tmp_path):
    path = tmp_path / "training_samples.csv"
    path.write_text("arbitrary,older\nCSV,format\n", encoding="utf-8")
    original = path.read_bytes()
    writer = TrainingSampleWriter(path)
    with pytest.warns(RuntimeWarning, match="preserved previous CSV"):
        writer.start(0)
    assert list(tmp_path.glob("training_samples.superseded-*.csv"))[0].read_bytes() == original
    writer.record(_episode(), _metrics(1))
    assert len(_read(path)[1]) == 1


def test_incomplete_nonzero_budget_resume_still_rolls_back_current_session(tmp_path):
    path = tmp_path / "training_samples.csv"
    initial = TrainingSampleWriter(path)
    with pytest.warns(RuntimeWarning, match="separate training_session_id"):
        initial.start(0, history_complete=False)
    for budget in range(1, 4):
        initial.record(_episode(), _metrics(budget))
    resumed = TrainingSampleWriter(path, append=True)
    with pytest.warns(RuntimeWarning) as warnings_seen:
        resumed.start(1, history_complete=False)
    assert len(warnings_seen) == 2  # Archive plus incomplete-history warning.
    assert resumed.session_id == initial.session_id
    resumed.record(_episode("replacement"), _metrics(2))
    rows = _read(path)[1]
    assert [row["training_budget_used"] for row in rows] == ["1", "2"]
    assert rows[-1]["training_samples_history_complete"] == "False"


def test_failed_atomic_schema_rewrite_keeps_previous_rows(tmp_path, monkeypatch):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    writer.start(0)
    writer.record(_episode(), _metrics(1))
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated interrupted rewrite")

    monkeypatch.setattr("rl_matdesign.utils.training_samples.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted rewrite"):
        writer.record(_episode(new_property=1), _metrics(2))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_resume_counters_leave_original_file_untouched(tmp_path):
    path = tmp_path / "training_samples.csv"
    path.write_text("formula,training_budget_used\nold,broken\n", encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="CSV training_budget_used"):
        TrainingSampleWriter(path, append=True).start(1)
    assert path.read_bytes() == original


def test_record_requires_start_and_exact_budget_progress(tmp_path):
    path = tmp_path / "training_samples.csv"
    writer = TrainingSampleWriter(path)
    with pytest.raises(RuntimeError, match="start"):
        writer.record(_episode(), _metrics(1))
    writer.start(0)
    with pytest.raises(RuntimeError, match="only be called once"):
        writer.start(0)
    with pytest.raises(ValueError, match="expected budget 1, got 2"):
        writer.record(_episode(), _metrics(2))
    assert _read(path)[1] == []
