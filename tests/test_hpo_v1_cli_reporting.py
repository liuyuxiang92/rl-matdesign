"""Public CLI and cost reports, including interrupted physical jobs."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rl_matdesign.hpo.artifacts import atomic_json
from rl_matdesign.hpo.cli import main
from rl_matdesign.hpo.reporting import export_reports


ROOT = Path(__file__).resolve().parents[1]


def test_validate_and_plan_use_scene_without_loading_predictor(capsys):
    config = ROOT / "configs/hpo/v1/dummy_a2c.yaml"
    assert main(["validate", "--hpo-config", str(config)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["valid"] and not value["predictor_evaluated"]
    assert value["budget_grids"] == {"train": [48, 64, 96], "generation": [8, 16, 32]}
    assert main(["plan", "--hpo-config", str(config)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schedule"]["train"] == [64, 48, 96]
    assert value["stages"] == ["ranges", "train-budget", "train-search", "generation"]


def test_cli_method_conflict_and_missing_campaign_are_clear(capsys, tmp_path):
    config = ROOT / "configs/hpo/v1/dummy_a2c.yaml"
    assert main(["validate", "--hpo-config", str(config), "--method", "dqn-mc"]) == 2
    assert "HPO error:" in capsys.readouterr().err
    assert main(["status", "--out", str(tmp_path)]) == 2
    assert "manifest" in capsys.readouterr().err


def test_unset_threshold_is_configuration_error(capsys, tmp_path):
    base = ROOT / "configs/test_dummy.yaml"
    config = tmp_path / "hpo.yaml"
    config.write_text(f"method: a2c\nbase_config: {base}\nobjective:\n  field: dp_mean\n  direction: maximize\n  threshold: null\n")
    assert main(["validate", "--hpo-config", str(config)]) == 2
    error = capsys.readouterr().err
    assert "objective.threshold" in error and "Traceback" not in error


def test_installed_entry_dispatch_preserves_legacy_experiments(monkeypatch):
    import rl_matdesign.cli as entry
    calls = []
    monkeypatch.setitem(sys.modules, "rl_matdesign.experiment", SimpleNamespace(main=lambda: calls.append("experiment")))
    monkeypatch.setattr(sys, "argv", ["rl-matdesign", "--config", "scene.yaml"])
    entry.main()
    assert calls == ["experiment"]
    monkeypatch.setattr("rl_matdesign.hpo.cli.main", lambda argv: calls.append(argv) or 3)
    monkeypatch.setattr(sys, "argv", ["rl-matdesign", "hpo", "status", "--out", "run"])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 3 and calls[-1] == ["status", "--out", "run"]


def test_reports_count_each_physical_job_once_and_preserve_unknown_time(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"domain": {"size": 1000}})
    job = {"job_id": "training", "phase": "train", "status": "COMPLETED", "budget": 10,
           "seed": 11, "parameters": {"lr": .01}, "eligible_for_selection": True, "score": .8,
           "ledger": {"attempts_started": 12, "unique_evaluated": 10}, "wall_seconds": 3.5}
    atomic_json(tmp_path / "jobs/training/result.json", job)
    atomic_json(tmp_path / "jobs/training/candidate_budget.json", {**job["ledger"], "phase": "train"})
    # Multiple logical measurements reuse the same completed physical job.
    for label in ("ranges", "search"):
        atomic_json(tmp_path / f"measurements/{label}.json", {"seed_results": [job]})
    atomic_json(tmp_path / "jobs/generation/result.json", {**job, "job_id": "generation", "phase": "generation",
                 "status": "INFEASIBLE_EFFICIENCY", "eligible_for_selection": False, "wall_seconds": None})
    atomic_json(tmp_path / "jobs/interrupted/candidate_budget.json", {
        "phase": "train", "attempts_started": 7, "unique_evaluated": 5, "status": "INTERRUPTED"})
    result = export_reports(tmp_path)
    train = result["physical_cost"]["train"]
    assert train["jobs"] == 2 and train["unfinished_jobs"] == 1
    assert train["attempts_started"] == 19 and train["unique_evaluated"] == 15
    assert train["wall_seconds"] is None and train["known_wall_seconds"] == 3.5
    assert result["physical_cost"]["generation"]["wall_seconds"] is None
    assert result["physical_cost"]["generation"]["attempts_started"] == 12
    assert len(result["unfinished_jobs_with_cost"]) == 1
    with (tmp_path / "trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and float(rows[0]["budget_fraction_of_space"]) == .01


def test_reports_empty_campaign_are_valid(tmp_path):
    result = export_reports(tmp_path)
    assert result["physical_cost"]["train"]["jobs"] == 0
    assert result["physical_cost"]["train"]["wall_seconds"] == 0
    assert json.loads((tmp_path / "selection.json").read_text()) == result
