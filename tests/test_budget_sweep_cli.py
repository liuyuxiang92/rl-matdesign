"""Public CLI and real-child coverage for budgets crossed with paired seeds."""
import json
from pathlib import Path

import pytest
import yaml

from rl_matdesign import seed_sweep
from tests.test_seed_sweep_cli import (
    cli_process, forbid_predictor, run_in_process, tiny_mc_config, write_config,
)


def test_glassnet_sized_grid_validates_before_predictor_construction(tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    seeds = list(range(2026112601, 2026112621))
    budgets = list(range(100, 2001, 100))
    path = write_config(tmp_path, {
        "method": "a2c", "train_seed": seeds, "eval_seed": 0,
        "train_budget": budgets, "parallel_jobs": 10, "num_gen_eps": 0,
    })
    output = tmp_path / "not_created"
    run_in_process(monkeypatch, path, "--out", output, "--validate-config")
    report = json.loads(capsys.readouterr().out)
    assert report["run_count"] == 400 and report["parallel_jobs"] == 10
    assert [(r["config"]["train_budget"], r["config"]["train_seed"])
            for r in report["runs"]] == [(b, s) for b in budgets for s in seeds]
    assert not output.exists()


def test_budget_only_list_uses_batch_dispatch(tmp_path, monkeypatch):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {"method": "a2c", "train_budget": [100, 200], "train_seed": 17})
    captured = []
    monkeypatch.setattr(seed_sweep, "run_seed_sweep", lambda args, plan: captured.append(plan) or 0)
    run_in_process(monkeypatch, path, "--out", tmp_path / "runs")
    assert len(captured) == 1
    assert [(r.config["train_budget"], r.config["train_seed"]) for r in captured[0].runs] == [(100, 17), (200, 17)]


def test_scalar_cli_overrides_restore_single_run_validation(tmp_path, monkeypatch, capsys):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {"method": "a2c", "train_budget": [100, 200], "train_seed": [17, 23]})
    run_in_process(monkeypatch, path, "--train-budget", "300", "--train-seed", "31", "--validate-config")
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"config", "sources"}
    assert (report["config"]["train_budget"], report["config"]["train_seed"]) == (300, 31)


@pytest.mark.parametrize("flag", ["--resume-training", "--only-generate"])
def test_budget_only_resume_requires_saved_scalar_config(tmp_path, monkeypatch, flag):
    forbid_predictor(monkeypatch)
    path = write_config(tmp_path, {"method": "a2c", "train_budget": [100, 200]})
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="saved scalar config"):
        run_in_process(monkeypatch, path, "--out", output, flag)
    assert not output.exists()


def test_real_budget_seed_grid_writes_four_independent_scalar_runs(tmp_path):
    config = tiny_mc_config()
    config.update(train_budget=[2, 3], value_normalization="none", gradient_clipping="none")
    path = write_config(tmp_path, config)
    output = tmp_path / "grid"
    result = cli_process(path, "--out", output, "--skip-generation", "--device", "cpu", timeout=120)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    manifest = json.loads((output / "seed_runs.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["run_count"] == 4 and manifest["parallel_jobs"] == 2
    assert manifest["train_budget_values"] == [2, 3]
    assert [(r["train_budget"], r["seeds"]["train"]) for r in manifest["runs"]] == [(2, 11), (2, 29), (3, 11), (3, 29)]
    for row in manifest["runs"]:
        directory = Path(row["out"])
        assert directory.parent == output / f"budget_{row['train_budget']}"
        saved = yaml.safe_load((directory / "config.yaml").read_text())
        assert saved["train_budget"] == row["train_budget"]
        assert saved["train_seed"] == row["seeds"]["train"]
        assert saved["dqn_num_train_eps"] == row["train_budget"] - 1
        assert "parallel_jobs" not in saved
        assert (directory / "qnet.pt").is_file()
        assert not (directory / "generated.csv").exists()
    assert not (output / ".seed_runs.lock").exists()
