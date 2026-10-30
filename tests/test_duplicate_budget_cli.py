"""Exercise duplicate accounting through the public CLI and saved checkpoints.

Tiny search spaces intentionally repeat candidates. These tests use the real
environments, training loops, serialization, and CSV output without DeepMD.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


_REPO = Path(__file__).resolve().parent.parent
_RUNNER = _REPO / "scripts" / "run_experiment.py"


def _config(*, method="dqn", count_duplicates=False, categorical=False):
    group = (
        {"name": "oxygen", "kind": "categorical",
         "choices": [{"element": "O", "values": [0.0, 1.0, 2.0]}]}
        if categorical else
        {"name": "metal", "kind": "composition", "species_set": ["Fe", "Co", "Ni"],
         "fraction_set": ["1.0"], "n_components": 1}
    )
    return {
        "groups": [group], "predictor": "dummy", "method": method,
        "train_count_duplicates": count_duplicates,
        "train_max_duplicate_attempts": 100,
        "dqn_warmup_eps": 1, "dqn_num_train_eps": 2,
        "dqn_batch_size": 1, "dqn_buffer_size": 1000,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 16,
        "dqn_eps_min": 1.0, "dqn_eps_anneal_eps": 10,
        "dqn_target_update_freq": 2,
        "pg_warmup_eps": 2, "pg_num_iters": 1, "pg_batch_eps": 3,
        "pg_entropy_min": 0.0, "pg_entropy_coef": 0.01,
        "num_gen_eps": 0, "exploration_gen_eps": 0,
    }


def _run(tmp_path, cfg, *extra_args, out=None):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = out or tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, str(_RUNNER), "--config", str(config),
         "--out", str(out), "--device", "cpu", "--train-seed", "0",
         "--skip-generation", "--save-checkpoint-freq", "1", *extra_args],
        capture_output=True, text=True, cwd=_REPO, timeout=120,
    )
    return proc, out


def _successful_rows(proc, out):
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-5000:]
    with (out / "training_log.csv").open() as f:
        return list(csv.DictReader(f))


def _counted(row):
    return row["budget_counted"].lower() == "true"


@pytest.mark.parametrize("target_mode", ["mc", "bootstrap"])
def test_dqn_unique_budget_keeps_duplicate_training_and_replay(tmp_path, target_mode):
    cfg = _config()
    cfg["dqn_target_mode"] = target_mode
    proc, out = _run(tmp_path, cfg)
    rows = _successful_rows(proc, out)
    episodes = [r for r in rows if r["phase"] in {"dqn_warmup", "dqn_train"}]
    assert sum(_counted(r) for r in episodes) == 3
    assert len({r["terminal_comp_key"] for r in episodes}) == 3
    duplicates = [r for r in episodes if not _counted(r)]
    assert duplicates, "the tiny seeded search space must exercise repeats"
    assert all(r["train_loss"] not in {"", "nan"} for r in duplicates)
    assert int(episodes[-1]["training_attempts"]) == len(episodes)
    assert int(episodes[-1]["training_budget_used"]) == 3
    assert int(episodes[-1]["training_unique_candidates"]) == 3
    assert int(episodes[-1]["training_duplicate_episodes"]) == len(duplicates)

    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    # One transition per candidate: replay retains every repeat, too.
    assert len(checkpoint["buffer"]) == len(episodes)
    assert checkpoint["episodes_completed"] == 2
    assert checkpoint["training_episodes_completed"] == len(episodes) - 1
    assert checkpoint["training_budget_state"]


@pytest.mark.parametrize("use_override", [False, True])
def test_cli_default_or_override_counts_duplicates_and_records_effective_config(tmp_path, use_override):
    cfg = _config()
    cfg["dqn_num_train_eps"] = 5  # More episodes than distinct candidates.
    if use_override:
        args = ["--train-count-duplicates", "--train-max-duplicate-attempts", "7"]
    else:
        del cfg["train_count_duplicates"]
        del cfg["train_max_duplicate_attempts"]
        args = []
    proc, out = _run(tmp_path, cfg, *args)
    rows = _successful_rows(proc, out)
    episodes = [r for r in rows if r["phase"] in {"dqn_warmup", "dqn_train"}]
    assert len(episodes) == 6
    assert all(_counted(r) for r in episodes)
    assert int(episodes[-1]["training_budget_used"]) == 6
    assert int(episodes[-1]["training_duplicate_episodes"]) >= 3
    effective = json.loads((out / "run_config.json").read_text())
    assert effective["train_count_duplicates"] is True
    assert effective["train_max_duplicate_attempts"] == (7 if use_override else 1000)


@pytest.mark.parametrize("method", ["reinforce", "a2c"])
def test_pg_unique_budget_uses_categorical_identity_and_extra_updates(tmp_path, method):
    cfg = _config(method=method, count_duplicates=True, categorical=True)
    proc, out = _run(tmp_path, cfg, "--no-train-count-duplicates")
    rows = _successful_rows(proc, out)
    episodes = [r for r in rows if r["phase"] == "pg_episode"]
    updates = [r for r in rows if r["phase"] == "pg_train"]
    assert sum(_counted(r) for r in episodes) == 3
    assert len({r["terminal_comp_key"] for r in episodes}) == 3
    assert len(episodes) > 3
    assert len(updates) > 1
    assert int(episodes[-1]["training_attempts"]) == len(episodes)
    assert int(episodes[-1]["training_unique_candidates"]) == 3
    assert int(episodes[-1]["training_budget_used"]) == 3
    assert int(episodes[-1]["training_duplicate_episodes"]) == len(episodes) - 3
    assert json.loads((out / "run_config.json").read_text())["train_count_duplicates"] is False


@pytest.mark.parametrize("key,value", [
    ("train_count_duplicates", "false"),
    ("train_count_duplicates", 0),
    ("train_max_duplicate_attempts", 0),
    ("train_max_duplicate_attempts", -1),
    ("train_max_duplicate_attempts", True),
    ("train_max_duplicate_attempts", 1.5),
])
def test_main_rejects_invalid_budget_config_before_predictor(tmp_path, monkeypatch, key, value):
    from rl_matdesign import experiment

    cfg = _config()
    cfg[key] = value
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["rl-matdesign", "--config", str(config),
                                     "--out", str(tmp_path / "out")])

    def unexpected_predictor(*args, **kwargs):
        pytest.fail("Invalid budget configuration reached predictor initialization")

    monkeypatch.setattr(experiment, "build_predictor", unexpected_predictor)
    with pytest.raises((ValueError, SystemExit), match=key):
        experiment.main()


def test_dqn_checkpoint_resume_retains_seen_candidates_and_rejects_mode_change(tmp_path):
    cfg = _config()
    cfg["dqn_num_train_eps"] = 1
    proc, out = _run(tmp_path, cfg)
    original = _successful_rows(proc, out)
    original_eps = [r for r in original if r["phase"] in {"dqn_warmup", "dqn_train"}]
    assert int(original_eps[-1]["training_budget_used"]) == 2

    cfg["dqn_num_train_eps"] = 2
    proc, out = _run(tmp_path, cfg, "--resume-training", out=out)
    resumed = _successful_rows(proc, out)
    episodes = [r for r in resumed if r["phase"] in {"dqn_warmup", "dqn_train"}]
    assert sum(r["phase"] == "dqn_warmup" for r in episodes) == 1
    assert sum(_counted(r) for r in episodes) == 3
    assert len({r["terminal_comp_key"] for r in episodes}) == 3
    assert int(episodes[-1]["training_attempts"]) == len(episodes)
    assert int(episodes[-1]["training_budget_used"]) == 3

    proc, _ = _run(tmp_path, cfg, "--resume-training", "--train-count-duplicates", out=out)
    assert proc.returncode != 0
    assert "train_count_duplicates" in proc.stdout + proc.stderr


def test_dqn_duplicate_guard_saves_resumable_state_before_regular_checkpoint(tmp_path):
    cfg = _config()
    cfg["train_max_duplicate_attempts"] = 1
    source_scaler = tmp_path / "source_scaler.bin"
    source_scaler.write_bytes(b"external scaler must remain unchanged")
    proc, out = _run(tmp_path, cfg, "--save-checkpoint-freq", "1000",
                     "--load-scaler", str(source_scaler))
    assert proc.returncode != 0
    assert "consecutive duplicate" in proc.stdout + proc.stderr
    assert source_scaler.read_bytes() == b"external scaler must remain unchanged"
    assert (out / "std_scaler.bin").exists()
    stopped = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    stopped_budget = stopped["training_budget_state"]
    assert stopped_budget["consecutive_duplicates"] == 1
    assert stopped_budget["counted"] < 3
    assert len(stopped["buffer"]) == stopped_budget["attempted"]

    cfg["train_max_duplicate_attempts"] = 100
    proc, out = _run(tmp_path, cfg, "--resume-training",
                     "--save-checkpoint-freq", "1000", out=out)
    rows = _successful_rows(proc, out)
    finished = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    final_budget = finished["training_budget_state"]
    assert final_budget["counted"] == final_budget["unique"] == 3
    assert set(stopped_budget["seen_keys"]) <= set(final_budget["seen_keys"])
    assert final_budget["attempted"] > stopped_budget["attempted"]
    assert len(finished["buffer"]) == final_budget["attempted"]
    episodes = [r for r in rows if r["phase"] == "dqn_train"]
    assert int(episodes[0]["training_attempts"]) == stopped_budget["attempted"] + 1
    assert int(episodes[-1]["training_budget_used"]) == 3
