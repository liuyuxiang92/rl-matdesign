"""Repeat complete CPU experiments, not just individual RNG calls."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "predictor": "dummy", "method": "a2c",
        "groups": [{"name": "metal", "kind": "composition",
                    "species_set": ["Ni", "Fe", "Co"],
                    "fraction_set": ["0.25", "0.50", "0.75"], "n_components": 2}],
        "eval_seed": 13, "train_seed": 17, "gen_seed": 19,
        "dqn_warmup_eps": 4, "dqn_num_train_eps": 4,
        "dqn_batch_size": 2, "dqn_buffer_size": 100,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 16,
        "dqn_target_update_freq": 2, "dqn_eps_anneal_eps": 4,
        "pg_warmup_eps": 4, "pg_num_iters": 2, "pg_batch_eps": 2,
        "pg_entropy_min": 0.0, "num_gen_eps": 3, "max_gen_attempts": 100,
        "exploration_gen_eps": 1,
    }


def _run(config, out, method, target, *extra):
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"),
               "--config", str(config), "--out", str(out), "--device", "cpu",
               "--method", method, "--dqn-target-mode", target, *extra]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-5000:]


def _scientific_rows(path):
    with path.open() as stream:
        return [{key: value for key, value in row.items()
                 if key in {"formula", "reward", "dp_mean", "dp_std", "phase", "return",
                            "terminal_reward", "terminal_comp_key", "epsilon_used"}}
                for row in csv.DictReader(stream)]


@pytest.mark.parametrize("method,target,weights", [
    ("dqn", "mc", "qnet.pt"),
    ("dqn", "bootstrap", "qnet.pt"),
    ("a2c", "bootstrap", "policy.pt"),
])
def test_three_algorithms_repeat_from_resolved_config(tmp_path, method, target, weights):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config()))
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        _run(config, out, method, target)
    left = torch.load(first / weights, map_location="cpu", weights_only=True)
    right = torch.load(second / weights, map_location="cpu", weights_only=True)
    assert left["environment_contract"] == right["environment_contract"]
    role = "qnet_state" if method == "dqn" else "policy_state"
    assert all(torch.equal(left[role][key], right[role][key]) for key in left[role])
    if method == "a2c":
        left_v = torch.load(first / "value_net.pt", map_location="cpu", weights_only=True)
        right_v = torch.load(second / "value_net.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(left_v["value_net_state"][key], right_v["value_net_state"][key])
                   for key in left_v["value_net_state"])
    assert _scientific_rows(first / "generated.csv") == _scientific_rows(second / "generated.csv")
    assert _scientific_rows(first / "training_log.csv") == _scientific_rows(second / "training_log.csv")
    effective = json.loads((first / "run_config.json").read_text())
    assert effective["method"] == method
    assert effective["dqn_target_mode"] == target
    assert (effective["eval_seed"], effective["train_seed"], effective["gen_seed"]) == (13, 17, 19)
    assert "dqn_lr" in effective and "pg_lr_actor" in effective
    manifests = list((first / "provenance").glob("*/manifest.json"))
    assert len(manifests) == 1
    evidence = json.loads(manifests[0].read_text())
    assert evidence["config_sources"]["method"] == "override"
    assert evidence["config"]["dqn_target_mode"] == target
    assert evidence["git"]["commit"]
    assert evidence["runtime"]["torch"]["deterministic_algorithms"] is True


def test_generation_reuses_saved_model_and_retains_training_evidence(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config()))
    out = tmp_path / "run"
    _run(config, out, "dqn", "mc")
    rows = _scientific_rows(out / "generated.csv")
    old = next((out / "provenance").glob("*/manifest.json"))
    old_bytes = old.read_bytes()
    training_evidence = {name: (out / name).read_bytes()
                         for name in ("training_log.csv", "episode_timing.csv", "timing.json")}
    _run(config, out, "dqn", "mc", "--only-generate")
    assert old.read_bytes() == old_bytes
    for name, content in training_evidence.items():
        assert (out / name).read_bytes() == content
    for name in ("generation_log.csv", "generation_episode_timing.csv", "generation_timing.json"):
        assert (out / name).exists()
    assert len(list((out / "provenance").glob("*/manifest.json"))) == 2
    assert _scientific_rows(out / "generated.csv") == rows
