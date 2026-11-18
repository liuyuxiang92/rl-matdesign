"""Actual CLI learning and generation with both new public group domains.

Only the property oracle is synthetic. Domain compilation, featurization,
replay/returns, neural updates, artifact contracts, and generation are real.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {
        "properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}],
        "groups": [
            {"name": "glass", "kind": "composition",
             "species_set": ["SiO2", "B2O3", "Na2O"],
             "amount_set": [0, .25, .5, .75, 1], "n_steps": 3, "total": 1,
             "element_bounds": {"SiO2": [.5, 1]}},
            {"name": "assignments", "kind": "categorical",
             "labels": ["left", "right", "cap"], "values": [0, True, "0"], "n_steps": 2},
        ],
        "eval_seed": 13, "train_seed": 17, "gen_seed": 19,
        "dqn_warmup_eps": 3, "dqn_num_train_eps": 3,
        "dqn_batch_size": 2, "dqn_buffer_size": 100,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 16,
        "dqn_target_update_freq": 2, "dqn_eps_anneal_eps": 4,
        "pg_warmup_eps": 3, "pg_num_iters": 1, "pg_batch_eps": 3,
        "pg_entropy_min": 0.0, "num_gen_eps": 3, "max_gen_attempts": 40,
        "exploration_gen_eps": 0,
    }


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("method,target", [("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap")])
def test_actual_training_and_generation_with_chemical_and_categorical_groups(tmp_path, method, target):
    cfg = configuration()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"),
               "--config", str(path), "--out", str(out), "--device", "cpu",
               "--method", method, "--dqn-target-mode", target]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout[-5000:] + process.stderr[-6000:]

    effective = json.loads((out / "run_config.json").read_text())
    assert effective["method"] == method and effective["dqn_target_mode"] == target
    artifact = torch.load(out / ("qnet.pt" if method == "dqn" else "policy.pt"),
                          map_location="cpu", weights_only=True)
    contract = artifact["environment_contract"]
    assert contract["max_steps"] == 5
    assert contract["design_spec"]["groups"] == cfg["groups"]
    assert contract["state"]["kind"] == "labelled_groups"
    assert [group["state"]["kind"] for group in contract["state"]["groups"]] == ["chemical_amounts", "labelled_field_assignments"]
    assert len(contract["groups"]) == 2
    state_key = "qnet_state" if method == "dqn" else "policy_state"
    assert all(torch.isfinite(value).all() for value in artifact[state_key].values())

    training = read_rows(out / "training_log.csv")
    phase = "dqn_train" if method == "dqn" else "pg_train"
    updates = [row for row in training if row["phase"] == phase]
    assert updates
    if method == "dqn":
        assert sum(int(row["optimizer_steps"]) for row in updates) == 3
        assert all(math.isfinite(float(row["train_loss"])) for row in updates)
        assert int(updates[-1]["buffer_rows"]) == 30  # six actual five-action episodes
    else:
        assert sum(int(row["actor_optimizer_steps"]) for row in updates) == 1
        assert sum(int(row["critic_optimizer_steps"]) for row in updates) == 1
        assert all(math.isfinite(float(row[field])) for row in updates for field in ("actor_loss", "critic_loss"))
        critic = torch.load(out / "value_net.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(value).all() for value in critic["value_net_state"].values())

    generated = read_rows(out / "generated.csv")
    assert len(generated) == 3
    for row in generated:
        candidate = json.loads(row["candidate_json"])
        assert set(candidate) == {"glass", "assignments"}
        amounts = candidate["glass"]
        assert set(amounts) <= {"SiO2", "B2O3", "Na2O"}
        assert sum(amounts.values()) == pytest.approx(1)
        assert .5 <= amounts["SiO2"] <= 1
        assert all(float(value)*4 == int(float(value)*4) for value in amounts.values())
        assigned = candidate["assignments"]
        assert len(assigned) == 2 and set(assigned) <= {"left", "right", "cap"}
        assert all((type(value), value) in [(int,0),(bool,True),(str,"0")] for value in assigned.values())
        assert math.isfinite(float(row["reward"]))
