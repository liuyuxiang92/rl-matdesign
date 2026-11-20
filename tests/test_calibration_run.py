"""The CLI settles calibration evaluations once and learns from frozen rewards."""
import csv
import json
import shutil
import sys

import pytest
import torch
import yaml

from rl_matdesign import calibration_run, experiment
from tests.test_experiment_reproducibility import _config, _run, _scientific_rows


def _calibration_config():
    cfg = _config()
    cfg["properties"][0]["normalization"] = {"mode": "auto_fixed"}
    # The fixture deliberately has no feasible candidate in the pilot.
    cfg["reward"] = {
        "type": "gated",
        "calibration": {"profile": "auto_gated_softplus_v1", "num_candidates": 3},
        "objective": {"terms": {"score": 1.0}},
        "gates": [{
            "name": "high_score", "score": {"terms": {"score": 1.0}},
            "min": 1000000.0, "priority": 1.0, "violation_normalization": "auto_fixed",
        }],
    }
    cfg.update(value_normalization="popart", train_budget=8, dqn_warmup_eps=1)
    cfg.pop("dqn_num_train_eps")
    cfg.pop("pg_num_iters")
    return cfg


@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap"),
])
def test_calibrated_runner_charges_pilot_once_and_loads_for_generation(tmp_path, method, target):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_calibration_config()))
    out = tmp_path / "run"
    _run(config, out, method, target, "--save-checkpoint-freq", "1")

    artifact = json.loads((out / "reward_calibration.json").read_text())
    rows = json.loads((out / "reward_calibration_observations.json").read_text())
    assert len(rows) == artifact["observation_count"] == 3
    assert all(row["reward_status"] == "final" and row["reward"] < 0 for row in rows)
    ledger = [json.loads(row) for row in (out / "evaluations.jsonl").read_text().splitlines()]
    pilot = [row for row in ledger if row["phase"] == "calibration"]
    assert len(pilot) == 3
    assert all(row["reward"] is None and row["reward_status"] == "pending" for row in pilot)
    with (out / "training_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert rows[2]["training_budget_used"] == "3"
    assert rows[-1]["training_budget_used"] == "8"

    checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
    assert checkpoint["execution_contract"]["evaluation"]["reward_calibration"] == artifact["content_hash"]
    if method == "dqn":
        assert checkpoint["episodes_completed"] == 5
        assert (checkpoint["target_net_state"] is None) == (target == "mc")
        assert checkpoint["qnet_state"]["value_normalizer.updates"].item() > 0
    else:
        assert checkpoint["value_net_state"]["value_normalizer.updates"].item() > 0
    generated = _scientific_rows(out / "generated.csv")
    _run(config, out, method, target, "--only-generate")
    assert _scientific_rows(out / "generated.csv") == generated
    assert json.loads((out / "reward_calibration.json").read_text()) == artifact

    resumed = tmp_path / "learner_resumed"
    resumed.mkdir()
    mid = "checkpoint-ep2.pt" if method == "dqn" else "checkpoint-iter1.pt"
    shutil.copyfile(out / mid, resumed / "checkpoint.pt")
    for name in ("std_scaler.bin", "reward_calibration.json"):
        shutil.copyfile(out / name, resumed / name)
    _run(config, resumed, method, target, "--resume-training", "--save-checkpoint-freq", "1")
    right = torch.load(resumed / "checkpoint.pt", weights_only=False)
    assert checkpoint["training_budget_state"] == right["training_budget_state"]
    assert checkpoint["predictor_ledger"]["phase_costs"] == right["predictor_ledger"]["phase_costs"]
    key = "qnet_state" if method == "dqn" else "value_net_state"
    assert all(torch.equal(value, right[key][name]) for name, value in checkpoint[key].items())
    assert _scientific_rows(resumed / "generated.csv") == generated


@pytest.mark.parametrize("after_freeze", [False, True])
def test_pilot_resume_preserves_paid_evaluations_and_model(tmp_path, monkeypatch, after_freeze):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_calibration_config()))
    reference_save = calibration_run._save_checkpoint

    def run(out, *flags):
        monkeypatch.setattr(sys, "argv", [
            "run_experiment", "--config", str(config), "--out", str(out),
            "--device", "cpu", "--method", "dqn", "--dqn-target-mode", "mc",
            "--skip-generation", "--save-checkpoint-freq", "1", *flags,
        ])
        experiment.main()

    full = tmp_path / "full"
    run(full)
    resumed = tmp_path / "resumed"

    def interrupt_at_settled_boundary(path, payload):
        reference_save(path, payload)
        hit = payload["frozen"] is not None if after_freeze else len(payload["entries"]) == 1
        if hit:
            raise KeyboardInterrupt

    monkeypatch.setattr(calibration_run, "_save_checkpoint", interrupt_at_settled_boundary)
    with pytest.raises(KeyboardInterrupt):
        run(resumed)
    assert not (resumed / "reward_calibration.json").exists()
    monkeypatch.setattr(calibration_run, "_save_checkpoint", reference_save)
    run(resumed, "--resume-training")

    for name in ("reward_calibration.json", "reward_calibration_observations.json"):
        assert json.loads((resumed / name).read_text()) == json.loads((full / name).read_text())
    ledger = [json.loads(row) for row in (resumed / "evaluations.jsonl").read_text().splitlines()]
    assert sum(row["phase"] == "calibration" for row in ledger) == 3
    a = torch.load(full / "qnet.pt", weights_only=True)["qnet_state"]
    b = torch.load(resumed / "qnet.pt", weights_only=True)["qnet_state"]
    assert all(torch.equal(a[key], b[key]) for key in a)
