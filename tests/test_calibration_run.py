"""Calibration is resumable overhead before the independent learner allowance."""
import csv
import json
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from rl_matdesign import calibration_run, experiment
from tests.test_experiment_reproducibility import _popart_config, _run, _scientific_rows


def _calibration_config():
    cfg = _popart_config()
    # An RL budget below the pilot size must still be available in full.
    cfg["train_budget"] = 9
    # The fixture deliberately has no feasible candidate in the pilot.
    cfg["reward"] = {
        "type": "gated",
        "failure": {"reward": -10},
        "objective": {"terms": {"score": 1.0}},
        "gates": [{
            "name": "high_score", "score": {"terms": {"score": 1.0}},
            "min": 1000000.0, "priority": 1.0,
        }],
    }
    return cfg


@pytest.mark.parametrize("normalization", ["none", "popart"])
def test_zero_dqn_warmup_fails_before_calibration(tmp_path, monkeypatch, normalization):
    cfg = _calibration_config()
    cfg.update(dqn_warmup_eps=0, value_normalization=normalization)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", "--config", str(config), "--out", str(out),
        "--device", "cpu", "--method", "dqn", "--skip-generation",
    ])
    with pytest.raises(ValueError, match="dqn_warmup_eps >= 1"):
        experiment.main()
    assert not (out / "calibration_checkpoint.pt").exists()


@pytest.mark.parametrize("missing_report", [False, True])
def test_pilot_resume_rejects_missing_or_advanced_budget_report(tmp_path, missing_report):
    # A failed checkpoint replacement after a duplicate/invalid rollout can
    # advance the report without changing the evaluation log. Never discard it.
    saved = {
        "type": "reward_calibration", "accounting": "outside_training_budget",
        "num_candidates": 100, "total_budget": 9, "execution_contract": {},
        "evaluation_log_size": 0, "entries": [], "attempts": 1,
        "evaluations": 0, "misses": 1, "frozen": None,
    }
    torch.save(saved, tmp_path / "calibration_checkpoint.pt")
    report = {
        "schema_version": 1, "accounting": "outside_training_budget",
        "target_candidates": 100, "valid_candidates": 0,
        "attempts_started": 2, "evaluations": 0, "status": "RUNNING",
    }
    path = tmp_path / "reward_calibration_budget.json"
    if not missing_report:
        path.write_text(json.dumps(report))
    predictor = SimpleNamespace(reward_calibration_num_candidates=100, evaluation_log_size=lambda: 0)
    with pytest.raises(ValueError, match="Calibration budget report differs"):
        calibration_run.prepare_reward_calibration(
            env=SimpleNamespace(), predictor=predictor, out=tmp_path,
            contract={}, total_budget=9, attempt_limit=100, resume_pilot=True,
        )
    if not missing_report:
        assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap"),
])
def test_calibrated_runner_excludes_pilot_and_loads_for_generation(tmp_path, method, target):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_calibration_config()))
    out = tmp_path / "run"
    _run(config, out, method, target, "--save-checkpoint-freq", "1")

    artifact = json.loads((out / "reward_calibration.json").read_text())
    rows = json.loads((out / "reward_calibration_observations.json").read_text())
    assert len(rows) == artifact["observation_count"] == 100
    assert all(row["reward_status"] == "final" and row["reward"] < 0 for row in rows)
    ledger = [json.loads(row) for row in (out / "evaluations.jsonl").read_text().splitlines()]
    pilot = [row for row in ledger if row["phase"] == "calibration"]
    assert len(pilot) == 100
    assert all(row["reward"] is None and row["reward_status"] == "pending" for row in pilot)
    with (out / "training_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert [int(row["training_budget_used"]) for row in rows] == list(range(1, 10))
    assert all(row["phase"] != "calibration" for row in rows)
    assert sum(row["phase"] == "dqn_warmup" for row in rows) == int(method == "dqn")
    overhead = json.loads((out / "reward_calibration_budget.json").read_text())
    assert overhead["accounting"] == "outside_training_budget"
    assert overhead["status"] == "COMPLETED"
    assert overhead["valid_candidates"] == overhead["target_candidates"] == 100
    assert overhead["evaluations"] == 100
    assert overhead["attempts_started"] >= 100

    checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
    assert checkpoint["training_budget_state"]["counted"] == 9
    assert checkpoint["execution_contract"]["evaluation"]["reward_calibration"] == artifact["content_hash"]
    if method == "dqn":
        assert checkpoint["episodes_completed"] == 8
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
@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap"),
])
def test_pilot_resume_preserves_paid_evaluations_and_model(tmp_path, monkeypatch, after_freeze, method, target):
    config = tmp_path / "config.yaml"
    cfg = _calibration_config()
    cfg.update(hpo_recording=True, hpo_phase="train", hpo_target_unique=9, hpo_ratio_limit=3)
    config.write_text(yaml.safe_dump(cfg))
    reference_save = calibration_run._save_checkpoint

    def run(out, *flags):
        monkeypatch.setattr(sys, "argv", [
            "run_experiment", "--config", str(config), "--out", str(out),
            "--device", "cpu", "--method", method, "--dqn-target-mode", target,
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
    pilot_checkpoint = torch.load(resumed / "calibration_checkpoint.pt", weights_only=False)
    assert pilot_checkpoint["accounting"] == "outside_training_budget"
    assert pilot_checkpoint["hpo_state"]["budget"]["attempts_started"] == 0
    assert pilot_checkpoint["hpo_state"]["budget"]["unique_evaluated"] == 0
    assert (resumed / "candidate_records.jsonl").read_text() == ""
    monkeypatch.setattr(calibration_run, "_save_checkpoint", reference_save)
    run(resumed, "--resume-training")

    for name in ("reward_calibration.json", "reward_calibration_observations.json", "reward_calibration_budget.json"):
        assert json.loads((resumed / name).read_text()) == json.loads((full / name).read_text())
    ledger = [json.loads(row) for row in (resumed / "evaluations.jsonl").read_text().splitlines()]
    assert sum(row["phase"] == "calibration" for row in ledger) == 100
    budget = json.loads((resumed / "candidate_budget.json").read_text())
    assert budget["status"] == "COMPLETED"
    assert budget["unique_evaluated"] == budget["target_unique"] == 9
    assert 9 <= budget["attempts_started"] < 27
    records = (resumed / "candidate_records.jsonl").read_text().splitlines()
    assert len(records) == 9
    weight_files = [("qnet.pt", "qnet_state")] if method == "dqn" else [
        ("policy.pt", "policy_state"), ("value_net.pt", "value_net_state"),
    ]
    for filename, field in weight_files:
        a = torch.load(full / filename, weights_only=True)[field]
        b = torch.load(resumed / filename, weights_only=True)[field]
        assert all(torch.equal(a[key], b[key]) for key in a)
