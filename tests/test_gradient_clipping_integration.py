"""The gradient pilot is separate overhead and exact resume retains adaptation."""
import csv
import json
import shutil

import numpy as np
import pytest
import torch
import yaml

from tests.test_experiment_reproducibility import _config, _popart_config, _run, _scientific_rows


def _controllers(checkpoint, method):
    state = checkpoint["gradient_clipping_state"]
    return {"q": state} if method == "dqn" else state


@pytest.mark.parametrize("normalization", ["none", "popart"])
@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap"),
])
def test_gradient_pilot_is_not_training_and_exact_resume_does_not_recalibrate(
        tmp_path, normalization, method, target):
    cfg = _popart_config()
    cfg.update(value_normalization=normalization, gradient_clipping="clipping", train_budget=9)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    _run(config, out, method, target, "--save-checkpoint-freq", "1")

    report_path = out / "gradient_calibration.json"
    report = json.loads(report_path.read_text())
    assert report["candidate_count"] == 100
    assert report["optimizer_updates"] == report["additional_predictor_calls"] == 0
    assert report["accounting"] == "outside_training_budget"
    roles = {"q"} if method == "dqn" else {"actor", "critic"}
    assert set(report["networks"]) == roles

    ledger = [json.loads(row) for row in (out / "evaluations.jsonl").read_text().splitlines()]
    # PopArt and clipping share one pilot; raw-reward clipping needs one, too.
    assert sum(row["phase"] == "calibration" for row in ledger) == 100
    with (out / "training_samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert all(row["phase"] != "calibration" for row in samples)
    counted = [row for row in samples if row["budget_counted"].lower() in ("true", "1")]
    assert len(counted) == 9
    assert int(samples[-1]["training_budget_used"]) == 9

    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["training_budget_state"]["counted"] == 9
    assert checkpoint["execution_contract"]["learning"]["gradient_clipping"] == "clipping"
    controllers = _controllers(checkpoint, method)
    assert set(controllers) == roles
    for role, state in controllers.items():
        pilot_state = report["networks"][role]
        assert pilot_state["history"] == []
        assert pilot_state["successful_updates"] == 0
        assert state["calibration_norms"] == pilot_state["calibration_norms"]
        positive = [x for x in pilot_state["calibration_norms"] if x > 0]
        if positive:
            assert pilot_state["startup_cap"] == pytest.approx(float(np.quantile(positive, .9)))
            assert state["startup_cap"] == pilot_state["startup_cap"]
        else:
            assert pilot_state["startup_cap"] is None
        assert state["successful_updates"] > 0
        assert len(state["history"]) == min(state["successful_updates"], 200)
    if method == "dqn":
        # Two transitions per sampled training episode, with no pilot replay rows.
        expected_rows = 2 * (checkpoint["training_episodes_completed"] + cfg["dqn_warmup_eps"])
        assert len(checkpoint["buffer"]) == min(cfg["dqn_buffer_size"], expected_rows)
        assert controllers["q"]["successful_updates"] == checkpoint["training_episodes_completed"]
        assert (checkpoint["target_net_state"] is None) == (target == "mc")
    else:
        assert controllers["actor"]["successful_updates"] == checkpoint["iteration"]
        assert controllers["critic"]["successful_updates"] == checkpoint["iteration"]

    generation = _scientific_rows(out / "generated.csv")
    preserved = {name: (out / name).read_bytes() for name in
                 ("gradient_calibration.json", "training_samples.csv", "training_log.csv")}
    _run(config, out, method, target, "--only-generate")
    assert _scientific_rows(out / "generated.csv") == generation
    for name, value in preserved.items():
        assert (out / name).read_bytes() == value

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    mid = "checkpoint-ep2.pt" if method == "dqn" else "checkpoint-iter1.pt"
    shutil.copyfile(out / mid, resumed / "checkpoint.pt")
    shutil.copyfile(out / "std_scaler.bin", resumed / "std_scaler.bin")
    if normalization == "popart":
        shutil.copyfile(out / "reward_calibration.json", resumed / "reward_calibration.json")
    # Neither pilot trajectories nor the gradient report are needed for resume.
    _run(config, resumed, method, target, "--resume-training", "--save-checkpoint-freq", "1")
    right = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert right["gradient_clipping_state"] == checkpoint["gradient_clipping_state"]
    assert right["training_budget_state"] == checkpoint["training_budget_state"]
    assert right["predictor_ledger"]["phase_costs"] == checkpoint["predictor_ledger"]["phase_costs"]
    roles_to_compare = ["qnet_state"] if method == "dqn" else ["policy_state", "value_net_state"]
    if method == "dqn" and target == "bootstrap":
        roles_to_compare.append("target_net_state")
    for role in roles_to_compare:
        assert all(torch.equal(value, right[role][key]) for key, value in checkpoint[role].items())
    assert _scientific_rows(resumed / "generated.csv") == generation
    assert not (resumed / "gradient_calibration.json").exists()
    resumed_ledger = [json.loads(row) for row in (resumed / "evaluations.jsonl").read_text().splitlines()]
    assert not any(row["phase"] == "calibration" for row in resumed_ledger)


@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap"),
])
def test_none_keeps_raw_reward_training_without_a_calibration_pilot(tmp_path, method, target):
    cfg = _config()
    cfg["gradient_clipping"] = "none"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    _run(config, out, method, target)
    assert not (out / "gradient_calibration.json").exists()
    assert not (out / "calibration_checkpoint.pt").exists()
    ledger = [json.loads(row) for row in (out / "evaluations.jsonl").read_text().splitlines()]
    assert not any(row["phase"] == "calibration" for row in ledger)
