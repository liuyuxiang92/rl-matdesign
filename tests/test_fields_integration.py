"""Named fields train and reload through the existing CLI and predictor API."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from tests.sequence_fixture import sequence_config
from tests.test_fields import field_config


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
def test_fields_runner_training_generation_and_checkpoint_reload(tmp_path, method):
    cfg = sequence_config()
    cfg["groups"] = field_config()["groups"]
    cfg["predictor"] = "tests.test_fields:DeterministicFieldPredictor"
    cfg["num_gen_eps"] = 2
    cfg["exploration_gen_eps"] = 0
    cfg["gen_epsilon"] = 1.0
    cfg["dqn_augment_permutations"] = 0
    config_path = tmp_path / "fields.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def run(out, *extra):
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")
        proc = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_experiment.py"),
            "--config", str(config_path), "--out", str(out), "--method", method,
            "--device", "cpu", "--dp-seed", "7", "--train-seed", "7", "--gen-seed", "7",
            *extra,
        ], cwd=ROOT, env=child_env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        assert (out / "design_schema.json").is_file()
        with (out / "generated.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        for row in rows:
            candidate = json.loads(row["candidate_json"])
            assert json.loads(row["formula"]) == candidate
            fields = candidate["conditions"]
            expected = fields["temperature"] / 100 + fields["water"]
            assert float(row["reward"]) == pytest.approx(expected)
            assert float(row["dp_mean"]) == pytest.approx(expected)
            assert float(row["dp_std"]) == 0.0
        return proc.stdout

    out = tmp_path / method
    run(out, "--save-checkpoint-freq", "1")
    assert (out / "training_log.csv").is_file()
    assert (out / "checkpoint.pt").is_file()
    cfg["dqn_num_train_eps"] = 4
    cfg["pg_num_iters"] = 3
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    resumed = run(out, "--resume-training", "--save-checkpoint-freq", "1")
    assert ("DQN resume from" if method == "dqn" else "Resuming from mid-training checkpoint") in resumed
    if method == "dqn":
        run(tmp_path / "reload", "--only-generate", "--load-qnet", str(out / "qnet.pt"),
            "--load-scaler", str(out / "std_scaler.bin"))


@pytest.mark.parametrize("method,std_mode", [
    ("dqn", "absent"), ("reinforce", "scalar"), ("a2c", "grid"),
])
def test_lookup_properties_keep_raw_q_and_uncertainty_in_generated_csv(tmp_path, method, std_mode):
    cfg = sequence_config()
    cfg["groups"] = field_config()["groups"]
    cfg["groups"][0]["choices"] = cfg["groups"][0]["choices"][:2]
    cfg.pop("predictor")
    # Reverse table axis order relative to the YAML field filling order.
    # The external model must resolve the complete candidate by field name.
    axes = [np.array([0.001, 0.002, 0.003]), np.array([100, 150, 200])]
    ii, jj = np.indices((3, 3))
    q_values = 1000.0 + 100.0 * ii + 10.0 * jj
    arrays = {"field_names": np.array(["water", "temperature"]),
              "axis_0": axes[0], "axis_1": axes[1], "mean": q_values}
    if std_mode == "scalar":
        arrays["std"] = np.asarray(0.75)
    elif std_mode == "grid":
        arrays["std"] = 0.1 * (ii + 1) + 0.01 * jj
    model_path = tmp_path / "lookup.npz"
    np.savez(model_path, **arrays)
    cfg["properties"] = [{
        "name": "q", "predictor": "rl_matdesign.predictors.lookup:LookupPredictor",
        "model": str(model_path), "direction": "max", "objective": "mean", "scale": 100,
    }]
    cfg.update(num_gen_eps=2, exploration_gen_eps=0, gen_epsilon=1.0,
               dqn_augment_permutations=0)
    config_path = tmp_path / "lookup_fields.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def run(out, *extra):
        proc = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_experiment.py"),
            "--config", str(config_path), "--out", str(out), "--method", method,
            "--device", "cpu", "--dp-seed", "7", "--train-seed", "7", "--gen-seed", "7",
            *extra,
        ], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        assert (out / "design_schema.json").is_file()
        with (out / "generated.csv").open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert reader.fieldnames == [
                "formula", "reward", "dp_mean", "candidate_json", "obj_q_mean", "obj_q_std",
            ]
        assert len(rows) == 2
        for row in rows:
            candidate = json.loads(row["candidate_json"])
            assert json.loads(row["formula"]) == candidate
            fields = candidate["conditions"]
            location = tuple(int(np.flatnonzero(axis == fields[name])[0])
                             for name, axis in zip(arrays["field_names"], axes))
            expected_q = q_values[location]
            expected_std = (0.0 if std_mode == "absent" else
                            float(arrays["std"]) if std_mode == "scalar" else
                            float(arrays["std"][location]))
            assert float(row["obj_q_mean"]) == pytest.approx(expected_q)
            assert float(row["obj_q_std"]) == pytest.approx(expected_std)
            assert float(row["reward"]) == pytest.approx(expected_q / 100)
            assert float(row["dp_mean"]) == pytest.approx(expected_q / 100)

    trained = tmp_path / method
    run(trained)
    weights_flag = "--load-qnet" if method == "dqn" else "--load-policy"
    weights_name = "qnet.pt" if method == "dqn" else "policy.pt"
    run(tmp_path / "reload", "--only-generate", weights_flag, str(trained / weights_name),
        "--load-scaler", str(trained / "std_scaler.bin"))
