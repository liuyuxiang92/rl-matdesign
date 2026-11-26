"""CLI acceptance uses tiny synthetic fixtures, independent of research YAMLs."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).parents[1]

# These deliberately small software examples have no scientific model or assets.
_CAP = {"name": "n_terminal", "kind": "categorical", "n_steps": 1,
        "choices": [{"name": "treatment", "values": ["H"]}]}
DOMAINS = {
    "blank": {"groups": [_CAP, {"name": "peptide", "kind": "categorical",
                               "labels": [1, 2], "values": ["X", "Y"], "n_steps": 2}],
              "structure": {"builder": "peptide", "length": 2}},
    "parent": {"groups": [_CAP, {"name": "peptide", "kind": "categorical",
                                "labels": [1, 2], "values": ["X", "Y"], "n_steps": 1}],
               "structure": {"builder": "peptide", "parent": ["X", "X"], "require_change": ["peptide"]}},
    "fields": {"groups": [{"name": "settings", "kind": "categorical", "n_steps": 2, "choices": [
        {"name": "level", "values": [1, 2]}, {"name": "switch", "values": [False, "off"]}]}]},
    "amounts": {"groups": [{"name": "alloy", "kind": "composition", "species_set": ["Fe", "Ni"],
                              "amount_set": [.5], "n_steps": 2, "total": 1}]},
}


@pytest.mark.parametrize("example", list(DOMAINS))
@pytest.mark.parametrize("method,target", [("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap")])
def test_synthetic_example_trains_and_generates(tmp_path, example, method, target):
    config = {
        "properties": [{"name": "quality", "predictor": "dummy", "direction": "max"}], **DOMAINS[example],
        "eval_seed": 23, "train_seed": 29, "gen_seed": 31,
        # This fixture checks fixed episode/update counts, including repeats.
        "train_count_duplicates": True,
        "dqn_warmup_eps": 2, "dqn_num_train_eps": 2,
        "dqn_batch_size": 2, "dqn_buffer_size": 16,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 8,
        "dqn_target_update_freq": 2,
        "pg_warmup_eps": 2, "pg_num_iters": 1, "pg_batch_eps": 2,
        "exploration_adaptive": False, "num_gen_eps": 2, "max_gen_attempts": 100,
        "gen_exploration_target": 1.0,
    }
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    process = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_experiment.py"),
        "--config", str(config_path), "--out", str(output),
        "--device", "cpu", "--method", method, "--dqn-target-mode", target,
    ], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-5000:]
    with (output / "generated.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert (output / ("qnet.pt" if method == "dqn" else "policy.pt")).is_file()
    for row in rows:
        candidate = json.loads(row["candidate_json"])
        if example == "fields":
            assert candidate["settings"]["level"] in [1, 2]
            assert (type(candidate["settings"]["switch"]), candidate["settings"]["switch"]) in [
                (bool, False), (str, "off"),
            ]
        elif example == "amounts":
            assert set(candidate) <= {"Fe", "Ni"}
            assert sum(candidate.values()) == pytest.approx(1)
        else:
            assert candidate["n_terminal"] == {"treatment": "H"}
            sequence = candidate["peptide"]
            assert len(sequence) == 2 and set(sequence) <= {"X", "Y"}
            if example == "parent":
                assert sum(token != "X" for token in sequence) == 1
