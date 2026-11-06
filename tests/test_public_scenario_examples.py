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
DOMAINS = {
    "blank": {"name": "chain", "kind": "sequence", "tokens": ["X", "Y"], "length": 2},
    "parent": {"name": "chain", "kind": "sequence", "tokens": ["X", "Y"],
               "parent": ["X", "X"], "num_mutations": 1},
    "fields": {"name": "settings", "kind": "categorical", "choices": [
        {"name": "level", "values": [1, 2]},
        {"name": "switch", "values": [False, "off"]},
    ]},
}


@pytest.mark.parametrize("example", list(DOMAINS))
@pytest.mark.parametrize("method,target", [("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap")])
def test_synthetic_example_trains_and_generates(tmp_path, example, method, target):
    config = {
        "predictor": "dummy", "groups": [DOMAINS[example]],
        "eval_seed": 23, "train_seed": 29, "gen_seed": 31,
        "dqn_warmup_eps": 2, "dqn_num_train_eps": 2,
        "dqn_batch_size": 2, "dqn_buffer_size": 16,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 8,
        "dqn_target_update_freq": 2, "dqn_eps_anneal_eps": 2,
        "pg_warmup_eps": 2, "pg_num_iters": 1, "pg_batch_eps": 2,
        "pg_entropy_min": 0.0, "num_gen_eps": 2, "max_gen_attempts": 100,
        "gen_epsilon": 1.0, "exploration_gen_eps": 0,
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
        else:
            sequence = candidate["chain"]
            assert len(sequence) == 2 and set(sequence) <= {"X", "Y"}
            if example == "parent":
                assert sum(token != "X" for token in sequence) == 1
