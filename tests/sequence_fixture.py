"""Portable synthetic sequence configuration shared by integration tests.

A/C/D are arbitrary token identifiers. This fixture defines neither molecular
structures nor thiol chemistry; those belong to the application adapter.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def sequence_config() -> dict:
    """Return a fresh configuration so tests can safely customize nested values."""
    return {
        "groups": [
            {
                "name": "n_terminal",
                "kind": "categorical",
                "choices": [{
                    "name": "treatment",
                    "element": "treatment",
                    "values": ["free", "acetyl"],
                }],
            },
            {
                "name": "peptide",
                "kind": "sequence",
                "length": 11,
                "tokens": ["A", "C", "D"],
                "count_constraints": [{"tokens": ["C"], "min": 2, "max": 2}],
            },
        ],
        "predictor": "dummy",
        # Sequence validity is intrinsic even when optional filters are disabled.
        "constrain_training": False,
        "pg_warmup_eps": 2,
        "pg_num_iters": 2,
        "pg_batch_eps": 2,
        "pg_lr_actor": 0.001,
        "pg_lr_critic": 0.001,
        "pg_entropy_coef": 0.01,
        "dqn_warmup_eps": 2,
        "dqn_num_train_eps": 3,
        "dqn_hidden_dim": 16,
        "dqn_buffer_size": 128,
        "dqn_batch_size": 8,
        "dqn_grad_steps_per_ep": 1,
        "dqn_target_update_freq": 2,
        "dqn_eps_anneal_eps": 3,
        "dqn_eps_min": 0.1,
        "gamma": 0.99,
        "gen_temperature": 1.0,
        "num_gen_eps": 3,
        "exploration_gen_eps": 1,
    }


def write_sequence_config(path: Path) -> Path:
    """Write the fixture in a test's temporary directory for the CLI runner."""
    path.write_text(yaml.safe_dump(sequence_config()), encoding="utf-8")
    return path


class CountedSequencePredictor:
    """Synthetic domain adapter: C is a counted token, with no chemistry claim."""

    def __init__(self, cfg=None, seed=None):
        self.calls = 0

    def sequence_constraints(self, group_name, tokens):
        if group_name != "peptide":
            return []
        if set(tokens) - {"A", "C", "D", "P"}:
            raise ValueError("Unknown fixture token")
        return [{"tokens": ["C"], "min": 2, "max": 2}]

    def predict(self, candidate):
        sequence = candidate["peptide"]
        assert len(sequence) == 5 and sequence.count("C") == 2
        self.calls += 1
        return float(sum(i + 1 for i, token in enumerate(sequence) if token == "D")), 0.25


def mutation_config() -> dict:
    cfg = sequence_config()
    cfg["groups"][1] = {
        "name": "peptide", "kind": "sequence", "tokens": ["A", "C", "D"],
        "parent": ["P", "C", "A", "C", "A"], "num_mutations": 2,
    }
    cfg.pop("predictor")
    cfg["properties"] = [{
        "name": "activity", "predictor": "tests.sequence_fixture:CountedSequencePredictor",
        "direction": "max", "objective": "mean_minus_kstd",
    }]
    cfg["dqn_target_mode"] = "mc"
    cfg["dqn_gamma"] = 1.0
    cfg["dqn_augment_permutations"] = 0
    return cfg
