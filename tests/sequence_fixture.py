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
