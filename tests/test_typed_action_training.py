"""Train and generate with real labelled environments and all benchmark arms."""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from rl_matdesign.action_features import value_alphabet
from rl_matdesign.env_multigroup import MultiGroupEnv
from rl_matdesign.training import (
    _precompute_elem_features, _rollout_random_episode, add_episode_to_buffer,
)

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "blank": {"name": "peptide", "kind": "sequence", "tokens": ["A", "B", "C"], "length": 3},
    "parent": {"name": "peptide", "kind": "sequence", "tokens": ["A", "B", "C"],
               "parent": ["A", "B", "C"], "num_mutations": 3},
    "fields": {"name": "conditions", "kind": "categorical", "choices": [
        {"name": "temperature", "values": [300, 350]},
        {"name": "linker", "values": [True, 1, "1"]},
    ]},
}


def test_stop_ends_random_rollout_and_mc_return_uses_actual_path():
    calls = []
    env = MultiGroupEnv(groups=[DOMAINS["parent"]], reward_fn=lambda candidate: calls.append(candidate) or 2.0)

    def choose():
        actions = env.allowed_actions()
        inner = env._active()
        if env.counter:
            return next(a for a in actions if np.argmax(env._from_union(a)[0]) == inner._stop_label)
        return actions[0]

    env.sample_random_action = choose
    _rollout_random_episode(env)
    assert env.is_terminal and len(env.path) == 2 < env.n_components
    assert len(calls) == 1
    assert sum(a != b for a, b in zip(calls[0]["peptide"], DOMAINS["parent"]["parent"])) == 1
    values = value_alphabet(env)
    buffer = collections.deque()
    labels, _ = _precompute_elem_features(env.species_set, env.state_featurizer, env=env)
    add_episode_to_buffer(env.path, buffer, labels, values, gamma=0.5)
    assert [row["mc_return"] for row in buffer] == [1.0, 2.0]
    assert [row["done"] for row in buffer] == [False, True]
    assert all(row["a_comp_val"] is None for row in buffer)
    assert all(row["a_comp_features"].shape == (values.feature_dim,) for row in buffer)


@pytest.mark.parametrize("domain", list(DOMAINS))
@pytest.mark.parametrize("method,target", [("dqn", "mc"), ("dqn", "bootstrap"), ("a2c", "bootstrap")])
def test_cli_trains_typed_domain(tmp_path, domain, method, target):
    cfg = {
        "predictor": "dummy", "groups": [DOMAINS[domain]],
        "eval_seed": 13, "train_seed": 17, "gen_seed": 19,
        "dqn_warmup_eps": 3, "dqn_num_train_eps": 3,
        "dqn_batch_size": 2, "dqn_buffer_size": 100,
        "dqn_grad_steps_per_ep": 1, "dqn_hidden_dim": 16,
        "dqn_target_update_freq": 2, "dqn_eps_anneal_eps": 4,
        "pg_warmup_eps": 3, "pg_num_iters": 1, "pg_batch_eps": 3,
        "pg_entropy_min": 0.0, "num_gen_eps": 2, "max_gen_attempts": 40,
        "exploration_gen_eps": 0,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"),
               "--config", str(path), "--out", str(out), "--device", "cpu",
               "--method", method, "--dqn-target-mode", target]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-5000:]
    weights = torch.load(out / ("qnet.pt" if method == "dqn" else "policy.pt"), weights_only=True)
    assert weights["environment_contract"]["artifact_schema_version"] == 3
    with (out / "generated.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        candidate = json.loads(row["candidate_json"])
        if domain == "fields":
            assert candidate["conditions"]["temperature"] in [300, 350]
            assert type(candidate["conditions"]["linker"]) in (bool, int, str)
        else:
            assert len(candidate["peptide"]) == 3
            if domain == "parent":
                assert 1 <= sum(a != b for a, b in zip(candidate["peptide"], DOMAINS["parent"]["parent"])) <= 3


def test_permutation_augmentation_retains_mc_targets_and_bounds_memory(monkeypatch):
    import itertools
    from rl_matdesign.training import _augment_episode_in_buffer
    spec = {"name": "sequence", "kind": "sequence", "tokens": ["A", "B"], "length": 10}
    env = MultiGroupEnv(groups=[spec], reward_fn=lambda _: 4.0)
    _rollout_random_episode(env)
    original = list(env.path)
    values = value_alphabet(env)
    labels = env.action_features()
    def no_enumeration(*args):
        raise AssertionError("Cannot enumerate 10! permutations for two augmentations")
    monkeypatch.setattr(itertools, "permutations", no_enumeration)
    buffer = collections.deque()
    _augment_episode_in_buffer(env, original, buffer, labels, values, K=2, gamma=0.5)
    assert 10 < len(buffer) <= 30
    for row in buffer:
        step = int(np.argmax(row["s_step"]))
        assert row["mc_return"] == 4.0 * 0.5 ** (9 - step)


@pytest.mark.parametrize("size", [1, 2])
def test_generation_uses_same_legacy_prediction_as_terminal_reward(size):
    from rl_matdesign.configuration import build_env
    from rl_matdesign.model import QRegressor
    from rl_matdesign.training import _fit_scaler_from_warmup, generate_candidates

    class CountingPredictor:
        def __init__(self):
            self.calls = 0
        def evaluation_batch_size(self):
            return size
        def predict(self, candidate):
            self.calls += 1
            return float(self.calls), 0.25

    predictor = CountingPredictor()
    env, _, _ = build_env({"groups": [DOMAINS["blank"]]}, predictor)
    scaler = _fit_scaler_from_warmup(env, 3)
    labels, _ = _precompute_elem_features(env.species_set, env.state_featurizer, env=env)
    values = value_alphabet(env)
    qnet = QRegressor(state_dim=scaler.n_features_in_, step_dim=env.n_components,
                     elem_dim=labels.shape[1], frac_dim=values.feature_dim, hidden_dim=8)
    rows = generate_candidates(env=env, predictor=predictor, scaler=scaler,
                               device=torch.device("cpu"), elem_feats_scaled=labels,
                               fraction_set=values, qnet=qnet, n_exploit=2, max_attempts=10)
    assert len(rows) == 2
    assert all(row["reward"] == row["dp_mean"] for row in rows)
    assert predictor.calls == int(max(row["reward"] for row in rows))


def test_metadata_retry_never_reuses_an_independent_old_raw_value(monkeypatch):
    from rl_matdesign.configuration import build_env
    from rl_matdesign.training import _fit_scaler_from_warmup, generate_candidates
    from rl_matdesign import training

    class Predictor:
        def __init__(self):
            self.calls = 0
            self.latest = None
        def predict(self, candidate):
            self.calls += 1
            self.latest = candidate, self.calls
            return float(self.calls), 0.0
        def predict_raw(self, candidate):
            assert candidate == self.latest[0]
            return float(self.latest[1]), 0.0
        def per_objective_stats(self, candidate):
            return {"value": self.predict_raw(candidate)}
        def candidate_metadata(self, candidate):
            if self.calls == 1:
                raise RuntimeError("First A evaluation has no metadata")
            return {"serial": self.latest[1]}

    predictor = Predictor()
    spec = {"name": "fields", "kind": "categorical",
            "choices": [{"name": "id", "values": ["A", "B"]}]}
    env, _, _ = build_env({"groups": [spec]}, predictor)
    scaler = _fit_scaler_from_warmup(env, 2)
    attempts = iter([0, 1, 0])
    def rollout(**kwargs):
        env.initialize()
        env.step(env.allowed_actions()[next(attempts)])
    monkeypatch.setattr(training, "_rollout_policy_episode", rollout)
    rows = generate_candidates(env=env, predictor=predictor, scaler=scaler,
                               device=torch.device("cpu"), elem_feats_scaled=env.action_features(),
                               fraction_set=value_alphabet(env), qnet=torch.nn.Linear(1, 1),
                               n_exploit=2, max_attempts=3)
    assert len(rows) == 2 and predictor.calls == 3
    retried = next(row for row in rows if json.loads(row["candidate_json"])["fields"]["id"] == "A")
    assert retried["reward"] == retried["dp_mean"] == retried["obj_value_mean"] == retried["meta_serial"] == 3
