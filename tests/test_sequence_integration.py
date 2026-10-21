"""Ordered, constrained sequence decisions through the real RL runner.

The fixture is deliberately synthetic: token C is a counted category, not a
claim about molecular structures or free thiols. Chemistry and GP feature
validation belong to the application-specific predictor/catalog adapter.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_matdesign.configuration import build_env, load_config
from rl_matdesign.encoding import decode_one_hot
from tests.sequence_fixture import sequence_config, write_sequence_config


_REPO = Path(__file__).resolve().parent.parent
_RUNNER = _REPO / "scripts" / "run_experiment.py"


class _RecordingPredictor:
    def __init__(self):
        self.candidates = []

    def predict(self, candidate):
        self.candidates.append(candidate)
        return 0.5, 0.1


def _env():
    predictor = _RecordingPredictor()
    env, _, _ = build_env(sequence_config(), predictor)
    env.initialize()
    return env, predictor


def _cap(env, cap="free"):
    # This fixture's categorical choices retain their declared order.
    actions = env.allowed_actions()
    env.step(actions[0 if cap == "free" else 1])


def _token(env, token):
    for action in env.allowed_actions():
        if decode_one_hot(action[0], env.species_set) == token:
            env.step(action)
            return
    raise AssertionError(f"Token {token!r} is unavailable at step {env.counter}")


def _allowed_tokens(env):
    return {
        decode_one_hot(action[0], env.species_set)
        for action in env.allowed_actions()
    }


def _candidate(env, sequence, cap="free"):
    env.initialize()
    _cap(env, cap)
    for token in sequence:
        _token(env, token)
    return env.terminal_cation_fractions()


def test_completed_cap_and_sequence_order_are_visible_to_policy():
    free, _ = _env()
    acetyl, _ = _env()
    _cap(free, "free")
    _cap(acetyl, "acetyl")
    assert not np.array_equal(
        free.current_state_features(), acetyl.current_state_features()
    ), "Later residue choices must be conditioned on the selected terminal treatment"

    forward, _ = _env()
    reverse, _ = _env()
    for env, tokens in ((forward, ["A", "D"]), (reverse, ["D", "A"])):
        _cap(env)
        for token in tokens:
            _token(env, token)
    assert not np.array_equal(
        forward.current_state_features(), reverse.current_state_features()
    ), "Two different ordered prefixes must not collapse to a composition histogram"


def test_arbitrary_token_action_features_preserve_identity():
    env, _ = _env()
    features = env.action_features()
    assert features is not None
    assert features.shape[0] == len(env.species_set)
    assert np.isfinite(features).all()
    assert np.unique(features, axis=0).shape[0] == len(env.species_set)


def test_intrinsic_count_masks_survive_disabled_optional_training_filters():
    env, _ = _env()
    env.constraints_enabled = False
    _cap(env)
    _token(env, "C")
    _token(env, "C")
    assert _allowed_tokens(env) == {"A", "D"}

    env.initialize()
    _cap(env)
    for _ in range(9):
        _token(env, "A")
    assert _allowed_tokens(env) == {"C"}, "Final two slots must satisfy the lower bound"
    _token(env, "C")
    assert _allowed_tokens(env) == {"C"}
    _token(env, "C")
    assert env.counter == 12  # One terminal treatment plus 11 token decisions.
    assert not env.allowed_actions()


def test_ordered_candidates_reach_predictor_and_have_distinct_dedup_keys():
    env, predictor = _env()
    sequence = ["A", "C", "D", "C"] + ["A"] * 7
    permuted = ["C", "A", "D", "C"] + ["A"] * 7
    keys = []
    for tokens, cap in ((sequence, "free"), (permuted, "free"), (sequence, "acetyl")):
        expected = {"n_terminal": {"treatment": cap}, "peptide": tokens}
        assert _candidate(env, tokens, cap) == expected
        assert predictor.candidates[-1] == expected
        keys.append(env.terminal_comp_key())
    assert len(set(keys)) == 3, "Deduplication must preserve sequence order and treatment"
    _candidate(env, sequence)
    assert env.terminal_comp_key() == keys[0]


def test_checkpoint_rejects_reordered_vocabulary_with_identical_dimensions(tmp_path):
    from rl_matdesign.utils.design_schema import save_or_check_design_schema

    original, _ = _env()
    checkpoint_dir = tmp_path / "trained"
    save_or_check_design_schema(original.design_schema(), str(checkpoint_dir))
    load_paths = [str(checkpoint_dir / "qnet.pt"), str(checkpoint_dir / "std_scaler.bin")]
    compatible_dir = tmp_path / "compatible"
    save_or_check_design_schema(
        original.design_schema(), str(compatible_dir), load_paths=load_paths
    )
    assert (compatible_dir / "sequence_schema.json").is_file()

    cfg = sequence_config()
    cfg["groups"][1]["tokens"] = ["D", "C", "A"]
    reordered, _, _ = build_env(cfg, _RecordingPredictor())
    reordered.initialize()
    assert original.current_state_features().shape == reordered.current_state_features().shape
    assert original.action_features().shape == reordered.action_features().shape
    rejected_dir = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Sequence design differs"):
        save_or_check_design_schema(
            reordered.design_schema(), str(rejected_dir), load_paths=load_paths
        )
    assert not (rejected_dir / "sequence_schema.json").exists()


def test_inline_tokens_work_independently_of_config_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "scenario"
    config_dir.mkdir()
    cfg = sequence_config()
    config_path = config_dir / "design.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    unrelated_dir = tmp_path / "working_directory"
    unrelated_dir.mkdir()
    monkeypatch.chdir(unrelated_dir)

    loaded = load_config(str(config_path))
    assert loaded["groups"][1]["tokens"] == ["A", "C", "D"]
    env, _, _ = build_env(loaded, _RecordingPredictor())
    sequence = ["C", "C"] + ["A"] * 9
    assert _candidate(env, sequence)["peptide"] == sequence


def test_ordered_sequence_rejects_reward_reuse_by_permutation_augmentation():
    from rl_matdesign.training import _augment_episode_in_buffer

    env, predictor = _env()
    candidate = _candidate(env, ["C", "A", "C"] + ["A"] * 8)
    buffer = deque()
    with pytest.raises(ValueError, match="ordered sequences"):
        _augment_episode_in_buffer(
            env, env.path, buffer, env.action_features(), env.fraction_set, K=1
        )
    assert not buffer
    assert env.terminal_cation_fractions() == candidate
    assert len(predictor.candidates) == 1


def _run(method, out, *extra, expect_success=True):
    config_path = write_sequence_config(out.parent / "sequence_fixture.yaml")
    proc = subprocess.run(
        [
            sys.executable, str(_RUNNER), "--config", str(config_path),
            "--method", method, "--out", str(out), "--device", "cpu",
            "--dp-seed", "0", "--train-seed", "0", "--gen-seed", "0",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO,
    )
    if expect_success:
        assert proc.returncode == 0, (
            f"Sequence runner failed for {method}:\n"
            f"stdout:\n{proc.stdout[-5000:]}\nstderr:\n{proc.stderr[-5000:]}"
        )
    return proc


def _assert_generated(out):
    with (out / "generated.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert {"candidate_json", "formula", "reward", "dp_mean", "dp_std"} <= set(
            reader.fieldnames or []
        )
        rows = list(reader)
    assert len(rows) == 4
    keys = []
    for row in rows:
        candidate = json.loads(row["candidate_json"])
        tokens = candidate["peptide"]
        assert isinstance(tokens, list)
        assert len(tokens) == 11
        assert set(tokens) <= {"A", "C", "D"}
        assert tokens.count("C") == 2
        assert candidate["n_terminal"]["treatment"] in {"free", "acetyl"}
        assert all(math.isfinite(float(row[col])) for col in ("reward", "dp_mean", "dp_std"))
        assert float(row["dp_std"]) >= 0.0
        keys.append(json.dumps(candidate, sort_keys=True))
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("method", ["dqn", "reinforce", "a2c"])
def test_sequence_runner_training_generation_and_reload(tmp_path, method):
    out = tmp_path / method
    _run(method, out)
    _assert_generated(out)
    assert (out / "std_scaler.bin").exists()
    assert (out / ("qnet.pt" if method == "dqn" else "policy.pt")).exists()
    if method == "a2c":
        assert (out / "value_net.pt").exists()
    if method == "dqn":
        reload_out = tmp_path / "dqn_reload"
        proc = _run(
            method, reload_out, "--only-generate",
            "--load-qnet", str(out / "qnet.pt"),
            "--load-scaler", str(out / "std_scaler.bin"),
        )
        assert "Loaded qnet" in proc.stdout
        _assert_generated(reload_out)


def test_a2c_weights_only_resume_rejects_incompatible_critic_schema(tmp_path):
    from rl_matdesign.utils.design_schema import save_or_check_design_schema

    trained = tmp_path / "trained"
    _run("a2c", trained, "--skip-generation")
    cfg = sequence_config()
    cfg["groups"][1]["tokens"] = ["D", "C", "A"]
    reordered, _, _ = build_env(cfg, _RecordingPredictor())
    incompatible = tmp_path / "other_vocabulary"
    save_or_check_design_schema(reordered.design_schema(), str(incompatible))
    # Identical tensors make this a semantic-schema check, not a size check.
    shutil.copyfile(trained / "value_net.pt", incompatible / "value_net.pt")

    resumed = tmp_path / "resume_rejected"
    proc = _run(
        "a2c", resumed, "--resume-training", "--skip-generation",
        "--load-policy", str(trained / "policy.pt"),
        "--load-scaler", str(trained / "std_scaler.bin"),
        "--load-value-net", str(incompatible / "value_net.pt"),
        expect_success=False,
    )
    assert proc.returncode != 0
    assert "Sequence design differs" in proc.stderr
    assert str(incompatible / "sequence_schema.json") in proc.stderr
    assert "Loaded policy" in proc.stdout
    assert "Loaded value_net" not in proc.stdout
    assert not (resumed / "training_log.csv").exists()
