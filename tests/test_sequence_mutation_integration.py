"""Parent editing through domain constraints, reward wrappers and the public CLI."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from rl_matdesign.configuration import build_env, build_predictor
from rl_matdesign.encoding import decode_one_hot
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.design_schema import save_or_check_design_schema
from rl_matdesign.utils.timing import PredictorTimer
from tests.sequence_fixture import mutation_config


def _step(env, decision):
    for action in env.allowed_actions():
        if decode_one_hot(action[0], env.species_set) == decision:
            env.step(action)
            return
    raise AssertionError(f"Unavailable decision {decision!r}")


def _decision(env, kind, value):
    # JSON IDs are internal decisions, not names sent to the predictor.
    for token in env.species_set:
        try:
            if json.loads(token) == [kind, value]:
                return token
        except (ValueError, TypeError):
            pass
    raise AssertionError(f"Missing {kind}: {value}")


def test_gp_contract_receives_parent_tokens_and_scores_only_complete_edits():
    cfg = mutation_config()
    original = deepcopy(cfg)
    predictor = PredictorTimer(build_predictor(cfg))
    env, _, _ = build_env(cfg, predictor)
    assert cfg == original
    assert env.n_components == 5  # cap + 2 * (position + replacement)
    assert env.design_schema()["groups"][1]["count_constraints"] == [
        {"tokens": ["C"], "min": 2, "max": 2}
    ]
    env.step(env.allowed_actions()[0])
    before = env.current_state_features().copy()
    _step(env, _decision(env, "position", 2))
    assert not np.array_equal(before, env.current_state_features())
    _step(env, _decision(env, "token", "D"))
    # Temporarily one counted residue: the final edit can move its position.
    assert predictor.n_calls == 0
    _step(env, _decision(env, "position", 3))
    assert predictor.n_calls == 0
    _step(env, _decision(env, "token", "C"))
    assert predictor.n_calls == 1
    assert env.terminal_cation_fractions() == {
        "n_terminal": {"treatment": "free"}, "peptide": ["P", "D", "C", "C", "A"]
    }
    assert [step.reward for step in env.path[:-1]] == [0.0] * 4
    first_key = env.terminal_comp_key()
    env.initialize()
    env.step(env.allowed_actions()[0])
    for position, token in [(3, "C"), (2, "D")]:
        _step(env, _decision(env, "position", position))
        _step(env, _decision(env, "token", token))
    assert env.terminal_comp_key() == first_key


def test_sequence_constraints_from_all_models_are_preserved_and_detached():
    class Leaf:
        def __init__(self, bound):
            self.bound = bound

        def sequence_constraints(self, group_name, tokens):
            assert group_name == "sequence" and tokens == ["A", "C", "D"]
            tokens.clear()  # A leaf must not mutate the next leaf's inputs.
            return [self.bound]

    bounds = [
        {"tokens": ["C"], "min": 1},
        {"tokens": ["A", "C"], "max": 2},
    ]
    predictor = StructureScorePredictor.__new__(StructureScorePredictor)
    predictor.properties = [
        {"instance": [Leaf(bounds[0]), Leaf(bounds[0])]},
        {"instance": [Leaf(bounds[1]), object()]},
    ]
    result = predictor.sequence_constraints("sequence", ["A", "C", "D"])
    assert result == bounds
    result[0]["tokens"].append("D")
    assert bounds[0]["tokens"] == ["C"]


@pytest.mark.parametrize("bad", [None, {}, ["C"]])
def test_bad_domain_constraint_hook_fails_at_startup(bad):
    class Predictor:
        def sequence_constraints(self, group_name, tokens):
            return bad

    with pytest.raises(ValueError, match="sequence_constraints must return"):
        build_env(mutation_config(), Predictor())


def test_public_env_rejects_legacy_csv_with_migration_guidance():
    cfg = mutation_config()
    cfg["groups"][1]["tokens"] = {"csv": "monomers.csv", "id_column": "monomer"}
    with pytest.raises(ValueError, match="CSV sequence tokens are no longer supported"):
        build_env(cfg, object())


def test_configured_bounds_cannot_disable_domain_constraints():
    cfg = mutation_config()
    cfg["groups"][1]["count_constraints"] = [{"tokens": ["C"], "max": 1}]
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match="admit no complete sequence"):
        build_env(cfg, build_predictor(cfg))
    assert cfg == before


def test_parent_identity_is_part_of_checkpoint_schema(tmp_path):
    cfg = mutation_config()
    original, _, _ = build_env(cfg, build_predictor(cfg))
    saved = tmp_path / "saved"
    save_or_check_design_schema(original.design_schema(), str(saved))
    cfg["groups"][1]["parent"] = ["P", "C", "D", "C", "A"]
    changed, _, _ = build_env(cfg, build_predictor(cfg))
    assert changed.current_state_features().shape == original.current_state_features().shape
    with pytest.raises(ValueError, match="Sequence design differs"):
        save_or_check_design_schema(
            changed.design_schema(), str(tmp_path / "changed"),
            load_paths=[str(saved / "qnet.pt")],
        )


def _cli(config_path, out, method, extra=()):
    executable = Path(sys.executable).parent / "rl-matdesign"
    command = str(executable) if executable.is_file() else shutil.which("rl-matdesign")
    if command is None:
        pytest.skip("Install rl-matdesign to exercise the public CLI")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The test predictor is deliberately outside the installed product package.
    env["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).resolve().parent.parent), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [command, "--config", str(config_path), "--method", method, "--out", str(out),
         "--device", "cpu", "--dp-seed", "0", "--train-seed", "0", "--gen-seed", "0", *extra],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result


def _assert_generated(out, parent):
    with (out / "generated.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    keys = set()
    for row in rows:
        candidate = json.loads(row["candidate_json"])
        seq = candidate["peptide"]
        assert len(seq) == len(parent)
        assert sum(a != b for a, b in zip(seq, parent)) == 2
        assert seq.count("C") == 2
        assert all(a in {"A", "C", "D"} for a, b in zip(seq, parent) if a != b)
        assert np.isfinite([float(row[c]) for c in ("reward", "dp_mean", "obj_activity_std")]).all()
        assert float(row["obj_activity_std"]) == pytest.approx(0.25)
        keys.add(json.dumps(candidate, sort_keys=True))
    assert len(keys) == len(rows)


@pytest.mark.parametrize("method,target", [
    ("dqn", "mc"), ("dqn", "bootstrap"), ("reinforce", "mc"), ("a2c", "mc"),
])
def test_public_cli_parent_edits_training_generation_and_reload(tmp_path, method, target):
    cfg = mutation_config()
    cfg["dqn_target_mode"] = target
    config_path = tmp_path / "mutations.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / f"{method}_{target}"
    result = _cli(config_path, out, method)
    assert "step_dim=5" in result.stdout
    _assert_generated(out, cfg["groups"][1]["parent"])
    if method == "dqn" and target == "mc":
        reloaded = tmp_path / "reloaded"
        _cli(config_path, reloaded, method, (
            "--only-generate", "--load-qnet", str(out / "qnet.pt"),
            "--load-scaler", str(out / "std_scaler.bin"),
        ))
        _assert_generated(reloaded, cfg["groups"][1]["parent"])
