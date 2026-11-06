"""Predictor resources are released even when experiment setup is interrupted."""
from __future__ import annotations

import sys
import json

import pytest

from rl_matdesign import experiment


class _ClosablePredictor:
    def __init__(self, close_error=None):
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _fail_during_env_setup(monkeypatch, tmp_path, predictor, error):
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--config", "unused.yaml",
        "--out", str(tmp_path / "run"), "--device", "cpu",
    ])
    monkeypatch.setattr(experiment, "load_yaml_config", lambda path: {})
    monkeypatch.setattr(experiment, "set_global_seed", lambda *a, **k: None)
    monkeypatch.setattr(
        experiment, "build_predictor", lambda cfg, seed: predictor,
    )

    def build_env(cfg, timed_predictor):
        # Exercise the real PredictorTimer delegation used by the main runner.
        assert timed_predictor.close_calls == 0
        raise error

    monkeypatch.setattr(experiment, "build_env", build_env)


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_main_closes_predictor_when_environment_setup_fails(
    monkeypatch, tmp_path, error_type,
):
    predictor = _ClosablePredictor()
    error = error_type("interrupted after predictor construction")
    _fail_during_env_setup(monkeypatch, tmp_path, predictor, error)

    with pytest.raises(error_type) as caught:
        experiment.main()

    assert caught.value is error
    assert predictor.close_calls == 1


def test_cleanup_failure_preserves_original_keyboard_interrupt(
    monkeypatch, tmp_path, capsys,
):
    predictor = _ClosablePredictor(close_error=RuntimeError("cleanup failed"))
    error = KeyboardInterrupt("user interrupted setup")
    _fail_during_env_setup(monkeypatch, tmp_path, predictor, error)

    with pytest.raises(KeyboardInterrupt) as caught:
        experiment.main()

    assert caught.value is error
    assert predictor.close_calls == 1
    assert "predictor.close() failed: RuntimeError: cleanup failed" in capsys.readouterr().out


def test_actual_source_identity_persisted_before_environment_setup(monkeypatch, tmp_path):
    predictor = _ClosablePredictor()
    identity = {"eval_seed": 42, "sources": [{"source_id": "resolved_source"}]}
    predictor.source_identity = lambda: identity
    _fail_during_env_setup(monkeypatch, tmp_path, predictor, ValueError("after source load"))
    for _ in range(2):
        predictor.close_calls = 0
        with pytest.raises(ValueError, match="after source load"):
            experiment.main()
    evidence = list((tmp_path / "run" / "provenance").glob("*/source_identity.json"))
    assert len(evidence) == 2
    assert all(json.loads(path.read_text()) == identity for path in evidence)
    assert predictor.close_calls == 1
