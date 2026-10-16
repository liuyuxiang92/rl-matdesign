"""The HPO runner uses the installed CLI with the caller's paths and environment."""

from __future__ import annotations

import os
import subprocess

import pytest

from rl_matdesign.hpo.runner import TrialRunError, run_trial_subprocess


def _run(**overrides):
    kwargs = dict(
        config_yaml="configs/trial config.yaml",
        out_dir="runs/trial result",
        method="dqn",
        train_seed=11,
        dp_seed=12,
        gen_seed=13,
    )
    kwargs.update(overrides)
    return run_trial_subprocess(**kwargs)


def test_cli_preserves_caller_paths_cwd_arguments_and_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HPO_RUNNER_INHERITED", "keep")
    monkeypatch.setenv("HPO_RUNNER_OVERRIDE", "original")
    monkeypatch.setenv("PATH", "/caller/python/bin")
    extra = {"HPO_RUNNER_OVERRIDE": "updated", "HPO_RUNNER_EXTRA": "new"}

    def fake_run(cmd, **kwargs):
        assert cmd == [
            "rl-matdesign",
            "--config", "configs/trial config.yaml",
            "--method", "a2c",
            "--out", "runs/trial result",
            "--train-seed", "11",
            "--dp-seed", "12",
            "--gen-seed", "13",
        ]
        # Inherited cwd keeps relative config/output paths valid outside the repo.
        assert kwargs.get("cwd") is None
        assert os.getcwd() == str(tmp_path)
        assert kwargs["env"]["PATH"] == "/caller/python/bin"
        assert kwargs["env"]["HPO_RUNNER_INHERITED"] == "keep"
        assert kwargs["env"]["HPO_RUNNER_OVERRIDE"] == "updated"
        assert kwargs["env"]["HPO_RUNNER_EXTRA"] == "new"
        assert kwargs["timeout"] == 47
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs.get("shell", False) is False
        return subprocess.CompletedProcess(cmd, 0, "done", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _run(method="a2c", timeout_s=47, env_extra=extra)

    assert (tmp_path / "runs/trial result").is_dir()
    assert result.generated_csv == os.path.join("runs/trial result", "generated.csv")
    assert os.environ["HPO_RUNNER_OVERRIDE"] == "original"
    assert extra == {"HPO_RUNNER_OVERRIDE": "updated", "HPO_RUNNER_EXTRA": "new"}


def test_default_environment_is_copied_and_timeout_is_unlimited(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HPO_RUNNER_INHERITED", "keep")

    def fake_run(cmd, **kwargs):
        assert kwargs["timeout"] is None
        assert kwargs["env"] == dict(os.environ)
        kwargs["env"]["HPO_RUNNER_INHERITED"] = "child-only"
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run()
    assert os.environ["HPO_RUNNER_INHERITED"] == "keep"


@pytest.mark.parametrize("returncode", [0, 7])
def test_process_result_preserves_exit_status_and_log_tails(
    tmp_path, monkeypatch, returncode
):
    monkeypatch.chdir(tmp_path)
    stdout = "discarded-output" + "x" * 1988 + "stdout-tail!"
    stderr = "discarded-error" + "y" * 1988 + "stderr-tail!"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, returncode, stdout, stderr),
    )

    result = _run()

    # Nonzero status remains available to the HPO driver, rather than raising here.
    assert result.returncode == returncode
    assert result.stdout_tail == stdout[-2000:]
    assert result.stderr_tail == stderr[-2000:]
    assert result.generated_csv == os.path.join("runs/trial result", "generated.csv")


def test_empty_process_output_becomes_empty_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, None, None),
    )

    result = _run()

    assert result.stdout_tail == ""
    assert result.stderr_tail == ""


def test_timeout_raises_trial_error_with_command_and_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    error = subprocess.TimeoutExpired("rl-matdesign", 47)

    def fake_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(TrialRunError) as exc_info:
        _run(timeout_s=47)

    assert exc_info.value.__cause__ is error
    assert "47" in str(exc_info.value)
    assert "rl-matdesign" in str(exc_info.value)
    assert "timed out" in str(exc_info.value).lower()


def test_missing_cli_raises_actionable_installation_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    error = FileNotFoundError(2, "No such file or directory", "rl-matdesign")

    def fake_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(TrialRunError) as exc_info:
        _run()

    message = str(exc_info.value).lower()
    assert exc_info.value.__cause__ is error
    assert "rl-matdesign" in message
    assert "install" in message
    assert "environment" in message
