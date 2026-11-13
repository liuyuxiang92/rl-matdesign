"""HPO keeps one resumable checkpoint without changing paid samples or training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import yaml

from rl_matdesign import training
from tests.test_experiment_reproducibility import _config


_ROOT = Path(__file__).resolve().parents[1]


def _run_recorded(config, out, *, interrupt=False, resume=False):
    # Interrupt only after a real, synchronized checkpoint has been committed.
    # Keep the original budget/archive intact: copying a checkpoint alone would
    # erase paid evaluations and would not exercise actual HPO continuation.
    wrapper = textwrap.dedent("""
        import json
        from pathlib import Path
        import sys
        sys.path.insert(0, sys.argv.pop(1))
        interrupt = sys.argv.pop(1) == 'yes'
        from rl_matdesign import experiment, training
        save = training._save_checkpoint
        def record_save(path, data, numbered_path=None):
            save(path, data, numbered_path=numbered_path)
            if data.get('hpo_state') is None:
                return
            with (Path(path).parent / 'checkpoint_events.jsonl').open('a') as stream:
                stream.write(json.dumps({
                    'numbered_path': numbered_path,
                    'budget': data['hpo_state']['budget'],
                }) + '\\n')
            if interrupt:
                raise SystemExit(73)
        training._save_checkpoint = record_save
        experiment.main()
    """)
    command = [sys.executable, "-c", wrapper, str(_ROOT / "src"),
               "yes" if interrupt else "no", "--config", str(config),
               "--out", str(out), "--device", "cpu", "--skip-generation",
               "--save-checkpoint-freq", "1"]
    if resume:
        command.append("--resume-training")
    return subprocess.run(command, cwd=_ROOT, text=True, capture_output=True, timeout=120)


def _assert_success(result):
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method,target", [
    ("a2c", "bootstrap"), ("dqn", "mc"), ("dqn", "bootstrap"),
])
def test_hpo_latest_checkpoint_retains_exact_resume(tmp_path, method, target):
    config = _config()
    config.update(method=method, dqn_target_mode=target,
                  dqn_warmup_eps=2, dqn_batch_size=1, dqn_eps_min=1., pg_warmup_eps=2,
                  pg_batch_eps=1, train_count_duplicates=False, train_budget=6,
                  hpo_recording=True, hpo_phase="train", hpo_target_unique=6,
                  hpo_ratio_limit=20., num_gen_eps=0, exploration_gen_eps=0)
    config.pop("pg_num_iters")
    config.pop("dqn_num_train_eps")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    _assert_success(_run_recorded(config_path, full))

    interrupted = _run_recorded(config_path, resumed, interrupt=True)
    assert interrupted.returncode == 73, interrupted.stdout[-3000:] + interrupted.stderr[-5000:]
    before_records = (resumed / "candidate_records.jsonl").read_bytes()
    before_budget = json.loads((resumed / "candidate_budget.json").read_text())
    stopped = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert before_budget["status"] == "INTERRUPTED"
    assert 0 < before_budget["unique_evaluated"] < 6
    assert before_budget["attempts_started"] == stopped["hpo_state"]["budget"]["attempts_started"]
    assert (resumed / "checkpoint.pt").is_file()
    assert not (resumed / "checkpoint.pt").is_symlink()
    _assert_success(_run_recorded(config_path, resumed, resume=True))
    assert (resumed / "candidate_records.jsonl").read_bytes().startswith(before_records)

    left = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    keys = ["training_budget_state", "hpo_state", "rng_state"]
    keys += (["policy_state", "value_net_state", "opt_actor_state", "opt_critic_state"]
             if method == "a2c" else ["qnet_state", "target_net_state", "opt_state", "buffer"])
    for key in keys:
        _assert_nested_equal(left[key], right[key])
    _assert_nested_equal(left["predictor_ledger"]["phase_costs"], right["predictor_ledger"]["phase_costs"])
    assert (full / "candidate_records.jsonl").read_bytes() == (resumed / "candidate_records.jsonl").read_bytes()
    assert json.loads((full / "candidate_budget.json").read_text()) == json.loads((resumed / "candidate_budget.json").read_text())

    for out in (full, resumed):
        events = [json.loads(line) for line in (out / "checkpoint_events.jsonl").read_text().splitlines()]
        assert len(events) > 1, "the run must exercise repeated checkpoint writes"
        assert all(event["numbered_path"] is None for event in events)
        assert not (out / "checkpoint.pt").is_symlink()
        assert not list(out.glob("checkpoint-ep*.pt"))
        assert not list(out.glob("checkpoint-iter*.pt"))
        assert json.loads((out / "candidate_budget.json").read_text())["status"] == "COMPLETED"


@pytest.mark.parametrize("legacy_symlink", [False, True])
@pytest.mark.parametrize("failure_point", ["serialize", "replace"])
def test_latest_checkpoint_failure_preserves_previous_and_can_recover(
    tmp_path, monkeypatch, legacy_symlink, failure_point,
):
    latest = tmp_path / "checkpoint.pt"
    old = tmp_path / "checkpoint-ep1.pt"
    training._save_checkpoint(str(latest), {"weight": torch.tensor([1.])},
                              numbered_path=str(old) if legacy_symlink else None)
    previous = latest.read_bytes()
    with monkeypatch.context() as patch:
        if failure_point == "serialize":
            def failed_save(data, path):
                Path(path).write_bytes(b"partial serialization")
                raise OSError("simulated disk full")
            patch.setattr(training.torch, "save", failed_save)
        else:
            def failed_replace(source, destination):
                raise OSError("simulated atomic replacement failure")
            patch.setattr(os, "replace", failed_replace)
        with pytest.raises(OSError, match="simulated"):
            training._save_checkpoint(str(latest), {"weight": torch.tensor([2.])})
    assert latest.read_bytes() == previous
    assert latest.is_symlink() is legacy_symlink
    assert torch.equal(torch.load(latest, weights_only=False)["weight"], torch.tensor([1.]))

    # Replacement also migrates an old numbered-file symlink to one regular
    # checkpoint, without modifying its former target or following the symlink.
    training._save_checkpoint(str(latest), {"weight": torch.tensor([2.])})
    assert not latest.is_symlink()
    assert torch.equal(torch.load(latest, weights_only=False)["weight"], torch.tensor([2.]))
    if legacy_symlink:
        assert old.read_bytes() == previous


def test_non_hpo_numbered_checkpoint_history_is_unchanged(tmp_path):
    latest = tmp_path / "checkpoint.pt"
    for episode in (1, 2):
        numbered = tmp_path / f"checkpoint-ep{episode}.pt"
        training._save_checkpoint(str(latest), {"episode": episode}, numbered_path=str(numbered))
    assert latest.is_symlink()
    assert os.readlink(latest) == "checkpoint-ep2.pt"
    assert torch.load(latest, weights_only=False)["episode"] == 2
    assert torch.load(tmp_path / "checkpoint-ep1.pt", weights_only=False)["episode"] == 1


def test_failed_checkpoint_cannot_roll_back_paid_hpo_attempts(tmp_path, monkeypatch):
    from rl_matdesign.utils.candidate_records import CandidateRecorder

    recorder = CandidateRecorder(tmp_path, target_unique=6, ratio_limit=20.)
    latest = tmp_path / "checkpoint.pt"
    saved_state = recorder.checkpoint_state()
    training._save_checkpoint(str(latest), {"hpo_state": saved_state})
    previous_checkpoint = latest.read_bytes()
    recorder.start()
    recorder.failure()  # This attempted prediction is paid even without success.

    def disk_full(data, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise OSError("simulated disk full")

    monkeypatch.setattr(training.torch, "save", disk_full)
    with pytest.raises(OSError, match="disk full"):
        training._save_checkpoint(str(latest), {"hpo_state": recorder.checkpoint_state()})
    recorder.close()
    previous_budget = (tmp_path / "candidate_budget.json").read_bytes()
    previous_records = (tmp_path / "candidate_records.jsonl").read_bytes()
    assert latest.read_bytes() == previous_checkpoint
    with pytest.raises(ValueError, match="advanced beyond checkpoint"):
        CandidateRecorder(tmp_path, target_unique=6, ratio_limit=20., resume_state=saved_state)
    assert (tmp_path / "candidate_budget.json").read_bytes() == previous_budget
    assert (tmp_path / "candidate_records.jsonl").read_bytes() == previous_records
    assert json.loads(previous_budget)["attempts_started"] == 1
