"""HPO subprocesses forward the canonical experiment seed options."""
from types import SimpleNamespace

import pytest

from rl_matdesign.hpo.runner import run_trial_subprocess


@pytest.mark.parametrize("eval_seed", [0, 42])
def test_trial_runner_forwards_canonical_seed_flags(tmp_path, monkeypatch, eval_seed):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="finished", stderr="")

    monkeypatch.setattr("rl_matdesign.hpo.runner.subprocess.run", run)
    result = run_trial_subprocess(
        config_yaml="scenario.yaml", out_dir=str(tmp_path / "trial"),
        method="dqn", train_seed=3, eval_seed=eval_seed, gen_seed=7,
    )
    cmd, _ = calls[0]
    assert cmd[cmd.index("--eval-seed") + 1] == str(eval_seed)
    assert cmd[cmd.index("--train-seed") + 1] == "3"
    assert cmd[cmd.index("--gen-seed") + 1] == "7"
    assert "--dp-seed" not in cmd
    assert result.returncode == 0
    assert result.generated_csv == str(tmp_path / "trial" / "generated.csv")
