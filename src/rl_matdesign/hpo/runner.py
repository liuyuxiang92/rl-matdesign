"""Subprocess runner for one (trial × seed) of the installed `rl-matdesign` CLI.

Each seed runs in a fresh process to isolate model-library state. The CLI is
resolved through PATH in the caller's environment; no source checkout is needed
by this runner. Process startup and predictor initialization count toward each
seed's runtime.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


class TrialRunError(RuntimeError):
    """Subprocess exited nonzero or produced no usable output."""


@dataclass
class TrialResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    generated_csv: str


def _tail(s: str, n: int = 2000) -> str:
    """Return the last ``n`` chars of ``s`` (for log surfacing)."""
    return s[-n:] if len(s) > n else s


def run_trial_subprocess(
    *,
    config_yaml: str,
    out_dir: str,
    method: str,
    train_seed: int,
    dp_seed: int,
    gen_seed: int,
    timeout_s: Optional[int] = None,
    env_extra: Optional[dict] = None,
) -> TrialResult:
    """Spawn `rl-matdesign` once; return the result without raising on nonzero exit.

    The caller decides whether to mark the Optuna trial FAIL — that lets the
    driver distinguish "missing generated.csv with exit 0" (bug, raise loudly)
    from "nonzero exit" (deterministic config issue, FAIL the trial).
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "rl-matdesign",
        "--config", config_yaml,
        "--method", method,
        "--out", out_dir,
        "--train-seed", str(train_seed),
        "--dp-seed", str(dp_seed),
        "--gen-seed", str(gen_seed),
    ]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise TrialRunError(
            "The rl-matdesign CLI was not found on PATH. Install rl-matdesign "
            "in the environment used for HPO (python -m pip install -e "
            "/path/to/rl-matdesign) and activate that environment."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise TrialRunError(
            f"trial timed out after {timeout_s}s: {' '.join(cmd)}"
        ) from e
    return TrialResult(
        returncode=proc.returncode,
        stdout_tail=_tail(proc.stdout or ""),
        stderr_tail=_tail(proc.stderr or ""),
        generated_csv=os.path.join(out_dir, "generated.csv"),
    )
