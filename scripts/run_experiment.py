#!/usr/bin/env python3
"""Compatibility entry point; experiment implementation lives in the package.

Existing commands and imports from ``run_experiment`` remain supported.
"""
from pathlib import Path
import sys

# Select this checkout even when another checkout is installed in editable mode.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rl_matdesign.experiment import *  # noqa: F401,F403,E402
from rl_matdesign.configuration import (  # noqa: F401,E402
    _FLAG_ALIASES, _FAILED_CANDIDATE_REWARD, _apply_flag_aliases,
)
from rl_matdesign.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
