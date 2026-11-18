"""Pytest shared setup.

Ensures `src/` and `scripts/` are importable so tests can pull from
`rl_matdesign.*`, `abcde_ooh.*`, and script entry points regardless of
installation mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
