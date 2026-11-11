"""Installed command entry point, including non-editable wheel installs."""
from __future__ import annotations


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "hpo":
        from .hpo.cli import main as run_hpo
        raise SystemExit(run_hpo(sys.argv[2:]))
    from .experiment import main as run_experiment

    run_experiment()
