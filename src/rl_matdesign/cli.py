"""Installed command entry point, including non-editable wheel installs."""
from __future__ import annotations


def main() -> None:
    from .experiment import main as run_experiment

    run_experiment()
