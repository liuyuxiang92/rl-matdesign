"""Optimizer exceptions must not look like successful GPU timing trials."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import LBFGS

from rl_matdesign.utils.structure import relax_structure, strict_relaxation_errors


@pytest.fixture
def failed_relaxation(monkeypatch):
    error = RuntimeError("CUDA out of memory during optimization")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(LBFGS, "run", fail)

    def run():
        return relax_structure(
            Atoms("H", positions=[[0, 0, 0]]), calc=Calculator(), relax_cell=False,
        )

    return run, error


def test_legacy_error_handling_returns_partial_structure(failed_relaxation, capsys):
    run, _ = failed_relaxation
    result = run()
    assert result.get_chemical_formula() == "H"
    assert "optimization raised and did not complete: CUDA out of memory" in capsys.readouterr().out


def test_strict_context_propagates_original_optimizer_exception(failed_relaxation):
    run, error = failed_relaxation
    with strict_relaxation_errors():
        with pytest.raises(RuntimeError) as caught:
            run()
    assert caught.value is error


def test_context_resets_after_optimizer_exception_escapes(failed_relaxation):
    run, error = failed_relaxation
    with pytest.raises(RuntimeError) as caught:
        with strict_relaxation_errors():
            run()
    assert caught.value is error
    assert len(run()) == 1


def test_nested_context_restores_outer_strict_setting(failed_relaxation):
    run, error = failed_relaxation
    with strict_relaxation_errors():
        with pytest.raises(RuntimeError) as inner:
            with strict_relaxation_errors():
                run()
        assert inner.value is error
        with pytest.raises(RuntimeError) as outer:
            run()
        assert outer.value is error
    assert len(run()) == 1


def test_context_resets_after_unrelated_body_exception(failed_relaxation):
    run, _ = failed_relaxation
    with pytest.raises(ValueError, match="builder failed before optimization"):
        with strict_relaxation_errors():
            raise ValueError("builder failed before optimization")
    assert len(run()) == 1


def test_successful_context_exit_restores_legacy_behavior(failed_relaxation):
    run, _ = failed_relaxation
    with strict_relaxation_errors():
        pass
    assert len(run()) == 1


def test_strict_context_keeps_step_limit_nonconvergence_as_warning(monkeypatch, capsys):
    monkeypatch.setattr(LBFGS, "run", lambda *args, **kwargs: False)
    atoms = Atoms("H", positions=[[0, 0, 0]])
    calculator = SinglePointCalculator(atoms, forces=np.array([[.1, 0, 0]]))

    with strict_relaxation_errors():
        result = relax_structure(atoms, calc=calculator, relax_cell=False, steps=2)

    assert len(result) == 1
    assert "did NOT converge" in capsys.readouterr().out
