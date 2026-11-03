"""Only converged, finite relaxations can be scored in strict contexts."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import LBFGS

from rl_matdesign.utils.structure import (
    RelaxationNotConverged,
    relax_structure,
    strict_relaxation_errors,
)


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


def test_legacy_step_limit_nonconvergence_remains_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(LBFGS, "run", lambda *args, **kwargs: False)
    atoms = Atoms("H", positions=[[0, 0, 0]])
    calculator = SinglePointCalculator(atoms, forces=np.array([[.1, 0, 0]]))

    result = relax_structure(atoms, calc=calculator, relax_cell=False, steps=2)

    assert len(result) == 1
    assert "did NOT converge" in capsys.readouterr().out


def test_strict_context_rejects_step_limit_without_an_extra_force_evaluation(monkeypatch):
    monkeypatch.setattr(LBFGS, "run", lambda *args, **kwargs: False)
    # This calculator cannot compute forces. The optimizer's false return is
    # sufficient to reject the result without obscuring it with another error.
    with strict_relaxation_errors(), pytest.raises(RelaxationNotConverged, match="2 steps"):
        relax_structure(Atoms("H"), calc=Calculator(), relax_cell=False, steps=2)


class _ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, *, force=0.0, stress=0.0):
        super().__init__()
        self.force = force
        self.stress = stress
        self.evaluations = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.evaluations += 1
        self.results = {
            "energy": 0.0,
            "forces": np.full((len(atoms), 3), self.force),
            "stress": np.full(6, self.stress),
        }


@pytest.mark.parametrize("relax_cell", [False, True])
def test_converged_relaxation_is_accepted_without_extra_calculator_evaluations(relax_cell):
    calculator = _ConstantCalculator()
    atoms = Atoms("H", cell=[4, 4, 4], pbc=True)
    with strict_relaxation_errors():
        result = relax_structure(atoms, calc=calculator, relax_cell=relax_cell, steps=0)

    assert np.allclose(result.positions, atoms.positions)
    assert calculator.evaluations == 1


@pytest.mark.parametrize("relax_cell,force,stress", [
    (False, .1, 0.0),
    (True, 0.0, .1),  # Converged atomic forces do not imply a converged cell.
])
def test_real_optimizer_rejects_nonconverged_forces_or_cell(relax_cell, force, stress):
    calculator = _ConstantCalculator(force=force, stress=stress)
    atoms = Atoms("H", cell=[4, 4, 4], pbc=True)
    with strict_relaxation_errors(), pytest.raises(RelaxationNotConverged):
        relax_structure(atoms, calc=calculator, relax_cell=relax_cell, steps=0)
    assert calculator.evaluations == 1


@pytest.mark.parametrize("field", [
    "positions", "cell", "energy", "free_energy", "forces", "stress",
])
def test_strict_context_rejects_nonfinite_final_results_even_if_optimizer_returns_true(
    monkeypatch, field,
):
    def fake_converged_run(optimizer, **kwargs):
        if field == "positions":
            optimizer.atoms.positions[0, 0] = np.nan
        elif field == "cell":
            optimizer.atoms.cell[0, 0] = np.inf
        else:
            optimizer.atoms.calc.results[field] = np.nan
        return True

    monkeypatch.setattr(LBFGS, "run", fake_converged_run)
    with strict_relaxation_errors(), pytest.raises(RuntimeError, match=f"non-finite {field}"):
        relax_structure(
            Atoms("H", cell=[4, 4, 4], pbc=True), calc=Calculator(), relax_cell=False,
        )
