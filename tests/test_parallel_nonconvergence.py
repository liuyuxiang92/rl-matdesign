"""Real worker processes discard unconverged configs in both relaxation paths."""
from __future__ import annotations

import pytest

from tests.test_parallel_structures import pool_factory


_EXTERNAL_RELAXATION = '''
_OriginalBuilder = Builder

class Builder(_OriginalBuilder):
    def _build(self, candidate, *, n_configs=1, rng=None):
        atoms = super()._build(candidate, n_configs=n_configs, rng=rng)[0]
        from ase.calculators.singlepoint import SinglePointCalculator
        from rl_matdesign.utils.structure import relax_structure
        candidate_id = candidate.get("id", "good")
        fail = candidate_id == "bad" or (candidate_id == "partial" and self.device == "3")
        force = float("nan") if candidate_id == "nonfinite" else (.1 if fail else 0.0)
        calc = SinglePointCalculator(atoms, energy=0.0,
                                     forces=np.full((len(atoms), 3), force))
        # This models an external plugin that imports the shared helper and
        # accepts no explicit strictness option of its own.
        result = relax_structure(atoms, calc=calc, relax_cell=False, steps=0)
        return [result]
'''


_TOP_LEVEL_RELAXATION = '''
_OriginalDP = DP

class DP(_OriginalDP):
    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        # The toy property depends on metadata, so its cache must account for
        # that too. Real DP properties depend on the geometry/species alone.
        if (self.atoms is not None and
                self.atoms.info.get("candidate_id") != atoms.info.get("candidate_id")):
            return all_changes
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        candidate_id = self.atoms.info.get("candidate_id", "good")
        fail = candidate_id == "bad" or (
            candidate_id == "partial" and os.environ["CUDA_VISIBLE_DEVICES"] == "3")
        force = float("nan") if candidate_id == "nonfinite" else (.1 if fail else 0.0)
        self.results["forces"] = np.full((len(self.atoms), 3), force)
'''


@pytest.fixture(params=["external_builder", "geo_opt"])
def relaxation_pool(request, pool_factory, tmp_path):
    modules = tmp_path / "fake_modules"
    cfg = {}
    if request.param == "external_builder":
        source = modules / "parallel_test_builder.py"
        source.write_text(source.read_text() + _EXTERNAL_RELAXATION)
    else:
        source = modules / "deepmd" / "calculator.py"
        source.write_text(source.read_text() + _TOP_LEVEL_RELAXATION)
        cfg["geo_opt"] = {
            "model": str(tmp_path / "calculator_initializations.jsonl"),
            "relax_cell": False,
            "steps": 0,
        }
    pool, _ = pool_factory(cfg_extra=cfg)
    return pool


def test_worker_skips_unconverged_config_and_keeps_converged_config(relaxation_pool):
    atoms = relaxation_pool.build({"id": "partial"}, n_configs=2)
    assert len(atoms) == 1
    assert atoms[0].info["device"] == "7"
    assert atoms[0].calc is None
    # Rejection is an ordinary task failure: both workers remain reusable.
    assert len(relaxation_pool.build({"id": "good"}, n_configs=2)) == 2


def test_all_unconverged_formula_fails_without_losing_next_formula(relaxation_pool):
    result = relaxation_pool.build_many([{"id": "bad"}, {"id": "good"}], n_configs=2)
    assert isinstance(result[0], RuntimeError)
    assert "RelaxationNotConverged" in str(result[0])
    assert len(result[1]) == 2
    assert all(atoms.info["candidate_id"] == "good" for atoms in result[1])


def test_worker_rejects_nonfinite_relaxation_results(relaxation_pool):
    with pytest.raises(RuntimeError, match="non-finite forces"):
        relaxation_pool.build({"id": "nonfinite"}, n_configs=2)
