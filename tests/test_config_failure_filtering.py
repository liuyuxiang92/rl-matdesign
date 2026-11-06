"""Only usable configurations contribute to a formula's properties and reward."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_cross_formula_predictor import enable_source_geo_opt, RecordingBuilder, _candidate, _predictor


@pytest.mark.parametrize("mode", ["scalar", "batch", "batch_scalar_builder"])
@pytest.mark.parametrize("shared", [True, False])
def test_top_level_relaxation_drops_failed_configs_once_and_averages_survivors(mode, shared):
    predictor = _predictor(n_random_configs=4, share_structure=shared)
    enable_source_geo_opt(predictor)
    relaxed = []
    scored = []
    builders = ([predictor._source_builders["default"]] if shared else list(predictor._source_builders.values()))
    if mode == "batch_scalar_builder":
        for builder in builders:
            builder.build_many_seeded = None

    def relax(cell, *, source_id):
        relaxed.append((cell.candidate, cell.index))
        if cell.candidate["Fe"] == 0.1 or cell.index < 2:
            raise RuntimeError("MD/optimizer failed or did not converge")
        return cell

    def score(prop, cells):
        scored.append((prop["name"], cells[0].candidate, [c.index for c in cells]))
        return float(np.mean([10.0 * c.index for c in cells])), 0.0

    predictor._relax = relax
    predictor._score = score
    bad, good = _candidate(0.1), _candidate(0.2)
    if mode == "scalar":
        with pytest.raises(ValueError, match="No usable configurations"):
            predictor.predict(bad)
        result = predictor.predict(good)
    else:
        results = predictor.batch_predict([bad, good], return_exceptions=True)
        assert isinstance(results[0], ValueError)
        result = results[1]

    # Each raw mean is (20+30)/2=25, with no denominator contribution from failures.
    assert result == pytest.approx((2 * 25 - 25 / 5, None))
    assert predictor.per_objective_stats(good) == {"benefit": (25.0, 0.0), "cost": (25.0, 0.0)}
    assert all(candidate == good and indices == [2, 3] for _, candidate, indices in scored)
    assert predictor._key(bad) not in predictor._stats_cache
    assert len(relaxed) == (8 if shared else 12)  # Failed first property stops that formula.
    # Diagnostics reuse the successful result, including the shortened ensemble.
    assert sum(candidate == good for candidate, _ in relaxed) == (4 if shared else 8)


@pytest.mark.parametrize("mode", ["scalar", "batch", "batch_scalar_builder"])
def test_serial_external_md_builder_observes_strict_nonconvergence(mode, monkeypatch):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.optimize import LBFGS
    from rl_matdesign.utils.structure import relax_structure

    monkeypatch.setattr(LBFGS, "run", lambda self, **kwargs: self.atoms.info["index"] >= 2)
    predictor = _predictor(n_random_configs=4)
    builder = RecordingBuilder()
    failures = []

    def build(candidate, *, n_configs, rng):
        usable = []
        for index in range(n_configs):
            try:
                if index == 0:
                    raise RuntimeError("MD process failed")
                atoms = Atoms("H", positions=[[0, 0, 0]], info={"index": index})
                atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((1, 3)))
                optimized = relax_structure(atoms, calc=atoms.calc, relax_cell=False)
                usable.append(SimpleNamespace(candidate=candidate, index=optimized.info["index"]))
            except Exception as exc:
                failures.append((index, str(exc)))
        return usable

    builder.build = build
    if mode == "batch_scalar_builder":
        builder.build_many_seeded = None
    predictor._source_builders["default"] = builder
    seen = []
    predictor._score = lambda prop, cells: (seen.append([c.index for c in cells]) or 1.0, 0.0)
    candidate = _candidate(0.2)
    if mode == "scalar":
        predictor.predict(candidate)
    else:
        predictor.batch_predict([candidate])
    assert seen == [[2, 3], [2, 3]]
    assert [index for index, _ in failures] == [0, 1]
    assert "converg" in failures[1][1]
