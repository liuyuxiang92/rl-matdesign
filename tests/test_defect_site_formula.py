"""Complete formulas must describe the actual defect-builder atom counts."""
from __future__ import annotations

import csv
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ase")

from rl_matdesign.predictors.builders.defect_site import DefectSiteBuilder


def _template(tmp_path, *, n_a=32, n_b=32, spectators=None):
    from ase import Atoms
    from ase.io import write

    spectators = {"O": 3 * n_a} if spectators is None else spectators
    symbols = ["Sr"] * n_a + ["Fe"] * n_b
    symbols += [element for element, count in spectators.items() for _ in range(count)]
    side = int(np.ceil(len(symbols) ** (1 / 3))) + 1
    positions = [(3 * (i % side), 3 * ((i // side) % side), 3 * (i // side**2))
                 for i in range(len(symbols))]
    path = tmp_path / "template.vasp"
    write(path, Atoms(symbols, positions=positions, cell=[3 * side] * 3, pbc=True),
          format="vasp")
    return str(path)


def _candidate(a=None, b=None, defect=None):
    candidate = {"A_dopant": a or {"Sr": 1.0}, "B_dopant": b or {"Fe": 1.0}}
    if defect is not None:
        candidate["A_defect"] = {"defect": defect}
    return candidate


def _parse(formula):
    # Require an ordinary chemical formula: no vacancy annotations, repeated
    # element tokens or scientific-notation coefficients hidden in the label.
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?", formula)
    assert "".join(element + amount for element, amount in tokens) == formula
    assert len({element for element, _ in tokens}) == len(tokens)
    return {element: float(amount or 1) for element, amount in tokens}


@pytest.mark.parametrize("candidate", [
    _candidate(),
    _candidate({"Sr": .5, "Ba": .5}, {"Fe": .75, "Co": .25}),
    _candidate({"Ba": 1.0}, {"Co": .8125, "Mn": .1875}),
    _candidate({"Sr": .5, "Ca": .5}, defect="Sr_-0.25"),
    _candidate({"Sr": .5, "Ca": .5}, defect="Ca_-0.25"),
    _candidate({"Sr": .75, "Ca": .25}, defect="Ca_-0.5"),
    _candidate({"Sr": .25, "Ca": .75}, defect="Sr_-0.5"),
    _candidate({"Sr": .5, "Ca": .5}, defect="Ba_-0.5"),
    _candidate({"Sr": .5, "Fe": .5}, {"Fe": .5, "Sr": .5}),
    _candidate({"Sr": .96875, "Ba": .03125}, {"Fe": .1875, "Co": .8125}),
], ids=["pristine", "hosts-retained", "full-substitution", "host-vacancy",
        "dopant-vacancy", "dopant-vacancy-clamped", "host-vacancy-clamped",
        "absent-species-vacancy", "elements-on-both-sites", "one-in-32-grid"])
def test_formula_matches_every_built_configuration(tmp_path, candidate):
    builder = DefectSiteBuilder({"base_poscar": _template(tmp_path)}, seed=12)
    formula = _parse(builder.composition_formula(candidate))
    for atoms in builder.build(candidate, n_configs=3):
        actual = {element: count / 32 for element, count in
                  Counter(atoms.get_chemical_symbols()).items()}
        assert formula == pytest.approx(actual, abs=1e-12)


def test_pristine_formula_and_fraction_precision(tmp_path):
    builder = DefectSiteBuilder({"base_poscar": _template(tmp_path)})
    assert builder.composition_formula(_candidate()) == "SrFeO3"
    formula = builder.composition_formula(_candidate(
        {"Sr": .96875, "Ba": .03125}, {"Fe": .1875, "Co": .8125}))
    assert "0.03125" in formula and "0.96875" in formula
    assert _parse(formula) == {"Ba": .03125, "Sr": .96875,
                              "Co": .8125, "Fe": .1875, "O": 3}


def test_mixed_anions_and_unequal_site_pools_use_one_normalization(tmp_path):
    builder = DefectSiteBuilder({"base_poscar": _template(
        tmp_path, n_a=8, n_b=4, spectators={"O": 16, "F": 4})})
    candidate = _candidate({"Sr": .75, "Ca": .25}, {"Fe": .5, "Co": .5})
    formula = _parse(builder.composition_formula(candidate))
    atoms = builder.build(candidate)[0]
    actual = {element: count / 8 for element, count in
              Counter(atoms.get_chemical_symbols()).items()}
    assert formula == pytest.approx(actual)
    assert formula == {"Sr": .75, "Ca": .25, "Fe": .25,
                       "Co": .25, "O": 2, "F": .5}


def test_formula_uses_realized_integer_counts_in_small_cells(tmp_path):
    builder = DefectSiteBuilder({"base_poscar": _template(tmp_path, n_a=3, n_b=3)})
    candidate = _candidate({"Sr": .8, "Ba": .2}, {"Fe": .8, "Co": .2})
    formula = _parse(builder.composition_formula(candidate))
    actual = Counter(builder.build(candidate)[0].get_chemical_symbols())
    assert formula == pytest.approx({element: count / 3 for element, count in actual.items()},
                                   abs=1e-12)
    assert formula["Ba"] == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize("species", ["Sr", "Ca"])
def test_interstitial_atom_is_included_as_element(tmp_path, species):
    builder = DefectSiteBuilder({"base_poscar": _template(tmp_path, n_a=4, n_b=4),
                                 "interstitial_min_dist": .1})
    candidate = _candidate(defect=f"{species}_0.25")
    formula = _parse(builder.composition_formula(candidate))
    actual = Counter(builder.build(candidate)[0].get_chemical_symbols())
    assert formula == pytest.approx({element: count / 4 for element, count in actual.items()})
    assert formula[species] == (1.25 if species == "Sr" else .25)


def test_formula_lookup_never_builds_or_relaxes(tmp_path, monkeypatch):
    import rl_matdesign.utils.structure as structure

    builder = DefectSiteBuilder({"base_poscar": _template(tmp_path)})

    def forbidden(*args, **kwargs):
        raise AssertionError("formula lookup must not run structure construction or physics")

    monkeypatch.setattr(builder, "build", forbidden)
    monkeypatch.setattr(structure, "build_substituted_structure", forbidden)
    monkeypatch.setattr(structure, "relax_structure", forbidden)
    assert builder.composition_formula(_candidate()) == "SrFeO3"


@pytest.mark.parametrize("share_structure", [True, False])
@pytest.mark.parametrize("charge_filter", [True, False])
def test_generation_exports_full_formula_through_predictor(
    tmp_path, monkeypatch, share_structure, charge_filter,
):
    import torch
    import rl_matdesign.training as training
    import rl_matdesign.constraints.charge as charge
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    predictor = StructureScorePredictor({
        "builder": "defect_site", "base_poscar": _template(tmp_path),
        "share_structure": share_structure, "n_random_configs": 2,
        "properties": [{"name": "delta", "predictor": "dp_property", "models": ["unused"],
                        "direction": "max", "objective": "mean"}],
    })
    scored = []

    def score(prop, structures, **kwargs):
        scored.extend(structures)
        return 5.0, 0.0

    monkeypatch.setattr(predictor, "_score", score)
    candidate = _candidate({"Ba": 1}, {"Co": .8125, "Mn": .1875})
    env = SimpleNamespace(
        terminal_formula="Ba1Co0.812Mn0.188", path=[],
        terminal_cation_fractions=lambda: candidate,
        terminal_comp_key=lambda: (("candidate", 1),),
    )

    def rollout(**kwargs):
        env.path = [SimpleNamespace(reward=predictor.predict(candidate)[0])]

    monkeypatch.setattr(training, "_rollout_policy_episode", rollout)
    charge_checked = []

    def charge_neutral(formula, **kwargs):
        charge_checked.append(_parse(formula))
        return True

    monkeypatch.setattr(charge, "charge_neutral", charge_neutral)
    try:
        rows = training.generate_candidates(
            env=env, predictor=predictor, scaler=None, device=torch.device("cpu"),
            elem_feats_scaled=np.empty((0, 0)), fraction_set=[],
            qnet=SimpleNamespace(eval=lambda: None), n_exploit=1, max_attempts=1,
            charge_filter=charge_filter,
        )
        path = tmp_path / "generated.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with path.open(newline="") as handle:
            exported = next(csv.DictReader(handle))
        assert _parse(exported["formula"]) == {"Ba": 1, "Co": .8125, "Mn": .1875, "O": 3}
        assert charge_checked == ([_parse(exported["formula"])] if charge_filter else [])
        assert float(exported["reward"]) == 5
        assert len(scored) == 2  # Raw stats and label lookup reuse the evaluated candidate.
        assert _parse(exported["formula"]) == pytest.approx({
            element: count / 32 for element, count in
            Counter(scored[0].get_chemical_symbols()).items()})
    finally:
        predictor.close()
