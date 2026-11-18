"""Complete formulas must describe the actual pipeline atom counts."""
from __future__ import annotations

import csv
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ase")

from rl_matdesign.predictors.builders.pipeline import PipelineBuilder


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


def _config(path, **step_overrides):
    return {"builder": "pipeline", "base_poscar": path, "steps": [
        {"type": "substitute", "a_site_symbol": "Sr", "a_dopant_group": "A_dopant", "a_defect_group": "A_defect",
         "b_site_symbol": "Fe", "b_dopant_group": "B_dopant", **step_overrides}]}


def _candidate(a=None, b=None, defect=None):
    candidate = {"A_dopant": a or {"Sr": 1.0}, "B_dopant": b or {"Fe": 1.0}, "A_defect": {}}
    if defect is not None:
        element, amount = defect.split("_")
        candidate["A_defect"] = {element: float(amount)}
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
    builder = PipelineBuilder(_config(_template(tmp_path)), seed=12)
    formula = _parse(builder.composition_formula(candidate))
    for atoms in builder.build(candidate, n_configs=3):
        actual = {element: count for element, count in
                  Counter(atoms.get_chemical_symbols()).items()}
        assert formula == pytest.approx(actual, abs=1e-12)


def test_pristine_formula_and_fraction_precision(tmp_path):
    builder = PipelineBuilder(_config(_template(tmp_path)))
    assert _parse(builder.composition_formula(_candidate())) == {"Sr":32,"Fe":32,"O":96}
    formula = builder.composition_formula(_candidate(
        {"Sr": .96875, "Ba": .03125}, {"Fe": .1875, "Co": .8125}))
    assert "Ba1" in formula and "Sr31" in formula
    assert _parse(formula) == {"Ba": 1, "Sr": 31, "Co": 26, "Fe": 6, "O": 96}


def test_mixed_anions_and_unequal_site_pools_use_one_normalization(tmp_path):
    builder = PipelineBuilder(_config(_template(
        tmp_path, n_a=8, n_b=4, spectators={"O": 16, "F": 4})))
    candidate = _candidate({"Sr": .75, "Ca": .25}, {"Fe": .5, "Co": .5})
    formula = _parse(builder.composition_formula(candidate))
    atoms = builder.build(candidate)[0]
    actual = {element: count for element, count in
              Counter(atoms.get_chemical_symbols()).items()}
    assert formula == pytest.approx(actual)
    assert formula == {"Sr": 6, "Ca": 2, "Fe": 2, "Co": 2, "O": 16, "F": 4}


def test_formula_uses_realized_integer_counts_in_small_cells(tmp_path):
    builder = PipelineBuilder(_config(_template(tmp_path, n_a=3, n_b=3)))
    candidate = _candidate({"Sr": .8, "Ba": .2}, {"Fe": .8, "Co": .2})
    formula = _parse(builder.composition_formula(candidate))
    actual = Counter(builder.build(candidate)[0].get_chemical_symbols())
    assert formula == pytest.approx({element: count for element, count in actual.items()},
                                   abs=1e-12)
    assert formula["Ba"] == 1


@pytest.mark.parametrize("species", ["Sr", "Ca"])
def test_interstitial_atom_is_included_as_element(tmp_path, species):
    builder = PipelineBuilder(_config(_template(tmp_path, n_a=4, n_b=4), interstitial_min_dist=.1))
    candidate = _candidate(defect=f"{species}_0.25")
    formula = _parse(builder.composition_formula(candidate))
    actual = Counter(builder.build(candidate)[0].get_chemical_symbols())
    assert formula == pytest.approx({element: count for element, count in actual.items()})
    assert formula[species] == (5 if species == "Sr" else 1)


def test_formula_lookup_never_builds_or_relaxes(tmp_path, monkeypatch):
    import rl_matdesign.utils.structure as structure

    builder = PipelineBuilder(_config(_template(tmp_path)))

    def forbidden(*args, **kwargs):
        raise AssertionError("formula lookup must not run structure construction or physics")

    monkeypatch.setattr(builder, "build", forbidden)
    monkeypatch.setattr(structure, "build_substituted_structure", forbidden)
    monkeypatch.setattr(structure, "relax_structure", forbidden)
    assert _parse(builder.composition_formula(_candidate())) == {"Sr":32,"Fe":32,"O":96}


@pytest.mark.parametrize("charge_filter", [True, False])
def test_generation_exports_full_formula_through_predictor(
    tmp_path, monkeypatch, charge_filter,
):
    import torch
    import rl_matdesign.training as training
    import rl_matdesign.constraints.charge as charge
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    cfg = {"structure": {**_config(_template(tmp_path)), "n_random_configs": 2},
           "properties": [{"name": "delta", "predictor": "dp_property", "models": ["unused"],
                           "direction": "max", "objective": "mean"}]}
    predictor = StructureScorePredictor(cfg)
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
            elem_feats_scaled=np.empty((0, 0)), action_values=[],
            qnet=SimpleNamespace(eval=lambda: None), num_gen_eps=1, max_attempts=1,
            charge_filter=charge_filter,
        )
        path = tmp_path / "generated.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with path.open(newline="") as handle:
            exported = next(csv.DictReader(handle))
        assert _parse(exported["formula"]) == {"Ba": 32, "Co": 26, "Mn": 6, "O": 96}
        assert charge_checked == ([_parse(exported["formula"])] if charge_filter else [])
        assert float(exported["reward"]) == 5
        assert len(scored) == 2  # Raw stats and label lookup reuse the evaluated candidate.
        assert _parse(exported["formula"]) == pytest.approx({
            element: count for element, count in
            Counter(scored[0].get_chemical_symbols()).items()})
    finally:
        predictor.close()


def test_four_block_substitution_counts_variation_lattice_and_formula(tmp_path):
    from ase import Atoms
    path = tmp_path / 'double-perovskite.vasp'
    symbols = ['Cs']*8 + ['Ag']*4 + ['Bi']*4 + ['Cl']*8
    template = Atoms(symbols, positions=np.arange(72).reshape(-1,3), cell=[40]*3, pbc=True)
    template.write(path, format='vasp')
    cfg = {'builder':'pipeline', 'base_poscar':str(path), 'steps':[{
        'type':'substitute', 'a_site_symbol':'Cs', 'a_dopant_group':'A',
        'b_site_symbol':'Ag', 'b_dopant_group':'B1', 'c_site_symbol':'Bi', 'c_dopant_group':'B3',
        'd_site_symbol':'Cl', 'd_dopant_group':'X'}]}
    builder = PipelineBuilder(cfg)
    candidate = {'A':{'Cs':.75,'Rb':.25}, 'B1':{'Ag':.5,'Cu':.25,'Na':.25},
                 'B3':{'Bi':.75,'Sb':.25}, 'X':{'Cl':.5,'Br':.25,'I':.25}}
    cells = builder.build(candidate,n_configs=4,rng=np.random.default_rng(1))
    expected = {'Cs':6,'Rb':2,'Ag':2,'Cu':1,'Na':1,'Bi':3,'Sb':1,'Cl':4,'Br':2,'I':2}
    assert _parse(builder.composition_formula(candidate)) == expected
    assert len({tuple(cell.get_chemical_symbols()) for cell in cells}) > 1
    for cell in cells:
        assert dict(Counter(cell.get_chemical_symbols())) == expected
        np.testing.assert_equal(cell.positions,template.positions)
        np.testing.assert_equal(cell.cell.array,template.cell.array)
    with pytest.raises((KeyError,ValueError),match='B1'):
        builder.build({'A':{'Cs':1}})
    with pytest.raises(ValueError,match='sum'):
        builder.build({**candidate,'A':{'Cs':.5}})


_SSE_VALENCES = {'Li':1,'P':5,'S':-2,'O':-2,'Cl':-1,'Br':-1,
                 'Mn':{'sulfide':2,'oxide':4},'W':{'sulfide':4,'oxide':6}}


def _sse(tmp_path,fu=500,*,primary=None,fallback=None,template_name='sse.vasp'):
    from ase import Atoms
    symbols=['Li']*(6*fu)+['P']*fu+['S']*(6*fu)
    path=tmp_path/template_name
    Atoms(symbols,positions=np.arange(len(symbols)*3).reshape(-1,3),cell=[50000]*3,pbc=True).write(path,format='vasp')
    step={'type':'sse_charge_balance','p_site_group':'P_site','halide_group':'halide',
          'valences':_SSE_VALENCES,'halide_total':1.7}
    if primary is not None:
        step['eligible_region']={'symbol':'S','take':'last','count':primary}
    if fallback is not None:
        step['eligible_region_fallback']={'symbol':'S','take':'first','count':fallback}
    return PipelineBuilder({'builder':'pipeline','base_poscar':str(path),'steps':[
        {'type':'substitute','a_site_symbol':'P','a_dopant_group':'P_site','amount_basis':'cation_fraction'},step]})


@pytest.mark.parametrize('amounts,metals,oxygen,delete', [
    ({'P':.95,'Mn':.05},{'Mn':25},0,275),
    ({'P':.94,'WO3':.06},{'W':30},90,380),
    ({'P':.89,'Mn':.05,'WO3':.06},{'Mn':25,'W':30},90,305),
    ({'P':.89,'MnO2':.05,'W':.06},{'Mn':25,'W':30},50,295),
])
def test_sse_explicit_compound_recipes_preserve_charge_derivation(tmp_path,amounts,metals,oxygen,delete):
    builder=_sse(tmp_path)
    candidate={'P_site':amounts,'halide':{'Cl':1.}}
    counts=builder.plan_candidate(candidate)['sse']['P_site']
    assert counts['formula_units']==500
    assert counts['metals']==metals and counts['O']==oxygen and counts['Li_delete']==delete
    cation=counts['Li']+5*counts['P']+sum(n*_SSE_VALENCES[m]['oxide' if counts['o_forms'][m] else 'sulfide'] for m,n in metals.items())
    anion=2*counts['S']+2*counts['O']+counts['Cl']+counts['Br']
    assert cation==anion
    assert counts['Br']==850-counts['Cl']


@pytest.mark.parametrize('primary,fallback,expected_size',[(60,120,60),(20,80,80)])
def test_sse_region_uses_original_ids_and_falls_back_only_on_overflow(tmp_path,primary,fallback,expected_size):
    from rl_matdesign.predictors.builders.pipeline import ORIGINAL_IDS
    builder=_sse(tmp_path,20,primary=primary,fallback=fallback)
    candidate={'P_site':{'P':.8,'MnO2':.1,'WO3':.1},'halide':{'Cl':1.}}
    plan=builder.plan_candidate(candidate)
    expected_ids=set(plan['sse']['P_site']['anion_original_ids'])
    assert len(expected_ids)==expected_size
    atoms=builder.build(candidate)[0]
    actual_ids={int(atoms.arrays[ORIGINAL_IDS][i]) for i,a in enumerate(atoms) if a.symbol in ('O','Cl','Br')}
    assert actual_ids and actual_ids <= expected_ids
    assert dict(Counter(atoms.get_chemical_symbols()))==plan['counts']


def test_sse_region_overflow_without_fallback_raises(tmp_path):
    builder=_sse(tmp_path,20,primary=20)
    with pytest.raises(ValueError,match='enough original host S sites'):
        builder.build({'P_site':{'P':.8,'MnO2':.1,'WO3':.1},'halide':{'Cl':1.}})


def test_sse_multiple_templates_infer_formula_units_independently(tmp_path):
    first=_sse(tmp_path,4,template_name='first.vasp')
    second=_sse(tmp_path,8,template_name='second.vasp')
    cfg={'builder':'pipeline','base_poscar':[{'path':first.specs[0].path,'n_configs':2},
                                          {'path':second.specs[0].path,'n_configs':3}],
         'steps':first.steps}
    builder=PipelineBuilder(cfg)
    candidate={'P_site':{'P':.75,'Mn':.25},'halide':{'Cl':1.}}
    plans=[builder.plan_candidate(candidate,template_index=i) for i in (0,1)]
    assert [p['sse']['P_site']['formula_units'] for p in plans]==[4,8]
    cells=builder.build(candidate,n_configs=99)
    assert [dict(Counter(cell.get_chemical_symbols())) for cell in cells]==[plans[0]['counts']]*2+[plans[1]['counts']]*3
