import pytest

from scripts.evaluate_formulas_dp import formula_candidate


def group():
    return dict(name='alloy',kind='composition',species_set=['Ni','Fe','Co'],amount_set=[.25,.5,.75],n_steps=2,total=1,unique=True)


def test_formula_maps_exact_amounts_into_named_group():
    assert formula_candidate('Ni0.5Fe0.5',group())=={'alloy':{'Ni':.5,'Fe':.5}}


@pytest.mark.parametrize('formula',['NiFe','Ni0.5Fe0.5O2H','Ni0.3Fe0.7'])
def test_formula_does_not_normalize_strip_scaffold_or_ignore_action_menus(formula):
    with pytest.raises(ValueError):
        formula_candidate(formula,group())


def test_elemental_formula_cannot_infer_precursor_form():
    spec={**group(),'species_set':['Ni','Ni2O3']}
    with pytest.raises(ValueError,match='precursor identity'):
        formula_candidate('Ni1',spec)
