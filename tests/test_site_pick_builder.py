"""SitePickBuilder tests — one-element-per-site substitution across N sites.

The multi-site generalization of SubstituteBuilder: each of several placeholder
site symbols gets exactly the one element its group picked (a MultiGroupEnv
candidate from one-slot categorical groups), no fractions, no vacancies.
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_PEROVSKITE_POSCAR = os.path.join(_REPO_ROOT, "perovskite.vasp")


def _two_site_template(tmp_path, n_a=3, n_b=2):
    from ase import Atoms
    from ase.io import write

    # n_a "Sr" placeholders (A-site) + n_b "Fe" placeholders (B-site), same
    # convention as the real perovskite.vasp fixture, + a couple spectator O.
    syms = ["Sr"] * n_a + ["Fe"] * n_b + ["O", "O"]
    atoms = Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))],
                  cell=[40, 40, 40], pbc=True)
    path = str(tmp_path / "tmpl.vasp")
    write(path, atoms, format="vasp")
    return path


def test_registry_resolves_site_pick_builder(tmp_path):
    from rl_matdesign.registry import resolve_builder, BUILDERS
    assert "site_pick" in BUILDERS
    b = resolve_builder("site_pick", {
        "base_poscar": _two_site_template(tmp_path), "site_map": {"g1": "Sr", "g2": "Fe"},
    })
    assert hasattr(b, "build")


def test_missing_poscar_raises():
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    with pytest.raises(ValueError) as info:
        SitePickBuilder({"site_map": {"g1": "Sr"}})
    assert "POSCAR" in str(info.value) or "base_poscar" in str(info.value)


def test_missing_site_map_raises(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    with pytest.raises(ValueError) as info:
        SitePickBuilder({"base_poscar": _two_site_template(tmp_path)})
    assert "site_map" in str(info.value)


def test_multi_template_list_rejected(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    with pytest.raises(TypeError):
        SitePickBuilder({
            "base_poscar": [_two_site_template(tmp_path)], "site_map": {"g1": "Sr"},
        })


def test_unknown_group_in_site_map_raises(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    b = SitePickBuilder({
        "base_poscar": _two_site_template(tmp_path), "site_map": {"ghost": "A"},
    })
    with pytest.raises(KeyError):
        b.build({"g1": {"A": "Fe"}}, n_configs=1, rng=np.random.default_rng(0))


def test_multi_slot_group_rejected(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    b = SitePickBuilder({
        "base_poscar": _two_site_template(tmp_path), "site_map": {"g1": "Sr"},
    })
    with pytest.raises(ValueError):
        b.build({"g1": {"A": "Fe", "extra": "Ni"}}, n_configs=1, rng=np.random.default_rng(0))


def test_unknown_site_symbol_raises(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    b = SitePickBuilder({
        "base_poscar": _two_site_template(tmp_path), "site_map": {"g1": "Zzz"},
    })
    with pytest.raises(ValueError):
        b.build({"g1": {"A": "Fe"}}, n_configs=1, rng=np.random.default_rng(0))


def test_build_places_one_element_per_site_all_placeholders(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder

    b = SitePickBuilder({
        "base_poscar": _two_site_template(tmp_path, n_a=3, n_b=2),
        "site_map": {"g1": "Sr", "g2": "Fe"},
    })
    candidate = {"g1": {"elem": "Fe"}, "g2": {"elem": "Ni"}}
    for st in b.build(candidate, n_configs=3, rng=np.random.default_rng(0)):
        cnt = Counter(st.get_chemical_symbols())
        assert cnt["Fe"] == 3 and cnt["Ni"] == 2
        assert cnt["O"] == 2
        assert "A" not in cnt and "B" not in cnt


def test_lattice_and_atom_count_unchanged(tmp_path):
    from ase.io import read as ase_read
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder

    poscar = _two_site_template(tmp_path, n_a=3, n_b=2)
    orig = ase_read(poscar)
    b = SitePickBuilder({"base_poscar": poscar, "site_map": {"g1": "Sr", "g2": "Fe"}})
    out = b.build({"g1": {"elem": "Fe"}, "g2": {"elem": "Ni"}},
                   n_configs=1, rng=np.random.default_rng(0))[0]
    assert len(out) == len(orig)
    assert (out.get_cell()[:] == orig.get_cell()[:]).all()


@pytest.fixture(params=[1, 4, "local"])
def perovskite_poscar(request, tmp_path):
    """Check a unit cell, a 320-atom supercell, and the optional local template."""
    if request.param == "local":
        if not os.path.exists(_PEROVSKITE_POSCAR):
            pytest.skip("perovskite.vasp not present")
        return _PEROVSKITE_POSCAR
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("SrFeO3", cell=[4, 4, 4], pbc=True, scaled_positions=[
        (0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0),
        (0.5, 0, 0.5), (0, 0.5, 0.5),
    ]).repeat((request.param,) * 3)
    path = str(tmp_path / "perovskite.vasp")
    write(path, atoms, format="vasp")
    return path


def test_perovskite_a_b_site_substitution(perovskite_poscar):
    """Real fixture: perovskite.vasp is SrFeO3 with Sr=A-site, Fe=B-site placeholders."""
    from ase.io import read as ase_read
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder

    orig = ase_read(perovskite_poscar)
    b = SitePickBuilder({
        "base_poscar": perovskite_poscar, "site_map": {"A_site": "Sr", "B_site": "Fe"},
    })
    candidate = {"A_site": {"A": "Ba"}, "B_site": {"B": "Nb"}}
    out = b.build(candidate, n_configs=2, rng=np.random.default_rng(0))

    source_counts = Counter(orig.get_chemical_symbols())
    assert set(source_counts) == {"Sr", "Fe", "O"}
    assert len(out) == 2
    for st in out:
        cnt = Counter(st.get_chemical_symbols())
        assert cnt == {"Ba": source_counts["Sr"], "Nb": source_counts["Fe"],
                       "O": source_counts["O"]}
        assert len(st) == len(orig)
    assert (out[0].get_cell()[:] == orig.get_cell()[:]).all()


def test_composition_formula_places_picks_at_real_site_counts(tmp_path):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder

    b = SitePickBuilder({
        "base_poscar": _two_site_template(tmp_path, n_a=3, n_b=2),
        "site_map": {"g1": "Sr", "g2": "Fe"},
    })
    formula = b.composition_formula({"g1": {"elem": "Fe"}, "g2": {"elem": "Ni"}})
    # 3 Sr-placeholders -> Fe, 2 Fe-placeholders -> Ni, 2 spectator O untouched.
    import re
    parsed = dict(re.findall(r"([A-Z][a-z]?)([0-9.]+)", formula))
    assert parsed == {"Fe": "3", "Ni": "2", "O": "2"}


def test_perovskite_composition_formula(perovskite_poscar):
    from rl_matdesign.predictors.builders.site_pick import SitePickBuilder
    import re

    b = SitePickBuilder({
        "base_poscar": perovskite_poscar, "site_map": {"A_site": "Sr", "B_site": "Fe"},
    })
    formula = b.composition_formula({"A_site": {"A": "Ba"}, "B_site": {"B": "Nb"}})
    parsed = dict(re.findall(r"([A-Z][a-z]?)([0-9.]+)", formula))
    from ase.io import read as ase_read
    source_counts = Counter(ase_read(perovskite_poscar).get_chemical_symbols())
    assert set(source_counts) == {"Sr", "Fe", "O"}
    assert {el: float(n) for el, n in parsed.items()} == {
        "Ba": source_counts["Sr"], "Nb": source_counts["Fe"], "O": source_counts["O"],
    }
