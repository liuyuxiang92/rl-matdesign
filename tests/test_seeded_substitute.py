"""Real ASE checks for the substitute builder's deterministic sampling contract."""
from collections import Counter

import pytest

pytest.importorskip('ase')
from ase import Atoms
from rl_matdesign.predictors.builders.substitute import SubstituteBuilder
from rl_matdesign.structure_sources import StructureSourceSpec
from rl_matdesign.utils.structure_samples import SAMPLE_INFO_KEY


def template(tmp_path, name, symbol='X', size=10):
    path = tmp_path / name
    atoms = Atoms([symbol] * size + ['O'], positions=[(i, 0, 0) for i in range(size + 1)],
                  cell=[size + 5, 10, 10], pbc=True)
    atoms.write(path, format='vasp')
    return str(path)


def signature(cells):
    return [(tuple(cell.get_chemical_symbols()), tuple(cell.positions.ravel()),
             cell.info[SAMPLE_INFO_KEY]) for cell in cells]


def _builder_and_source(tmp_path):
    cfg = {'builder': 'substitute', 'n_random_configs': 3,
           'base_poscar': [{'path': template(tmp_path, 'a.vasp'), 'n_configs': 2},
                           {'path': template(tmp_path, 'b.vasp', 'Fe', 20), 'site_symbol': 'Fe'}]}
    return SubstituteBuilder(cfg), StructureSourceSpec('bulk', cfg)


def test_serial_seeded_multi_template_scalar_batch_reorder_and_mapping_parity(tmp_path):
    builder, source = _builder_and_source(tmp_path)
    a, b = {'Ni': .5, 'Co': .5}, {'Ni': .7, 'Co': .3}
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=11)
    first = builder.build_many_seeded([a, b], **kwargs)
    assert [len(group) for group in first] == [5, 5]
    assert [len(cell) for cell in first[0]] == [11, 11, 21, 21, 21]
    assert [r['status'] for r in builder.last_sample_records[0]] == ['succeeded'] * 5
    second = builder.build_many_seeded([b, dict(reversed(list(a.items())))], **kwargs)
    assert signature(first[0]) == signature(second[1]) == signature(builder.build_seeded(a, **kwargs))
    assert signature(first[1]) == signature(second[0])
    for cells in first:
        assert all(Counter(cell.get_chemical_symbols())['O'] == 1 for cell in cells)
    assert [cell.info[SAMPLE_INFO_KEY]['template_id']['site_symbol'] for cell in first[0]] == ['X'] * 2 + ['Fe'] * 3


def test_site_map_seeded_groups_and_species_order_are_irrelevant(tmp_path):
    path = tmp_path / 'multi.vasp'
    atoms = Atoms(['Cs'] * 8 + ['Cl'] * 8, positions=[(i, 0, 0) for i in range(16)],
                  cell=[25, 10, 10], pbc=True)
    atoms.write(path, format='vasp')
    cfg = {'builder': 'substitute', 'base_poscar': [{'path': str(path), 'n_configs': 2}],
           'site_map': {'cation': 'Cs', 'anion': 'Cl'}}
    source = StructureSourceSpec('bulk', cfg)
    builder = SubstituteBuilder(cfg)
    candidate = {'cation': {'Cs': .5, 'Rb': .5}, 'anion': {'Cl': .5, 'Br': .5}}
    reverse = {group: dict(reversed(list(values.items()))) for group, values in reversed(list(candidate.items()))}
    kwargs = dict(n_configs=9, source_spec=source, eval_seed=5)
    before = builder.build_seeded(candidate, **kwargs)
    after = builder.build_seeded(reverse, **kwargs)
    assert len(before) == 2
    assert signature(before) == signature(after)


def test_seeded_partial_failure_keeps_sample_ids_and_successful_template(tmp_path):
    cfg = {'builder': 'substitute', 'base_poscar': [template(tmp_path, 'valid.vasp'),
                                                   str(tmp_path / 'missing.vasp')]}
    source = StructureSourceSpec('bulk', cfg)
    builder = SubstituteBuilder(cfg)
    result = builder.build_seeded({'Ni': 1}, n_configs=2, source_spec=source, eval_seed=4)
    assert len(result) == 2
    records = builder.last_sample_records[0]
    assert [r['status'] for r in records] == ['succeeded', 'succeeded', 'failed', 'failed']
    assert [r['template_id']['index'] for r in records] == [0, 0, 1, 1]
    assert [r['sample_index'] for r in records] == [0, 1, 0, 1]
    assert all('FileNotFoundError' in r['error'] for r in records[2:])
