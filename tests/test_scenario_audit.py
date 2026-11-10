"""Audit failures are executable checks, not self-assertions in a static report."""
from pathlib import Path
import runpy

import yaml

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/audit_scenarios.py'))['audit_inventory']


def test_audit_detects_missing_hpo_reference(tmp_path):
    (tmp_path / 'hpo.yaml').write_text('base_config: missing.yaml\nsearch_space: {}\n')
    report = AUDIT(tmp_path, ['hpo.yaml'])
    assert report['failed'] == 1
    assert 'base_config does not exist' in report['entries'][0]['error']


def test_audit_detects_singleton_uncertainty(tmp_path):
    cfg = {'properties': [{'name': 'e', 'predictor': 'dp_energy', 'models': ['x.pt'],
                          'direction': 'min', 'objective': 'mean_minus_kstd',
                          'uncertainty': 'model_spread', 'source': 'bulk'}],
           'structure_sources': {'bulk': {'builder': 'substitute', 'base_poscar': 'missing'}}}
    (tmp_path / 'test.yaml').write_text(yaml.safe_dump(cfg))
    report = AUDIT(tmp_path, ['test.yaml'])
    assert report['failed'] == 1
    assert 'singleton model_spread' in report['entries'][0]['error']


def test_audit_keeps_invalid_filters_visible(tmp_path):
    cfg = {'groups': [{'name': 'alloy', 'kind': 'composition', 'species_set': ['Ni', 'Fe'],
                      'n_components': 2, 'fraction_set': ['0.5'],
                      'filters': [{'type': 'misspelled_filter'}]}], 'predictor': 'dummy'}
    (tmp_path / 'test.yaml').write_text(yaml.safe_dump(cfg))
    report = AUDIT(tmp_path, ['test.yaml'], domains=True)
    assert report['failed'] == 1
    assert 'misspelled_filter' in report['entries'][0]['error']


def test_hpo_accepts_absolute_base_outside_repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    base = tmp_path / 'external.yaml'
    base.write_text('predictor: dummy\nmethod: dqn\n')
    (root / 'hpo.yaml').write_text(yaml.safe_dump({'base_config': str(base), 'method': 'dqn',
        'search_space': {'dqn_lr': {'dist': 'uniform', 'low': 0.0001, 'high': 0.001}}}))
    report = AUDIT(root, ['hpo.yaml'])
    assert report['failed'] == 0, report
    assert report['entries'][0]['base_config'] == str(base)
