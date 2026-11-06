"""OOH uses the common candidate/source/sample random protocol."""
from dataclasses import replace
import random

from abcde_ooh.dp_predictor import DPConfig, _choose_counts_from_fractions, ooh_structure_source
from tests.test_refactor_contracts import _sampling_probe


def test_count_allocation_resolves_equal_remainders_by_label():
    a = {'Ni': .5, 'Fe': .5}
    assert _choose_counts_from_fractions(3, a) == _choose_counts_from_fractions(3, dict(reversed(list(a.items()))))
    assert _choose_counts_from_fractions(3, a) == {'Fe': 2, 'Ni': 1}


def test_source_seed_excludes_model_output_debug_and_reward_settings(tmp_path):
    template = tmp_path / 'slab'
    template.write_bytes(b'fixed slab contents')
    cfg = DPConfig(base_poscar=str(template))
    source = ooh_structure_source(cfg)
    other = ooh_structure_source(replace(cfg, model_files=('other.pt',), output_index=2,
                                         debug_dir='other_dir', debug_max_configs=1))
    candidate = {'Ni': .5, 'Fe': .5}
    assert source.to_schema() == other.to_schema()
    assert source.seed_for(candidate, eval_seed=1, sample_index=0) != source.seed_for(candidate, eval_seed=1, sample_index=1)
    assert source.seed_for(candidate, eval_seed=1, sample_index=0) != source.seed_for(candidate, eval_seed=2, sample_index=0)
    template.write_bytes(b'changed slab contents')
    changed = ooh_structure_source(cfg)
    assert source.protocol_fingerprint != changed.protocol_fingerprint


def test_bare_slab_ignores_unused_adsorbate_geometry():
    bare = DPConfig(adsorbates=())
    assert ooh_structure_source(bare).to_schema() == ooh_structure_source(
        replace(bare, ads_height=100, ads_dz=100)).to_schema()


def test_direct_sampling_retains_stable_sample_records_and_ignores_zero_components():
    predictor, sites = _sampling_probe(7)
    a = {'Ni': .5, 'Fe': .5}
    predictor.predict_overpotential(a)
    first = list(sites)
    records = list(predictor.last_sample_records)
    sites.clear()
    state = random.getstate()
    predictor.predict_overpotential({'Co': 0., 'Fe': .5, 'Ni': .5})
    assert random.getstate() == state
    assert sites == first
    assert predictor.last_sample_records == records
    assert len({r['seed'] for r in records}) == len(records)


def test_wrapper_cache_key_preserves_finer_than_historical_grid():
    from rl_matdesign.predictors.ooh import OOHCatalystPredictor
    predictor = object.__new__(OOHCatalystPredictor)
    predictor._settings_key = ((), 0)
    first = {'Ni': .501, 'Fe': .499}
    second = {'Ni': .502, 'Fe': .498}
    assert predictor._comp_key(first) != predictor._comp_key(second)
    assert predictor._comp_key(first) == predictor._comp_key({'Co': 0, **first})


def test_debug_configuration_limit_does_not_change_scientific_samples_or_result():
    candidate = {'Ni': .5, 'Fe': .5}
    ordinary, ordinary_sites = _sampling_probe(7)
    expected = ordinary.predict_overpotential(candidate)
    limited, limited_sites = _sampling_probe(7)
    limited.cfg = replace(limited.cfg, debug_dir='unused_debug_path', debug_max_configs=1)
    dumped = []
    limited._maybe_dump_frames = lambda **kw: dumped.append(kw['config_idx'])
    assert limited.predict_overpotential(candidate) == expected
    assert limited_sites == ordinary_sites and len(limited_sites) == 4
    assert limited.last_sample_records == ordinary.last_sample_records
    assert dumped == [0]
    # Even without writing files, the old code incorrectly truncated sampling.
    limited_sites.clear()
    limited.cfg = replace(limited.cfg, debug_dir=None, debug_max_configs=0)
    assert limited.predict_overpotential(candidate) == expected
    assert limited_sites == ordinary_sites


def test_relaxation_model_bytes_belong_to_source_only_when_relaxation_is_enabled(tmp_path):
    template, model = tmp_path / 'slab', tmp_path / 'relax.pt'
    template.write_bytes(b'slab')
    model.write_bytes(b'relax model A')
    cfg = DPConfig(base_poscar=str(template), geo_opt=True, geo_opt_model=str(model))
    source = ooh_structure_source(cfg)
    disabled = ooh_structure_source(replace(cfg, geo_opt=False))
    hashes = source.physical_protocol['asset_sha256']
    assert set(hashes) == {str(template), str(model)}
    model.write_bytes(b'relax model B')
    assert ooh_structure_source(cfg).protocol_fingerprint != source.protocol_fingerprint
    assert ooh_structure_source(replace(cfg, geo_opt=False)).to_schema() == disabled.to_schema()


def test_adsorbate_random_consumption_cannot_move_alloy_decoration_stream():
    candidate = {'Ni': .5, 'Fe': .5}
    plain, plain_sites = _sampling_probe(17)
    expected = plain.predict_overpotential(candidate)
    extra, extra_sites = _sampling_probe(17)
    prepare = extra._build_dp_inputs_for_one_doped_slab
    adsorbate_draws = []
    def consume(doped, *, anchor_elems, rng):
        adsorbate_draws.append([rng.random() for _ in range(100)])
        return prepare(doped, anchor_elems=anchor_elems, rng=rng)
    extra._build_dp_inputs_for_one_doped_slab = consume
    assert extra.predict_overpotential(candidate) == expected
    assert extra_sites == plain_sites
    source = extra.structure_source
    for index, draws in enumerate(adsorbate_draws):
        seed = source.seed_for(candidate, eval_seed=17, purpose='adsorbates',
                               template_id='slab', sample_index=index)
        assert draws[0] == random.Random(seed).random()
        assert seed != extra.last_sample_records[index]['seed']
