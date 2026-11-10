"""Runtime source ownership, deterministic native ensembles and failure boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
from rl_matdesign.structure_sources import (
    LegacySourceMigrationRequired, StructureSourceSpec, resolve_source_assets,
)


class NativeBuilder:
    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.calls = []
        self.closed = 0
        self.samples = []

    def build(self, candidate, *, n_configs, rng):
        from ase import Atoms
        self.calls.append(deepcopy(candidate))
        # Both random draws and iteration order expose incorrect canonicalization.
        assert list(candidate) == sorted(candidate)
        self.samples = [Atoms("H", positions=[[rng.random(), 0, 0]],
                              info={"nested": {"untouched": [1]}}) for _ in range(n_configs)]
        return self.samples

    def build_many(self, *args, **kwargs):
        raise AssertionError("A native plugin's shared-RNG batch must not be called")

    def close(self):
        self.closed += 1
        if self.cfg.get("close_fails"):
            raise RuntimeError("close failed")


class StructureLeaf:
    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.seen = []
        self.closed = 0

    def score_structures(self, structures):
        self.seen.append([(cell.positions.copy(), deepcopy(cell.info)) for cell in structures])
        values = [len(cell) if self.cfg.get("atom_count") else cell.positions[0, 0]
                  for cell in structures]
        if self.cfg.get("mutates"):
            for cell in structures:
                cell.positions[:] = -999
                cell.info["nested"]["untouched"].append(999)
            structures.clear()
        return values

    def close(self):
        self.closed += 1


class CompositionLeaf:
    def score(self, candidate):
        return sum(candidate.values())


@pytest.fixture
def factories(monkeypatch):
    import rl_matdesign.registry as registry
    builders, leaves = [], []
    real_builder_factory = registry.resolve_builder

    def builder(kind, cfg, *, seed=None):
        if cfg.get("fail_init"):
            raise RuntimeError("builder initialization failed")
        result = NativeBuilder(cfg)
        builders.append(result)
        return result

    def leaf(kind, cfg, *, seed=None):
        if cfg.get("fail_init"):
            raise RuntimeError("leaf initialization failed")
        result = CompositionLeaf() if cfg.get("composition_only") else StructureLeaf(cfg)
        leaves.append(result)
        return result

    monkeypatch.setattr(registry, "resolve_builder", builder)
    monkeypatch.setattr(registry, "resolve_predictor", leaf)
    return SimpleNamespace(builders=builders, leaves=leaves,
                           real_builder_factory=real_builder_factory, registry=registry)


def config(*, independent=False, geo_opt=False):
    source = {"builder": "substitute", "base_poscar": "missing-template", "n_random_configs": 3}
    if geo_opt:
        source["geo_opt"] = {"enabled": True}
    ids = ["a", "b"] if independent else ["bulk"]
    return {
        "structure_sources": {name: deepcopy(source) for name in ids},
        "properties": [{"name": name, "predictor": "example:Leaf", "direction": "max",
                        "source": name if independent else "bulk", "objective": "mean"}
                       for name in ("a", "b")],
    }


def test_shared_source_builds_relaxes_once_and_cache_charges_no_work(factories):
    p = StructureScorePredictor(config(geo_opt=True), seed=11)
    relaxed = []
    p._relax = lambda cell, *, source_id: (relaxed.append(source_id) or cell)
    candidate = {"Ni": .6, "Fe": .4}
    p.predict(candidate)
    assert len(factories.builders[0].calls) == 1
    assert relaxed == ["bulk"] * 3
    assert p.last_evaluation_timings[0]["config_succeeded"] == 3
    assert set(p.last_evaluation_timings[0]["sources"]) == {"bulk"}
    stats = p.per_objective_stats(candidate)
    assert stats["a"] == stats["b"]
    diagnostics = p.source_diagnostics(candidate)
    assert diagnostics["bulk"]["sample_identity"] == "native_ensemble"
    assert len(diagnostics["bulk"]["samples"]) == 3
    assert diagnostics["bulk"]["samples"][0]["seed_scope"] == "native_ensemble"
    p.batch_predict([candidate, dict(reversed(list(candidate.items())))])
    assert len(factories.builders[0].calls) == 1
    assert all(row["cache_hit"] and row["config_succeeded"] == 0 for row in p.last_evaluation_timings)


def test_equal_definitions_under_distinct_names_do_not_share(factories):
    p = StructureScorePredictor(config(independent=True), seed=11)
    stats = p.per_objective_stats({"Fe": .4, "Ni": .6})
    assert len(factories.builders) == 2
    assert [len(b.calls) for b in factories.builders] == [1, 1]
    assert stats["a"] != stats["b"]
    assert p.composition_formula({"Fe": .4, "Ni": .6}) is None
    assert set(p.last_evaluation_timings[0]["sources"]) == {"a", "b"}


def test_order_batch_partition_cache_eviction_weights_and_property_order_do_not_change_samples(factories):
    cfg = config()
    cfg["predict_cache_size"] = 1
    first = {"Ni": .6, "Fe": .4}
    second = {"Fe": .2, "Ni": .8}
    p = StructureScorePredictor(cfg, seed=17)
    expected = p.per_objective_stats(first)
    p.predict(second)
    assert p.per_objective_stats(dict(reversed(list(first.items())))) == expected
    q = StructureScorePredictor(cfg, seed=17)
    q.batch_predict([second, first, first.copy()])
    assert q.per_objective_stats(first) == expected
    reversed_cfg = deepcopy(cfg)
    reversed_cfg["properties"].reverse()
    reversed_cfg["properties"][0]["weight"] = 19
    r = StructureScorePredictor(reversed_cfg, seed=17)
    r.batch_predict([second])
    r.batch_predict([first])
    assert r.per_objective_stats(first) == expected
    assert p.source_diagnostics(first) == q.source_diagnostics(first) == r.source_diagnostics(first)


def test_each_leaf_model_gets_detached_atoms_arrays_info_and_list(factories):
    cfg = config()
    cfg["properties"][0].update(mutates=True, model=["one", "two"])
    p = StructureScorePredictor(cfg, seed=2)
    stats = p.per_objective_stats({"H": 1})
    assert stats["a"][0] == stats["b"][0]
    assert stats["a"][1] == 0.0  # Two identical members give a measured zero spread.
    assert stats["b"][1] is None  # A singleton has no model-spread estimate.
    assert len(factories.leaves) == 3
    for leaf in factories.leaves:
        for position, info in leaf.seen[0]:
            assert position[0, 0] >= 0
            assert info["nested"]["untouched"] == [1]
    assert all(cell.info["nested"]["untouched"] == [1] for cell in factories.builders[0].samples)
    assert all(cell.positions[0, 0] >= 0 for cell in factories.builders[0].samples)


def test_failed_constructor_closes_previously_created_leaves(factories):
    cfg = config()
    cfg["properties"][1]["fail_init"] = True
    with pytest.raises(RuntimeError, match="leaf initialization"):
        StructureScorePredictor(cfg, seed=1)
    assert factories.leaves[0].closed == 1
    assert not factories.builders


def test_failed_second_source_closes_first_source_and_all_leaves(factories):
    cfg = config(independent=True)
    cfg["structure_sources"]["a"]["close_fails"] = True
    cfg["structure_sources"]["b"]["fail_init"] = True
    with pytest.raises(RuntimeError, match="builder initialization"):
        StructureScorePredictor(cfg, seed=1)
    assert factories.builders[0].closed == 1
    assert all(leaf.closed == 1 for leaf in factories.leaves)


def test_close_is_idempotent_and_deduplicates_owned_resources(factories):
    p = StructureScorePredictor(config(), seed=1)
    p._own(factories.builders[0])
    p.close()
    p.close()
    assert all(item.closed == 1 for item in [*factories.builders, *factories.leaves])


def test_legacy_false_requires_deliberate_migration_and_cleans_up(factories):
    cfg = {"share_structure": False, "base_poscar": "same-template",
           "properties": [{k: v for k, v in prop.items() if k != "source"}
                          for prop in config()["properties"]]}
    with pytest.raises(LegacySourceMigrationRequired) as error:
        StructureScorePredictor(cfg, seed=1)
    assert len(error.value.migration_report["structure_sources"]) == 2
    assert all(leaf.closed == 1 for leaf in factories.leaves)
    assert not factories.builders


def test_source_config_is_detached_and_none_seed_is_recorded_once(factories):
    cfg = config()
    p = StructureScorePredictor(cfg)
    identity = p.source_identity()
    cfg["structure_sources"]["bulk"]["n_random_configs"] = 99
    cfg["properties"][0]["source"] = "another"
    exported = p.source_plan.sources[0].effective_config
    exported["n_random_configs"] = 88
    p.predict({"H": 1})
    assert len(factories.builders[0].samples) == 3
    assert p.source_identity() == identity
    q = StructureScorePredictor(config(), seed=identity["eval_seed"])
    assert q.per_objective_stats({"H": 1}) == p.per_objective_stats({"H": 1})


def test_composition_only_predictor_does_not_resolve_any_source(factories):
    p = StructureScorePredictor({"properties": [{"name": "p", "predictor": "example:Leaf",
        "composition_only": True, "direction": "max"}]}, seed=0)
    assert p.predict({"H": 1}) == (1.0, None)
    assert p.source_identity()["sources"] == []
    assert not factories.builders


def test_source_asset_hash_is_recorded_and_changed_asset_invalidates_cache(factories, tmp_path):
    asset = tmp_path / "template.vasp"
    asset.write_text("first physical template")
    cfg = config()
    cfg["structure_sources"]["bulk"]["base_poscar"] = str(asset)
    p = StructureScorePredictor(cfg, seed=11)
    p.predict({"H": 1})
    old = p.source_identity()
    assert old["asset_resolution"]["bulk"][0]["status"] == "resolved"
    assert old["sources"][0]["physical_protocol"]["asset_sha256"]
    asset.write_text("different physical template")
    with pytest.raises(EvaluationProtocolError, match="asset changed"):
        p.predict({"H": 1})
    q = StructureScorePredictor(cfg, seed=11)
    assert q.source_identity()["sources"][0]["protocol_fingerprint"] != old["sources"][0]["protocol_fingerprint"]


def test_source_resolver_handles_template_dicts_and_unknown_relative_paths(tmp_path):
    (tmp_path / "a.vasp").write_text("a")
    (tmp_path / "b.vasp").write_text("b")
    spec = StructureSourceSpec("bulk", {"base_poscar": [{"path": "a.vasp", "n_configs": 3},
                                                       {"path": "b.vasp", "n_configs": 2}],
                                        "geo_opt": {"model": "missing.pt"}})
    frozen, references = resolve_source_assets(spec, base_dir=tmp_path)
    assert [r["status"] for r in references] == ["resolved", "resolved", "missing"]
    assert len(frozen.physical_protocol["asset_sha256"]) == 2
    external = StructureSourceSpec("bulk", {"builder": "external:Builder", "base_poscar": "a.vasp"})
    frozen, references = resolve_source_assets(external, base_dir=tmp_path)
    assert references[0]["status"] == "unresolved" and references[0]["path"] is None
    assert not frozen.physical_protocol["asset_sha256"]


def test_multitemplate_counts_and_parent_relaxation_sample_diagnostics(factories, monkeypatch, tmp_path):
    from ase import Atoms
    from ase.io import write
    monkeypatch.setattr(factories.registry, "resolve_builder", factories.real_builder_factory)
    paths = []
    for number in (4, 8):
        path = tmp_path / f"template{number}.vasp"
        write(path, Atoms("Ni" * number, positions=np.arange(number * 3).reshape(-1, 3),
                         cell=[40, 40, 40], pbc=True), format="vasp")
        paths.append(str(path))
    cfg = config(geo_opt=True)
    cfg["structure_sources"]["bulk"].update(base_poscar=[
        {"path": paths[0], "n_configs": 3}, {"path": paths[1], "n_configs": 2}], site_symbol="Ni")
    for prop in cfg["properties"]:
        prop["atom_count"] = True
    p = StructureScorePredictor(cfg, seed=7)
    calls = []
    def relax(cell, *, source_id):
        sample = cell.info["_rl_matdesign_sample"]
        calls.append((sample["template_id"]["index"], sample["sample_index"]))
        if calls[-1] == (0, 0):
            raise RuntimeError("not converged")
        return cell
    p._relax = relax
    stats = p.per_objective_stats({"Fe": .5, "Co": .5})
    assert stats == {"a": (6.0, None), "b": (6.0, None)}
    assert p.cached_outputs({"Fe": .5, "Co": .5})["a"].config_spread == 2.0
    assert p.last_evaluation_timings[0]["config_attempted"] == 5
    assert p.last_evaluation_timings[0]["config_succeeded"] == 4
    assert calls == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    diag = p.source_diagnostics({"Co": .5, "Fe": .5})["bulk"]
    assert diag["sample_identity"] == "explicit_plan"
    assert len(diag["samples"]) == 5
    assert len(diag["surviving_samples"]) == 4
    assert diag["parent_relaxation_failures"][0]["status"] == "relaxation_failed"
    assert diag["parent_relaxation_failures"][0]["template_id"]["index"] == 0


def test_only_selected_source_runs_parent_relaxation(factories):
    cfg = config(independent=True)
    cfg["structure_sources"]["a"]["geo_opt"] = {"enabled": True}
    p = StructureScorePredictor(cfg, seed=11)
    calls = []
    p._relax = lambda cell, *, source_id: (calls.append(source_id) or cell)
    p.predict({"H": 1})
    assert calls == ["a"] * 3


def test_lazy_model_load_failure_cleans_partial_models_without_double_close(factories, monkeypatch):
    import rl_matdesign.utils.dp_eval as dp_eval
    cfg = config()
    cfg["properties"] = [{"name": "p", "predictor": "dp_energy", "models": ["first", "broken"],
                          "direction": "min", "source": "bulk"}]
    loaded = []
    def load(paths, **kwargs):
        if paths == ["broken"]:
            raise RuntimeError("load failed")
        model = NativeBuilder({})
        loaded.append(model)
        return [model]
    monkeypatch.setattr(dp_eval, "load_ase_calculators", load)
    p = StructureScorePredictor(cfg, seed=0)
    with pytest.raises(RuntimeError, match="load failed"):
        p._get_calculators(p.properties[0])
    assert loaded[0].closed == 1
    assert not p._calc_cache
    p.close()
    assert loaded[0].closed == 1


@pytest.mark.parametrize("builder", ["substitute", "site_pick", "defect_site", "sse"])
def test_search_grid_changes_do_not_change_builtin_source_physics(builder):
    cfg = {"builder": builder, "groups": [
        {"name": "P_site", "kind": "composition", "fraction_set": ["0.1", "0.2"]},
        {"name": "S_site", "kind": "categorical", "choices": [
            {"name": "O_a", "element": "O", "values": [0, 0.1]},
            {"name": "Cl", "element": "Cl", "values": [0, 0.1]}],
         "filters": [{"type": "sse_doping", "metal_only": ["Ru"]}]},
    ]}
    first = StructureSourceSpec("bulk", cfg)
    changed = deepcopy(cfg)
    changed["groups"][0]["fraction_set"] = ["0.05", "0.10", "0.15", "0.20"]
    changed["groups"][1]["choices"][0]["values"] = [0, 0.05, 0.1]
    changed["groups"][1]["filters"] = []
    second = StructureSourceSpec("bulk", changed)
    assert first.protocol_fingerprint == second.protocol_fingerprint
    assert first.seed_for({"H": 1}, eval_seed=3) == second.seed_for({"H": 1}, eval_seed=3)
    if builder == "sse":
        changed["groups"][1]["choices"][0]["name"] = "O_b"
        assert StructureSourceSpec("bulk", changed).protocol_fingerprint != first.protocol_fingerprint


def test_relative_source_assets_cannot_silently_follow_a_changed_working_directory(factories, tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "template.vasp").write_text(str(directory))
    monkeypatch.chdir(first)
    cfg = config()
    cfg["structure_sources"]["bulk"]["base_poscar"] = "template.vasp"
    p = StructureScorePredictor(cfg, seed=11)
    monkeypatch.chdir(second)
    with pytest.raises(EvaluationProtocolError, match="Runtime directory changed"):
        p.predict({"H": 1})


def test_explicit_plugin_sample_capability_uses_common_serial_executor(factories, monkeypatch):
    from ase import Atoms
    class SampleBuilder(NativeBuilder):
        def build(self, *args, **kwargs):
            raise AssertionError("The explicit sample contract takes precedence over native ensembles")

        def build_sample(self, candidate, *, sample):
            self.calls.append(sample.sample_index)
            return Atoms("H", positions=[[np.random.default_rng(sample.seed).random(), 0, 0]])

    monkeypatch.setattr(factories.registry, "resolve_builder", lambda kind, cfg, **kwargs: SampleBuilder(cfg))
    p = StructureScorePredictor(config(), seed=19)
    q = StructureScorePredictor(config(), seed=19)
    candidate = {"Ni": .6, "Fe": .4}
    expected = p.predict(candidate)
    assert q.batch_predict([{"Ni": .8, "Fe": .2}, candidate])[1] == expected
    assert p.source_diagnostics(candidate) == q.source_diagnostics(candidate)
    assert p.source_diagnostics(candidate)["bulk"]["sample_identity"] == "explicit_plan"
    assert p._source_builders["bulk"].calls == [0, 1, 2]


@pytest.mark.parametrize("seed", [True, -1, 1.5, 1.0, "1"])
def test_evaluation_seed_rejects_lossy_or_boolean_values(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        StructureScorePredictor(config(), seed=seed)


@pytest.mark.parametrize("mode", ["scalar", "batch"])
@pytest.mark.parametrize("boundary", ["build", "relax", "property"])
def test_evaluation_protocol_errors_are_fatal_not_candidate_failures(factories, mode, boundary):
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    p = StructureScorePredictor(config(geo_opt=boundary == "relax"), seed=11)
    def broken(*args, **kwargs):
        raise EvaluationProtocolError("incompatible evaluation protocol")
    if boundary == "build":
        p._source_builders["bulk"].build = broken
    elif boundary == "relax":
        p._relax = broken
    else:
        p._score = broken
    with pytest.raises(EvaluationProtocolError, match="incompatible"):
        if mode == "scalar":
            p.predict({"H": 1})
        else:
            p.batch_predict([{"H": 1}, {"H": 2}], return_exceptions=True)
    assert not p._stats_cache


def test_malformed_batch_length_is_a_fatal_plugin_contract_error(factories):
    from rl_matdesign.utils.rollout_errors import EvaluationProtocolError
    p = StructureScorePredictor(config(), seed=11)
    p._source_builders["bulk"].build_many_seeded = lambda *args, **kwargs: []
    with pytest.raises(EvaluationProtocolError, match="one result per candidate"):
        p.batch_predict([{"H": 1}], return_exceptions=True)
    assert not p._stats_cache



def test_candidate_plan_failure_preserves_neighbors_and_reports_no_build_attempt(factories, monkeypatch):
    from ase import Atoms
    from rl_matdesign.utils.structure_samples import sample_plan
    class PlannedBuilder(NativeBuilder):
        def sample_plan(self, candidate, **kwargs):
            if candidate["H"] == 1:
                raise ValueError("candidate has no usable template")
            return sample_plan(candidate, **kwargs)

        def build_sample(self, candidate, *, sample):
            return Atoms("H")

    monkeypatch.setattr(factories.registry, "resolve_builder", lambda kind, cfg, **kwargs: PlannedBuilder(cfg))
    p = StructureScorePredictor(config(), seed=19)
    result = p.batch_predict([{"H": 1}, {"H": 2}], return_exceptions=True)
    assert isinstance(result[0], ValueError)
    assert result[1] == (0.0, None)
    assert p.last_evaluation_timings[0]["config_attempted"] == 0
    assert p.last_evaluation_timings[1]["config_attempted"] == 3
    assert p._key({"H": 1}) not in p._stats_cache
    assert p._key({"H": 2}) in p._stats_cache
