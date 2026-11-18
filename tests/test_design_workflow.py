"""New configuration interfaces exercise real builders with mocked expensive models."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from rl_matdesign.configuration import load_config, build_env, build_predictor
from rl_matdesign.config_schema import ConfigError, validate_config
from rl_matdesign.predictors.structure_score import StructureScorePredictor
from rl_matdesign.rewards import RewardCombiner
from rl_matdesign.structure_sources import resolve_structure_plan, resolve_source_assets


def template(tmp_path, symbols="Co O H", counts="4 2 1"):
    n = sum(map(int, counts.split()))
    path = tmp_path / "POSCAR"
    text = ["fixture", "1", "20 0 0", "0 20 0", "0 0 20", symbols, counts, "Direct"]
    text += [f"{(i+1)/(n+2)} .4 .4" for i in range(n)]
    path.write_text("\n".join(text)+"\n")
    return str(path)


def pipeline(path, count=2):
    return {"builder": "pipeline", "base_poscar": path, "n_random_configs": count,
            "steps": [{"type": "substitute", "a_site_symbol": "Co", "a_dopant_group": "alloy"}]}


def install_models(monkeypatch, factory):
    # Test only the model boundary; all pipeline/source/reward code stays real.
    for name in ["deepmd", "deepmd.pt", "deepmd.pt.infer", "deepmd.pt.infer.deep_eval"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["deepmd.pt.infer.deep_eval"].DeepProperty = factory


def test_best_shared_alternative_and_stable_tie():
    cfg = {"type": "max", "alternatives": [
        {"name": f"T{t}", "terms": {f"c{t}": .5, f"s{t}": .5}}
        for t in [460, 470, 480, 490]]}
    values = dict(zip([f"c{t}" for t in [460, 470, 480, 490]], [8, 1, 4, 2]))
    values.update(dict(zip([f"s{t}" for t in [460, 470, 480, 490]], [0, 9, 3, 1])))
    combiner = RewardCombiner(cfg, list(values))
    result = combiner.evaluate({}, values)
    assert result["reward"] == 5
    assert result["selected_alternative"] == "T470"
    assert [row["score"] for row in result["alternatives"]] == [4, 5, 3.5, 1.5]
    tied = combiner.evaluate({}, {name: 1 for name in values})
    assert tied["selected_alternative"] == "T460"


@pytest.mark.parametrize("bad", [{}, {"alternatives": []},
    {"alternatives": [{"name": "x", "terms": {"missing": 1}}]},
    {"alternatives": [{"name": "x", "terms": {"p": 1}}] * 2}])
def test_invalid_alternatives_fail(bad):
    with pytest.raises(ValueError):
        RewardCombiner({"type": "max", **bad}, ["p"])


def test_lips_explicit_temperatures_load_two_ensembles_and_build_once(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).parents[1] / "configs/lips_sse.yaml")
    # Retain all eight real property definitions and reward terms. A small
    # substitution fixture isolates source sharing from SSE population tests.
    cfg["structure"] = pipeline(template(tmp_path))
    loaded, forwards = [], []
    class Model:
        def __init__(self, model_file, **kwargs):
            self.model_file = model_file
            loaded.append(model_file)
        def get_type_map(self):
            return ["Co", "Ni", "Fe", "O", "H"]
    install_models(monkeypatch, Model)
    import rl_matdesign.utils.dp_eval as dp
    def eval_outputs(structures, models, elem_to_type, *, fparam, aparam):
        forwards.append((models[0].model_file, tuple(fparam), len(structures)))
        index = [460, 470, 480, 490].index(fparam[1])
        values = [1, 9, 3, 1] if len(fparam) == 4 else [8, 1, 4, 2]
        return [[[np.log(values[index])] for _ in structures] for _ in models]
    monkeypatch.setattr(dp, "eval_property_outputs", eval_outputs)
    predictor = StructureScorePredictor(cfg, seed=3)
    builder = predictor._source_builders["structure"]
    calls = []
    original = builder.build_seeded
    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(builder, "build_seeded", build)
    candidate = {"alloy": {"Ni": .5, "Fe": .5}}
    prediction = predictor.predict(candidate)
    assert prediction == (pytest.approx(5), None)
    assert len(loaded) == 10 and len(forwards) == 8 and len(calls) == 1
    assert all(call[2] == 2 for call in forwards)
    record = predictor.prediction_record(candidate, prediction)
    assert record["selected_alternative"] == "T470"
    assert len(predictor.predict_outputs(candidate)) == 8
    assert predictor.batch_predict([candidate, candidate]) == [prediction, prediction]
    assert len(forwards) == 8 and len(calls) == 1


def test_ooh_pipeline_consumes_prebuilt_masked_frames_once_per_parent(tmp_path, monkeypatch):
    path = template(tmp_path)
    models = []
    for i in range(2):
        model = tmp_path / f"m{i}.pt"
        model.write_text(f"fake {i}")
        models.append(str(model))
    forwards = []
    class Model:
        def __init__(self, model_file, **kwargs):
            self.value = 2. if model_file == models[0] else 4.
        def eval(self, *, coords, cells, atom_types, mixed_type):
            forwards.append((coords.copy(), atom_types.copy()))
            assert mixed_type and coords.shape == (3, 10, 3)
            assert sorted((row == -1).sum() for row in atom_types) == [0, 1, 2]
            return np.asarray([[self.value]])
    install_models(monkeypatch, Model)
    structure = pipeline(path)
    structure["steps"].append({"type": "ooh_adsorbates", "adsorbates": ["O*", "OH*", "OOH*"]})
    cfg = {"structure": structure, "properties": [{"name": "overpotential",
           "predictor": "ooh_frames", "models": models, "direction": "min",
           "objective": "mean_plus_kstd", "uncertainty": "model_spread"}], "k": 1}
    predictor = build_predictor(cfg, seed=8)
    candidate = {"alloy": {"Ni": .5, "Fe": .5}}
    assert predictor.predict(candidate) == (-2., None)  # -mean(2,4) + model spread(2,4)
    output = predictor.predict_outputs(candidate)["overpotential"]
    assert output.mean == 3 and output.model_spread == 1
    assert len(output.samples) == 2 and len(output.samples[0].configuration_ids) == 2
    assert len(forwards) == 4  # two models x two parent samples, not six configurations
    predictor.predict(candidate)
    assert len(forwards) == 4


def test_singular_structure_and_nested_assets(tmp_path):
    structure = pipeline(template(tmp_path))
    model = tmp_path / "relax.pt"
    model.write_text("fake relaxation weights")
    structure["steps"].append({"type": "geo_opt", "model": str(model)})
    cfg = {"structure": structure, "properties": [{"name": "a"}, {"name": "b"}]}
    plan = resolve_structure_plan(cfg, needs_structure=["a", "b"])
    assert plan.property_sources == {"a": "structure", "b": "structure"}
    frozen, references = resolve_source_assets(plan.sources[0])
    assert len(references) == 2 and all(row["status"] == "resolved" for row in references)
    assert "steps[1].model[0]" in frozen.physical_protocol["asset_sha256"]
    changed = deepcopy(cfg)
    changed["properties"][0]["source"] = "x"
    with pytest.raises(ValueError, match="implicitly"):
        resolve_structure_plan(changed, needs_structure=["a", "b"])


@pytest.mark.parametrize("field,value", [
    ("anion_formula", "O3"), ("scaffold_formula", "O3"),
    ("scaffold_poscar", "POSCAR"), ("scaffold_site_symbol", "Fe"),
    ("constraint_filter", "smact_charge"), ("episode_style", "interleaved"),
    ("required_elements", ["O"]), ("nonzero_ratio_at_last", True),
])
def test_new_design_rejects_ignored_legacy_chemical_fields(field, value):
    cfg = {"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}], "groups": [{"name": "alloy", "kind": "composition",
           "species_set": ["Ni", "Fe"], "amount_set": [1], "n_steps": 1}], field: value}
    with pytest.raises(ConfigError, match="Removed top-level"):
        validate_config(cfg)


@pytest.mark.parametrize("multiple_groups", [False, True])
def test_new_design_cannot_silently_drop_top_level_filters(multiple_groups):
    group = {"name": "first", "kind": "composition", "species_set": ["Fe"],
             "amount_set": [1], "n_steps": 1}
    cfg = {"properties": [{"name": "score", "predictor": "dummy", "direction": "max", "objective": "mean"}], "groups": [group], "filters": ["smact_charge"]}
    if multiple_groups:
        cfg["groups"].append({**group, "name": "second"})
    else:
        group["filters"] = ["smact_pauling"]
    with pytest.raises(ConfigError, match="Removed top-level"):
        validate_config(cfg)


@pytest.mark.parametrize("location", ["top", "property"])
@pytest.mark.parametrize("field,value", [
    ("structure_parallel", {"devices": [0]}), ("ads_height", 2.4),
    ("base_poscar", "other.POSCAR"),
])
def test_singular_structure_rejects_settings_outside_its_owner(location, field, value):
    cfg = {"structure": pipeline("POSCAR"), "properties": [{"name": "energy", "predictor": "dp_energy"}]}
    owner = cfg if location == "top" else cfg["properties"][0]
    owner[field] = value
    with pytest.raises(ConfigError, match="conflict|structure-owned|Removed top-level"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="conflict|structure-owned|Removed top-level"):
        resolve_structure_plan(cfg, needs_structure=["energy"])
