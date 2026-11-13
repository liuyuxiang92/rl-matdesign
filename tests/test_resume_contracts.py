"""Native resume accepts identical evidence and rejects scientific changes."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from rl_matdesign.utils.execution_contract import require_execution_contract


def execution():
    return {
        "evaluation": {
            "source_sha256": "unchanged-source",
            "configuration": {"property": "target", "threshold": 0},
            "assets": [{"sha256": "frozen-oracle"}],
            "numerical_runtime": {"platform": "original-platform"},
        },
        "learning": {"train_seed": 42, "checkpoint_frequency": 1},
    }


def generation():
    return {
        "execution": execution(),
        "sampling": {"gen_seed": 101, "gen_temperature": .5},
        "frozen_artifacts": {"policy.pt": "frozen-model", "std_scaler.bin": "frozen-scaler"},
    }


def replace(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("evaluation", "source_sha256"), "changed-source"),
    (("evaluation", "configuration", "threshold"), 1),
    (("evaluation", "configuration", "threshold"), False),
    (("evaluation", "configuration", "property"), "different-target"),
    (("evaluation", "assets", 0, "sha256"), "new-model"),
    (("evaluation", "numerical_runtime", "platform"), "new-platform"),
    (("learning", "train_seed"), 43),
    (("learning", "checkpoint_frequency"), 10),
])
def test_execution_rejects_changed_evidence_without_mutating_saved_state(path, value):
    saved = execution()
    before = copy.deepcopy(saved)
    require_execution_contract(saved, copy.deepcopy(saved))
    changed = copy.deepcopy(saved)
    replace(changed, path, value)
    with pytest.raises(ValueError, match="first differing fields") as error:
        require_execution_contract(saved, changed)
    assert str(path[-1]) in str(error.value)
    assert saved == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_execution_rejects_nonfinite_contract_even_when_both_sides_match(value):
    contract = execution()
    contract["evaluation"]["configuration"]["threshold"] = value
    with pytest.raises(ValueError, match="protocol differs"):
        require_execution_contract(contract, copy.deepcopy(contract))


@pytest.mark.parametrize("hpo", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_model_loading_rejects_changed_source_for_hpo_and_normal_runs(tmp_path, hpo, checkpoint):
    import numpy as np
    import torch
    from rl_matdesign.env import CompositionEnv
    from rl_matdesign.utils.model_contract import load_artifact, save_weights

    env = CompositionEnv(species_set=("Ni", "Fe", "Co"), fraction_set=["0.25", "0.50", "0.75"],
                         total_units=4, n_components=2, state_featurizer=lambda _: np.zeros(2))
    if hpo:
        env.hpo_recorder = object()
    env.evaluation_contract = execution()["evaluation"]
    path = tmp_path / "policy.pt"
    save_weights(path, torch.nn.Linear(2, 1), env, "policy")
    if checkpoint:
        raw = torch.load(path, weights_only=True)
        raw["execution_contract"] = execution()
        torch.save(raw, path)
    before = path.read_bytes()
    load_artifact(path, env, checkpoint=checkpoint)
    env.evaluation_contract["source_sha256"] = "changed-source"
    with pytest.raises(ValueError, match="evaluation contract differs"):
        load_artifact(path, env, checkpoint=checkpoint)
    assert path.read_bytes() == before


@pytest.mark.parametrize("path,value", [
    (("execution", "evaluation", "source_sha256"), "changed-source"),
    (("execution", "evaluation", "numerical_runtime", "platform"), "new-platform"),
    (("execution", "learning", "train_seed"), 43),
    (("sampling", "gen_seed"), 102),
    (("sampling", "gen_temperature"), .9),
    (("frozen_artifacts", "policy.pt"), "new-model"),
    (("frozen_artifacts", "std_scaler.bin"), "new-scaler"),
])
def test_generation_validates_before_restoring_ledger_or_rng(monkeypatch, path, value):
    from rl_matdesign.utils import generation_resume

    loaded, restored_rng = [], []
    predictor = SimpleNamespace(load_state_dict=loaded.append)
    monkeypatch.setattr(generation_resume, "restore_rng_state", restored_rng.append)
    saved = {"generation_contract": generation(),
             "predictor_ledger": {"calls": 7}, "rng_state": {"state": "saved"}}
    changed = generation()
    replace(changed, path, value)
    with pytest.raises(ValueError, match="Generation resume changed"):
        generation_resume.restore_generation(predictor, saved, changed)
    assert loaded == restored_rng == []
    generation_resume.restore_generation(predictor, saved, generation())
    assert loaded == [{"calls": 7}]
    assert restored_rng == [{"state": "saved"}]
