"""HPO keeps sequence CSV references attached to the YAML that defines them."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import hpo
from rl_matdesign.configuration import load_config
from rl_matdesign.sequence import SequenceGroup


def _write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_tokens(path, values):
    path.write_text(
        "monomer,label\n" + "".join(f"{token},{token}-label\n" for token in values),
        encoding="utf-8",
    )


@pytest.mark.parametrize("source", ["relative", "absolute", "inline", "fixed_override"])
def test_both_hpo_stages_preserve_sequence_token_source(tmp_path, monkeypatch, source):
    """Run the real HPO config pipeline, replacing only expensive trial execution."""
    base_dir = tmp_path / "base scenario"
    hpo_dir = tmp_path / "hpo definitions"
    caller_dir = tmp_path / "unrelated caller"
    out_dir = tmp_path / "results"
    for directory in (base_dir, hpo_dir, caller_dir):
        directory.mkdir()

    expected_tokens = ["ZZ", "AA", "Q"]  # Order is part of the action space.
    base_csv = base_dir / "new_monomers.csv"
    override_csv = hpo_dir / "new_monomers.csv"
    _write_tokens(base_csv, expected_tokens)
    _write_tokens(override_csv, ["HPO_DECOY"])
    _write_tokens(caller_dir / "new_monomers.csv", ["CWD_DECOY"])
    tokens = {"csv": "new_monomers.csv", "id_column": "monomer"}
    if source == "absolute":
        tokens["csv"] = str(base_csv)
    elif source == "inline":
        tokens = expected_tokens

    base_cfg = {
        "groups": [{"name": "sequence", "kind": "sequence", "length": 2, "tokens": tokens}],
        "dqn_num_train_eps": 20,
        "dqn_warmup_eps": 4,
        # Other loaders/predictor plug-ins own these path conventions.
        "dp_poscar": "structures/POSCAR",
        "dp_model": ["models/model.pt"],
        "properties": [{"predictor": "custom.api:Predictor", "new_monomers": "plugin.csv"}],
    }
    fixed = {}
    expected_csv = None if source == "inline" else base_csv
    if source == "fixed_override":
        expected_tokens = ["K", "B", "Z"]
        _write_tokens(override_csv, expected_tokens)
        fixed = {
            "groups": [{
                "name": "override_sequence", "kind": "sequence", "length": 2,
                "tokens": {"csv": "new_monomers.csv", "id_column": "monomer"},
            }],
        }
        expected_csv = override_csv

    base_path = base_dir / "scenario.yaml"
    hpo_path = hpo_dir / "search.yaml"
    _write_yaml(base_path, base_cfg)
    _write_yaml(hpo_path, {
        "base_config": "../base scenario/scenario.yaml",
        "study_name": "sequence_paths",
        "method": "dqn",
        "fixed_overrides": fixed,
        "search_space": {"dqn_batch_size": {"dist": "categorical", "choices": [4]}},
        "stage1": {
            "n_trials": 1, "n_parallel_trials": 1, "seeds_per_trial": 1,
            "budget_scaling": {"dqn_num_train_eps": 0.5},
        },
        "stage2": {"n_top_to_confirm": 1, "seeds_per_trial": 1},
    })
    original_base = base_path.read_bytes()
    original_hpo = hpo_path.read_bytes()
    monkeypatch.chdir(caller_dir)
    monkeypatch.setattr(hpo, "parse_args", lambda: SimpleNamespace(
        hpo_config=str(hpo_path), out=str(out_dir), gpu_ids=None,
        timeout_s=None, skip_stage2=False, no_plots=True,
    ))
    create_study = hpo.optuna.create_study

    def memory_study(**kwargs):
        kwargs["storage"] = None
        return create_study(**kwargs)

    monkeypatch.setattr(hpo.optuna, "create_study", memory_study)
    monkeypatch.setattr(hpo, "_write_analysis_outputs", lambda **kwargs: None)
    monkeypatch.setattr(hpo, "_write_final_report", lambda **kwargs: None)
    observed = []

    def check_trial(**kwargs):
        trial_path = Path(kwargs["trial_yaml"])
        emitted = yaml.safe_load(trial_path.read_text(encoding="utf-8"))
        # Require the written YAML itself to be portable to either stage.
        token_spec = emitted["groups"][0]["tokens"]
        if expected_csv is not None:
            assert token_spec["csv"] == str(expected_csv)
        else:
            assert token_spec == expected_tokens
        effective = load_config(str(trial_path))
        group = SequenceGroup(effective["groups"][0])
        assert group.species_set == expected_tokens
        if expected_csv is not None:
            assert group.token_metadata[expected_tokens[0]]["label"] == f"{expected_tokens[0]}-label"
        for key in ("dp_poscar", "dp_model", "properties"):
            assert effective[key] == base_cfg[key]
        observed.append((trial_path.relative_to(out_dir).parts[0], effective["dqn_num_train_eps"]))
        return 1.0, 2

    monkeypatch.setattr(hpo, "_run_one_seed", check_trial)
    hpo.main()

    assert observed == [("stage1", 10), ("stage2", 20)]
    assert base_path.read_bytes() == original_base
    assert hpo_path.read_bytes() == original_hpo


def test_sequence_path_resolution_does_not_mutate_source_mapping(tmp_path):
    from rl_matdesign.configuration import resolve_sequence_catalog_paths

    # YAML aliases can share a path dictionary with an application-specific key.
    shared = {"csv": "new_monomers.csv", "id_column": "monomer"}
    cfg = {
        "groups": [{"kind": "sequence", "tokens": shared}],
        "plugin_options": shared,
    }
    before = deepcopy(cfg)
    resolved = resolve_sequence_catalog_paths(cfg, str(tmp_path / "scenario.yaml"))

    assert cfg == before
    assert resolved["groups"][0]["tokens"]["csv"] == str(tmp_path / "new_monomers.csv")
    assert resolved["plugin_options"] == before["plugin_options"]


@pytest.mark.parametrize(
    "fixed,sampled,expected",
    [
        ({"num_train_eps": 200}, {}, 200),
        ({}, {"num_train_eps": 300}, 300),
        ({"num_train_eps": 200}, {"num_train_eps": 300}, 300),
    ],
)
def test_hpo_preserves_deprecated_flag_override_precedence(tmp_path, fixed, sampled, expected):
    """Resolve files early, but retain base < fixed < sampled flag precedence."""
    base_path = tmp_path / "base.yaml"
    trial_path = tmp_path / "trial.yaml"
    _write_yaml(base_path, {
        "groups": [{"name": "sequence", "kind": "sequence", "length": 2, "tokens": ["A", "B"]}],
        "num_train_eps": 100,
    })
    trial = hpo._build_trial_yaml(
        base_cfg=hpo._load_base_config(str(base_path)),
        fixed_overrides=fixed,
        sampled=sampled,
        budget_scaling={},
        method="dqn",
    )
    hpo._dump_trial_yaml(str(trial_path), trial)

    with pytest.warns(DeprecationWarning, match="num_train_eps"):
        effective = load_config(str(trial_path))

    assert effective["dqn_num_train_eps"] == expected
    assert "num_train_eps" not in effective
