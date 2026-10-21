"""Direct GP API checks; install GP in the SAME Python environment as RL.

Set GP_PHYSCHEM_TEST_ROOT for expensive original-data parity checks. There is
no worker, separate interpreter, prepared catalog or disk model cache.
"""
from __future__ import annotations

import collections
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gp_api():
    return pytest.importorskip("gp_physchem.api")


def test_direct_api_does_not_import_unrelated_top_level_modules(gp_api, monkeypatch):
    unrelated = {}
    for name in ("config", "constants", "chemistry", "features", "model"):
        unrelated[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, unrelated[name])
    for name in list(sys.modules):
        if name == "gp_physchem" or name.startswith("gp_physchem."):
            monkeypatch.delitem(sys.modules, name)
    imported = importlib.import_module("gp_physchem.api")
    assert callable(imported.PeptidePredictor)
    assert all(sys.modules[name] is sentinel for name, sentinel in unrelated.items())


def test_bundled_catalog_default_is_relative_to_gp_package(gp_api, monkeypatch, tmp_path):
    monkeypatch.delenv("GP_PHYSCHEM_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    values = runpy.run_path(gp_api.config.__file__)
    package_root = Path(gp_api.config.__file__).resolve().parent
    assert values["EXAMPLES"] == package_root / "examples"
    assert values["NEW_MONOMERS"] == values["EXAMPLES"] / "new_monomers.csv"
    assert values["NEW_MONOMERS"].is_absolute()


@pytest.fixture
def mock_gp_inputs(gp_api, monkeypatch, tmp_path):
    """Exercise the public API without conformers, chemistry assembly or fitting."""
    calls = collections.Counter()
    source = tmp_path / "new_monomers.csv"
    source.write_text(
        "monomer,has_free_thiol,mapped_thiol,can_assemble\n"
        "BB1,1,1,1\nBB2,0,0,1\nBB3,0,0,1\nBB6,1,1,1\n"
    )
    dictionary, train = tmp_path / "dictionary.csv", tmp_path / "train.csv"
    dictionary.touch()
    train.touch()

    class Library:
        def __init__(self):
            self.records = {
                name: SimpleNamespace(name=name, has_free_thiol=thiol,
                    smiles_mapped=str(thiol), can_assemble=True, vector=np.zeros(1))
                for name, thiol in (("A", 0), ("C", 1), ("b-HCys", 1))
            }

        def register_extra_row(self, row):
            calls["descriptor"] += 1
            record = SimpleNamespace(
                name=row.monomer, has_free_thiol=int(row.has_free_thiol),
                smiles_mapped=str(int(row.mapped_thiol)), can_assemble=bool(row.can_assemble),
                vector=np.zeros(1),
            )
            self.records[record.name] = record
            return record

        def get(self, name):
            if name == "V" and name not in self.records:
                self.records[name] = SimpleNamespace(name=name, has_free_thiol=0,
                    smiles_mapped="0", can_assemble=True, vector=np.ones(1))
            return self.records[name]

    def load_library(path):
        calls["library"] += 1
        assert path == dictionary
        return Library()

    def fit(matrix, target):
        calls["fit"] += 1
        np.testing.assert_array_equal(target, [7.0, 8.0])
        return object(), SimpleNamespace(transform=lambda values: values)

    def validate(frame, library):
        calls["validate"] += 1
        row = frame.iloc[0]
        return ([row.monomers.split("|")], np.array([row.disulfide_pos_j - row.disulfide_pos_i + 1]),
                np.array([11]), np.array([float(row.n_term == "Ac")]))

    def predict(model, scaler, x):
        calls["predict"] += 1
        return np.array([7.0 + x[0, 0]]), np.array([0.5])

    def build(seq, acetyl, library):
        calls["assemble"] += 1
        sites = tuple(index + 1 for index, token in enumerate(seq) if library.get(token).has_free_thiol)
        return "C", sites

    monkeypatch.setattr(gp_api, "load_library", load_library)
    monkeypatch.setattr(gp_api, "has_map", lambda mapped, atom: mapped == "1")
    monkeypatch.setattr(gp_api, "load_training_table", lambda path: gp_api.pd.DataFrame({"pIC50_from_1000nM": [7.0, 8.0]}))
    monkeypatch.setattr(gp_api, "train_matrix", lambda frame, library: (np.zeros((2, 1)), {}, {}, [["BB2"] * 11]))
    monkeypatch.setattr(gp_api, "fit_gp", fit)
    monkeypatch.setattr(gp_api, "validate_candidates", validate)
    monkeypatch.setattr(gp_api, "featurise", lambda seqs, columns, table, ring, nres, acetyl: np.asarray(acetyl).reshape(-1, 1))
    monkeypatch.setattr(gp_api, "predict_gp", predict)
    monkeypatch.setattr(gp_api, "build_peptide", build)
    monkeypatch.setattr(gp_api.config, "NEW_MONOMERS", source)
    cfg = {"new_monomers": str(source), "dictionary": str(dictionary), "train": str(train)}
    return gp_api, cfg, calls


@pytest.mark.parametrize("text, message", [
    ("wrong\nBB1\n", "monomer column"),
    ("monomer\nBB1\nBB1\n", "duplicate"),
    ("monomer\n BB1\n", "canonical"),
    ("monomer\n1\n", "canonical"),
    ("monomer\n", "non-empty"),
])
def test_direct_api_rejects_invalid_source_ids_before_chemistry(mock_gp_inputs, text, message):
    api, cfg, calls = mock_gp_inputs
    Path(cfg["new_monomers"]).write_text(text)
    with pytest.raises(ValueError, match=message):
        api.PeptidePredictor(cfg)
    assert calls["library"] == calls["descriptor"] == calls["fit"] == 0


@pytest.mark.parametrize("tokens", [["BB1"], ["BB1", "BB2"], ["BB1", "BB6", "BB99"]])
def test_direct_api_rejects_stale_thiol_mask_before_fit(mock_gp_inputs, tokens):
    api, cfg, calls = mock_gp_inputs
    with pytest.raises(ValueError, match="do not match"):
        api.PeptidePredictor({**cfg, "thiol_tokens": tokens})
    assert calls["fit"] == 0


@pytest.mark.parametrize("replacement, message", [
    ("BB1,1,0,1", "annotation disagrees"),
    ("BB1,1,1,0", "backbone mapping"),
])
def test_direct_api_rejects_inconsistent_source_chemistry(mock_gp_inputs, replacement, message):
    api, cfg, calls = mock_gp_inputs
    source = Path(cfg["new_monomers"])
    source.write_text(source.read_text().replace("BB1,1,1,1", replacement))
    with pytest.raises(ValueError, match=message):
        api.PeptidePredictor(cfg)
    assert calls["fit"] == 0


def _candidate():
    return {"peptide": ["BB1"] + ["BB2"] * 9 + ["BB6"], "n_terminal": {"treatment": "H"}}


def test_direct_api_fits_once_reuses_in_memory_and_preserves_prediction_metadata(mock_gp_inputs):
    api, cfg, calls = mock_gp_inputs
    predictor = api.PeptidePredictor({**cfg, "thiol_tokens": ["BB6", "BB1"]}, seed=987)
    assert predictor.tokens == ["BB1", "BB2", "BB3", "BB6"]
    assert predictor.thiol_tokens == ["BB1", "BB6"]
    first = _candidate()
    assert predictor.predict(first) == (7.0, 0.5)
    metadata = predictor.candidate_metadata(first)
    metadata["posterior_sd"] = -1
    assert predictor.candidate_metadata(first)["posterior_sd"] == 0.5
    assert predictor.batch_predict([first, first]) == [(7.0, 0.5), (7.0, 0.5)]
    assert calls["predict"] == calls["assemble"] == 1
    changed_cap = {**first, "n_terminal": {"treatment": "Ac"}}
    assert predictor.predict(changed_cap) == (8.0, 0.5)
    changed_order = _candidate()
    changed_order["peptide"][0], changed_order["peptide"][1] = changed_order["peptide"][1], changed_order["peptide"][0]
    predictor.predict(changed_order)
    assert predictor.candidate_metadata(changed_order)["disulfide_pos_i"] == 2
    assert calls["predict"] == calls["assemble"] == 3
    assert calls["fit"] == calls["library"] == 1
    assert calls["descriptor"] == 4
    assert sorted(path.name for path in Path(cfg["new_monomers"]).parent.iterdir()) == ["dictionary.csv", "new_monomers.csv", "train.csv"]


@pytest.mark.parametrize("candidate, message", [
    (None, "exactly peptide"),
    ({**_candidate(), "extra": True}, "exactly peptide"),
    ({**_candidate(), "peptide": ["BB1"] * 10}, "exactly 11"),
    ({**_candidate(), "peptide": tuple(_candidate()["peptide"])}, "ordered list"),
    ({**_candidate(), "peptide": [1] + ["BB2"] * 9 + ["BB6"]}, "monomer IDs"),
    ({**_candidate(), "peptide": ["BB999"] + ["BB2"] * 9 + ["BB6"]}, "unknown monomer"),
    ({**_candidate(), "peptide": ["BB2"] * 11}, "exactly two"),
    ({**_candidate(), "peptide": ["BB1"] * 3 + ["BB2"] * 8}, "exactly two"),
    ({**_candidate(), "n_terminal": {"treatment": "Boc"}}, "H or Ac"),
    ({**_candidate(), "n_terminal": {"treatment": "H", "extra": True}}, "H or Ac"),
])
def test_direct_api_rejects_invalid_candidates_before_prediction(mock_gp_inputs, candidate, message):
    api, cfg, calls = mock_gp_inputs
    predictor = api.PeptidePredictor(cfg)
    with pytest.raises(ValueError, match=message):
        predictor.predict(candidate)
    assert calls["predict"] == calls["validate"] == 0
    assert predictor.predict(_candidate()) == (7.0, 0.5)



def test_direct_api_loads_gp_default_catalog_from_any_working_directory(mock_gp_inputs, monkeypatch, tmp_path):
    api, cfg, calls = mock_gp_inputs
    source = Path(cfg.pop("new_monomers"))
    workdir = tmp_path / "rl_working_directory"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    predictor = api.PeptidePredictor(cfg)
    assert predictor.new_monomers == source
    assert predictor.tokens == ["BB1", "BB2", "BB3", "BB6"]
    assert predictor.sequence_constraints("peptide", predictor.tokens) == [
        {"tokens": ["BB1", "BB6"], "min": 2, "max": 2}
    ]
    assert predictor.predict(_candidate()) == (7.0, 0.5)
    assert calls["descriptor"] == 4
    assert calls["fit"] == 1
    assert list(workdir.iterdir()) == []


def test_direct_api_missing_default_catalog_fails_before_fitting(mock_gp_inputs, monkeypatch, tmp_path):
    api, cfg, calls = mock_gp_inputs
    cfg.pop("new_monomers")
    monkeypatch.setattr(api.config, "NEW_MONOMERS", tmp_path / "missing_new_monomers.csv")
    with pytest.raises(FileNotFoundError, match="missing_new_monomers.csv"):
        api.PeptidePredictor(cfg)
    assert calls["library"] == calls["descriptor"] == calls["fit"] == 0


@pytest.mark.parametrize("missing_default", [False, True])
def test_direct_api_explicit_catalog_overrides_gp_default(mock_gp_inputs, monkeypatch, tmp_path, missing_default):
    api, cfg, calls = mock_gp_inputs
    if missing_default:
        monkeypatch.setattr(api.config, "NEW_MONOMERS", tmp_path / "missing_default.csv")
    override = tmp_path / "custom_monomers.csv"
    override.write_text("monomer,has_free_thiol,mapped_thiol,can_assemble\nBB8,0,0,1\n")
    predictor = api.PeptidePredictor({**cfg, "new_monomers": str(override)})
    assert predictor.new_monomers == override
    assert predictor.tokens == ["BB8"]
    candidate = {"peptide": ["C"] + ["BB8"] * 9 + ["b-HCys"], "n_terminal": {"treatment": "H"}}
    assert predictor.predict(candidate) == (7.0, 0.5)
    with pytest.raises(ValueError, match="unknown monomer"):
        predictor.sequence_constraints("peptide", ["BB2", "C"])
    assert calls["descriptor"] == calls["fit"] == 1


def test_direct_api_uses_native_dictionary_when_extra_catalog_explicitly_disabled(mock_gp_inputs):
    api, cfg, calls = mock_gp_inputs
    cfg["new_monomers"] = None
    predictor = api.PeptidePredictor(cfg)
    assert predictor.new_monomers is None
    assert predictor.tokens == predictor.thiol_tokens == []
    parent = {"peptide": ["C"] + ["A"] * 9 + ["b-HCys"], "n_terminal": {"treatment": "H"}}
    assert predictor.predict(parent) == (7.0, 0.5)
    assert calls["descriptor"] == 0
    assert calls["fit"] == 1


def test_direct_api_extra_definitions_are_not_a_candidate_whitelist(mock_gp_inputs):
    api, cfg, calls = mock_gp_inputs
    source = Path(cfg["new_monomers"])
    source.write_text("monomer,has_free_thiol,mapped_thiol,can_assemble\nBB2,0,0,1\n")
    predictor = api.PeptidePredictor(cfg)
    assert predictor.tokens == ["BB2"]
    assert predictor.thiol_tokens == []
    candidate = {"peptide": ["C", "BB2"] + ["A"] * 8 + ["b-HCys"], "n_terminal": {"treatment": "H"}}
    assert predictor.predict(candidate) == (7.0, 0.5)
    assert predictor.sequence_constraints("peptide", ["BB2", "C", "A", "b-HCys"]) == [
        {"tokens": ["C", "b-HCys"], "min": 2, "max": 2}
    ]


def test_direct_api_constraints_resolve_only_requested_universe(mock_gp_inputs):
    api, cfg, calls = mock_gp_inputs
    predictor = api.PeptidePredictor(cfg)
    assert predictor.sequence_constraints("other", ["unknown"]) == []
    assert predictor.sequence_constraints("peptide", ["BB2", "C"]) == [
        {"tokens": ["C"], "min": 2, "max": 2}
    ]
    with pytest.raises(ValueError, match="unknown monomer"):
        predictor.sequence_constraints("peptide", ["C", "undefined"])
    with pytest.raises(ValueError, match="no free-thiol"):
        predictor.sequence_constraints("peptide", ["A", "BB2"])
    with pytest.raises(ValueError, match="duplicate"):
        predictor.sequence_constraints("peptide", ["C", "C"])
    assert calls["predict"] == 0


def test_direct_api_adds_lazy_builtin_descriptors_without_refitting(mock_gp_inputs):
    api, cfg, calls = mock_gp_inputs
    predictor = api.PeptidePredictor(cfg)
    assert "V" not in predictor.table
    predictor.sequence_constraints("peptide", ["V", "C"])
    np.testing.assert_array_equal(predictor.table["V"], np.ones(1))
    candidate = {"peptide": ["C"] + ["V"] * 9 + ["b-HCys"], "n_terminal": {"treatment": "H"}}
    assert predictor.predict(candidate) == (7.0, 0.5)
    assert calls["fit"] == 1


@pytest.fixture
def chemistry_library(gp_api):
    chemistry = importlib.import_module("gp_physchem.chemistry")
    return chemistry, chemistry.load_library(gp_api.config.MONOMER_DICTIONARY)


@pytest.mark.parametrize("name", ["A", "D-HCys", "HCys", "C", "c", "b-HCys", "p", "Aib"])
def test_duplicate_known_chemical_definition_reuses_training_descriptors(chemistry_library, monkeypatch, name):
    chemistry, library = chemistry_library
    existing = library.get(name)
    vector = existing.vector.copy()
    before_computed = list(library.computed_names)
    def unexpected_recompute(*args, **kwargs):
        raise AssertionError("known descriptors must be reused")
    monkeypatch.setattr(chemistry, "descriptors_from_smiles", unexpected_recompute)
    # Missing chirality uses the existing record, including non-standard D names.
    row = chemistry.pd.Series({"monomer": name, "smiles_free_aa": existing.smiles_free_aa})
    assert library.register_extra_row(row) is existing
    np.testing.assert_array_equal(existing.vector, vector)
    assert library.computed_names == before_computed


@pytest.mark.parametrize("changes", [
    {"smiles_free_aa": "N[C@H](C)C(=O)O"},  # opposite enantiomer
    {"chirality": "D"},
    {"backbone": "beta3"},
    {"has_free_thiol": 1},
    {"has_free_thiol": 0.5},
    {"has_free_thiol": 1.5},
    {"has_free_thiol": float("inf")},
    {"smiles_mapped": "[N:1][C@@H](C)[C:2](=O)O"},  # missing leaving map
    {"smiles_mapped": "[N:1][C@@H](C)[C:2](=[O:4])O"},  # wrong oxygen role
    {"sidechain": "CC"},
])
def test_conflicting_known_definitions_fail_without_mutating_library(chemistry_library, changes):
    chemistry, library = chemistry_library
    existing = library.get("A")
    row = chemistry.pd.Series({"monomer": "A", "smiles_free_aa": existing.smiles_free_aa, **changes})
    before_records = dict(library.records)
    before_sidechains = dict(library.extra_sidechains)
    before_computed = list(library.computed_names)
    before_vector = existing.vector.copy()
    with pytest.raises(ValueError):
        library.register_extra_row(row)
    assert library.records.keys() == before_records.keys()
    assert all(library.records[name] is record for name, record in before_records.items())
    assert library.extra_sidechains == before_sidechains
    assert library.computed_names == before_computed
    np.testing.assert_array_equal(existing.vector, before_vector)


def test_unused_builtin_name_cannot_be_shadowed_by_conflicting_csv(chemistry_library, monkeypatch):
    chemistry, library = chemistry_library
    name = "D-S"
    library.records.pop(name, None)
    monkeypatch.setattr(chemistry, "descriptors_from_smiles", lambda smiles: {
        column: 0.0 for column in chemistry.DESC_COLS
    })
    row = chemistry.pd.Series({"monomer": name, "smiles_free_aa": "N[C@@H](C)C(=O)O", "chirality": "L"})
    before = dict(library.records)
    with pytest.raises(ValueError, match="conflicts"):
        library.register_extra_row(row)
    assert library.records.keys() == before.keys()
    assert name not in library.computed_names

def _source_snapshot(root):
    # Ignore interpreter bytecode housekeeping; detect model/data artifacts.
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    }


@pytest.fixture(scope="session")
def real_gp_report(tmp_path_factory):
    if not os.environ.get("GP_PHYSCHEM_TEST_ROOT"):
        pytest.skip("opt-in real GP checks: set GP_PHYSCHEM_TEST_ROOT")
    api = pytest.importorskip("gp_physchem.api")
    from gp_physchem.chemistry import build_peptide, load_library
    from gp_physchem.features import featurise, load_training_table, train_matrix, validate_candidates
    from gp_physchem.model import fit_gp, predict_gp
    import pandas as pd
    from rdkit import Chem

    root = Path(os.environ["GP_PHYSCHEM_TEST_ROOT"]).expanduser().resolve()
    required = [root / "examples/new_monomers.csv", root / "data/monomer_dictionary.csv",
                root / "data/long_table.csv", root / "examples/reference_predictions.csv"]
    assert all(path.is_file() for path in required), "GP test root lacks source data"
    workdir = tmp_path_factory.mktemp("direct_gp_inference")
    before = _source_snapshot(root)
    # The reference library independently loads the original RAW CSV; reusing
    # the API library could conceal initialization and descriptor errors.
    report = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)
        patch.setattr(sys, "dont_write_bytecode", True)
        patch.setattr(api.config, "NEW_MONOMERS", root / "examples/new_monomers.csv")
        original = load_library(root / "data/monomer_dictionary.csv", root / "examples/new_monomers.csv")
        raw = pd.read_csv(root / "examples/new_monomers.csv", encoding="utf-8-sig")
        tokens = raw.monomer.tolist()
        assert len(tokens) == len(set(tokens)) == 138
        thiols = [name for name in tokens if original.get(name).has_free_thiol == 1]
        assert thiols == ["BB1", "BB6", "BB55", "BB56", "BB103", "BB137"]
        train = load_training_table(root / "data/long_table.csv")
        matrix, columns, table, _ = train_matrix(train, original)
        model, scaler = fit_gp(matrix, train.pIC50_from_1000nM.to_numpy(float))

        predictor = api.PeptidePredictor({
            "dictionary": str(root / "data/monomer_dictionary.csv"),
            "train": str(root / "data/long_table.csv"),
        })
        assert list(predictor.tokens) == tokens
        assert list(predictor.thiol_tokens) == thiols

        raw_vectors = np.asarray([original.get(name).vector for name in tokens])
        cached_vectors = np.asarray([predictor.library.get(name).vector for name in tokens])
        np.testing.assert_allclose(cached_vectors, raw_vectors, rtol=0, atol=1e-10)
        report["descriptor_max_abs_error"] = float(np.max(np.abs(cached_vectors - raw_vectors)))
        report["catalog_count"] = len(tokens)
        report["thiols"] = thiols

        def direct(frame):
            seqs, ring, nres, acetyl = validate_candidates(frame, original)
            x = featurise(seqs, columns, table, ring, nres, acetyl)
            mean, std = predict_gp(model, scaler, x)
            return seqs, x, mean, std

        known = pd.read_csv(root / "examples/candidates.csv", encoding="utf-8-sig")
        _, _, known_mean, known_std = direct(known)
        reference = pd.read_csv(root / "examples/reference_predictions.csv", encoding="utf-8-sig").set_index("compound_id").loc[known.compound_id]
        # Match the supplied GP self-test's sklearn 1.7 reference tolerance; API
        # versus freshly evaluated original APIs below uses the stricter 1e-9.
        np.testing.assert_allclose(known_mean, reference.pred_pIC50.to_numpy(), rtol=0, atol=1e-4)
        np.testing.assert_allclose(known_std, reference.posterior_sd.to_numpy(), rtol=0, atol=1e-4)
        report["known_reference_count"] = len(known)
        report["known_reference_max_abs_error"] = float(max(np.max(np.abs(known_mean-reference.pred_pIC50)), np.max(np.abs(known_std-reference.posterior_sd))))

        raw_examples = pd.read_csv(root / "examples/candidates_with_new_monomers.csv", encoding="utf-8-sig")
        raw_seqs, _, raw_mean, raw_std = direct(raw_examples)
        assert np.isfinite(raw_mean).all() and np.isfinite(raw_std).all() and (raw_std >= 0).all()
        example_bb = sorted({token for seq in raw_seqs for token in seq if token.startswith("BB")})
        assert {"BB2", "BB3", "BB5"}.issubset(example_bb)
        report["raw_example_count"] = len(raw_examples)
        report["raw_example_bb_tokens"] = example_bb

        candidates, rows = [], []
        def add(seq, cap):
            candidate = {"peptide": list(seq), "n_terminal": {"treatment": cap}}
            sites = [i+1 for i, token in enumerate(seq) if original.get(token).has_free_thiol]
            assert len(sites) == 2
            rows.append({"compound_id": "case_"+str(len(rows)), "monomers": "|".join(seq), "n_term": cap, "c_term": "NH2", "disulfide_pos_i": sites[0], "disulfide_pos_j": sites[1]})
            candidates.append(candidate)

        baseline = ["BB96", "BB97", "BB91", "BB109", "BB105", "BB45", "BB45", "BB96", "BB111", "BB48", "BB90"]
        covered = set()
        for index, token in enumerate(tokens):
            seq = list(baseline)
            position = index % 11
            seq[position] = token
            occupied = {position} if token in thiols else set()
            for anchor in (1, 10, 0, 8):
                if len(occupied) == 2:
                    break
                if anchor == position:
                    continue
                seq[anchor] = "BB56" if not occupied else "BB103"
                occupied.add(anchor)
            assert seq[position] == token
            covered.add(token)
            add(seq, "H")
            add(seq, "Ac")
        assert covered == set(tokens)

        # These four cases share residue counts. Two swap only non-thiol order;
        # another moves both bridge endpoints. Retain paired H/Ac treatments.
        anchor = ["BB96", "BB56", "BB91", "BB109", "BB105", "BB45", "BB45", "BB96", "BB111", "BB48", "BB103"]
        order_swap = list(anchor)
        order_swap[2], order_swap[8] = order_swap[8], order_swap[2]
        bridge_move = list(anchor)
        bridge_move[0], bridge_move[1] = bridge_move[1], bridge_move[0]
        bridge_move[8], bridge_move[10] = bridge_move[10], bridge_move[8]
        assert collections.Counter(anchor) == collections.Counter(order_swap) == collections.Counter(bridge_move)
        special_start = len(candidates)
        for seq, cap in ((anchor,"H"), (order_swap,"H"), (bridge_move,"H"), (anchor,"Ac")):
            add(seq, cap)

        # Native parent residues can remain even when absent from replacement CSV.
        native_parent = ["A", "HCys", "F", "K", "V", "W", "W", "A", "P", "R", "b-HCys"]
        add(native_parent, "H")
        add(native_parent, "Ac")
        assert predictor.sequence_constraints("peptide", ["A", "HCys", "b-HCys"]) == [
            {"tokens": ["HCys", "b-HCys"], "min": 2, "max": 2}
        ]
        frame = pd.DataFrame(rows)
        direct_seqs, direct_x, direct_mean, direct_std = direct(frame)
        assert not np.array_equal(direct_x[special_start], direct_x[special_start+1])
        assert not np.array_equal(direct_x[special_start], direct_x[special_start+2])
        assert not np.array_equal(direct_x[special_start], direct_x[special_start+3])
        errors_mean, errors_std = [], []
        for index, candidate in enumerate(candidates):
            actual_mean, actual_std = predictor.predict(candidate)
            np.testing.assert_allclose([actual_mean, actual_std], [direct_mean[index], direct_std[index]], rtol=0, atol=1e-9)
            errors_mean.append(abs(actual_mean-direct_mean[index]))
            errors_std.append(abs(actual_std-direct_std[index]))
            metadata = predictor.candidate_metadata(candidate)
            row = rows[index]
            for key in ("monomers", "n_term", "c_term", "disulfide_pos_i", "disulfide_pos_j"):
                assert metadata[key] == row[key], (index, key, metadata, row)
            smiles, pair = build_peptide(direct_seqs[index], row["n_term"] == "Ac", original)
            assert Chem.MolFromSmiles(metadata["smiles"]) is not None
            assert metadata["smiles"] == smiles
            assert tuple(pair) == (row["disulfide_pos_i"], row["disulfide_pos_j"])
            assert metadata["pred_pIC50"] == actual_mean
            assert metadata["posterior_sd"] == actual_std
        report.update(prediction_count=len(candidates), covered_tokens=len(covered), mean_max_abs_error=float(max(errors_mean)), std_max_abs_error=float(max(errors_std)), caps=sorted(set(frame.n_term)), topology_count=len(set(zip(frame.disulfide_pos_i, frame.disulfide_pos_j))), order_swap_checked=True, smiles_checked=len(candidates))

        valid = candidates[special_start]
        def payload(seq=None, cap="H", **extra):
            return {"peptide": list(anchor if seq is None else seq), "n_terminal": {"treatment": cap}, **extra}
        invalid = {
            "unknown_token": payload(["BB999"]+anchor[1:]),
            "length_10": payload(anchor[:-1]),
            "length_12": payload(anchor+["BB96"]),
            "thiols_0": payload(["BB96"]*11),
            "thiols_1": payload(["BB1"]+["BB96"]*10),
            "thiols_3": payload(["BB1"]*3+["BB96"]*8),
            "invalid_cap": payload(cap="Boc"),
            "extra_group": payload(disulfide=[2,11]),
            "extra_terminal_key": {"peptide": anchor, "n_terminal": {"treatment":"H", "ignored":True}},
        }
        for name, candidate in invalid.items():
            try:
                predictor.predict(candidate)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid candidate accepted: "+name)
        report["rejections"] = sorted(invalid)

        # Invalid input must leave the direct Python object healthy.
        np.testing.assert_allclose(predictor.predict(valid),
            [direct_mean[special_start], direct_std[special_start]], rtol=0, atol=1e-9)
        expected = predictor.candidate_metadata(valid)
        returned = predictor.candidate_metadata(valid)
        returned["monomers"] = "corrupted"
        returned["posterior_sd"] = -1
        assert predictor.candidate_metadata(valid) == expected
        assert _source_snapshot(root) == before
        assert list(workdir.iterdir()) == []
        report["no_generated_artifacts"] = True
        report["metadata_copy_checked"] = True
    print("GP_DIRECT_VALIDATION " + json.dumps(report, sort_keys=True, allow_nan=False))
    return report


def test_real_gp_raw_descriptors_and_original_reference_fixtures(real_gp_report):
    assert real_gp_report["catalog_count"] == 138
    assert real_gp_report["descriptor_max_abs_error"] <= 1e-10
    assert real_gp_report["known_reference_count"] == 3
    assert real_gp_report["known_reference_max_abs_error"] <= 1e-4
    assert real_gp_report["raw_example_count"] == 3
    assert {"BB2", "BB3", "BB5"}.issubset(real_gp_report["raw_example_bb_tokens"])


def test_real_gp_all_138_candidates_match_direct_original_predictions(real_gp_report):
    assert real_gp_report["covered_tokens"] == 138
    assert real_gp_report["prediction_count"] == 282
    assert real_gp_report["mean_max_abs_error"] <= 1e-9
    assert real_gp_report["std_max_abs_error"] <= 1e-9


def test_real_gp_metadata_preserves_caps_order_topology_and_smiles(real_gp_report):
    assert real_gp_report["caps"] == ["Ac", "H"]
    assert real_gp_report["topology_count"] > 2
    assert real_gp_report["order_swap_checked"]
    assert real_gp_report["smiles_checked"] == 282
    assert real_gp_report["metadata_copy_checked"]


def test_real_gp_rejects_invalid_actions_and_remains_usable(real_gp_report):
    assert set(real_gp_report["rejections"]) == {
        "unknown_token", "length_10", "length_12",
        "thiols_0", "thiols_1", "thiols_3", "invalid_cap", "extra_group", "extra_terminal_key",
    }


def test_real_gp_creates_no_intermediate_files(real_gp_report):
    assert real_gp_report["no_generated_artifacts"]
