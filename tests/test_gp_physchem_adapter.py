"""Adapter contracts and opt-in parity checks against the actual external GP.

Real checks require GP_PHYSCHEM_TEST_PYTHON, GP_PHYSCHEM_TEST_ROOT, and
GP_PHYSCHEM_TEST_PREPARED. Without all three, the expensive checks are skipped.
The original GP project is always read only; cache mutations use temporary copies.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[1] / "integrations/gp_physchem/worker.py"
_SPEC = importlib.util.spec_from_file_location("gp_physchem_worker_contract", _PATH)
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "original.py"
    source.write_text("original model")
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("monomer,has_free_thiol\nBB1,1\n")
    manifest = {
        "schema_version": worker.SCHEMA_VERSION,
        "inputs": {"model": worker.fingerprint(source)},
        "artifacts": {"catalog": {"path": "catalog.csv", "sha256": worker.sha256(catalog)}},
        "versions": {"python": "test"},
    }
    worker.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(worker, "versions", lambda: {"python": "test"})
    return tmp_path, manifest


def test_manifest_accepts_unchanged_artifacts(prepared):
    path, manifest = prepared
    assert worker.verify_manifest(path) == manifest


@pytest.mark.parametrize("file_name", ["original.py", "catalog.csv"])
def test_manifest_rejects_changed_source_or_catalog(prepared, file_name):
    path, _ = prepared
    (path / file_name).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        worker.verify_manifest(path)


def test_manifest_rejects_runtime_drift(prepared, monkeypatch):
    path, _ = prepared
    monkeypatch.setattr(worker, "versions", lambda: {"python": "different"})
    with pytest.raises(ValueError, match="versions changed"):
        worker.verify_manifest(path)


@pytest.fixture
def predictor():
    instance = worker.Predictor.__new__(worker.Predictor)
    instance.manifest = {"n_residues": 11}
    instance.allowed = {"BB1", "BB2"}
    instance.library = SimpleNamespace(get=lambda token: SimpleNamespace(has_free_thiol=int(token == "BB1")))
    return instance


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ({"peptide": ["BB1"] * 10, "n_terminal": {"treatment": "H"}}, "exactly 11"),
        ({"peptide": ["C"] * 11, "n_terminal": {"treatment": "H"}}, "outside source catalog"),
        ({"peptide": ["BB1"] * 11, "n_terminal": {"treatment": "H"}, "ignored": 1}, "exactly peptide"),
        ({"peptide": ["BB1"] * 11, "n_terminal": {"treatment": "H", "ignored": 1}}, "H or Ac"),
        ({"peptide": ["BB2"] * 11, "n_terminal": {"treatment": "other"}}, "H or Ac"),
        ({"peptide": ["BB1"] * 11, "n_terminal": {"treatment": "H"}}, "exactly two"),
        ({"peptide": ["BB2"] * 11, "n_terminal": {"treatment": "H"}}, "exactly two"),
    ],
)
def test_predict_rejects_invalid_candidate_before_gp(predictor, candidate, message):
    with pytest.raises(ValueError, match=message):
        predictor.predict(candidate)


def test_protocol_recovers_after_invalid_request(monkeypatch, capsys, tmp_path):
    class FakePredictor:
        def __init__(self, _):
            print("model diagnostics")
            self.metadata = {"target": "pIC50_from_1000nM"}

        def predict(self, candidate):
            print("prediction diagnostics")
            return {"mean": 7.0, "std": 0.3, "metadata": candidate}

    monkeypatch.setattr(worker, "Predictor", FakePredictor)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO('not JSON\n{"op":"predict","candidate":{"id":1}}\n'))
    worker.serve(tmp_path)
    captured = capsys.readouterr()
    results = [json.loads(line) for line in captured.out.splitlines()]
    assert results[0]["status"] == "ready"
    assert results[1]["error_type"] == "JSONDecodeError"
    assert results[2] == {"mean": 7.0, "std": 0.3, "metadata": {"id": 1}}
    assert "model diagnostics" in captured.err
    assert "prediction diagnostics" in captured.err


@pytest.fixture(scope="session")
def real_gp_runtime():
    names = ("GP_PHYSCHEM_TEST_PYTHON", "GP_PHYSCHEM_TEST_ROOT", "GP_PHYSCHEM_TEST_PREPARED")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip("opt-in real GP checks: set " + ", ".join(missing))
    # Resolving a venv executable symlink would select its base interpreter and
    # lose the venv packages. Preserve the invoked interpreter path.
    python = Path(os.path.abspath(Path(os.environ[names[0]]).expanduser()))
    root, prepared_dir = (Path(os.environ[name]).expanduser().resolve() for name in names[1:])
    required = [python, root / "chemistry.py", root / "examples/new_monomers.csv", prepared_dir / "manifest.json"]
    if not all(path.is_file() for path in required):
        pytest.skip("opt-in GP interpreter, project or prepared cache is unavailable")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return python, root, prepared_dir, env


# Run in the GP's own pinned interpreter. The reference library is intentionally
# loaded from the RAW source CSV: using the prepared catalog here would conceal
# descriptor drift introduced by preparation, serialization or runtime changes.
_REAL_MODEL_CHECK = r'''
import collections
import contextlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

root, prepared, worker_path, repo = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from chemistry import build_peptide, load_library
from features import featurise, load_training_table, train_matrix, validate_candidates
from model import fit_gp, predict_gp

report = {}
with contextlib.redirect_stdout(sys.stderr):
    original = load_library(root / "data/monomer_dictionary.csv", root / "examples/new_monomers.csv")
    raw = pd.read_csv(root / "examples/new_monomers.csv", encoding="utf-8-sig")
    tokens = raw.monomer.tolist()
    assert len(tokens) == len(set(tokens)) == 138
    thiols = [name for name in tokens if original.get(name).has_free_thiol == 1]
    assert thiols == ["BB1", "BB6", "BB55", "BB56", "BB103", "BB137"]
    train = load_training_table(root / "data/long_table.csv")
    matrix, columns, table, _ = train_matrix(train, original)
    model, scaler = fit_gp(matrix, train.pIC50_from_1000nM.to_numpy(float))

    spec = importlib.util.spec_from_file_location("real_gp_worker", worker_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    predictor = adapter.Predictor(prepared)
    assert predictor.tokens == tokens

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
    # Match the supplied GP self-test's sklearn 1.7 reference tolerance; worker
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

    frame = pd.DataFrame(rows)
    direct_seqs, direct_x, direct_mean, direct_std = direct(frame)
    assert not np.array_equal(direct_x[special_start], direct_x[special_start+1])
    assert not np.array_equal(direct_x[special_start], direct_x[special_start+2])
    assert not np.array_equal(direct_x[special_start], direct_x[special_start+3])
    errors_mean, errors_std = [], []
    for index, candidate in enumerate(candidates):
        result = predictor.predict(candidate)
        np.testing.assert_allclose([result["mean"], result["std"]], [direct_mean[index], direct_std[index]], rtol=0, atol=1e-9)
        errors_mean.append(abs(result["mean"]-direct_mean[index]))
        errors_std.append(abs(result["std"]-direct_std[index]))
        metadata = result["metadata"]
        row = rows[index]
        for key in ("monomers", "n_term", "c_term", "disulfide_pos_i", "disulfide_pos_j"):
            assert metadata[key] == row[key], (index, key, metadata, row)
        smiles, pair = build_peptide(direct_seqs[index], row["n_term"] == "Ac", original)
        assert Chem.MolFromSmiles(metadata["smiles"]) is not None
        assert metadata["smiles"] == smiles
        assert tuple(pair) == (row["disulfide_pos_i"], row["disulfide_pos_j"])
        assert metadata["pred_pIC50"] == result["mean"]
        assert metadata["posterior_sd"] == result["std"]
    report.update(prediction_count=len(candidates), covered_tokens=len(covered), mean_max_abs_error=float(max(errors_mean)), std_max_abs_error=float(max(errors_std)), caps=sorted(set(frame.n_term)), topology_count=len(set(zip(frame.disulfide_pos_i, frame.disulfide_pos_j))), order_swap_checked=True, smiles_checked=len(candidates))

    valid = candidates[special_start]
    def payload(seq=None, cap="H", **extra):
        return {"peptide": list(anchor if seq is None else seq), "n_terminal": {"treatment": cap}, **extra}
    invalid = {
        "dictionary_token_outside_catalog": payload(["A"]+anchor[1:]),
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

    # Exercise the genuine JSON transport, including cache keys for same-count
    # reorderings and recovery after a rejected candidate, in the pinned runtime.
    sys.path.insert(0, str(repo / "src"))
    from rl_matdesign.predictors.json_process import JsonProcessPredictor
    with JsonProcessPredictor({"command": [sys.executable, "-u", str(worker_path), "serve", "--prepared", str(prepared)], "startup_timeout":300, "request_timeout":120}) as transport:
        for candidate in invalid.values():
            try:
                transport.predict(candidate)
            except RuntimeError as exc:
                assert "rejected candidate" in str(exc)
            else:
                raise AssertionError("JSON transport accepted invalid candidate")
        for index in range(special_start, len(candidates)):
            candidate = candidates[index]
            np.testing.assert_allclose(transport.predict(candidate), [direct_mean[index],direct_std[index]], rtol=0, atol=1e-9)
            metadata = transport.candidate_metadata(candidate)
            assert metadata["monomers"] == rows[index]["monomers"]
            assert metadata["n_term"] == rows[index]["n_term"]
            assert (metadata["disulfide_pos_i"],metadata["disulfide_pos_j"]) == (rows[index]["disulfide_pos_i"],rows[index]["disulfide_pos_j"])
        assert len(transport._cache) == 4
    report["transport_cases"] = 4
print(json.dumps(report, allow_nan=False))
'''


@pytest.fixture(scope="session")
def real_gp_report(real_gp_runtime, tmp_path_factory):
    python, root, prepared_dir, env = real_gp_runtime
    result = subprocess.run(
        [str(python), "-c", _REAL_MODEL_CHECK, str(root), str(prepared_dir), str(_PATH), str(_PATH.parents[2])],
        capture_output=True, text=True, env=env, timeout=1200,
    )
    assert result.returncode == 0, f"Independent GP validation failed:\n{result.stderr}\n{result.stdout}"
    report = json.loads(result.stdout)
    (tmp_path_factory.mktemp("gp_parity") / "report.json").write_text(json.dumps(report, indent=2)+"\n")
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
    assert real_gp_report["prediction_count"] == 280
    assert real_gp_report["mean_max_abs_error"] <= 1e-9
    assert real_gp_report["std_max_abs_error"] <= 1e-9


def test_real_gp_metadata_preserves_caps_order_topology_and_smiles(real_gp_report):
    assert real_gp_report["caps"] == ["Ac", "H"]
    assert real_gp_report["topology_count"] > 2
    assert real_gp_report["order_swap_checked"]
    assert real_gp_report["smiles_checked"] == 280
    assert real_gp_report["transport_cases"] == 4


def test_real_gp_rejects_invalid_actions_and_transport_recovers(real_gp_report):
    assert set(real_gp_report["rejections"]) == {
        "dictionary_token_outside_catalog", "unknown_token", "length_10", "length_12",
        "thiols_0", "thiols_1", "thiols_3", "invalid_cap", "extra_group", "extra_terminal_key",
    }


@pytest.mark.parametrize("mutation", ["catalog_hash", "source_hash", "runtime_version", "schema_version", "artifact_escape"])
def test_real_gp_worker_rejects_tampered_cache(real_gp_runtime, tmp_path, mutation):
    python, _, prepared_dir, env = real_gp_runtime
    copied = tmp_path / "prepared"
    shutil.copytree(prepared_dir, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "catalog_hash":
        with (copied / manifest["catalog_file"]).open("a") as stream:
            stream.write("\n")
        expected = "prepared artifact changed"
    elif mutation == "source_hash":
        manifest["inputs"]["dictionary"]["sha256"] = "0"*64
        expected = "GP input changed"
    elif mutation == "runtime_version":
        manifest["versions"]["rdkit"] = "different-version"
        expected = "versions changed"
    elif mutation == "schema_version":
        manifest["schema_version"] = -1
        expected = "unsupported prepared manifest schema"
    else:
        outside = tmp_path / "outside.csv"
        shutil.copyfile(copied / "catalog.csv", outside)
        manifest["artifacts"]["catalog"]["path"] = "../outside.csv"
        expected = "prepared artifact changed"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [str(python), "-u", str(_PATH), "serve", "--prepared", str(copied)],
        input="", capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode != 0
    assert '"status": "ready"' not in result.stdout
    assert expected in result.stderr
