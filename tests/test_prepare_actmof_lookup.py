"""Validate archive construction on a tiny synthetic grid, without downloading."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_matdesign.predictors.lookup import LookupPredictor


@pytest.fixture
def prepared_source(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_actmof_lookup.py"
    spec = importlib.util.spec_from_file_location("actmof_prepare_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIELD_NAMES = ("temperature", "duration")
    module.RANGES = {"temperature": (10, 11), "duration": (1, 2)}
    module.SHAPE = (2, 2)
    module.GRID_SIZE = 4
    module.DATA_FILES = ("data/part.csv",)
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    data = source / "data/part.csv"
    pd.DataFrame({
        "temperature": [11, 10, 11, 10], "duration": [2, 1, 1, 2],
        "q_final": [4, 1, 3, 0], "emulated_intensity": [999] * 4,
    }).to_csv(data, index=False)
    (source / "LICENSE").write_text("Synthetic fixture license\n")
    module.SOURCE_SHA256 = {
        name: module.sha256_file(source / name)
        for name in (*module.DATA_FILES, "LICENSE")
    }
    return module, source, data


def test_prepare_writes_exact_dense_grid_and_verifiable_manifest(prepared_source, tmp_path):
    module, source, _ = prepared_source
    output = tmp_path / "lookup"
    result = module.prepare(source, output, chunksize=2)
    predictor = LookupPredictor({"model": output / "lookup.npz"})
    assert predictor.predict({"duration": 2, "temperature": 10}) == (0., None)
    assert predictor.predict({"conditions": {"temperature": 11, "duration": 2}}) == (4., None)
    assert result["score"]["count"] == 4
    assert result["score"]["zero_count"] == 1
    assert result["lookup"]["sha256"] == module.sha256_file(output / "lookup.npz")
    assert result["uncertainty"]["published"] is False
    assert json.loads((output / "manifest.json").read_text()) == result
    assert (output / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    with np.load(output / "lookup.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"field_names", "mean", "axis_0", "axis_1"}


@pytest.mark.parametrize("case,message", [
    ("duplicate", "Duplicate condition"), ("missing", "Incomplete ActMOF grid"),
    ("outside", "outside"), ("fractional", "Non-integer"),
    ("nonfinite", "Non-finite"), ("negative", "Negative q_final"),
    ("inexact", "represented exactly"),
])
def test_invalid_or_incomplete_source_never_becomes_a_lookup(prepared_source, case, message):
    module, source, data = prepared_source
    frame = pd.read_csv(data)
    if case == "duplicate":
        frame.loc[1] = frame.loc[0]
    elif case == "missing":
        frame = frame.iloc[:-1]
    elif case == "outside":
        frame.loc[0, "temperature"] = 12
    elif case == "fractional":
        frame["duration"] = frame["duration"].astype(float)
        frame.loc[0, "duration"] = 1.5
    elif case == "nonfinite":
        frame["q_final"] = frame["q_final"].astype(float)
        frame.loc[0, "q_final"] = np.inf
    elif case == "negative":
        frame.loc[0, "q_final"] = -1
    else:
        frame.loc[0, "q_final"] = 2**24 + 1
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match=message):
        module.build_dense_lookup(source, chunksize=2)


def test_hash_mismatch_fails_before_replacing_existing_archive(prepared_source, tmp_path):
    module, source, data = prepared_source
    output = tmp_path / "lookup"
    output.mkdir()
    previous = output / "lookup.npz"
    previous.write_bytes(b"previous archive")
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.prepare(source, output)
    assert previous.read_bytes() == b"previous archive"
