"""Generated CSV projection preserves raw metrics and HPO reward semantics."""
from __future__ import annotations

from copy import deepcopy
import csv

import pytest

from rl_matdesign.hpo.metric import top_k_mean_from_csv
from rl_matdesign.utils.metrics import RunMetrics, validate_generated_columns


_COLUMNS = {
    "formula": "formula",
    "reward": "reward",
    "pIC50": "obj_pIC50_mean",
    "pIC50_uncertainty": "obj_pIC50_std",
    "meta_monomers": "meta_monomers",
    "meta_smiles": "meta_smiles",
}


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def _metrics():
    metrics = RunMetrics()
    metrics.log(phase="train", reward=1000.0, loss=0.2)
    for index, reward in enumerate((1.0, 5.0, 3.0)):
        metrics.log(
            phase="generate", formula=f"sequence_{index}", reward=reward,
            # Deliberately different: exported activity must be the raw
            # per-property mean, never the aggregate or transformed reward.
            dp_mean=-10.0 * index, obj_pIC50_mean=6.0 + index,
            obj_pIC50_std=0.1 * (index + 1),
            meta_monomers='A|特殊残基,"X"\n|C',
            meta_smiles="N[C@@H](C)C(=O)O",
            candidate_json='{"peptide": ["A", "C"]}',
            metadata_json='{"diagnostics": true}',
        )
    return metrics


def test_projection_uses_raw_stats_preserves_metrics_and_hpo(tmp_path):
    metrics = _metrics()
    before = deepcopy(metrics.rows)
    columns = deepcopy(_COLUMNS)
    before_log = tmp_path / "before.csv"
    after_log = tmp_path / "after.csv"
    output = tmp_path / "generated.csv"
    metrics.to_csv(before_log)
    metrics.to_generated_csv(output, columns=columns)
    metrics.to_csv(after_log)

    headers, rows = _read(output)
    assert headers == list(_COLUMNS)
    assert [row["formula"] for row in rows] == ["sequence_1", "sequence_2", "sequence_0"]
    assert [float(row["pIC50"]) for row in rows] == [7.0, 8.0, 6.0]
    assert [float(row["pIC50_uncertainty"]) for row in rows] == pytest.approx([0.2, 0.3, 0.1])
    assert all(row["meta_monomers"] == before[1]["meta_monomers"] for row in rows)
    assert all(row["meta_smiles"] == before[1]["meta_smiles"] for row in rows)
    assert metrics.rows == before
    assert columns == _COLUMNS
    assert before_log.read_bytes() == after_log.read_bytes()
    assert len(rows) == 3
    assert top_k_mean_from_csv(str(output), k=2) == (4.0, 2)


def test_default_export_preserves_optional_fields_from_all_sorted_rows(tmp_path):
    metrics = _metrics()
    # A top-ranked row lacking optional metadata cannot erase other rows' data.
    metrics.rows[1]["only_in_later_row"] = "keep this diagnostic"
    output = tmp_path / "generated.csv"
    metrics.to_generated_csv(output)
    headers, rows = _read(output)
    assert headers == [key for key in metrics.rows[2] if key != "phase"] + ["only_in_later_row"]
    assert rows[-1]["only_in_later_row"] == "keep this diagnostic"
    assert rows[0]["only_in_later_row"] == ""
    assert "candidate_json" in headers and "metadata_json" in headers
    assert [float(row["reward"]) for row in rows] == [5.0, 3.0, 1.0]


@pytest.mark.parametrize("columns", [
    {}, [], "formula,reward", {"formula": "formula"},
    {"formula": "formula", "reward": "dp_mean"},
    {"formula": "candidate_json", "reward": "reward"},
    {**_COLUMNS, "": "dp_mean"}, {**_COLUMNS, "  ": "dp_mean"},
    {**_COLUMNS, "activity": ""}, {**_COLUMNS, "activity": "  "},
    {**_COLUMNS, "activity": 1}, {**_COLUMNS, 1: "dp_mean"},
])
def test_invalid_column_mapping_fails_validation(columns):
    with pytest.raises(ValueError):
        validate_generated_columns(columns)


def test_optional_default_mapping_is_valid():
    validate_generated_columns(None)
    validate_generated_columns(_COLUMNS)


@pytest.mark.parametrize("preexisting", [False, True])
def test_missing_source_in_later_row_fails_before_opening_output(tmp_path, preexisting):
    metrics = _metrics()
    del metrics.rows[1]["obj_pIC50_std"]
    output = tmp_path / "generated.csv"
    if preexisting:
        output.write_text("previous complete results\n", encoding="utf-8")
    with pytest.raises(ValueError, match="obj_pIC50_std"):
        metrics.to_generated_csv(output, columns=_COLUMNS)
    if preexisting:
        assert output.read_text(encoding="utf-8") == "previous complete results\n"
    else:
        assert not output.exists()
