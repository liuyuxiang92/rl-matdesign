#!/usr/bin/env python3
"""Prepare and serve the user's unmodified GP in its own Python environment.

Preparation writes a portable, inspectable descriptor CSV and provenance manifest.
Serving is a line-delimited JSON protocol; model fitting happens once at startup.
The GP project is read only. No pickle or generated Python is loaded from caches.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
GP_MODULES = ("config", "constants", "chemistry", "features", "model")
PACKAGES = ("numpy", "pandas", "sklearn", "scipy", "rdkit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: importlib.import_module(name).__version__ for name in PACKAGES},
    }


def load_gp(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    for name in GP_MODULES:
        if not (root / f"{name}.py").is_file():
            raise ValueError(f"missing GP module: {root / (name + '.py')}")
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve() != root / f"{name}.py":
            raise RuntimeError(f"module {name!r} is already loaded from another project")
    sys.path.insert(0, str(root))
    return {name: importlib.import_module(name) for name in GP_MODULES}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _catalog_ids(frame: Any, chemistry: Any) -> list[str]:
    if "monomer" not in frame:
        raise ValueError("new-monomer CSV needs a monomer column")
    result = []
    for row_number, raw in enumerate(frame["monomer"], start=2):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"new-monomer CSV row {row_number}: empty monomer ID")
        canonical = chemistry.normalise_name(raw)
        if canonical != raw:
            raise ValueError(
                f"new-monomer CSV row {row_number}: use canonical ID {canonical!r} "
                f"instead of {raw!r}; aliases cannot be distinct actions"
            )
        if raw in result:
            raise ValueError(f"new-monomer CSV row {row_number}: duplicate ID {raw!r}")
        result.append(raw)
    if not result:
        raise ValueError("new-monomer CSV is empty")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.gp_root.expanduser().resolve(strict=True)
    source = (args.new_monomers or root / "examples/new_monomers.csv").expanduser().resolve(strict=True)
    train_path = (args.train or root / "data/long_table.csv").expanduser().resolve(strict=True)
    dictionary_path = (args.dictionary or root / "data/monomer_dictionary.csv").expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    # Descriptor output must never overwrite inputs or the provided GP project.
    if out == root or root in out.parents:
        raise ValueError("prepared output must be outside the original GP project")
    if (out / "manifest.json").exists():
        previous = verify_manifest(out)
        expected = {"train": train_path, "dictionary": dictionary_path, "new_monomers": source}
        if previous["gp_root"] != str(root) or any(previous["inputs"][key]["path"] != str(path) for key, path in expected.items()):
            raise ValueError("existing preparation uses different GP inputs; choose a fresh output directory")
        return {"status": "prepared", "reused": True, "directory": str(out), "metadata": previous}
    if (out / "catalog.csv").exists():
        raise ValueError(f"prepared files already exist in {out}; choose a fresh output directory")
    gp = load_gp(root)
    pd = importlib.import_module("pandas")
    chemistry, constants = gp["chemistry"], gp["constants"]
    raw = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False)
    tokens = _catalog_ids(raw, chemistry)
    library = chemistry.load_library(dictionary_path)
    records = []
    failures = []
    for row_number, (_, row) in enumerate(raw.iterrows(), start=2):
        try:
            rec = library.register_extra_row(row)
            if not rec.can_assemble:
                raise ValueError("monomer has no supported peptide backbone atom mapping")
            mapped_thiol = int(chemistry.has_map(rec.smiles_mapped, 3))
            if rec.has_free_thiol not in (0, 1) or mapped_thiol != rec.has_free_thiol:
                raise ValueError("free-thiol annotation disagrees with the mapped chemical structure")
            entry = {f"source_{key}": value for key, value in row.items()}
            entry.update({
                "monomer": rec.name,
                "smiles_free_aa": rec.smiles_free_aa,
                "smiles_mapped": rec.smiles_mapped,
                "chirality": rec.chirality,
                "backbone": rec.backbone,
                "has_free_thiol": rec.has_free_thiol,
                **{name: float(rec.vector[index]) for index, name in enumerate(constants.DESC_COLS)},
            })
            records.append(entry)
        except Exception as exc:
            failures.append({"row": row_number, "monomer": row["monomer"], "error": str(exc)})
    if failures:
        raise ValueError("monomer audit failed; no rows were dropped: " + json.dumps(failures, ensure_ascii=False))
    thiols = [token for token in tokens if library.get(token).has_free_thiol == 1]
    if not thiols:
        raise ValueError("source catalog has no free-thiol monomers")
    # Validate all training inputs before publishing the prepared manifest.
    training = gp["features"].load_training_table(train_path)
    matrix, columns, _, _ = gp["features"].train_matrix(training, library)
    out.mkdir(parents=True, exist_ok=True)
    catalog_path = out / "catalog.csv"
    pd.DataFrame(records).to_csv(catalog_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gp_root": str(root),
        "inputs": {
            **{f"module_{name}": fingerprint(root / f"{name}.py") for name in GP_MODULES},
            "adapter": fingerprint(Path(__file__)),
            "train": fingerprint(train_path),
            "dictionary": fingerprint(dictionary_path),
            "new_monomers": fingerprint(source),
        },
        "artifacts": {"catalog": {"path": "catalog.csv", "sha256": sha256(catalog_path)}},
        "catalog_file": "catalog.csv",
        "versions": versions(),
        "source_tokens": tokens,
        "thiol_tokens": thiols,
        "n_tokens": len(tokens),
        "n_residues": int(constants.N_RESIDUES),
        "n_terminal_treatments": ["H", "Ac"],
        "c_term": "NH2",
        "target": "pIC50_from_1000nM",
        "direction": "maximize",
        "uncertainty": "posterior_sd including fitted WhiteKernel observation noise",
        "training_rows": len(training),
        "feature_width": int(matrix.shape[1]),
        "onehot_width": len(columns),
        "descriptor_columns": list(constants.DESC_COLS),
    }
    write_json(out / "manifest.json", manifest)
    return {"status": "prepared", "directory": str(out), "metadata": manifest}


def verify_manifest(prepared: Path) -> dict[str, Any]:
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prepared manifest schema")
    for label, item in manifest["inputs"].items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"GP input changed since preparation: {label} ({path}); prepare a fresh cache")
    for label, item in manifest["artifacts"].items():
        path = (prepared / item["path"]).resolve()
        if prepared not in path.parents or not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"prepared artifact changed: {label}; prepare a fresh cache")
    current_versions = versions()
    if current_versions != manifest["versions"]:
        raise ValueError(f"GP runtime versions changed: prepared={manifest['versions']}, current={current_versions}")
    return manifest


class Predictor:
    def __init__(self, prepared: Path):
        prepared = prepared.expanduser().resolve(strict=True)
        self.manifest = verify_manifest(prepared)
        self.gp = load_gp(Path(self.manifest["gp_root"]))
        self.pd = importlib.import_module("pandas")
        self.np = importlib.import_module("numpy")
        self.tokens = self.manifest["source_tokens"]
        self.allowed = set(self.tokens)
        self.library = self.gp["chemistry"].load_library(
            Path(self.manifest["inputs"]["dictionary"]["path"]), prepared / "catalog.csv"
        )
        if len(self.tokens) != len(self.allowed) or len(self.tokens) != self.manifest["n_tokens"]:
            raise ValueError("prepared catalog has inconsistent token identities")
        catalog = self.pd.read_csv(prepared / "catalog.csv", encoding="utf-8-sig")
        if catalog["monomer"].tolist() != self.tokens:
            raise ValueError("prepared catalog order/identities disagree with manifest")
        thiols = [token for token in self.tokens if self.library.get(token).has_free_thiol == 1]
        if thiols != self.manifest["thiol_tokens"]:
            raise ValueError("prepared thiol classifications disagree with manifest")
        train = self.gp["features"].load_training_table(Path(self.manifest["inputs"]["train"]["path"]))
        matrix, self.columns, self.table, train_seqs = self.gp["features"].train_matrix(train, self.library)
        if matrix.shape != (self.manifest["training_rows"], self.manifest["feature_width"]):
            raise ValueError("prepared training feature dimensions changed")
        self.model, self.scaler = self.gp["model"].fit_gp(matrix, train[self.manifest["target"]].to_numpy(float))
        self.train_scaled = self.scaler.transform(matrix)
        self.observed_pairs = {(position, token) for seq in train_seqs for position, token in enumerate(seq[:11])}
        self.metadata = {
            key: self.manifest[key] for key in (
                "n_tokens", "thiol_tokens", "target", "direction", "n_residues", "n_terminal_treatments",
                "c_term", "uncertainty", "training_rows", "feature_width", "versions",
            )
        }
        self.metadata["fitted_kernel"] = str(self.model.kernel_)

    def predict(self, candidate: Any) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        if set(candidate) != {"peptide", "n_terminal"}:
            raise ValueError("candidate must contain exactly peptide and n_terminal action groups")
        seq = candidate.get("peptide")
        length = self.manifest["n_residues"]
        if not isinstance(seq, list) or len(seq) != length or any(not isinstance(item, str) for item in seq):
            raise ValueError(f"candidate.peptide must be an ordered list of exactly {length} monomer IDs")
        unknown = sorted(set(seq) - self.allowed)
        if unknown:
            raise ValueError(f"candidate contains monomers outside source catalog: {unknown}")
        terminal = candidate.get("n_terminal")
        if not isinstance(terminal, dict) or set(terminal) != {"treatment"} or terminal.get("treatment") not in ("H", "Ac"):
            raise ValueError("candidate.n_terminal.treatment must be H or Ac")
        treatment = terminal["treatment"]
        thiols = [index + 1 for index, token in enumerate(seq) if self.library.get(token).has_free_thiol == 1]
        if len(thiols) != 2:
            raise ValueError(f"candidate must contain exactly two free-thiol residues; found positions {thiols}")
        row = {
            "compound_id": "rl_candidate", "monomers": "|".join(seq), "n_term": treatment,
            "c_term": "NH2", "disulfide_pos_i": thiols[0], "disulfide_pos_j": thiols[1],
        }
        frame = self.pd.DataFrame([row])
        seqs, ring, n_residues, acetylated = self.gp["features"].validate_candidates(frame, self.library)
        x = self.gp["features"].featurise(seqs, self.columns, self.table, ring, n_residues, acetylated)
        mean, std = self.gp["model"].predict_gp(self.model, self.scaler, x)
        mean, std = float(mean[0]), float(std[0])
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise ValueError("GP returned non-finite mean or invalid uncertainty")
        smiles, pair = self.gp["chemistry"].build_peptide(seq, treatment == "Ac", self.library)
        if tuple(thiols) != pair:
            raise ValueError("assembled disulfide positions disagree with predicted positions")
        scaled = self.scaler.transform(x)[0]
        distance = float(self.np.sqrt(((self.train_scaled - scaled) ** 2).sum(axis=1)).min())
        unseen = "|".join(f"p{index + 1}={token}" for index, token in enumerate(seq) if (index, token) not in self.observed_pairs) or "none"
        metadata = {
            **{key: value for key, value in row.items() if key != "compound_id"},
            "pred_pIC50": mean, "posterior_sd": std,
            "nearest_training_distance_scaled": distance,
            "unseen_position_monomer": unseen,
            "mutation_count_vs_parent": self.gp["features"].mutation_count(seq),
            "applicability_note": "seen position×monomer" if unseen == "none" else "unseen position×monomer; use for ranking only",
            "smiles": smiles,
        }
        return {"mean": mean, "std": std, "metadata": metadata}


def serve(prepared: Path) -> None:
    with contextlib.redirect_stdout(sys.stderr):
        predictor = Predictor(prepared)
    print(json.dumps({"status": "ready", "metadata": predictor.metadata}, ensure_ascii=False, allow_nan=False), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("op") != "predict":
                raise ValueError("expected request object with op=predict")
            with contextlib.redirect_stdout(sys.stderr):
                result = predictor.predict(request.get("candidate"))
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="audit source monomers and cache descriptors")
    prep.add_argument("--gp-root", type=Path, required=True)
    prep.add_argument("--new-monomers", type=Path, help="default: GP_ROOT/examples/new_monomers.csv")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--train", type=Path)
    prep.add_argument("--dictionary", type=Path)
    server = commands.add_parser("serve", help="start the persistent line-delimited JSON predictor")
    server.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            with contextlib.redirect_stdout(sys.stderr):
                result = prepare(args)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        else:
            serve(args.prepared)
    except Exception as exc:
        print(f"gp_physchem {args.command} failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
