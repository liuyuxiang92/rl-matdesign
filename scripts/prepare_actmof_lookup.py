#!/usr/bin/env python3
"""Validate the pinned ActMOF CSVs and build a dense LookupPredictor archive.

Examples::

    python scripts/prepare_actmof_lookup.py --source /path/to/ActMOF --out data/actmof
    python scripts/prepare_actmof_lookup.py --download --out data/actmof

Only the official q_final column supplies scores. The emulated intensity and
FWHM are deliberately not converted back into scores. No oracle uncertainty is
published, so the archive omits std (the lookup predictor's deterministic zero).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

import numpy as np
import pandas as pd

SOURCE_REPOSITORY = "https://github.com/deep-synthesis/ActMOF"
SOURCE_COMMIT = "f1733f05bf8f18a585d50088b8aae3f86a2c5078"
RAW_BASE_URL = f"https://raw.githubusercontent.com/deep-synthesis/ActMOF/{SOURCE_COMMIT}"
FIELD_NAMES = (
    "metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature",
)
RANGES = dict(zip(FIELD_NAMES, ((5, 75), (5, 15), (0, 30), (1, 12), (10, 30))))
SHAPE = tuple(hi - lo + 1 for lo, hi in RANGES.values())
GRID_SIZE = math.prod(SHAPE)
DATA_FILES = tuple(f"data/design_grid_part{i:03d}.csv" for i in range(1, 8))

# Verify local sources as well as downloads against the pinned public version.
SOURCE_SHA256 = {
    DATA_FILES[0]: "9f2f1b9828bfc1af5a2b7756535da7f0a260a4e85aae8494f156d06168d97760",
    DATA_FILES[1]: "6f71a07524af4f703fae61adc71c7adc5576a749fc2cebcd5a07db44605bb5c8",
    DATA_FILES[2]: "b306420da115bbad03dddfbf50e32256d340f584bd54f17c44f40f554c357595",
    DATA_FILES[3]: "b47d963bf01b39edeb857fcbabf7bc8e7380e3be068f6af1284bf0c92b799213",
    DATA_FILES[4]: "5661378812385f89c7478673ec501c433f90c6d23a10d8fed2aa62060006e079",
    DATA_FILES[5]: "24ad3ba363b953f0669a5022dea0cc5f0920a81f0ec708ff986524817ca81508",
    DATA_FILES[6]: "b66d5a3391fe2e045556890753aed4aa1b9dd0678c3c11395925a5b691e7fd12",
    "LICENSE": "e9d1369f0ca6df7359478877ffadcd07e7968413c75364081b71b6444418a08f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, relative: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official source file: {path}")
    actual = sha256_file(path)
    expected = SOURCE_SHA256[relative]
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, found {actual}. "
            f"This lookup requires the files at commit {SOURCE_COMMIT}."
        )
    return {
        "path": relative,
        "local_path": str(path.resolve()),
        "url": f"{RAW_BASE_URL}/{relative}",
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def download_source(source: Path, relative: str, timeout: float = 120.0) -> None:
    """A download is published into the cache only after its hash is verified."""
    destination = source / relative
    if destination.exists():
        verify_source(destination, relative)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(
                f"{RAW_BASE_URL}/{relative}", headers={"User-Agent": "rl-matdesign-actmof"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        verify_source(temporary, relative)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_dense_lookup(source: Path, *, chunksize: int = 200_000) -> tuple[np.ndarray, dict]:
    """Require one finite integer score for every point of the complete grid."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    scores = np.empty(GRID_SIZE, dtype=np.float32)
    seen = np.zeros(GRID_SIZE, dtype=bool)
    columns = list(FIELD_NAMES) + ["q_final"]
    lower = np.array([RANGES[name][0] for name in FIELD_NAMES], dtype=np.int64)
    upper = np.array([RANGES[name][1] for name in FIELD_NAMES], dtype=np.int64)
    rows = 0
    file_rows = {}
    for relative in DATA_FILES:
        path = source / relative
        count = 0
        # Parse as float64 first, so fractional coordinates or scores cannot be
        # silently truncated by an integer dtype cast.
        for chunk in pd.read_csv(path, usecols=columns, dtype=np.float64, chunksize=chunksize):
            values = chunk.loc[:, columns].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite coordinate or q_final in {path}")
            if not np.equal(values, np.rint(values)).all():
                raise ValueError(f"Non-integer coordinate or q_final in {path}")
            coordinates = values[:, :len(FIELD_NAMES)]
            q = values[:, -1]
            if np.any(coordinates < lower) or np.any(coordinates > upper):
                raise ValueError(f"Coordinate outside the declared ActMOF grid in {path}")
            if np.any(q < 0):
                raise ValueError(f"Negative q_final in {path}")
            indices = np.ravel_multi_index((coordinates.astype(np.int64) - lower).T, SHAPE)
            if np.unique(indices).size != indices.size or seen[indices].any():
                raise ValueError(f"Duplicate condition in {path}, near source row {count + 1}")
            q32 = q.astype(np.float32)
            if not np.array_equal(q32.astype(np.float64), q):
                raise ValueError(f"q_final cannot be represented exactly as float32 in {path}")
            scores[indices] = q32
            seen[indices] = True
            count += len(chunk)
        file_rows[relative] = count
        rows += count
        print(f"Validated {relative}: {count:,} rows", flush=True)
    missing = int((~seen).sum())
    if rows != GRID_SIZE or missing:
        raise ValueError(f"Incomplete ActMOF grid: {rows:,} rows; {missing:,} missing conditions")
    return scores.reshape(SHAPE), {
        "count": rows,
        "min": int(scores.min()),
        "max": int(scores.max()),
        "zero_count": int(np.count_nonzero(scores == 0)),
        "nonzero_count": int(np.count_nonzero(scores)),
        "file_rows": file_rows,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(source: Path, out: Path, *, download: bool = False, chunksize: int = 200_000) -> dict:
    """Create lookup.npz, LICENSE and a provenance manifest."""
    source, out = Path(source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_records = []
    for relative in DATA_FILES + ("LICENSE",):
        if download:
            download_source(source, relative)
        source_records.append(verify_source(source / relative, relative))
    mean, score_summary = build_dense_lookup(source, chunksize=chunksize)

    lookup_path = out / "lookup.npz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=out, prefix=".lookup.", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            payload = {"field_names": np.asarray(FIELD_NAMES, dtype=str), "mean": mean}
            payload.update({
                f"axis_{index}": np.arange(RANGES[name][0], RANGES[name][1] + 1, dtype=np.int64)
                for index, name in enumerate(FIELD_NAMES)
            })
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        lookup_sha256 = sha256_file(temporary)
        os.replace(temporary, lookup_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    _atomic_copy(source / "LICENSE", out / "LICENSE")
    license_record = verify_source(out / "LICENSE", "LICENSE")
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_directory": str(source),
        "source_files": source_records,
        "field_names": list(FIELD_NAMES),
        "ranges": {name: list(RANGES[name]) for name in FIELD_NAMES},
        "shape": list(SHAPE),
        "score": {"column": "q_final", "dtype": "float32", **score_summary},
        "lookup": {"path": "lookup.npz", "sha256": lookup_sha256, "bytes": lookup_path.stat().st_size},
        "uncertainty": {"published": False, "std_field": "omitted; deterministic lookup returns zero"},
        "license": license_record,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out, prefix=".manifest.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out / "manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="Local official ActMOF checkout or downloaded source directory")
    mode.add_argument("--download", action="store_true", help="Download the pinned public CSVs and license, without cloning")
    parser.add_argument("--out", type=Path, required=True, help="Output cache directory")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV rows per validation chunk (default: 200000)")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    source = args.source if args.source is not None else args.out / "source"
    manifest = prepare(source, args.out, download=args.download, chunksize=args.chunksize)
    print(f"Prepared {manifest['score']['count']:,} conditions: {args.out.resolve() / 'lookup.npz'}")


if __name__ == "__main__":
    main()
