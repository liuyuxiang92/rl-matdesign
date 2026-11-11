"""Durable HPO state and fingerprints; no predictor or tensor imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def package_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): file_digest(p)
                   for p in sorted(root.rglob("*.py"))})


def parameter_id(parameters: dict) -> str:
    return digest(parameters)[:16]


def asset_fingerprints(plan: dict) -> list[dict]:
    from ..utils.provenance import referenced_files
    records = referenced_files(plan["base_config_data"], base_dir=plan["base_config_dir"])
    result = []
    for row in records:
        path = row.get("path")
        result.append({"field": row.get("field"), "path": path,
                       "status": row.get("status"),
                       "sha256": file_digest(path) if path and Path(path).is_file() else None})
    return result


def checkpoint_files(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    return {name: str(directory / name) for name in
            ("qnet.pt", "policy.pt", "value_net.pt", "std_scaler.bin", "checkpoint.pt")
            if (directory / name).is_file()}
