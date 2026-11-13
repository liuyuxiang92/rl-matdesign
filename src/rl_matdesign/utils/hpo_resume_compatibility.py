"""Narrow, release-reviewed source upgrades for HPO checkpoint storage fixes.

The registry contains exact old/new fingerprints, not version ranges or a flag
that disables provenance checks. Only the source field at a known contract
location can differ; every scientific/runtime field must still match. Registry
JSON is packaged separately so its target hashes do not refer to themselves.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re


_REGISTRY_PATH = Path(__file__).with_name("hpo_resume_compatibility.json")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _valid_digest(value) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _same(actual, expected) -> bool:
    """Compare serialized evidence without bool/int coercion or NaN equality."""
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def approved_upgrade(old_digest, new_digest, kind="package") -> dict | None:
    """Return the exact reviewed, forward-only upgrade, or no exception.

    Missing target hashes deliberately disable an entry while a release is
    being prepared. Both package and worker hashes must be finalized before
    either kind of compatibility can be granted.
    """
    if kind not in {"package", "source"}:
        raise ValueError(f"Unknown HPO software fingerprint kind: {kind}")
    if old_digest == new_digest or not all(map(_valid_digest, (old_digest, new_digest))):
        return None
    registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if (not isinstance(registry, dict) or registry.get("schema_version") != 1
            or not isinstance(registry.get("migrations"), list)):
        raise ValueError("Invalid HPO source compatibility registry")
    for entry in registry["migrations"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid HPO source compatibility registry entry")
        old, new = entry.get("from"), entry.get("to")
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        if not all(_valid_digest(value.get(key))
                   for value in (old, new) for key in ("package", "source")):
            continue
        if old[kind] == old_digest and new[kind] == new_digest:
            if not isinstance(entry.get("id"), str) or not entry["id"]:
                raise ValueError("HPO source compatibility entry needs a migration ID")
            return copy.deepcopy(entry)
    return None


def evaluation_contracts_equal(actual, expected, *, allow_source_upgrade=False) -> bool:
    if _same(actual, expected):
        return True
    if not allow_source_upgrade or not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if approved_upgrade(actual.get("source_sha256"), expected.get("source_sha256"), kind="source") is None:
        return False
    return _same({key: value for key, value in actual.items() if key != "source_sha256"},
                 {key: value for key, value in expected.items() if key != "source_sha256"})


def execution_contracts_equal(actual, expected, *, allow_source_upgrade=False) -> bool:
    if _same(actual, expected):
        return True
    if not allow_source_upgrade or not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if set(actual) != set(expected) or "evaluation" not in actual:
        return False
    return (evaluation_contracts_equal(actual["evaluation"], expected["evaluation"],
                                       allow_source_upgrade=True)
            and _same({key: value for key, value in actual.items() if key != "evaluation"},
                      {key: value for key, value in expected.items() if key != "evaluation"}))


def generation_contracts_equal(actual, expected, *, allow_source_upgrade=False) -> bool:
    if _same(actual, expected):
        return True
    if not allow_source_upgrade or not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if set(actual) != set(expected) or "execution" not in actual:
        return False
    return (execution_contracts_equal(actual["execution"], expected["execution"],
                                      allow_source_upgrade=True)
            and _same({key: value for key, value in actual.items() if key != "execution"},
                      {key: value for key, value in expected.items() if key != "execution"}))
