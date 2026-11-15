"""One audited source-only transition for the DQN storage performance release.

This is not a general migration registry. The old revision is fixed below, every
installed Python source must match the release inventory, and only the source
digest may differ in a resumed contract. Runtime, oracle, learning, sampling and
frozen-artifact fields remain exact. The JSON inventory is release data so its
target hashes can include this module without a self-referential hash cycle.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


UPGRADE_ID = "dqn-checkpoint-storage-performance-v1"
BASELINE_REVISION = "ba6bc1cf3dec6e050d44fd278e1e41243f54dbca"
BASELINE_PACKAGE_SHA256 = "d992dc1e785b44ae206433abeb083811c1b480646a5d63724da1cfed19d3e7d4"
BASELINE_EXECUTION_SHA256 = "9310a1952de5716f22fb390d4b01b9418ddcf6a5ce6961615f749cae636ad9b0"
LEGACY_PACKAGE_SHA256 = "0570e95addf4c00909a422be689acee0a035ca5948249339de8ef90096de81d9"
LEGACY_EXECUTION_SHA256 = "8315d282a7c9a4ca04475a5e27bf3794581f80b61b13e2387651e2194b5017a6"
# Historical, unregistered predictor modules remaining in the user's checkout.
# The built-in GlassNet factory does not import them. They may remain byte-exact
# or all be absent after checkout cleanup; no other extra source is admitted.
LEGACY_PREDICTOR_FILES = {
    "rl_matdesign/predictors/composite.py": "87e59d55d57835f2082410fe15ffaa426dad69272961698f36ca4f456cbeef96",
    "rl_matdesign/predictors/dp_property.py": "e995cf4ff6e4cb979206c311b285cded2075698c49c916247cf271b569be6dd9",
    "rl_matdesign/predictors/dp_structure.py": "35a7d921af1304d32debd2cc2ec66865f9719841f84a4057fa2ee7ae4644e346",
    "rl_matdesign/predictors/hea.py": "752c7784a2fed02988aa4dbd0abff27b9882deb1115abd3a18afdbbd7a66f682",
    "rl_matdesign/predictors/perovskite.py": "0e098ee8c0ead5351fc56227a25ac793e16ea4a1626d9da1c6d0f0458076c7ba",
    "rl_matdesign/predictors/sinter_calcine.py": "46f69b8d5163ddc42972fb0546a019eb6449bbcd226a3a80ea4847ac058e6306",
}
# Adding a file here requires review of both its behavior and exact target hash.
# Scientific scoring, reward, domain and algorithm configuration are excluded.
_PERMITTED_FILES = frozenset({
    "rl_matdesign/training.py",
    "rl_matdesign/hpo/controller.py",
    "rl_matdesign/hpo/cli.py",
    "rl_matdesign/utils/candidate_records.py",
    "rl_matdesign/utils/checkpoint_serialization.py",
    "rl_matdesign/utils/timing.py",
    "rl_matdesign/utils/stage_timing.py",
    "rl_matdesign/utils/td_target_cache.py",
    "rl_matdesign/utils/execution_contract.py",
    "rl_matdesign/utils/model_contract.py",
    "rl_matdesign/utils/generation_resume.py",
    "rl_matdesign/utils/source_compatibility.py",
})


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path() -> Path:
    return Path(__file__).with_name("dqn_performance_source.json")


def source_inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def inventory_digests(inventory: dict[str, str]) -> dict[str, str]:
    """Reproduce the existing campaign and execution digest algorithms."""
    package = {path.removeprefix("rl_matdesign/"): value
               for path, value in inventory.items() if path.startswith("rl_matdesign/")}
    package_hash = hashlib.sha256(json.dumps(package, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    execution = hashlib.sha256()
    for path in sorted(inventory, key=Path):
        execution.update(path.encode())
        execution.update(bytes.fromhex(inventory[path]))
    return {"package": package_hash, "execution": execution.hexdigest()}


def approved_performance_upgrade(actual, expected, *, kind: str) -> dict | None:
    """Return evidence only for the exact old -> installed audited release.

    This read-only check deliberately performs no environment or source repair.
    Unknown baselines, reverse upgrades, extra source files, modified release
    files and missing inventory data all fail closed.
    """
    known = {"clean": {"package": BASELINE_PACKAGE_SHA256, "execution": BASELINE_EXECUTION_SHA256},
             "legacy_predictors": {"package": LEGACY_PACKAGE_SHA256, "execution": LEGACY_EXECUTION_SHA256}}
    if kind not in known["clean"] or not isinstance(expected, str):
        return None
    saved_variant = next((name for name, hashes in known.items() if actual == hashes[kind]), None)
    if saved_variant is None:
        return None
    try:
        raw = _manifest_path().read_bytes()
        record = json.loads(raw)
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("schema_version") != 1 or record.get("id") != UPGRADE_ID
                or record.get("baseline_revision") != BASELINE_REVISION):
            return None
        baseline, overrides = record["baseline_files"], record["target_files"]
        if not isinstance(baseline, dict) or not isinstance(overrides, dict):
            return None
        if not overrides or not set(overrides) <= _PERMITTED_FILES:
            return None
        if inventory_digests(baseline) != known["clean"]:
            return None
        if inventory_digests({**baseline, **LEGACY_PREDICTOR_FILES}) != known["legacy_predictors"]:
            return None
        target = {**baseline, **overrides}
        variants = {"clean": target, "legacy_predictors": {**target, **LEGACY_PREDICTOR_FILES}}
        installed = source_inventory(_source_root())
        current_variant = next((name for name, inventory in variants.items()
            if inventory == installed and inventory_digests(inventory)[kind] == expected), None)
        if current_variant is None or expected == actual:
            return None
        return {"id": UPGRADE_ID, "baseline_revision": BASELINE_REVISION,
                "kind": kind, "saved_source_sha256": actual,
                "current_source_sha256": expected,
                "saved_inventory_variant": saved_variant,
                "current_inventory_variant": current_variant,
                "legacy_predictor_files": dict(LEGACY_PREDICTOR_FILES),
                "inventory_sha256": hashlib.sha256(raw).hexdigest(),
                "changed_source_files": sorted(overrides),
                "scope": "audited source-only performance update; all other contract fields unchanged"}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _exact(actual, expected) -> bool:
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False


def _source_only_equal(actual, expected, path: tuple[str, ...]) -> bool:
    if _exact(actual, expected):
        return True
    try:
        changed = copy.deepcopy(actual)
        old, new = changed, expected
        for name in path[:-1]:
            old, new = old[name], new[name]
        saved_digest, current_digest = old[path[-1]], new[path[-1]]
        old[path[-1]] = current_digest
        return (_exact(changed, expected)
                and approved_performance_upgrade(saved_digest, current_digest, kind="execution") is not None)
    except (KeyError, TypeError, IndexError):
        return False


def evaluation_contracts_equal(actual, expected) -> bool:
    return _source_only_equal(actual, expected, ("source_sha256",))


def execution_contracts_equal(actual, expected) -> bool:
    return _source_only_equal(actual, expected, ("evaluation", "source_sha256"))


def generation_contracts_equal(actual, expected) -> bool:
    return _source_only_equal(actual, expected, ("execution", "evaluation", "source_sha256"))
