"""Audited full-state continuation into the DQN performance v2 protocol.

This is not a general migration registry. Two exact historical releases and
their recorded legacy-file variants may continue into one exact v2 release.
Runtime, oracle, configured learning/sampling values and frozen-artifact fields
remain exact. The batched bootstrap implementation can change floating-point
trajectories, and HPO validation/termination behavior changes explicitly; this
is an approved protocol transition, not a claim of bitwise continuation.

Inventories are release data so target hashes can include this module without
a self-referential hash cycle. Existing v1 campaign receipts are not rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


UPGRADE_ID = "dqn-batched-targets-persistence-v2"
BASELINE_REVISION = "ba6bc1cf3dec6e050d44fd278e1e41243f54dbca"
BASELINE_PACKAGE_SHA256 = "d992dc1e785b44ae206433abeb083811c1b480646a5d63724da1cfed19d3e7d4"
BASELINE_EXECUTION_SHA256 = "9310a1952de5716f22fb390d4b01b9418ddcf6a5ce6961615f749cae636ad9b0"
LEGACY_PACKAGE_SHA256 = "0570e95addf4c00909a422be689acee0a035ca5948249339de8ef90096de81d9"
LEGACY_EXECUTION_SHA256 = "8315d282a7c9a4ca04475a5e27bf3794581f80b61b13e2387651e2194b5017a6"
V1_REVISION = "b947208aa196eb08500722f95005e333b4a8051e"
V1_PACKAGE_SHA256 = "4db450af563d86684ec83a349f598176cbd70997d89993dbfaecb0cffebdf0a3"
V1_EXECUTION_SHA256 = "5e47246afb8325373a7189407fedfe30981611332d2cfebab4adc0e9eb126162"
V1_LEGACY_PACKAGE_SHA256 = "c729cb22bdcb0f967f3e9a16a8844f0d6671a775c636adc18bb7561f63e1266f"
V1_LEGACY_EXECUTION_SHA256 = "90effe1f21dc15f7ff5ad2389f7c75d094d587dd247d7a1093fc21e8cd05d9cb"
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
# Scientific scoring, reward, domain and user configuration are excluded.
# The listed training/validation/termination implementations are the explicitly
# approved v2 protocol changes, rather than numerically invisible optimizations.
_PERMITTED_FILES = frozenset({
    "rl_matdesign/training.py",
    "rl_matdesign/experiment.py",
    "rl_matdesign/hpo/controller.py",
    "rl_matdesign/hpo/artifacts.py",
    "rl_matdesign/hpo/cli.py",
    "rl_matdesign/hpo/restart.py",
    "rl_matdesign/utils/candidate_records.py",
    "rl_matdesign/utils/candidate_budget.py",
    "rl_matdesign/utils/episode_batch.py",
    "rl_matdesign/utils/checkpoint_serialization.py",
    "rl_matdesign/utils/checkpoint_store.py",
    "rl_matdesign/utils/timing.py",
    "rl_matdesign/utils/stage_timing.py",
    "rl_matdesign/utils/td_target_cache.py",
    "rl_matdesign/utils/td_targets.py",
    "rl_matdesign/utils/execution_contract.py",
    "rl_matdesign/utils/model_contract.py",
    "rl_matdesign/utils/generation_resume.py",
    "rl_matdesign/utils/source_compatibility.py",
})


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path() -> Path:
    return Path(__file__).with_name("dqn_performance_v2_source.json")


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
    known = {
        BASELINE_REVISION: {
            "clean": {"package": BASELINE_PACKAGE_SHA256, "execution": BASELINE_EXECUTION_SHA256},
            "legacy_predictors": {"package": LEGACY_PACKAGE_SHA256, "execution": LEGACY_EXECUTION_SHA256}},
        V1_REVISION: {
            "clean": {"package": V1_PACKAGE_SHA256, "execution": V1_EXECUTION_SHA256},
            "legacy_predictors": {"package": V1_LEGACY_PACKAGE_SHA256, "execution": V1_LEGACY_EXECUTION_SHA256}},
    }
    if kind not in ("package", "execution") or not isinstance(expected, str):
        return None
    saved = next(((revision, variant) for revision, variants in known.items()
                  for variant, hashes in variants.items() if actual == hashes[kind]), None)
    if saved is None:
        return None
    saved_revision, saved_variant = saved
    try:
        raw = _manifest_path().read_bytes()
        record = json.loads(raw)
        if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                or record.get("schema_version") != 2 or record.get("id") != UPGRADE_ID
                or record.get("baseline_revision") != BASELINE_REVISION
                or record.get("previous_release_revision") != V1_REVISION):
            return None
        baseline, previous, overrides = (record[name] for name in
            ("baseline_files", "previous_release_files", "target_files"))
        if any(not isinstance(value, dict) for value in (baseline, previous, overrides)):
            return None
        if not overrides or not set(overrides) <= _PERMITTED_FILES:
            return None
        previous_release = {**baseline, **previous}
        for revision, inventory in ((BASELINE_REVISION, baseline), (V1_REVISION, previous_release)):
            if inventory_digests(inventory) != known[revision]["clean"]:
                return None
            if inventory_digests({**inventory, **LEGACY_PREDICTOR_FILES}) != known[revision]["legacy_predictors"]:
                return None
        target = {**previous_release, **overrides}
        variants = {"clean": target, "legacy_predictors": {**target, **LEGACY_PREDICTOR_FILES}}
        installed = source_inventory(_source_root())
        current_variant = next((name for name, inventory in variants.items()
            if inventory == installed and inventory_digests(inventory)[kind] == expected), None)
        if current_variant is None or expected == actual:
            return None
        saved_inventory = baseline if saved_revision == BASELINE_REVISION else previous_release
        if saved_variant == "legacy_predictors":
            saved_inventory = {**saved_inventory, **LEGACY_PREDICTOR_FILES}
        changed_files = sorted(path for path in set(saved_inventory) | set(installed)
                               if saved_inventory.get(path) != installed.get(path))
        return {"id": UPGRADE_ID, "baseline_revision": saved_revision,
                "kind": kind, "saved_source_sha256": actual,
                "current_source_sha256": expected,
                "saved_inventory_variant": saved_variant,
                "current_inventory_variant": current_variant,
                "legacy_predictor_files": dict(LEGACY_PREDICTOR_FILES),
                "inventory_sha256": hashlib.sha256(raw).hexdigest(),
                "changed_source_files": changed_files,
                "migration_kind": "approved_code_and_protocol_update",
                "bitwise_continuation": False,
                "checkpoint_handoff": "native full-state restart; preserve saved network, optimizer, RNG and paid-candidate accounting",
                "protocol_changes": {
                    "bootstrap_targets": "batch next-state action evaluations with within-minibatch deduplication; floating-point trajectories may differ",
                    "budget_validation": "validate only the top three configurations at each candidate budget",
                    "efficiency_stop": "stop earlier when the strict attempts-per-unique limit can no longer be met",
                    "persistence": "reduce persistence work while preserving synchronized full-state checkpoints and paid observations",
                },
                "scope": "audited code/protocol transition; all non-source contract fields unchanged; bitwise trajectory identity is not guaranteed"}
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
