#!/usr/bin/env python3
"""Read-only diagnosis of the campaign gate, using the installed RL package.

Does not instantiate Campaign/JobRunner, run a predictor, load .pt files, change
fingerprints, or authorize recovery. No sys.path override: the imported package
location is part of the diagnosis. Use the same Python environment as the CLI.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import sys


CONTRACT_KEYS = ("plan", "domain", "software_digest", "assets", "configuration_sources")
_MISSING = object()


def differences(saved, current, path=""):
    """Yield JSON field paths, preserving missing versus null distinctions."""
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(set(saved) | set(current)):
            yield from differences(saved.get(key, _MISSING), current.get(key, _MISSING),
                                   f"{path}.{key}" if path else key)
    elif isinstance(saved, list) and isinstance(current, list):
        for i in range(max(len(saved), len(current))):
            yield from differences(saved[i] if i < len(saved) else _MISSING,
                                   current[i] if i < len(current) else _MISSING, f"{path}[{i}]")
    elif type(saved) is not type(current) or saved != current:
        def preview(value):
            if value is _MISSING:
                return {"missing": True}
            text = json.dumps(value, ensure_ascii=False)
            return value if len(text) <= 240 else text[:240] + "..."
        yield {"path": path, "saved": preview(saved), "current": preview(current)}


def inspect_campaign(out):
    report = {"out": str(Path(out).resolve()), "read_only": True,
              "python": sys.executable, "cli_path": shutil.which("rl-matdesign"),
              "campaign_contract_accepted": False, "checkpoint_state_checked": False,
              "errors": [], "differences": []}
    try:
        from rl_matdesign.hpo import artifacts
        from rl_matdesign.hpo.domain import design_space_info
        from rl_matdesign.hpo.spec import load_plan
        from rl_matdesign.utils.hpo_resume_compatibility import approved_upgrade
        report["package"] = {"path": str(Path(artifacts.__file__).resolve().parents[1]),
                             "digest": artifacts.package_digest()}
        saved = artifacts.read_json(Path(out) / "manifest.json")
        original = {key: saved.get(key) for key in CONTRACT_KEYS}
        report["manifest_integrity"] = artifacts.digest(original) == saved.get("contract_digest")
        report["saved_software_digest"] = original["software_digest"]
        same_software = original["software_digest"] == report["package"]["digest"]
        upgrade = approved_upgrade(original["software_digest"], report["package"]["digest"])
        report["software_identical"] = same_software
        report["software_upgrade_registered"] = upgrade is not None
        report["software_upgrade_id"] = upgrade["id"] if upgrade else None
        # Report raw source changes even when the edited YAML no longer parses.
        sources = {}
        report["configuration_sources"] = sources
        for name, row in (original["configuration_sources"] or {}).items():
            path = row["path"]
            entry = {"path": path, "saved_sha256": row.get("sha256")}
            sources[name] = entry
            try:
                entry["current_sha256"] = artifacts.file_digest(path)
                entry["unchanged"] = entry["current_sha256"] == entry["saved_sha256"]
            except OSError as exc:
                entry.update(unchanged=False, error=str(exc))
        plan = load_plan(saved["plan"]["source_path"], method=saved["plan"]["method"])
        source_paths = {"hpo_yaml": plan["source_path"], "base_yaml": plan["base_config"]}
        if not plan["parameter_profile"].startswith("builtin:"):
            source_paths["profile_yaml"] = plan["parameter_profile"]
        current = {"plan": plan, "domain": design_space_info(plan),
                   "software_digest": report["package"]["digest"],
                   "assets": artifacts.asset_fingerprints(plan),
                   "configuration_sources": {
                       name: {"path": path, "sha256": artifacts.file_digest(path)}
                       for name, path in source_paths.items()}}
        report["changed_contract_sections"] = [key for key in CONTRACT_KEYS if original[key] != current[key]]
        diff = list(differences(original, current))
        report["difference_count"] = len(diff)
        report["differences"] = diff[:50]
        report["differences_truncated"] = len(diff) > 50
        nonsoftware_equal = all(original[key] == current[key]
                                for key in CONTRACT_KEYS if key != "software_digest")
        # Mirror the actual Campaign gate; this is NOT checkpoint readiness.
        report["campaign_contract_accepted"] = bool(report["manifest_integrity"] and (
            artifacts.digest(current) == saved.get("contract_digest")
            or (upgrade is not None and nonsoftware_equal)))
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    report["next_step"] = (
        "Campaign gate matches. Run hpo restart --check-only to check actual job states."
        if report["campaign_contract_accepted"] else
        "Share this diagnostic output. Preserve original YAML/assets/manifest/checkpoints; "
        "do not edit hashes or bypass the compatibility check.")
    return report


def main(argv=None):
    # A diagnostic must not create package bytecode caches on a full disk.
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing HPO output directory")
    args = parser.parse_args(argv)
    # Keep stdout machine-readable, including legacy import/config messages.
    with redirect_stdout(sys.stderr):
        report = inspect_campaign(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["campaign_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
