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
import hashlib
import re
import subprocess
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


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20).stdout


def _tree_inventory(repo, revision, relative_package):
    """Hash committed blobs, including files ignored by archive export rules."""
    rows = _git(repo, "ls-tree", "-r", "-z", revision, "--", relative_package)
    result = {}
    prefix = relative_package.rstrip("/") + "/"
    for row in rows.split(b"\0"):
        if not row:
            continue
        meta, raw_path = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = raw_path.decode()
        if kind != "blob" or not name.endswith(".py") or not name.startswith(prefix):
            continue
        if mode == "120000":
            raise ValueError("Committed Python symlink needs manual inspection: " + name)
        result[name[len(prefix):]] = hashlib.sha256(_git(repo, "cat-file", "blob", oid)).hexdigest()
    return result


def _inventory_digest(inventory):
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_evidence(package, out, saved_digest):
    """Localize a non-release fingerprint; this never authorizes an upgrade."""
    result = {"available": False, "historical_sample": [], "historical_sample_limit": 12}
    try:
        root = Path(package).resolve()
        inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob("*.py"))}
        result["actual_package_digest"] = _inventory_digest(inventory)
        result["python_file_count"] = len(inventory)
        repo = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
        relative = root.relative_to(repo).as_posix()
        head = _git(repo, "rev-parse", "HEAD").decode().strip()
        committed = _tree_inventory(repo, "HEAD", relative)
        extra = {name: value for name, value in inventory.items() if name not in committed}
        result.update(available=True, repository=str(repo), head_commit=head,
                      head_package_digest=_inventory_digest(committed),
                      modified_python_files=[{"path": name, "head_sha256": committed[name],
                                              "current_sha256": inventory[name]}
                                             for name in sorted(set(inventory) & set(committed))
                                             if inventory[name] != committed[name]],
                      missing_python_files=sorted(set(committed) - set(inventory)),
                      extra_python_files=[{"path": name, "sha256": value}
                                          for name, value in sorted(extra.items())])
        # Old invocations recorded their commit and dirty patch even if campaign
        # manifests only kept one aggregate source hash. Inspect a bounded sample.
        seen, trees = set(), {}
        manifests = Path(out).glob("jobs/*/provenance/*/manifest.json")
        for index, path in enumerate(manifests):
            if index >= result["historical_sample_limit"]:
                result["historical_sample_truncated"] = True
                break
            document = json.loads(path.read_text())
            if not isinstance(document, dict):
                raise ValueError(f"Invalid provenance object: {path}")
            git = document.get("git", {})
            if not isinstance(git, dict):
                raise ValueError(f"Invalid provenance git object: {path}")
            key = (git.get("commit"), git.get("patch_sha256"))
            if key in seen:
                continue
            seen.add(key)
            row = {"manifest": str(path), "commit": git.get("commit"),
                   "dirty_tracked": git.get("dirty_tracked"),
                   "patch_sha256": git.get("patch_sha256"),
                   "untracked_file_count": git.get("untracked_file_count")}
            result["historical_sample"].append(row)
            commit = git.get("commit")
            # Only pass actual recorded object IDs to Git, not arbitrary refs.
            if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
                continue
            try:
                if commit not in trees:
                    trees[commit] = _tree_inventory(repo, commit, relative)
                tree = trees[commit]
                row["committed_package_digest"] = _inventory_digest(tree)
                row["committed_tree_matches_saved_digest"] = row["committed_package_digest"] == saved_digest
                # This is a test of one explicit hypothesis, not proof that
                # untracked files are harmless or that training can resume.
                combined = {**tree, **{k: v for k, v in extra.items() if k not in tree}}
                row["committed_tree_plus_current_extra_files_matches_saved_digest"] = (
                    _inventory_digest(combined) == saved_digest)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row["inspection_error"] = str(exc)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["inspection_error"] = str(exc)
    return result


def inspect_campaign(out):
    report = {"out": str(Path(out).resolve()), "read_only": True,
              "python": sys.executable, "cli_path": shutil.which("rl-matdesign"),
              "campaign_contract_accepted": False, "checkpoint_state_checked": False,
              "errors": [], "differences": []}
    try:
        from rl_matdesign.hpo import artifacts
        from rl_matdesign.hpo.domain import design_space_info
        from rl_matdesign.hpo.spec import load_plan
        from rl_matdesign.utils.hpo_resume_compatibility import approved_upgrade, normalize_legacy_serial_contract
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
        if not same_software and upgrade is None:
            report["source_evidence"] = source_evidence(
                report["package"]["path"], out, original["software_digest"])
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
        comparable, plan_migrations = normalize_legacy_serial_contract(original, current)
        report["normalizable_plan_defaults"] = plan_migrations
        comparable["software_digest"] = current["software_digest"]
        # Mirror the actual Campaign gate; this is NOT checkpoint readiness.
        report["campaign_contract_accepted"] = bool(report["manifest_integrity"] and (
            artifacts.digest(current) == saved.get("contract_digest")
            or (upgrade is not None and artifacts.digest(comparable) == artifacts.digest(current))))
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
