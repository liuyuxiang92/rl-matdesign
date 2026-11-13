#!/usr/bin/env python3
"""Prune superseded numbered checkpoints from a stopped HPO campaign.

Standalone, standard-library-only tool: it does not import rl_matdesign, load
pickle/PT files, change campaign configuration, or rewrite result fingerprints.
Stop the HPO controller AND all workers before both inspection and deletion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys

NUMBERED = re.compile(r"checkpoint-(ep|iter)([0-9]+)\.pt\Z")
MENTION = re.compile(r"(?<![A-Za-z0-9_-])(checkpoint-(?:ep|iter)[0-9]+\.pt)(?![A-Za-z0-9_.-])")
JOB_ID = re.compile(r"[0-9a-f]{24}\Z")


class UnsafeCampaign(RuntimeError):
    """Inspection cannot establish that deletion is safe."""


def identity(path: Path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def require_directory(path: Path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise UnsafeCampaign(f"Not a real directory (symlinks are refused): {path}")
    return identity(path)[:3]


def read_object(path: Path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise UnsafeCampaign(f"Metadata is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise UnsafeCampaign(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise UnsafeCampaign(f"Expected a JSON object: {path}")
    return value


def metadata_snapshot(root: Path):
    """Protect any numbered filename mentioned in JSON/YAML, conservatively.

    Filename-only protection deliberately applies across all jobs. This also
    covers relative paths, dictionary keys, YAML fields and old/custom metadata
    formats without needing PyYAML. No symlink directories are traversed.
    """
    signatures, mentions = {}, set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            if (base / name).is_symlink():
                raise UnsafeCampaign(f"Symlink directory in campaign; inspection refused: {base / name}")
        for name in files:
            if Path(name).suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            path = base / name
            before = identity(path)
            if not stat.S_ISREG(before[2]):
                raise UnsafeCampaign(f"Metadata is not a regular file: {path}")
            content = path.read_text(encoding="utf-8")
            if identity(path) != before:
                raise UnsafeCampaign(f"Metadata changed during inspection: {path}")
            signatures[str(path)] = before
            mentions.update(MENTION.findall(content))
    return signatures, mentions


def checkpoint_pointer(directory: Path):
    alias = directory / "checkpoint.pt"
    signature = identity(alias)
    mode = signature[2]
    if stat.S_ISREG(mode):
        return {"alias": signature, "link": None, "target": None, "index": None}
    if not stat.S_ISLNK(mode):
        raise UnsafeCampaign(f"checkpoint.pt is neither a regular file nor a safe symlink: {alias}")
    link = os.readlink(alias)
    match = NUMBERED.fullmatch(link)
    # The existing writer creates a basename-only relative link in this job.
    if not match:
        raise UnsafeCampaign(f"Unusual or escaping checkpoint.pt symlink: {alias} -> {link}")
    target = identity(directory / link)
    if not stat.S_ISREG(target[2]):
        raise UnsafeCampaign(f"checkpoint.pt target is not a regular file: {directory / link}")
    return {"alias": signature, "link": link, "target": target,
            "index": (match[1], int(match[2]))}


def plan_pruning(campaign):
    # Do not resolve away an explicitly supplied symlink before checking it.
    root = Path(os.path.abspath(os.fspath(campaign)))
    root_id = require_directory(root)
    manifest = read_object(root / "manifest.json")
    if manifest.get("protocol") not in {"hpo_v1", "hpo_v2"} or not manifest.get("contract_digest"):
        raise UnsafeCampaign("Expected an HPO manifest with protocol and contract_digest")
    jobs = root / "jobs"
    signatures, mentions = metadata_snapshot(root)
    plan = {"root": root, "root_identity": root_id, "jobs_identity": None,
            "metadata": signatures, "candidates": [], "skipped": [], "kept_numbered": 0}
    if not jobs.exists():
        return plan
    plan["jobs_identity"] = require_directory(jobs)
    for directory in sorted(jobs.iterdir()):
        if not JOB_ID.fullmatch(directory.name):
            plan["skipped"].append(f"Unrecognized job directory: {directory}")
            continue
        try:
            job_identity = require_directory(directory)
            job = read_object(directory / "job.json")
            if job.get("phase") != "train":
                continue
            histories = sorted(p for p in directory.iterdir() if NUMBERED.fullmatch(p.name))
            if not histories:
                continue
            pointer = checkpoint_pointer(directory)
        except (OSError, UnsafeCampaign) as exc:
            plan["skipped"].append(f"Kept job {directory.name}: {exc}")
            continue
        for path in histories:
            before = identity(path)
            match = NUMBERED.fullmatch(path.name)
            index = pointer["index"]
            older = index is None or (match[1] == index[0] and int(match[2]) < index[1])
            if not stat.S_ISREG(before[2]) or not older or path.name in mentions:
                plan["kept_numbered"] += 1
                continue
            plan["candidates"].append({"path": path, "identity": before,
                                       "job_identity": job_identity, "pointer": pointer,
                                       "bytes": before[3]})
    return plan


def apply_pruning(plan):
    """Delete only inspected files, after rechecking pointers and identities.

    This is a stopped-campaign operation, not synchronization with live workers.
    A detected change aborts further deletion; files already deleted stay deleted.
    """
    root = plan["root"]
    if require_directory(root) != plan["root_identity"]:
        raise UnsafeCampaign("Campaign directory changed after inspection")
    if plan["jobs_identity"] is not None and require_directory(root / "jobs") != plan["jobs_identity"]:
        raise UnsafeCampaign("Jobs directory changed after inspection")
    current, _ = metadata_snapshot(root)
    if current != plan["metadata"]:
        raise UnsafeCampaign("Campaign metadata changed after inspection; stop all workers and inspect again")
    # Validate the entire selected set before the first irreversible deletion.
    for item in plan["candidates"]:
        _recheck(item)
    removed, size = 0, 0
    for item in plan["candidates"]:
        _recheck(item)
        item["path"].unlink()
        removed += 1
        size += item["bytes"]
    return {"removed_files": removed, "removed_bytes": size}


def _recheck(item):
    path = item["path"]
    if require_directory(path.parent) != item["job_identity"]:
        raise UnsafeCampaign(f"Job directory changed after inspection: {path.parent}")
    if checkpoint_pointer(path.parent) != item["pointer"]:
        raise UnsafeCampaign(f"Latest checkpoint changed after inspection: {path.parent}")
    if identity(path) != item["identity"]:
        raise UnsafeCampaign(f"Checkpoint file changed after inspection: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing HPO output directory; stop its controller and all workers first")
    parser.add_argument("--apply", action="store_true", help="Delete inspected superseded files (default: dry-run only)")
    args = parser.parse_args(argv)
    try:
        plan = plan_pruning(args.out)
        candidates = plan["candidates"]
        print("APPLY" if args.apply else "DRY RUN — no files deleted")
        print(f"Campaign: {plan['root']}")
        print(f"Superseded files: {len(candidates)}; bytes: {sum(c['bytes'] for c in candidates)}")
        print(f"Protected numbered files: {plan['kept_numbered']}")
        for reason in plan["skipped"]:
            print(f"SKIP: {reason}")
        if args.apply:
            result = apply_pruning(plan)
            print(f"Removed {result['removed_files']} files ({result['removed_bytes']} bytes).")
        else:
            print("Stop the HPO controller AND all workers, review this summary, then rerun with --apply.")
        print("Latest checkpoints, final models, scalers, ledgers and result metadata are preserved.")
        return 0
    except (OSError, UnicodeError, UnsafeCampaign) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
