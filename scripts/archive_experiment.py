#!/usr/bin/env python3
"""Create or verify an experiment bundle without asserting historical provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_matdesign.utils.provenance import create_bundle, verify_bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser("create", help="Copy explicit runs, assets and current tracked source")
    archive.add_argument("--run-dir", action="append", required=True)
    archive.add_argument("--config", action="append", default=[])
    archive.add_argument("--asset", action="append", default=[], help="Explicit input/model/environment file; repeat as needed")
    archive.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    archive.add_argument("--out", required=True, help="New output directory outside the selected runs")
    verify = commands.add_parser("verify", help="Check checksums and report unresolved evidence")
    verify.add_argument("bundle")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            path = create_bundle(args.out, args.run_dir, configs=args.config,
                                 assets=args.asset, repo_dir=args.repo_dir)
            report = verify_bundle(path)
            report["bundle"] = str(path)
        else:
            report = verify_bundle(args.bundle)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["integrity_ok"] and report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
