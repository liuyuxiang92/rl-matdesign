"""Append-only run evidence and portable, checksum-verified experiment bundles.

This module uses only the standard library. Capturing evidence does not seed or
consume Python, NumPy, or Torch random streams, import Torch, or initialize CUDA.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import uuid

SCHEMA_VERSION = 1
DETERMINISM_ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_SCHEDULE", "OMP_PROC_BIND", "OMP_PLACES",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_CBWR", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "KMP_AFFINITY", "KMP_DETERMINISTIC_REDUCTION", "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_DEFAULT_MATH", "CUDA_DEVICE_MAX_CONNECTIONS",
)
ENVIRONMENT_RECORD_NAMES = frozenset({
    "environment.yml", "environment.yaml", "conda-explicit.txt", "pip-freeze.txt",
    "conda-lock.yml", "conda-lock.yaml", "requirements-lock.txt", "uv.lock", "poetry.lock",
})
# Only fields whose contract denotes a file. Plugins with other fields must pass
# input_files / --asset explicitly. Never infer paths from arbitrary strings.
FILE_KEYS = frozenset({
    "model", "models", "dp_model", "dp_models", "model_file", "model_files",
    "model_path", "checkpoint", "checkpoint_path", "weights", "base_poscar",
    "poscar", "poscar_template", "scaffold_poscar", "geo_opt_model",
    "train_csv", "data_csv", "dataset_path", "feature_schema", "scaler_path",
    "load_qnet", "load_scaler", "load_policy", "load_value", "config_file",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=60).stdout


def _repository(repo_dir: str | Path | None) -> Path:
    # Installed packages without a Git checkout are valid; capture reports that.
    return Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[3]


def _git_evidence(repo: Path, destination: Path) -> dict[str, Any]:
    try:
        root = Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel")).strip())
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        patch = _git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
        tracked = _git(root, "status", "--porcelain=v1", "--untracked-files=no").decode()
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        (destination / "git.patch").write_bytes(patch)
        return {
            "available": True, "root": str(root), "commit": commit,
            "dirty_tracked": bool(tracked), "tracked_status": tracked,
            "patch_file": "git.patch", "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "untracked_file_count": sum(bool(p) for p in untracked),
            "untracked_policy": "not captured automatically; declare assets explicitly",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}


def _runtime() -> dict[str, Any]:
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    result = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": dict(sorted(packages.items())),
        "determinism_environment": {key: os.environ[key] for key in DETERMINISM_ENV_KEYS if key in os.environ},
        "environment_collection": "determinism whitelist only; all other variables omitted",
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch"] = {
            "version": str(torch.__version__),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "built_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_initialized": torch.cuda.is_initialized(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            query = subprocess.run(
                [nvidia_smi, "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3, check=True,
            )
            result["nvidia_smi"] = {"status": "available", "columns": "index,name,uuid,driver_version",
                                    "output": query.stdout[:16384].decode(errors="replace"),
                                    "truncated": len(query.stdout) > 16384}
        except (OSError, subprocess.SubprocessError) as exc:
            result["nvidia_smi"] = {"status": "unavailable", "reason": type(exc).__name__}
    else:
        result["nvidia_smi"] = {"status": "not_installed"}
    return result


def referenced_files(config: Mapping[str, Any], *, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Resolve declared files using their owners, without importing backends.

    Most built-ins open paths from runtime cwd. MGTransformer instead runs its
    server under mgt_repo. Unknown plugins may implement other conventions:
    their relative file values are recorded unresolved, never guessed as cwd.
    Explicit structure sources use their lightweight runtime resolver, including
    content hashes for templates and relaxation models, so provenance cannot
    interpret source-owned paths differently from evaluation.
    """
    base = Path.cwd() if base_dir is None else Path(base_dir).absolute()
    found: list[dict[str, Any]] = []
    known_predictors = {None, "structure_score", "rf_magpie", "ooh", "dummy",
                        "dp_energy", "dp_property", "dp_classification"}
    mgt_predictors = {"mgtransformer", "rl_matdesign.predictors.mgtransformer:MGTransformerPredictor"}

    def visit(value: Any, location: str, owner: str | None = None,
              owner_config: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, Mapping):
            if not location and isinstance(value.get("properties"), list) and value["properties"]:
                # build_reward routes a nonempty properties list to the shared
                # engine even if a legacy top-level predictor name is present.
                owner = "structure_score"
                owner_config = value
            elif "predictor" in value:
                owner = value["predictor"]
                owner_config = value
            for key, child in value.items():
                label = f"{location}.{key}" if location else str(key)
                if not location and key == "structure_sources":
                    from ..structure_sources import StructureSourceSpec, resolve_source_assets
                    if not isinstance(child, Mapping):
                        raise ValueError("structure_sources must be a mapping of named definitions")
                    for source_id, definition in child.items():
                        source = StructureSourceSpec(source_id, definition)
                        _, references = resolve_source_assets(source, base_dir=base)
                        for reference in references:
                            record = {**reference,
                                      "field": f"{label}.{source_id}.{reference['field']}",
                                      "source": source_id, "builder": source.builder,
                                      "source_asset_resolution": reference["status"]}
                            if reference["path"] is None:
                                record["status"] = "unresolved_path"
                            else:
                                record["status"] = {"resolved": "present", "missing": "missing_or_not_file",
                                                    "unresolved": "unreadable"}[reference["status"]]
                                record["path_resolution"] = ("declared_absolute_path"
                                    if Path(reference["declared"]).is_absolute() else "runtime_cwd")
                            found.append(record)
                    # The general walker must not reinterpret an external
                    # builder's unresolved paths or duplicate nested templates.
                    continue
                is_mgt_file = owner in mgt_predictors and key in {"model", "models", "config"}
                if key in FILE_KEYS or is_mgt_file:
                    for item in child if isinstance(child, (list, tuple)) else [child]:
                        if not isinstance(item, (str, os.PathLike)) or not str(item):
                            continue
                        path = Path(item)
                        record = {"field": label, "declared": str(item), "predictor": owner}
                        if path.is_absolute():
                            record.update(path=str(path), path_resolution="declared_absolute_path")
                        elif is_mgt_file:
                            mgt_repo = (owner_config or value).get("mgt_repo")
                            if mgt_repo:
                                root = (base / str(mgt_repo)).absolute()
                                record.update(path=str((root / path).absolute()),
                                              path_resolution="mgtransformer_repo")
                            else:
                                record.update(path=None, status="unresolved_path",
                                              reason="MGTransformer relative model/config requires mgt_repo")
                        elif owner not in known_predictors and owner not in mgt_predictors:
                            record.update(path=None, status="unresolved_path",
                                          reason="Unknown plugin owns relative path resolution; declare resolved input_files/--asset")
                        else:
                            record.update(path=str((base / path).absolute()), path_resolution="runtime_cwd")
                        found.append(record)
                visit(child, label, owner, owner_config)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]", owner, owner_config)

    visit(config, "")
    return found


def _file_record(path: str | Path, **extra: Any) -> dict[str, Any]:
    p = Path(path).expanduser().absolute()
    record = {"path": str(p), **extra}
    try:
        if not p.is_file():
            return {**record, "status": "missing_or_not_file"}
        return {**record, "status": "present", "size": p.stat().st_size,
                "sha256": sha256_file(p)}
    except OSError as exc:
        return {**record, "status": "unreadable", "reason": str(exc)}


def write_run_manifest(
    output_dir: str | Path, config: Mapping[str, Any], *,
    argv: Sequence[str] | None = None, config_path: str | Path | None = None,
    seeds: Mapping[str, Any] | None = None, phase: str = "run",
    input_files: Sequence[str | Path] = (), repo_dir: str | Path | None = None,
    config_sources: Mapping[str, Any] | None = None,
    runtime_args: Mapping[str, Any] | None = None,
) -> Path:
    """Capture a NEW invocation; never overwrite earlier training/resume evidence.

    Call before training and before overwriting prior run outputs. This captures
    observations at invocation time, not a claim that a historical run used this
    checkout. The manifest records asset hashes but does not copy large assets.
    """
    now = datetime.now(timezone.utc)
    evidence = Path(output_dir) / "provenance" / (now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    references = referenced_files(config)
    references.extend({"field": "explicit_input", "path": str(Path(p).expanduser().absolute())}
                      for p in input_files)
    if config_path is not None:
        references.append({"field": "config_path", "path": str(Path(config_path).expanduser().absolute())})
        config_file = Path(config_path)
        if config_file.is_file():
            shutil.copyfile(config_file, evidence / "input_config.yaml")
    file_cache: dict[str, dict[str, Any]] = {}
    files = []
    for ref in references:
        if ref.get("path") is None:
            files.append(dict(ref))
            continue
        if ref["path"] not in file_cache:
            if ref.get("source_asset_resolution") == "resolved":
                # The shared source resolver verified these bytes in this
                # invocation; avoid re-reading a potentially large MD model.
                file_cache[ref["path"]] = {key: ref[key] for key in ("path", "status", "size", "sha256")}
            else:
                file_cache[ref["path"]] = _file_record(ref["path"])
        files.append({**file_cache[ref["path"]], **{k: v for k, v in ref.items()
                     if k not in {"path", "status", "size", "sha256"}}})
    manifest = {
        "schema_version": SCHEMA_VERSION, "evidence_scope": "captured_at_invocation",
        "created_utc": now.isoformat(), "phase": phase, "cwd": str(Path.cwd()),
        "argv": list(sys.argv if argv is None else argv), "config": dict(config),
        "config_sources": dict(config_sources or {}), "runtime_args": dict(runtime_args or {}),
        "seeds": dict(seeds or {}), "git": _git_evidence(_repository(repo_dir), evidence),
        "runtime": _runtime(), "input_files": files,
        "missing_inputs": [row for row in files if row["status"] != "present"],
        "limitations": ["Implicit plugin/downloaded assets require explicit input_files or --asset.",
                        "No hardware-independent bitwise reproducibility claim.",
                        "Package versions are evidence, not a complete environment lockfile."],
    }
    path = evidence / "manifest.json"
    _json(path, manifest)
    return path


def _copy_tree(source: Path, destination: Path, problems: list[str]) -> None:
    """Copy one explicitly selected tree without following directory symlinks."""
    destination.mkdir(parents=True, exist_ok=False)
    for item in sorted(source.iterdir()):
        target = destination / item.name
        if item.is_symlink():
            problems.append(f"Skipped symlink: {item}")
        elif item.is_dir():
            _copy_tree(item, target, problems)
        elif item.is_file():
            shutil.copy2(item, target)
        else:
            problems.append(f"Skipped special file: {item}")


def _load_config(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        result = json.loads(path.read_text())
    else:
        import yaml  # Existing application dependency; not needed by manifest capture.
        result = yaml.safe_load(path.read_text())
    if not isinstance(result, Mapping):
        raise ValueError(f"Config must contain a mapping: {path}")
    return result


def create_bundle(
    output_dir: str | Path, run_dirs: Sequence[str | Path], *,
    configs: Sequence[str | Path] = (), assets: Sequence[str | Path] = (),
    repo_dir: str | Path | None = None,
) -> Path:
    """Copy explicit experiment material and the CURRENT packaging checkout.

    Original run code provenance comes only from each run's recorded evidence;
    the current source snapshot is clearly labeled as packaging-time evidence.
    """
    out = Path(output_dir).expanduser().resolve()
    runs = [Path(p).expanduser().resolve() for p in run_dirs]
    if not runs:
        raise ValueError("At least one run directory is required")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {out}")
    for run in runs:
        if not run.is_dir():
            raise ValueError(f"Run directory does not exist: {run}")
        if out == run or run in out.parents:
            raise ValueError("Bundle output must not be inside a selected run directory")
    out.mkdir(parents=True, exist_ok=False)
    problems: list[str] = []
    source_dir = out / "source"
    source_dir.mkdir()
    git = _git_evidence(_repository(repo_dir), source_dir)
    if git["available"]:
        root = Path(git["root"])
        with (source_dir / "head.tar").open("wb") as handle:
            subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "HEAD"],
                           stdout=handle, stderr=subprocess.PIPE, check=True, timeout=120)
        working_tree = source_dir / "working_tree"
        working_tree.mkdir()
        for raw in _git(root, "ls-files", "-z").split(b"\0"):
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            src = root / relative
            if src.is_symlink():
                problems.append(f"Skipped tracked source symlink: {relative}")
            elif src.is_file():
                target = working_tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            elif src.exists():
                problems.append(f"Unsupported tracked source entry (possibly submodule): {relative}")
        if git["untracked_file_count"]:
            problems.append("Untracked source files were not included automatically; inspect and declare required files as assets.")
    else:
        problems.append("Packaging checkout has no available Git source evidence.")
    historical_sources: dict[str, str] = {}
    run_records = []
    explicit_environment = [str(Path(p).expanduser().absolute()) for p in assets
                            if Path(p).name in ENVIRONMENT_RECORD_NAMES and Path(p).is_file()]
    requests: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        relative = Path("runs") / f"{index:03d}-{run.name}"
        _copy_tree(run, out / relative, problems)
        invocations = []
        evidence_dir = run / "provenance"
        if evidence_dir.is_dir() and not evidence_dir.is_symlink():
            for evidence in sorted(evidence_dir.glob("*/manifest.json")):
                if evidence.is_symlink() or evidence.parent.is_symlink():
                    continue
                try:
                    invocation = json.loads(evidence.read_text())
                    original_git = invocation.get("git") or {}
                    commit = original_git.get("commit", "")
                    if (invocation.get("evidence_scope") != "captured_at_invocation"
                            or not original_git.get("available")
                            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)):
                        problems.append(f"Invocation has no verifiable recorded source revision: {evidence}")
                    elif commit not in historical_sources:
                        archive = source_dir / "historical" / f"{commit}.tar"
                        try:
                            if not git["available"]:
                                raise ValueError("packaging repository unavailable")
                            archive.parent.mkdir(exist_ok=True)
                            with archive.open("wb") as handle:
                                subprocess.run(["git", "-C", git["root"], "archive", "--format=tar", commit],
                                               stdout=handle, stderr=subprocess.PIPE, check=True, timeout=120)
                            historical_sources[commit] = str(archive.relative_to(out))
                        except (OSError, ValueError, subprocess.SubprocessError):
                            archive.unlink(missing_ok=True)
                            problems.append(f"Recorded source revision unavailable locally: {commit}")
                    if original_git.get("untracked_file_count", 0):
                        problems.append(f"Original invocation had untracked files requiring review: {evidence}")
                    patch = evidence.parent / "git.patch"
                    expected_patch = original_git.get("patch_sha256")
                    if expected_patch and (not patch.is_file() or patch.is_symlink()
                                           or sha256_file(patch) != expected_patch):
                        problems.append(f"Recorded source patch missing or changed: {patch}")
                    invocations.append({"path": str(evidence.relative_to(run)),
                                        "git": original_git, "phase": invocation.get("phase")})
                    for ref in invocation.get("input_files", []):
                        requests.append({**ref, "reference_origin": str(evidence)})
                except (ValueError, OSError) as exc:
                    problems.append(f"Unreadable invocation evidence {evidence}: {exc}")
        if not invocations:
            problems.append(f"Historical source version unknown for {run}; current checkout is NOT proof of its original code.")
            legacy_config = run / "run_config.json"
            if legacy_config.is_file() and not legacy_config.is_symlink():
                cfg = _load_config(legacy_config)
                # A legacy file has no recorded runtime cwd. Resolve explicitly
                # against packaging cwd, and label that uncertainty.
                requests.extend({**ref, "reference_origin": str(legacy_config),
                                 "path_resolution": "packaging_cwd_original_cwd_unknown"}
                                for ref in referenced_files(cfg))
        environment_records = [str(p.relative_to(out)) for p in (out / relative).iterdir()
                               if p.is_file() and p.name in ENVIRONMENT_RECORD_NAMES]
        if not environment_records and not explicit_environment:
            problems.append(f"No explicit environment export/lock archived for run: {run}; package metadata is not an environment lock.")
        run_records.append({"original_path": str(run), "bundle_path": str(relative),
                            "historical_provenance": "recorded_invocations" if invocations else "unknown",
                            "environment_records": environment_records,
                            "explicit_environment_assets": explicit_environment,
                            "invocations": invocations})
    for config in configs:
        path = Path(config).expanduser().absolute()
        requests.append({"path": str(path), "reference_origin": "explicit_config"})
        if path.is_file():
            requests.extend({**ref, "reference_origin": str(path)}
                            for ref in referenced_files(_load_config(path)))
    requests.extend({"path": str(Path(p).expanduser().absolute()), "reference_origin": "explicit_asset"}
                    for p in assets)
    copied: dict[str, str] = {}
    asset_records = []
    file_cache = {}
    for ref in requests:
        if ref.get("path") is None:
            asset_records.append(dict(ref))
            problems.append(f"Unresolved asset path ({ref.get('field')}): {ref.get('declared')}; {ref.get('reason')}")
            continue
        if ref["path"] not in file_cache:
            file_cache[ref["path"]] = _file_record(ref["path"])
        record = {**file_cache[ref["path"]], "reference_origin": ref.get("reference_origin"),
                  "field": ref.get("field")}
        original = record["path"]
        if record["status"] == "present":
            if ref.get("sha256") and ref["sha256"] != record["sha256"]:
                problems.append(f"Asset changed since invocation: {original}")
                record["recorded_sha256"] = ref["sha256"]
                record["matches_invocation"] = False
            if original not in copied:
                relative = Path("assets") / f"{len(copied):04d}" / Path(original).name
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                if sha256_file(target) != record["sha256"]:
                    problems.append(f"Asset changed during packaging: {original}")
                copied[original] = str(relative)
            record["bundle_path"] = copied[original]
        else:
            problems.append(f"Missing or unreadable referenced asset: {original}")
        asset_records.append(record)
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "packaging_source": git, "source_scope": "current_checkout_at_packaging_not_historical_run_attestation",
        "historical_sources": historical_sources,
        "runs": run_records, "assets": asset_records, "problems": problems,
        "complete": not problems, "reproduction_validated": False,
        "limitations": ["Review plugin caches/downloaded assets not declared in configuration.",
                        "Bundle checksums verify file integrity, not successful scientific reproduction.",
                        "Paths in original configuration must be remapped using assets in this manifest.",
                        "Archive a tested environment lock/export separately via --asset."],
    }
    _json(out / "bundle.json", manifest)
    checksums = {str(p.relative_to(out)): sha256_file(p)
                 for p in sorted(out.rglob("*")) if p.is_file()}
    _json(out / "checksums.json", checksums)
    return out


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify every archived file, reject traversal/symlinks, report completeness."""
    bundle = Path(bundle_dir).resolve()
    checksum_file = bundle / "checksums.json"
    if checksum_file.is_symlink():
        raise ValueError("Checksum manifest must not be a symlink")
    expected = json.loads(checksum_file.read_text())
    if not isinstance(expected, dict):
        raise ValueError("Checksum manifest must be a mapping")
    errors = []
    for relative, checksum in expected.items():
        path = Path(relative)
        target = bundle / path
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"Unsafe checksum path: {relative}")
        elif any(p.is_symlink() for p in [target, *target.parents] if p != bundle and bundle in p.parents):
            errors.append(f"Symlink: {relative}")
        elif not target.is_file():
            errors.append(f"Missing: {relative}")
        elif sha256_file(target) != checksum:
            errors.append(f"Checksum mismatch: {relative}")
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file() or p.is_symlink()}
    extras = actual - set(expected) - {"checksums.json"}
    errors.extend(f"Unexpected file: {name}" for name in sorted(extras))
    metadata_path = bundle / "bundle.json"
    metadata = {}
    try:
        if metadata_path.is_symlink():
            raise ValueError("Bundle metadata must not be a symlink")
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("Bundle metadata must be a mapping")
    except (OSError, ValueError) as exc:
        errors.append(f"Unreadable bundle metadata: {exc}")
        metadata = {}
    return {"integrity_ok": not errors, "errors": errors,
            "complete": metadata.get("complete", False), "problems": metadata.get("problems", [])}
