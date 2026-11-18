"""Optional LAMMPS averaging operation on an already-built structure.

The external CLC package supplies trajectory I/O only. Its template-building
classes are deliberately not used, so original site/sample identity survives.
"""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def md_average(atoms, cfg, *, rng):
    model = str(cfg["model"])
    if model.endswith((".ckpt", ".ckpt.pt")):
        raise ValueError("md_average requires a frozen LAMMPS model, not a training checkpoint; freeze the requested head first")
    if not cfg.get("lmp_bin"):
        raise ValueError("md_average requires lmp_bin")
    model = str(Path(model).expanduser().resolve())
    if not Path(model).is_file():
        raise FileNotFoundError(f"md_average model does not exist: {model}")
    binary = str(cfg["lmp_bin"])
    if os.sep in binary:
        binary = str(Path(binary).expanduser().resolve())
    elif shutil.which(binary) is None:
        raise FileNotFoundError(f"LAMMPS executable not found: {binary}")
    try:
        from clc_workflow.lammps_io import write_conf_lmp, render_input_lammps, read_frames_since, average_frames
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError("md_average requires clc_workflow.lammps_io and pymatgen") from exc
    minimum = float(cfg.get("min_dist", 1.7))
    if len(atoms) > 1:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        if float(distances.min()) < minimum:
            raise ValueError(f"md_average input has an interatomic distance below min_dist={minimum}")
    nsteps = int(cfg.get("nsteps", 100000))
    timestep = float(cfg.get("timestep_ps", .002))
    window = float(cfg.get("average_last_ps", 20))
    if nsteps <= 0 or timestep <= 0 or window <= 0:
        raise ValueError("md_average nsteps, timestep_ps and average_last_ps must be positive")
    type_map = sorted(set(atoms.get_chemical_symbols()))
    expected_types = np.asarray([type_map.index(s) + 1 for s in atoms.get_chemical_symbols()])
    parent_dir = cfg.get("scratch_dir")
    if parent_dir:
        Path(parent_dir).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="rl-pipeline-md-", dir=parent_dir))
    succeeded = False
    try:
        write_conf_lmp(scratch / "conf.lmp", AseAtomsAdaptor.get_structure(atoms), type_map,
                       source="rl_matdesign pipeline")
        seed = int(cfg.get("velocity_seed", rng.integers(1, 2**31-1)))
        plugin = cfg.get("plugin")
        if plugin and os.sep in str(plugin):
            plugin = str(Path(plugin).expanduser().resolve())
        deck = render_input_lammps(type_map=type_map, model_path=model,
            temperature_k=float(cfg.get("temperature_k", 300)), nsteps=nsteps, timestep_ps=timestep,
            thermo_freq=int(cfg.get("thermo_freq", 10)), dump_freq=int(cfg.get("dump_freq", 100)),
            pressure_bar=float(cfg.get("pressure_bar", 0)), tau_t=float(cfg.get("tau_t", .1)),
            tau_p=float(cfg.get("tau_p", .5)), seed=seed, model_head=cfg.get("head"),
            pair_style=cfg.get("pair_style", "deepmd"), atom_style=cfg.get("atom_style", "atomic"),
            plugin=plugin)
        (scratch / "input.lammps").write_text(deck)
        from ...utils.stage_timing import timed_stage
        with timed_stage("md"):
            result = subprocess.run([binary, *map(str, cfg.get("lmp_args", [])), "-in", "input.lammps"],
                                    cwd=scratch, capture_output=True, text=True,
                                    env={**os.environ, **{str(k): str(v) for k, v in (cfg.get("env") or {}).items()}})
        (scratch / "stdout.log").write_text(result.stdout)
        (scratch / "stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"LAMMPS failed with exit code {result.returncode}; logs: {scratch}")
        min_step = max(0, nsteps - round(window / timestep))
        frames = read_frames_since(scratch / "traj.lammpstrj", min_step)
        average, stats = average_frames(frames, rms_max=cfg.get("avg_rms_max", .5))
        # CLC's parser sorts by LAMMPS id, which the writer assigns in input order.
        # NPT dynamics changes neither identities nor types. Reject any mismatch.
        if average["n_atoms"] != len(atoms) or not np.array_equal(average["types"], expected_types):
            raise ValueError("MD trajectory changed atom populations/order; original site identity cannot be retained")
        output = atoms.copy()
        output.set_cell(average["cell"], scale_atoms=False)
        output.set_positions(average["coords"])
        output.info = deepcopy(atoms.info)
        output.info["_rl_matdesign_md"] = dict(stats, velocity_seed=seed, temperature_k=float(cfg.get("temperature_k", 300)))
        if cfg.get("keep_scratch", False):
            output.info["_rl_matdesign_md"]["scratch_dir"] = str(scratch)
        succeeded = True
        return output
    except Exception as exc:
        if hasattr(exc, "add_note"):
            exc.add_note(f"MD operation scratch directory retained at {scratch}")
        raise
    finally:
        if succeeded and not cfg.get("keep_scratch", False):
            shutil.rmtree(scratch)
