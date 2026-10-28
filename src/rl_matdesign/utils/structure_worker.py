"""Private subprocess entry point; CUDA environment is set before this imports.

Dedicated inherited pipe descriptor keeps plugin/ASE/LAMMPS stdout out of IPC.
This module must not import experiment.py or execute the user's training script.
"""
from __future__ import annotations

import sys
import traceback
from multiprocessing.connection import Connection


def main(fd: int) -> None:
    connection = Connection(fd)
    builder = None
    try:
        setup = connection.recv()
        try:
            import numpy as np
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "A structure worker requires exactly one visible CUDA GPU; "
                    "check the GPU index, allocation and CUDA-enabled Python environment"
                )
            torch.set_num_threads(setup["threads"])
            from rl_matdesign.registry import resolve_builder

            cfg = setup["cfg"]
            builder = resolve_builder(setup["kind"], cfg, seed=setup["seed"])
            geo_opt = cfg.get("geo_opt") or {}
            geo_enabled = bool(geo_opt) and bool(geo_opt.get("enabled", True))
            geo_calc = None
            connection.send({"ok": True})
        except Exception:
            connection.send({"ok": False, "error": traceback.format_exc()})
            return

        while True:
            try:
                task = connection.recv()
            except EOFError:
                break
            if task["op"] == "close":
                break
            try:
                if task["op"] == "formula":
                    fn = getattr(builder, "composition_formula", None)
                    formula = fn(task["candidate"]) if callable(fn) else None
                    connection.send({"ok": True, "formula": formula})
                    continue
                from rl_matdesign.utils.structure import strict_relaxation_errors

                with strict_relaxation_errors():
                    atoms_list = builder.build(
                        task["candidate"], n_configs=1, rng=np.random.default_rng(task["seed"])
                    )
                    if len(atoms_list) != 1:
                        raise ValueError(
                            "structure_parallel requires a builder returning exactly one "
                            f"structure for n_configs=1; got {len(atoms_list)}. "
                            "Multi-template/expanded ensembles must use the serial path."
                        )
                    atoms = atoms_list[0]
                    if geo_enabled:
                        from rl_matdesign.utils.structure import relax_structure
                        if geo_calc is None:
                            from deepmd.calculator import DP
                            head = geo_opt.get("head")
                            geo_calc = DP(model=geo_opt.get("model", "models/DPA-3.1-3M.pt"),
                                          **({"head": head} if head else {}))
                        atoms = relax_structure(
                            atoms, calc=geo_calc, fmax=float(geo_opt.get("fmax", 0.001)),
                            steps=int(geo_opt.get("steps", 1000)),
                            relax_cell=bool(geo_opt.get("relax_cell", True)),
                        )
                # ASE copy drops the calculator. Do not serialize a live DeepMD
                # model/CUDA allocation or move it back into the parent process.
                detached = atoms.copy()
                detached.calc = None
                connection.send({"ok": True, "atoms": detached})
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        try:
            close = getattr(builder, "close", None)
            if callable(close):
                close()
        finally:
            connection.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
