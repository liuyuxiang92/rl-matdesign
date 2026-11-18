"""OOH model adapter consuming frames produced by the structure pipeline.

Each parent slab is one configuration. Its ordered, padded adsorbate frames
are a single model input, exactly as in the original DeepProperty contract.
No substitution, adsorbate placement, MD, or relaxation occurs in this adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from numbers import Real

import numpy as np

from .outputs import OutputSchema, ModelSamples, aggregate_samples
from ..utils.rollout_errors import EvaluationProtocolError


class OOHFramesPredictor:
    output_schema = {"overpotential": OutputSchema("overpotential", None, "physical")}

    def __init__(self, cfg, *, seed=None):
        paths = cfg.get("models")
        if not isinstance(paths, (list, tuple)) or not paths or len(set(paths)) != len(paths):
            raise ValueError("OOH models must be a non-empty list of distinct checkpoints")
        index = cfg.get("output_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("OOH output_index must be a nonnegative integer")
        self.output_index = index
        self._assets = {}
        resolved_paths = []
        for raw in paths:
            path = Path(raw).resolve(strict=True)
            if str(path) in self._assets:
                raise ValueError("OOH checkpoints must resolve to distinct model files")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self._assets[str(path)] = (digest, self._stamp(path))
            resolved_paths.append(str(path))
        from deepmd.pt.infer.deep_eval import DeepProperty
        self.models = []
        self._closed = False
        try:
            for path in resolved_paths:
                self.models.append(DeepProperty(
                    model_file=str(path), auto_batch_size=False, head="property"))
        except BaseException:
            self.close()
            raise
        self.model_ids = tuple(f"deepmd:{i}:sha256:{digest}"
                               for i, (digest, _) in enumerate(self._assets.values()))

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def oracle_identity(self):
        for name, (digest, stamp) in self._assets.items():
            path = Path(name)
            if self._stamp(path) != stamp:
                raise EvaluationProtocolError(f"OOH checkpoint changed: {name}; create a new predictor")
        return {"adapter": "ooh_pipeline_frames_v1", "output_index": self.output_index,
                "models": {path: digest for path, (digest, _) in self._assets.items()}}

    @staticmethod
    def prepared_inputs(parent):
        from ase.data import chemical_symbols
        from .builders.pipeline import pipeline_frames

        frames = pipeline_frames(parent)
        if not frames:
            raise EvaluationProtocolError("OOH requires an ooh_adsorbates pipeline step (use adsorbates: [] for bare)")
        symbols = frames[0][1].get_chemical_symbols()
        if any(frame.get_chemical_symbols() != symbols for _, frame, _ in frames):
            raise EvaluationProtocolError("OOH frames must have identical atom counts and symbol order")
        periodic_index = {symbol: i for i, symbol in enumerate(s for s in chemical_symbols if s != "X")}
        try:
            types = np.asarray([periodic_index[s] for s in symbols], dtype=np.int32)
        except KeyError as exc:
            raise EvaluationProtocolError(f"Unknown OOH atom symbol {exc.args[0]}") from exc
        coords = np.asarray([frame.get_positions() for _, frame, _ in frames], dtype=np.float64)
        cells = np.asarray([frame.get_cell().array for _, frame, _ in frames], dtype=np.float64)
        atom_types = np.tile(types, (len(frames), 1))
        for i, (_, _, mask) in enumerate(frames):
            if any(isinstance(j, bool) or not isinstance(j, int) or j < 0 or j >= len(types) for j in mask):
                raise EvaluationProtocolError("Invalid OOH inactive-atom mask")
            atom_types[i, mask] = -1
        return coords, cells, atom_types

    def predict_structure_outputs(self, structures):
        from ..utils.dp_eval import pick_scalar

        self.oracle_identity()
        if not structures:
            raise EvaluationProtocolError("OOH requires at least one parent sample")
        rows = [[] for _ in self.models]
        ids = []
        for index, parent in enumerate(structures):
            sample = parent.info.get("_rl_matdesign_sample", {"returned_index": index})
            ids.append(json.dumps(sample, sort_keys=True, separators=(",", ":")))
            coords, cells, atom_types = self.prepared_inputs(parent)
            for row, model in zip(rows, self.models):
                raw = model.eval(coords=coords, cells=cells, atom_types=atom_types, mixed_type=True)
                try:
                    selected = np.asarray(raw, dtype=object).reshape(-1)[self.output_index]
                    if isinstance(selected, (bool, np.bool_)) or not isinstance(selected, Real):
                        raise ValueError("OOH model output must be a real numeric scalar")
                    if not np.isfinite(selected):
                        raise ValueError("OOH model output must be finite")
                    row.append(pick_scalar(raw, output_index=self.output_index))
                except (ValueError, TypeError, IndexError, OverflowError) as exc:
                    raise EvaluationProtocolError("OOH model cannot supply the configured scalar output") from exc
        output = aggregate_samples(self.output_schema["overpotential"],
                                   [ModelSamples(model_id, tuple(ids), tuple(row))
                                    for model_id, row in zip(self.model_ids, rows)])
        self.oracle_identity()
        return {"overpotential": output}

    def close(self):
        if self._closed:
            return
        self._closed = True
        for model in reversed(self.models):
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
