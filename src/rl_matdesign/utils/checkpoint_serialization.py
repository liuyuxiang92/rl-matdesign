"""Lossless column packing for full, trusted training checkpoints.

Replay rows are unpacked by pickle into the original list of dictionaries, so
existing ``torch.load(weights_only=False)`` readers need no migration logic.
Model-only exports are separate and do not use this representation.
"""
from __future__ import annotations

import os
import time
import zlib
from typing import Any

import numpy as np
import torch


_CHECKPOINT_PICKLE_PROTOCOL = 5
_COLUMN_COMPRESSION_LEVEL = 1
_MIN_COMPRESS_BYTES = 64 * 1024


def _pack_storage(array):
    """Use fast lossless compression only when it meaningfully reduces bytes."""
    if _COLUMN_COMPRESSION_LEVEL and array.nbytes >= _MIN_COMPRESS_BYTES:
        compressed = zlib.compress(memoryview(array).cast("B"), _COLUMN_COMPRESSION_LEVEL)
        if len(compressed) < array.nbytes * .9:
            return ("zlib", array.dtype.str, array.shape, compressed)
    return ("tensor", torch.from_numpy(array))


def _unpack_storage(packed):
    if packed[0] == "tensor":
        return _cpu_array(packed[1])
    if packed[0] == "zlib":
        _, dtype, shape, compressed = packed
        return np.frombuffer(zlib.decompress(compressed), dtype=np.dtype(dtype)).reshape(shape)
    raise ValueError(f"Unsupported packed replay storage: {packed[0]}")


def _cpu_array(tensor):
    # A checkpoint reader may map network and optimizer tensors to CUDA. Replay
    # data stays NumPy, as it did before column packing was introduced.
    return tensor.detach().cpu().numpy()


def _pack_column(key, values):
    first = values[0]
    if all(value is None for value in values):
        return 'none', None
    if type(first) is np.ndarray and all(
        type(value) is np.ndarray and value.shape == first.shape
        and value.dtype == first.dtype for value in values
    ) and first.dtype.isnative and first.dtype.kind in 'biufc':
        try:
            stacked = _pack_storage(np.stack(values))
        except (TypeError, ValueError):
            # E.g. platform-specific float128: retain NumPy's exact encoding.
            pass
        else:
            return 'array', stacked
    if type(first) in (bool, int, float) and all(type(value) is type(first) for value in values):
        kind = {bool: 'bool', int: 'int', float: 'float'}[type(first)]
        if kind != 'int' or all(-(1 << 63) <= value < (1 << 63) for value in values):
            dtype = {'bool': np.bool_, 'int': np.int64, 'float': np.float64}[kind]
            return kind, _pack_storage(np.asarray(values, dtype=dtype))
    if key == 'next_allowed_idx' and all(
        type(value) is list and all(
            type(pair) is tuple and len(pair) == 2
            and all(type(index) is int and -(1 << 63) <= index < (1 << 63) for index in pair)
            for pair in value
        ) for value in values
    ):
        lengths = np.fromiter((len(value) for value in values), dtype=np.int64, count=len(values))
        offsets = np.empty(len(values) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        flattened = np.asarray([pair for value in values for pair in value], dtype=np.int64).reshape(-1, 2)
        return 'pairs', (_pack_storage(flattened), _pack_storage(offsets))
    # Unknown extras, mixed scalar types, exotic arrays, and legacy fields retain
    # their ordinary pickle representation instead of being coerced.
    return 'raw', values


class PackedReplaySnapshot:
    """Serialization-only view; constructing it never mutates the live buffer."""
    def __init__(self, rows: list[dict]):
        self.row_count = len(rows)
        schemas = []
        schema_indices = []
        schema_lookup = {}
        for row in rows:
            schema = tuple(row)
            if schema not in schema_lookup:
                schema_lookup[schema] = len(schemas)
                schemas.append(schema)
            schema_indices.append(schema_lookup[schema])
        self.schemas = schemas
        self.schema_indices = (None if len(schemas) == 1
                               else torch.tensor(schema_indices, dtype=torch.int64, device="cpu"))
        self.columns = []
        keys = dict.fromkeys(key for schema in schemas for key in schema)
        for key in keys:
            positions = [index for index, row in enumerate(rows) if key in row]
            values = [rows[index][key] for index in positions]
            kind, packed = _pack_column(key, values)
            indices = (None if len(positions) == len(rows)
                       else torch.tensor(positions, dtype=torch.int64, device="cpu"))
            self.columns.append((key, indices, kind, packed))

    def __reduce_ex__(self, protocol):
        return unpack_replay_rows, (1, self.row_count, self.schemas,
                                    self.schema_indices, self.columns)


def unpack_replay_rows(version, row_count, schemas, schema_indices, columns):
    """Restore the same ordinary replay schema at the existing load boundary."""
    if version != 1:
        raise ValueError(f'Unsupported packed replay version: {version}')
    if schema_indices is None:
        rows = [dict.fromkeys(schemas[0]) for _ in range(row_count)]
    else:
        rows = [dict.fromkeys(schemas[index]) for index in _cpu_array(schema_indices).tolist()]
    for key, indices, kind, packed in columns:
        positions = range(row_count) if indices is None else _cpu_array(indices).tolist()
        if kind == 'array':
            array = _unpack_storage(packed)
            values = (array[index:index + 1].reshape(array.shape[1:]).copy()
                      for index in range(len(array)))
        elif kind in ('bool', 'int', 'float'):
            values = _unpack_storage(packed).tolist()
        elif kind == 'pairs':
            pairs, offsets = map(_unpack_storage, packed)
            # Convert each integer only once, retaining list/tuple/int types.
            tuples = list(map(tuple, pairs.tolist()))
            values = (tuples[start:end] for start, end in zip(offsets[:-1], offsets[1:]))
        elif kind == 'none':
            values = (None for _ in positions)
        elif kind == 'raw':
            values = packed
        else:
            raise ValueError(f'Unsupported packed replay column kind: {kind}')
        for position, value in zip(positions, values):
            rows[position][key] = value
    return rows


def save_checkpoint_payload(path: str, payload: dict[str, Any]) -> dict[str, float | int]:
    """Pack and serialize one payload; its caller owns atomic replacement.

    Measurements cover packing and writing the temporary file, without claiming
    that buffered writes have reached physical storage. Unsupported replay
    containers retain their original representation.
    """
    started = time.perf_counter()
    rows = payload.get('buffer')
    if type(rows) is list and rows and all(type(row) is dict for row in rows):
        payload = {**payload, 'buffer': PackedReplaySnapshot(rows)}
    packed = time.perf_counter()
    torch.save(payload, path, pickle_protocol=_CHECKPOINT_PICKLE_PROTOCOL)
    serialized = time.perf_counter()
    return {'checkpoint_pack_s': packed - started,
            'checkpoint_serialize_s': serialized - packed,
            'checkpoint_bytes': os.path.getsize(path)}
