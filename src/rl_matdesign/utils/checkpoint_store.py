"""Incremental, portable backing for bulky HPO checkpoint state.

Every learning unit still publishes a complete logical checkpoint. Immutable
content records are committed before its small .pt manifest is atomically
replaced; old records are only reclaimed after publication. No paid observation
or learning update is rolled back to make a checkpoint loadable.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import pickle
import sqlite3
import struct
import zlib

import numpy as np
import torch


_MARKER = "_hpo_checkpoint_storage"
_DATABASE = "checkpoint_data.sqlite3"
_WRITERS = OrderedDict()


def _portable(value, active=None):
    """Raw record pickles must not bypass torch.load's tensor map_location.

    Unknown objects stay in the ordinary Torch payload, where their pre-existing
    serialization contract applies. Numeric arrays are checked without visiting
    their elements. Cyclic containers also take the ordinary-pickle path.
    """
    kind = type(value)
    if kind in (str, bytes, bool, int, float, complex, type(None)):
        return True
    if isinstance(value, (np.ndarray, np.generic)):
        return not value.dtype.hasobject
    if kind not in (dict, list, tuple, set, frozenset):
        return False
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        values = (item for pair in value.items() for item in pair) if kind is dict else iter(value)
        return all(_portable(item, active) for item in values)
    finally:
        active.remove(identity)


def _records(value):
    """Stable content-defined chunks amortize record/SQLite overhead.

    Rows are independently pickled before grouping, so loading cannot introduce
    mutable aliases between replay rows. Digest-selected boundaries mostly survive
    FIFO eviction; the cap keeps a chunk's memory and worst-case rewrite bounded.
    """
    items = value.items() if type(value) is dict else value
    block = []
    for item in items:
        raw = pickle.dumps(item, protocol=5)
        block.append(raw)
        if hashlib.sha256(raw).digest()[0] % 32 == 0 or len(block) >= 128:
            yield pickle.dumps(tuple(block), protocol=5)
            block = []
    if block:
        yield pickle.dumps(tuple(block), protocol=5)


def _paths(payload):
    for path in (("buffer",), ("dp_cache",), ("visit_counts",),
                 ("training_budget_state", "seen_keys"), ("hpo_state", "seen"),
                 ("predictor_ledger", "seen")):
        node = payload
        for key in path[:-1]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        value = node.get(path[-1]) if isinstance(node, dict) else None
        if type(value) in (list, dict) and value and _portable(value):
            yield path, value
    costs = (payload.get("predictor_ledger") or {}).get("phase_costs", {})
    for name, cost in costs.items():
        if (isinstance(cost, dict) and type(cost.get("seen")) is list
                and cost["seen"] and _portable(cost["seen"])):
            yield ("predictor_ledger", "phase_costs", name, "seen"), cost["seen"]


def _replace(payload, path, value):
    node = payload
    for key in path[:-1]:
        node[key] = dict(node[key])
        node = node[key]
    node[path[-1]] = value


def _database_path(directory):
    path = Path(directory) / _DATABASE
    if path.is_symlink():
        raise ValueError("Checkpoint backing database must not be a symlink")
    return path


class _Writer:
    def __init__(self, directory):
        self.path = _database_path(directory)
        self.connection = sqlite3.connect(self.path)
        # DELETE journaling keeps a committed store self-contained when the job
        # directory is copied. FULL ensures blobs precede checkpoint publication.
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS objects "
                                "(id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL, data BLOB NOT NULL)")
        self.connection.commit()
        self.known = {bytes(digest): index for index, digest in
                      self.connection.execute("SELECT id,digest FROM objects")}
        self.signature = self._signature()
        self.publications = 0
        self.pending = None

    def _signature(self):
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def prepare(self, payload):
        if self._signature() != self.signature:
            raise ValueError("Checkpoint backing database changed outside its writer")
        prepared = dict(payload)
        references, live, added = [], set(), {}
        written = 0
        try:
            with self.connection:
                for path, value in _paths(payload):
                    ids, integrity = [], hashlib.sha256()
                    for raw in _records(value):
                        # Hash actual contents, not object identity: mutations of
                        # nested NumPy arrays must never reuse stale stored data.
                        digest = hashlib.sha256(raw).digest()
                        index = added.get(digest, self.known.get(digest))
                        if index is None:
                            data = zlib.compress(raw, 1)
                            cursor = self.connection.execute(
                                "INSERT INTO objects(digest,data) VALUES (?,?)", (digest, data))
                            index = cursor.lastrowid
                            added[digest] = index
                            written += len(data)
                        ids.append(index)
                        live.add(index)
                        integrity.update(struct.pack("<Q", index))
                        integrity.update(digest)
                    references.append({"path": list(path), "kind":
                                       ("chunked_dict" if type(value) is dict else "chunked_list"),
                                       "count": len(value), "object_count": len(ids),
                                       "ids": zlib.compress(struct.pack(f"<{len(ids)}Q", *ids), 1),
                                       "sha256": integrity.hexdigest()})
                    _replace(prepared, path, None)
        except BaseException:
            self.signature = self._signature()
            raise
        self.known.update(added)
        self.signature = self._signature()
        self.pending = live
        prepared[_MARKER] = {"version": 1, "database": _DATABASE, "references": references}
        return prepared, {"checkpoint_new_blob_bytes": written, "checkpoint_new_blob_count": len(added),
                          "checkpoint_store_bytes": self.path.stat().st_size}

    def published(self):
        self.publications += 1
        # Batched reclamation limits dead replay data without a second database
        # fsync every episode. Freed SQLite pages are reused on later inserts.
        if self.pending is None or self.publications % 64:
            return
        if self._signature() != self.signature:
            raise ValueError("Checkpoint backing database changed outside its writer")
        stale = [(digest, index) for digest, index in self.known.items() if index not in self.pending]
        with self.connection:
            self.connection.executemany("DELETE FROM objects WHERE id=?", [(i,) for _, i in stale])
        for digest, _ in stale:
            del self.known[digest]
        self.signature = self._signature()

    def close(self):
        self.connection.close()


def _writer(directory):
    key = str(Path(directory).resolve())
    writer = _WRITERS.get(key)
    if writer is None:
        writer = _WRITERS[key] = _Writer(key)
    _WRITERS.move_to_end(key)
    while len(_WRITERS) > 4:
        _, old = _WRITERS.popitem(last=False)
        old.close()
    return writer


def prepare_checkpoint(path, payload):
    """Only real HPO training checkpoints use the incremental representation."""
    hpo = payload.get("hpo_state")
    if payload.get("type") not in {"dqn", "pg"} or not isinstance(hpo, dict) or not isinstance(hpo.get("budget"), dict):
        return payload, {}
    if _MARKER in payload:
        raise ValueError("Checkpoint storage marker is reserved")
    return _writer(Path(path).parent).prepare(payload)


def checkpoint_published(path):
    writer = _WRITERS.get(str(Path(path).parent.resolve()))
    if writer is not None:
        # A database commit may outlive a non-durable directory rename. Publish
        # the .pt directory entry durably before reclaiming the previous .pt's
        # backing records, otherwise a power loss could resurrect broken refs.
        descriptor = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        writer.published()


def load_checkpoint(path, *, map_location="cpu"):
    """Load old standalone or new incremental checkpoints without writing files.

    As with the old full training checkpoints, this loads trusted pickle data.
    Model-only exports still use the restricted PyTorch model loader.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or _MARKER not in payload:
        return payload
    storage = payload.pop(_MARKER)
    if (not isinstance(storage, dict)
            or set(storage) != {"version", "database", "references"}
            or type(storage.get("version")) is not int or storage["version"] != 1
            or storage.get("database") != _DATABASE
            or not isinstance(storage.get("references"), list)):
        raise ValueError("Unsupported checkpoint storage descriptor")
    database = _database_path(path.parent)
    if not database.is_file():
        raise ValueError(f"Checkpoint backing database is missing: {database}")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    cached = {}
    seen_paths = set()
    try:
        for reference in storage["references"]:
            if not isinstance(reference, dict) or set(reference) != {"path", "kind", "count", "object_count", "ids", "sha256"}:
                raise ValueError("Invalid checkpoint storage reference")
            if not isinstance(reference["path"], list):
                raise ValueError("Invalid checkpoint storage field")
            field = tuple(reference["path"])
            if not field or not all(isinstance(key, str) for key in field) or field in seen_paths:
                raise ValueError("Invalid checkpoint storage field")
            ordinary = {("buffer",), ("dp_cache",), ("visit_counts",),
                        ("training_budget_state", "seen_keys"), ("hpo_state", "seen"),
                        ("predictor_ledger", "seen")}
            phase_seen = (len(field) == 4 and field[:2] == ("predictor_ledger", "phase_costs")
                          and field[-1] == "seen")
            if field not in ordinary and not phase_seen:
                raise ValueError("Invalid checkpoint storage field")
            node = payload
            for key in field[:-1]:
                node = node.get(key) if isinstance(node, dict) else None
            if not isinstance(node, dict) or field[-1] not in node or node[field[-1]] is not None:
                raise ValueError("Checkpoint storage field has no reserved placeholder")
            seen_paths.add(field)
            count = reference["count"]
            object_count = reference["object_count"]
            packed = zlib.decompress(reference["ids"])
            if (type(count) is not int or count <= 0 or type(object_count) is not int
                    or not 0 < object_count <= count or len(packed) != object_count * 8):
                raise ValueError("Invalid checkpoint storage reference length")
            ids = struct.unpack(f"<{object_count}Q", packed)
            integrity, items = hashlib.sha256(), []
            for index in ids:
                if index not in cached:
                    row = connection.execute("SELECT digest,data FROM objects WHERE id=?", (index,)).fetchone()
                    if row is None:
                        raise ValueError("Checkpoint backing record is missing")
                    digest, data = row
                    raw = zlib.decompress(data)
                    if hashlib.sha256(raw).digest() != digest:
                        raise ValueError("Checkpoint backing record checksum differs")
                    cached[index] = digest, raw
                digest, raw = cached[index]
                integrity.update(struct.pack("<Q", index))
                integrity.update(digest)
                # Each occurrence restores an independent value, as the packed
                # replay loader did; no unintended mutable aliases are created.
                restored = pickle.loads(raw)
                if reference["kind"] in {"chunked_list", "chunked_dict", "replay_list"}:
                    if (not isinstance(restored, tuple) or not 0 < len(restored) <= 128
                            or not all(isinstance(item, bytes) for item in restored)):
                        raise ValueError("Invalid checkpoint content block")
                    items.extend(pickle.loads(item) for item in restored)
                else:
                    items.append(restored)
            if integrity.hexdigest() != reference["sha256"]:
                raise ValueError("Checkpoint backing references checksum differs")
            if len(items) != count:
                raise ValueError("Invalid checkpoint restored item count")
            if reference["kind"] in {"dict", "chunked_dict"}:
                value = dict(items)
                if len(value) != count:
                    raise ValueError("Checkpoint backing mapping contains duplicate keys")
            elif reference["kind"] in {"list", "chunked_list", "replay_list"}:
                value = items
            else:
                raise ValueError("Unsupported checkpoint backing container")
            _replace(payload, field, value)
    finally:
        connection.close()
    return payload
