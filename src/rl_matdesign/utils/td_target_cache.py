"""Bounded memoization of scalar, deterministic DQN target evaluations.

One cache belongs to one fixed target network, scaler and action encoding.
Clear it immediately after copying new weights into that network. This keeps
the original per-state forward shape and floating-point evaluation unchanged.
The cache is only an optimization; it is deliberately not checkpoint state.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class TDTargetCache:
    def __init__(self, max_entries=4096, max_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._bytes = 0
        self._values = OrderedDict()

    @staticmethod
    def key(row):
        def array_key(value):
            a = np.asarray(value)
            return a.dtype.str, a.shape, a.tobytes()
        return (array_key(row["s_mat_next_raw"]), array_key(row["s_step_next"]),
                np.asarray(row["next_allowed_idx"], dtype=np.int64).tobytes())

    @staticmethod
    def _size(key):
        return len(key[0][2]) + len(key[1][2]) + len(key[2]) + 512

    def get(self, key):
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key, value):
        if key not in self._values:
            self._bytes += self._size(key)
        self._values[key] = value
        self._values.move_to_end(key)
        while (len(self._values) > self.max_entries or self._bytes > self.max_bytes):
            evicted, _ = self._values.popitem(last=False)
            self._bytes -= self._size(evicted)

    def clear(self):
        self._values.clear()
        self._bytes = 0
