"""Persistent JSON-lines transport for predictors in a separate environment.

The command must print ``{"status": "ready", "metadata": {...}}`` once, then
answer each ``{"op": "predict", "candidate": ...}`` line with scalar ``mean``
and ``std`` fields and optional ``metadata``. Diagnostics belong on stderr.
No shell, chemistry encoding, reward transformation, or seed arguments are
introduced by this transport.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import math
from queue import Empty, Queue
import subprocess
from threading import RLock, Thread
import time
from typing import Any

from ..utils.composition import composition_key


def _read_stdout(stream, messages: Queue) -> None:
    """Keep blocking pipe reads away from the caller's deadline."""
    try:
        for line in stream:
            messages.put(line)
    except (OSError, ValueError) as exc:
        messages.put(exc)
    finally:
        messages.put(None)


def _write_stdin(stream, payload: bytes, completion: Queue) -> None:
    # Unbuffered pipes avoid a buffered stream lock during timeout cleanup.
    try:
        remaining = memoryview(payload)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("predictor stdin closed")
            remaining = remaining[written:]
        completion.put(None)
    except (OSError, ValueError) as exc:
        completion.put(exc)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


class JsonProcessPredictor:
    """Load a predictor worker once and cache its raw distributions/metadata.

    Required config: ``command`` (a nonempty list of strings). Optional config:
    ``cwd``, ``startup_timeout`` (600 seconds), ``request_timeout`` (120 seconds),
    and positive ``cache_size`` (10,000 entries). Each request's timeout includes
    both writing and reading. ``seed`` is accepted for registry compatibility;
    configure worker seeds explicitly in ``command`` when supported.
    """

    def __init__(self, cfg, seed=None):
        self._process = None
        self._reader = None
        self._closed = False
        self._lock = RLock()
        self._messages = Queue()
        self._cache = OrderedDict()
        self.seed = seed

        command = cfg.get("command")
        if (not isinstance(command, list) or not command
                or not all(isinstance(arg, str) for arg in command) or not command[0]):
            raise ValueError("command must be a nonempty list of strings with an executable")
        self.startup_timeout = self._timeout(cfg, "startup_timeout", 600.0)
        self.request_timeout = self._timeout(cfg, "request_timeout", 120.0)
        self.cache_size = cfg.get("cache_size", 10000)
        if (isinstance(self.cache_size, bool)
                or not isinstance(self.cache_size, int) or self.cache_size < 1):
            raise ValueError("cache_size must be a positive integer")

        try:
            self._process = subprocess.Popen(
                command, cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=None, bufsize=0,
            )
            self._reader = Thread(
                target=_read_stdout, args=(self._process.stdout, self._messages),
                name="json-predictor-stdout", daemon=True,
            )
            self._reader.start()
            ready = self._receive(time.monotonic() + self.startup_timeout, "startup")
            if ready.get("status") != "ready":
                raise RuntimeError(f"predictor startup failed: {ready!r}")
            self.startup_metadata = self._metadata(ready)
        except BaseException:
            self._close(force=True)
            raise

    @staticmethod
    def _timeout(cfg, name, default):
        value = cfg.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return value

    @staticmethod
    def _metadata(response):
        metadata = response.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError("predictor metadata must be a JSON object")
        return metadata

    @staticmethod
    def _wait(queue, deadline, phase):
        try:
            return queue.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty as exc:
            raise TimeoutError(f"predictor {phase} timed out") from exc

    def _receive(self, deadline, phase):
        line = self._wait(self._messages, deadline, phase)
        if line is None:
            raise RuntimeError(f"predictor exited before a {phase} response")
        if isinstance(line, Exception):
            raise RuntimeError(f"cannot read predictor {phase} response") from line
        try:
            response = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(f"predictor {phase} response is not valid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"predictor {phase} response must be a JSON object")
        return response

    def _result(self, candidate):
        if self._closed:
            raise RuntimeError("predictor is closed")
        # The list wrapper asks composition_key for typed keys even for payloads
        # that contain only categorical mappings (e.g. the IDs "01" and "1").
        key = composition_key([candidate])
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        payload = (json.dumps(
            {"op": "predict", "candidate": candidate}, allow_nan=False,
        ) + "\n").encode("utf-8")
        deadline = time.monotonic() + self.request_timeout
        try:
            completion = Queue()
            writer = Thread(
                target=_write_stdin,
                args=(self._process.stdin, payload, completion),
                name="json-predictor-stdin", daemon=True,
            )
            writer.start()
            write_error = self._wait(completion, deadline, "request write")
            if write_error is not None:
                raise RuntimeError("cannot write to predictor") from write_error
            response = self._receive(deadline, "prediction")
        except BaseException:
            # A late reply cannot safely be associated with the next request.
            self._close(force=True)
            raise
        if "error" in response:
            raise RuntimeError(f"predictor rejected candidate: {response['error']}")
        try:
            values = [response.get(name) for name in ("mean", "std")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                raise ValueError("mean and std must be numeric scalars")
            mean, std = map(float, values)
            if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                raise ValueError("mean and std must be finite, with std >= 0")
            result = (mean, std, self._metadata(response))
        except (ValueError, OverflowError, RuntimeError) as exc:
            self._close(force=True)
            raise RuntimeError(f"invalid predictor distribution: {exc}") from exc
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def predict(self, candidate) -> tuple[float, float]:
        with self._lock:
            mean, std, _ = self._result(candidate)
            return mean, std

    def batch_predict(self, candidates):
        return [self.predict(candidate) for candidate in candidates]

    def candidate_metadata(self, candidate) -> dict[str, Any]:
        """Return matching response metadata, requesting a prediction if needed."""
        with self._lock:
            return deepcopy(self._result(candidate)[2])

    def _close(self, *, force=False):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        if force and process.poll() is None:
            process.kill()
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()

    def close(self):
        """Close stdin, then kill a worker that does not exit within two seconds."""
        with self._lock:
            self._close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
