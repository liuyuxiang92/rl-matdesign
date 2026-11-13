"""Cooperative signals for native HPO workers, without changing normal runs.

The first SIGINT/SIGTERM requests a stop after a fully committed training or
generation checkpoint.  Collection and learning finish their current unit so
the paid candidate ledger never has to be rolled back.  Repeated SIGTERM is
idempotent because schedulers can signal both coordinator and worker directly.
A later SIGINT forces immediate exit, with no resumability guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager
import signal
import threading


_active_stop = None
_TERMINAL = {"COMPLETED", "INFEASIBLE_EFFICIENCY", "INFEASIBLE_NUMERICAL"}


class _StopRequest:
    def __init__(self):
        self.signum = None

    def handle(self, signum, _frame):
        if self.signum is not None:
            if signum == signal.SIGINT:
                raise SystemExit(128 + signum)
            return
        # Keep the handler free of I/O: a signal can arrive during an unrelated
        # Python stream write, where printing could itself interrupt cleanup.
        self.signum = signum


@contextmanager
def cooperative_worker_stop():
    """Install worker-only handlers before importing the experiment runtime.

    Prior handlers are restored on every exit.  Explicit nesting and use from
    non-main threads are errors rather than silently changing an embedding
    application's signal behavior.  Importing this module has no side effects.
    """
    global _active_stop
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Cooperative HPO signals require the main thread")
    if _active_stop is not None:
        raise RuntimeError("Cooperative HPO signals are already installed")
    request = _StopRequest()
    previous = {}
    _active_stop = request
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request.handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _active_stop = None


def checkpoint_committed(data):
    """Honor a pending stop only after a resumable HPO checkpoint is on disk.

    Call after serialization, atomic replacement, and any latest-file link
    replacement succeed.  Zero-attempt initialization checkpoints are not yet
    resumable jobs.  Terminal budgets must finish their normal final model and
    report exports instead of being interrupted between checkpoint and export.
    """
    request = _active_stop
    if request is None or request.signum is None or not isinstance(data, dict):
        return
    hpo = data.get("hpo_state")
    budget = hpo.get("budget") if isinstance(hpo, dict) else None
    if not isinstance(budget, dict):
        return
    counters = ("pending", "attempts_started", "unique_evaluated", "target_unique", "max_attempts")
    if any(type(budget.get(name)) is not int for name in counters):
        return
    if (budget["pending"] != 0 or budget["attempts_started"] <= 0
            or budget["unique_evaluated"] >= budget["target_unique"]
            or budget.get("status") in _TERMINAL):
        return
    if (budget["attempts_started"] + budget["target_unique"] - budget["unique_evaluated"]
            > budget["max_attempts"]):
        # The next reservation will mark terminal efficiency failure.  Let its
        # normal exception/finalization path preserve that terminal outcome.
        return
    raise SystemExit(128 + request.signum)
