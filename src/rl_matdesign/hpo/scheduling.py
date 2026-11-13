"""Bounded independent jobs; all shared scoring is finalized in input order."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import warnings


def run_jobs(jobs, runner, parallel_jobs, *, on_start, on_result, on_progress):
    """Run independent worker processes without concurrent scientific decisions.

    Threads only wait on isolated subprocesses. The coordinator alone finalizes
    calibration, scores and progress, so completion order cannot choose a scale.
    """
    if parallel_jobs == 1:
        results = []
        for index, job in enumerate(jobs):
            on_start(index, job)
            row = runner(**job)
            results.append(row)
            on_result(index, row)
        return results
    if not jobs:
        return []
    execute = getattr(runner, "execute", runner)
    finalize = getattr(runner, "finalize", lambda value: value)
    preserve = getattr(runner, "preserve", finalize)
    pool = ThreadPoolExecutor(max_workers=parallel_jobs, thread_name_prefix="hpo-worker")
    pending, launched, ready = {}, {}, {}
    finalized = set()
    results = [None] * len(jobs)
    next_submit = next_finalize = 0

    def fill():
        nonlocal next_submit
        while next_submit < len(jobs) and len(pending) < parallel_jobs:
            index = next_submit
            on_start(index, jobs[index])
            future = pool.submit(execute, **jobs[index])
            pending[future] = index
            launched[index] = future
            next_submit += 1

    try:
        fill()
        while pending:
            done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                index = pending[future]
                ready[index] = future.result()
                del pending[future]
            while next_finalize in ready:
                raw = ready.pop(next_finalize)
                finalized.add(next_finalize)
                row = finalize(raw)
                results[next_finalize] = row
                on_result(next_finalize, row)
                next_finalize += 1
            fill()
            if done:
                on_progress([pending[f] for f in pending if not f.done()],
                            len(launched) - len(pending), len(jobs) - next_submit)
    except BaseException as failure:
        # Stop active workers before joining their waiting threads. Queued jobs
        # that never started spend no candidate budget and must not be counted.
        cancel = (getattr(runner, "stop_all", None) if isinstance(failure, KeyboardInterrupt) else None)
        if cancel is None:
            cancel = getattr(runner, "cancel_all", None)
        if cancel is not None:
            try:
                cancel()
            except BaseException as exc:
                warnings.warn(f"HPO worker cancellation reported: {exc}", RuntimeWarning)
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        # Preserve completed/interrupted artifacts from every launched worker,
        # including later jobs that finished while an earlier result was pending.
        # Cleanup must not calibrate from a later job while an earlier reference
        # is unresolved. Ordered resume will score these preserved observations.
        for index, future in sorted(launched.items()):
            if index in finalized or future.cancelled():
                continue
            try:
                preserve(future.result())
            except BaseException as exc:
                warnings.warn(f"HPO job {index} cleanup reported: {exc}", RuntimeWarning)
        raise
    else:
        pool.shutdown(wait=True)
    return results
