"""Concurrent execution must preserve ordered scoring and stop owned work."""
from threading import Barrier, Event, Lock, get_ident

import pytest

from rl_matdesign.hpo.scheduling import run_jobs


def callbacks():
    return {"on_start": lambda *args: None, "on_result": lambda *args: None,
            "on_progress": lambda *args: None}


def test_pool_refills_while_first_job_waits_and_finalizes_on_coordinator():
    first_started, release_first = Event(), Event()
    lock = Lock()
    coordinator = get_ident()

    class Runner:
        active = peak = 0
        finalized = []

        def execute(self, index):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if index == 0:
                    first_started.set()
                    assert release_first.wait(5), "later completed slots were not refilled"
                else:
                    assert first_started.wait(5)
                    if index == 3:
                        release_first.set()
                return {"index": index}
            finally:
                with lock:
                    self.active -= 1

        def finalize(self, raw):
            assert get_ident() == coordinator
            self.finalized.append(raw["index"])
            return raw

    runner = Runner()
    rows = run_jobs([{"index": i} for i in range(5)], runner, 2, **callbacks())
    assert runner.peak == 2
    assert runner.active == 0
    assert runner.finalized == list(range(5))
    assert [row["index"] for row in rows] == list(range(5))


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_fatal_result_cancels_active_jobs_and_preserves_later_artifacts(failure):
    sibling_started, cancelled = Event(), Event()

    class Runner:
        started = []
        persisted = []

        def execute(self, index):
            self.started.append(index)
            if index == 0:
                assert sibling_started.wait(5)
            else:
                sibling_started.set()
                assert cancelled.wait(5), "worker was not cancelled before pool join"
            return {"index": index, "interrupted": index != 0}

        def finalize(self, raw):
            self.persisted.append(raw)
            return raw

        def cancel_all(self):
            cancelled.set()

    def result(index, row):
        if index == 0:
            raise failure("original failure")

    runner = Runner()
    options = callbacks()
    options["on_result"] = result
    with pytest.raises(failure, match="original failure"):
        run_jobs([{"index": i} for i in range(5)], runner, 2, **options)
    assert cancelled.is_set()
    assert sorted(runner.started) == [0, 1]
    assert runner.persisted == [{"index": 0, "interrupted": False},
                                {"index": 1, "interrupted": True}]


def test_serial_keeps_direct_runner_contract_and_input_order():
    calls = []

    def runner(index):
        calls.append(index)
        return index

    assert run_jobs([{"index": i} for i in range(3)], runner, 1, **callbacks()) == [0, 1, 2]
    assert calls == [0, 1, 2]


def test_cleanup_defers_scoring_and_does_not_replace_original_failure():
    sibling_started, cancelled = Event(), Event()

    class Runner:
        finalized = []
        preserved = []

        def execute(self, index):
            if index == 0:
                assert sibling_started.wait(5)
            else:
                sibling_started.set()
                assert cancelled.wait(5)
            return {"index": index}

        def finalize(self, raw):
            self.finalized.append(raw["index"])
            return raw

        def preserve(self, raw):
            self.preserved.append(raw["index"])
            raise SystemExit("secondary cleanup failure")

        def cancel_all(self):
            cancelled.set()

    def result(index, row):
        raise RuntimeError("primary scientific failure")

    runner = Runner()
    options = callbacks()
    options["on_result"] = result
    with pytest.warns(RuntimeWarning, match="secondary cleanup failure"):
        with pytest.raises(RuntimeError, match="primary scientific failure"):
            run_jobs([{"index": 0}, {"index": 1}], runner, 2, **options)
    assert runner.finalized == [0]
    assert runner.preserved == [1]


def test_failed_reference_cannot_be_replaced_by_cleanup_of_later_job(tmp_path, monkeypatch):
    from threading import Barrier
    from tests.test_hpo_v2_job_runner import FakeWorker, public_plan_path
    from rl_matdesign.hpo.artifacts import read_json
    from rl_matdesign.hpo.job_runner import JobRunner
    from rl_matdesign.hpo.spec import load_plan

    source = public_plan_path(tmp_path, threshold=0)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    worker.values = [0] * 8
    first = runner.execute({"pg_batch_eps": 2}, 8, 11, "train")
    worker.values = [-4, -3, -2, -1, 0, 1, 2, 3]
    later = runner.execute({"pg_batch_eps": 2}, 8, 29, "train")
    ready = Barrier(2, timeout=5)

    class PreparedRunner:
        finalize = staticmethod(runner.finalize)
        preserve = staticmethod(runner.preserve)
        cancel_all = staticmethod(runner.cancel_all)

        def execute(self, index):
            ready.wait()
            return [first, later][index]

    def result(index, row):
        assert index == 0 and row["status"] == "QUALITY_CALIBRATION_REQUIRED"
        raise RuntimeError("reference cannot calibrate")

    options = callbacks()
    options["on_result"] = result
    with pytest.raises(RuntimeError, match="reference cannot calibrate"):
        run_jobs([{"index": 0}, {"index": 1}], PreparedRunner(), 2, **options)
    assert not (runner.out / "quality_calibration.json").exists()
    pending = read_json(runner.out / "jobs" / later["job_id"] / "result.json")
    assert pending["status"] == "INCOMPLETE"
    assert not pending["eligible_for_selection"]
    assert pending["ledger"]["unique_evaluated"] == 8
    assert pending.get("quality_normalization", {}).get("scale") is None


@pytest.mark.parametrize("job_count", [0, 1, 3])
def test_parallel_limit_larger_than_ready_jobs_does_not_wait_or_pad(job_count):
    ready = Barrier(max(job_count, 1), timeout=5)
    lock = Lock()

    class Runner:
        active = peak = 0

        def execute(self, index):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                ready.wait()
                return index
            finally:
                with lock:
                    self.active -= 1

    runner = Runner()
    indices = list(range(job_count))
    rows = run_jobs([{"index": index} for index in indices], runner, 10, **callbacks())
    assert rows == indices
    assert runner.peak == job_count
    assert runner.active == 0
