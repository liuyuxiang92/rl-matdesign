"""Cooperative coordinator cancellation preserves work until worker checkpoints."""
from concurrent.futures import CancelledError
from threading import Event
import signal

import pytest

from rl_matdesign.hpo import cli
from rl_matdesign.hpo.job_runner import JobRunner
from rl_matdesign.hpo.scheduling import run_jobs
from rl_matdesign.hpo.spec import load_plan
from tests.test_hpo_v2_job_runner import public_plan_path


def test_cooperative_stop_signals_only_workers_and_waits_without_timeout(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path/'campaign')
    calls = []
    class Process:
        def __init__(self, index): self.index = index
        def send_signal(self, value): calls.append(('pid', self.index, value))
        def wait(self, timeout=None):
            assert timeout is None
            calls.append(('wait', self.index))
    events = [Event(), Event()]
    runner._processes = {Process(i): event for i, event in enumerate(events)}
    monkeypatch.setattr(runner, '_signal_process', lambda *a: pytest.fail('Predictor group was killed'))
    runner.stop_all()
    assert calls == [('pid', 0, signal.SIGTERM), ('pid', 1, signal.SIGTERM), ('wait', 0), ('wait', 1)]
    assert all(event.is_set() for event in events)
    with pytest.raises(CancelledError):runner.execute({}, 8, 1, 'train')


def test_second_interrupt_forces_cleanup_of_every_group(tmp_path, monkeypatch):
    runner = JobRunner(load_plan(public_plan_path(tmp_path)), tmp_path/'campaign')
    killed = []
    class Process:
        def __init__(self, index): self.index, self.waits = index, 0
        def send_signal(self, value): pass
        def wait(self, timeout=None):
            self.waits += 1
            if self.index == 0 and self.waits == 1:raise KeyboardInterrupt
    processes = [Process(0), Process(1)]
    runner._processes = {process: Event() for process in processes}
    monkeypatch.setattr(runner, '_signal_process', lambda process, sig:killed.append((process.index,sig)))
    runner.stop_all()
    assert killed == [(0,signal.SIGKILL),(1,signal.SIGKILL)]
    assert [p.waits for p in processes] == [2,1]


def test_scheduler_prefers_cooperative_stop_for_user_interrupt():
    from tests.test_hpo_parallel_scheduling import callbacks
    calls = []
    class Runner:
        def execute(self, **kwargs):raise KeyboardInterrupt
        def stop_all(self):calls.append('checkpoint stop')
        def cancel_all(self):pytest.fail('User interruption force-killed workers')
    with pytest.warns(RuntimeWarning, match='cleanup'):
        with pytest.raises(KeyboardInterrupt):
            run_jobs([{}],Runner(),2,**callbacks())
    assert calls == ['checkpoint stop']


def test_sigterm_routes_to_keyboard_interrupt_and_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    if previous != signal.SIG_DFL:pytest.skip('Embedding has a custom handler')
    with pytest.raises(KeyboardInterrupt):
        with cli._coordinator_termination(True):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_custom_sigterm_handler_is_preserved():
    previous = signal.getsignal(signal.SIGTERM)
    handler = lambda *args:None
    try:
        signal.signal(signal.SIGTERM,handler)
        with cli._coordinator_termination(True):
            assert signal.getsignal(signal.SIGTERM) is handler
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGTERM,previous)


@pytest.mark.parametrize('status,returncode', [
    ('COMPLETED', 0), ('INFEASIBLE_EFFICIENCY', 1), ('INFEASIBLE_NUMERICAL', 1),
])
def test_serial_interrupt_after_worker_completed_is_finalized_without_new_sampling(tmp_path, monkeypatch, status, returncode):
    import subprocess
    from rl_matdesign.hpo.artifacts import read_json
    from rl_matdesign.hpo.restart import _inspect_job
    from tests.test_hpo_v2_job_runner import FakeWorker
    plan = load_plan(public_plan_path(tmp_path))
    out = tmp_path/'campaign'
    runner = JobRunner(plan, out)
    worker = FakeWorker(monkeypatch)
    worker.ledger_updates = dict(phase='train', ratio_limit=1.5, max_attempts=11,
                                pending=0, failed_attempts=0, preparation_only_attempts=0,
                                status=status)
    if status != 'COMPLETED':
        worker.ledger_updates.update(unique_evaluated=7, attempts_started=11)
        worker.archive_transform = lambda rows: rows[:7]
    original = subprocess.Popen
    class InterruptedWait(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.first = True
            self.returncode = returncode
        def send_signal(self,sig):pass  # Worker finished while Ctrl+C arrived.
        def wait(self,timeout=None):
            if self.first:
                self.first = False
                raise KeyboardInterrupt
            return self.returncode
    monkeypatch.setattr('rl_matdesign.hpo.job_runner.subprocess.Popen',InterruptedWait)
    with pytest.raises(KeyboardInterrupt):runner({'pg_batch_eps':2},8,11,'train')
    directory = worker.calls[0]['directory']
    saved = read_json(directory/'result.json')
    assert saved['returncode'] == returncode
    assert saved['coordinator_interruption']['type'] == 'KeyboardInterrupt'
    assert 'execution_failure' not in saved
    assert _inspect_job(directory,'a2c')['classification'] == 'RECOVERABLE_TERMINAL'
    before = (directory/'candidate_budget.json').read_bytes()
    result = JobRunner(load_plan(plan['source_path']),out)({'pg_batch_eps':2},8,11,'train')
    assert result['status'] == status
    assert len(worker.calls) == 1
    assert (directory/'candidate_budget.json').read_bytes() == before
    assert _inspect_job(directory,'a2c')['classification'] == 'CACHED'
