"""Exercise real worker processes without requiring CUDA or DeepMD.

The temporary torch module is visible only to child interpreters; the test
process keeps its own imports unchanged. Builders return ASE cells just as the
real MD/optimization plugin does, but coordinate through files instead of GPUs.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")


_FAKE_TORCH = '''
import os
IMPORT_ENV = dict(os.environ)
if os.environ.get("FAKE_CUDA_TRACE"):
    with open(os.environ["FAKE_CUDA_TRACE"], "a") as trace:
        trace.write(str(os.getpid()) + "\\n")
class _Cuda:
    @staticmethod
    def is_available():
        return os.environ.get("FAKE_CUDA_AVAILABLE", "1") == "1"
    @staticmethod
    def device_count():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        return int(os.environ.get("FAKE_CUDA_COUNT", str(len(visible.split(",")))))
    @staticmethod
    def set_device(device):
        assert device == 0
cuda = _Cuda()
def set_num_threads(count):
    assert count > 0
    global THREADS
    THREADS = count
'''


_FAKE_BUILDER = '''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from ase import Atoms

class UnpickleableCalculator:
    def __getstate__(self):
        raise RuntimeError("a GPU calculator must never cross the worker boundary")

class Builder:
    def __init__(self, cfg, *, seed=None):
        self.cfg = cfg
        if cfg.get("test_preload_marker"):
            loaded = json.loads(Path(cfg["test_preload_marker"]).read_text())
            assert set(loaded) == {"delta", "pure"}, "parent models must load before builder"
        self.calls = 0
        self.root = Path(cfg["test_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.rng = np.random.default_rng(seed)
        with (self.root / "initializations.jsonl").open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "device": self.device}) + "\\n")
        if cfg.get("test_init_mode") == "hang":
            while True:
                time.sleep(.1)

    def composition_formula(self, candidate):
        return "H2"

    def build(self, candidate, *, n_configs=1, rng=None):
        assert n_configs == 1, "one MD+optimization task per dispatched configuration"
        self.calls += 1
        with (self.root / "build_calls.jsonl").open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "device": self.device,
                                  "call": self.calls}) + "\\n")
        (self.root / ("started-" + str(os.getpid()))).touch()
        mode = candidate.get("mode", self.cfg.get("test_mode", "normal"))
        if mode == "crash":
            os._exit(17)
        if mode == "barrier":
            # Serial execution cannot pass this rendezvous, regardless of speed.
            deadline = time.monotonic() + 8
            while len(list(self.root.glob("started-*"))) < 2:
                if time.monotonic() > deadline:
                    raise RuntimeError("second worker never reached the barrier")
                time.sleep(.01)
        if mode == "all_fail" or (mode == "one_fail" and self.device == "3"):
            raise RuntimeError("simulated MD failure on GPU " + self.device)
        if mode == "empty":
            return []
        if mode == "pair_fail":
            marker = self.root / ("active-" + self.device + "-" + str(os.getpid()))
            marker.touch()
            try:
                time.sleep(.15)
                if len(list(self.root.glob("active-" + self.device + "-*"))) > 1:
                    raise RuntimeError("simulated GPU OOM during concurrent pair")
            finally:
                marker.unlink()
        if mode == "hang":
            child_script = (
                "import os, sys, time\\n"
                "from pathlib import Path\\n"
                "root = Path(sys.argv[1])\\n"
                "(root / 'grandchild.pid').write_text(str(os.getpid()))\\n"
                "while True:\\n"
                "    (root / 'heartbeat').write_text(str(time.monotonic()))\\n"
                "    time.sleep(.03)\\n"
            )
            subprocess.Popen([sys.executable, "-c", child_script, str(self.root)])
            while True:
                time.sleep(.1)
        time.sleep(float(self.cfg.get("test_delays", {}).get(self.device, 0)))
        rng = self.rng if rng is None else rng
        value = int(rng.integers(0, 2**31))
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .75]])
        if mode == "internal_relax_oom":
            from ase.calculators.calculator import Calculator
            from rl_matdesign.utils.structure import relax_structure
            class OOMCalculator(Calculator):
                implemented_properties = ["energy", "forces"]
                def calculate(self, *args, **kwargs):
                    raise RuntimeError("CUDA out of memory in external builder optimization")
            # External plugins historically call this without strict kwargs.
            atoms = relax_structure(atoms, calc=OOMCalculator(), relax_cell=False, steps=1)
        import torch
        keys = ["CUDA_VISIBLE_DEVICES", "LOCAL_RANK", "DEVICE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
        atoms.info.update(
            pid=os.getpid(), device=self.device, call=self.calls, draw=value,
            env={key: os.environ.get(key) for key in keys},
            import_env={key: torch.IMPORT_ENV.get(key) for key in keys},
            torch_threads=torch.THREADS,
        )
        if mode == "calculator":
            atoms.calc = UnpickleableCalculator()
        return [atoms]
'''


_FAKE_DP = '''
import json
import os
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

class DP(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, head=None, **kwargs):
        super().__init__(**kwargs)
        with Path(model).open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "head": head,
                                  "device": os.environ["CUDA_VISIBLE_DEVICES"]}) + "\\n")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3)),
                        "stress": np.zeros(6)}
'''


@pytest.fixture
def pool_factory(tmp_path, monkeypatch):
    # Predictor reward folding imports training.py lazily. Keep the genuine
    # module cached in the parent before exposing the child-only shim path.
    pytest.importorskip("torch")
    modules = tmp_path / "fake_modules"
    modules.mkdir()
    (modules / "torch.py").write_text(_FAKE_TORCH)
    (modules / "parallel_test_builder.py").write_text(_FAKE_BUILDER)
    (modules / "deepmd").mkdir()
    (modules / "deepmd" / "__init__.py").write_text("")
    (modules / "deepmd" / "calculator.py").write_text(_FAKE_DP)
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("LOCAL_RANK", "6")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "19")
    monkeypatch.delenv("FAKE_CUDA_AVAILABLE", raising=False)
    monkeypatch.delenv("FAKE_CUDA_COUNT", raising=False)
    monkeypatch.delenv("FAKE_CUDA_TRACE", raising=False)
    pools = []

    def make(*, cfg_extra=None, parallel_extra=None, seed=2026):
        from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder

        cfg = {
            "test_root": str(tmp_path / ("run-" + str(len(pools)))),
            "structure_parallel": {
                "devices": [0, 1],
                "workers_per_device": 1,
                "threads_per_worker": 1,
                "timeout_s": 15,
                "startup_timeout_s": 15,
                "min_successful_configs": 1,
                **(parallel_extra or {}),
            },
            **(cfg_extra or {}),
        }
        pool = ParallelStructureBuilder("parallel_test_builder:Builder", cfg, seed=seed)
        pools.append(pool)
        return pool, Path(cfg["test_root"])

    yield make
    for pool in reversed(pools):
        pool.close()


def _initializations(root):
    path = root / "initializations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_defect_formula_survives_external_wrapper_and_worker_ipc(pool_factory, tmp_path):
    """Match the MD plugin's forwarding contract without running any physics."""
    from tests.test_defect_site_formula import _candidate, _parse, _template
    from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder

    # pool_factory installs the child-only CUDA shim and module search path.
    module = tmp_path / "fake_modules" / "formula_forwarding_builder.py"
    module.write_text('''
from rl_matdesign.predictors.builders.defect_site import DefectSiteBuilder

class Builder:
    def __init__(self, cfg, *, seed=None):
        self._inner = DefectSiteBuilder(cfg, seed=seed)

    def build(self, *args, **kwargs):
        raise AssertionError("formula reporting must never run MD or optimization")

    def composition_formula(self, candidate):
        return self._inner.composition_formula(candidate)
''')
    pool = ParallelStructureBuilder("formula_forwarding_builder:Builder", {
        "base_poscar": _template(tmp_path),
        "structure_parallel": {"devices": [0], "workers_per_device": 1,
                               "startup_timeout_s": 15, "timeout_s": 15},
    })
    try:
        formula = pool.composition_formula(_candidate(
            {"Sr": .5, "Ba": .5}, {"Fe": .75, "Co": .25}))
        assert _parse(formula) == {"Sr": .5, "Ba": .5, "Fe": .75, "Co": .25, "O": 3}
        pids = [worker.process.pid for worker in pool._workers.values()]
    finally:
        pool.close()
    _wait_stopped(pids)


def _process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed grandchild may briefly remain as an adopted zombie on Linux.
    # Avoid relying on `ps`, which is blocked in some desktop sandboxes.
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return True


def _wait_stopped(pids, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_process_running(pid) for pid in pids):
            return
        time.sleep(.05)
    assert not [pid for pid in pids if _process_running(pid)], "worker process leaked"


def test_two_workers_run_concurrently_with_isolated_gpu_environment(pool_factory):
    parent_env = dict(os.environ)
    pool, _ = pool_factory()
    atoms = pool.build({"mode": "barrier"}, n_configs=2, rng=np.random.default_rng(9))

    assert len(atoms) == 2
    assert len({cell.info["pid"] for cell in atoms}) == 2
    assert {cell.info["device"] for cell in atoms} == {"3", "7"}
    for cell in atoms:
        env = cell.info["env"]
        assert env["LOCAL_RANK"] == "0"
        assert env["DEVICE"] == "cuda"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            assert env[key] == "1"
        assert cell.info["import_env"] == env  # Binding happened before torch import.
        assert cell.info["torch_threads"] == 1
    assert dict(os.environ) == parent_env


def test_workers_reuse_builder_across_candidates_and_close_cleanly(pool_factory):
    pool, root = pool_factory()
    first = pool.build({}, n_configs=5)
    second = pool.build({}, n_configs=5)
    initializations = _initializations(root)
    pids = {row["pid"] for row in initializations}
    assert len(initializations) == len(pids) == 2
    assert {cell.info["pid"] for cell in first} == pids
    assert {cell.info["pid"] for cell in second} == pids
    for pid in pids:
        calls = [cell.info["call"] for cell in first + second if cell.info["pid"] == pid]
        assert sorted(calls) == list(range(1, len(calls) + 1))
    assert pool.composition_formula({}) == "H2"
    pool.close()
    pool.close()  # Idempotent cleanup also runs in fixture finalization.
    _wait_stopped(pids)


@pytest.mark.parametrize("device_count", [2, 4])
def test_two_workers_per_device_fills_slots_without_calibration(
    pool_factory, monkeypatch, device_count,
):
    physical = ["3", "7", "9", "11"][:device_count]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(physical))
    pool, root = pool_factory(parallel_extra={
        "devices": list(range(device_count)), "workers_per_device": 2,
    })
    atoms = pool.build({"mode": "barrier"}, n_configs=2 * device_count)
    pids_by_device = {}
    for cell in atoms:
        pids_by_device.setdefault(cell.info["device"], set()).add(cell.info["pid"])
    assert {device: len(pids) for device, pids in pids_by_device.items()} == {
        device: 2 for device in physical
    }
    calls = [json.loads(line) for line in (root / "build_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2 * device_count


def test_auto_discovers_visible_devices_and_reserves_logical_gpu(pool_factory):
    pool, _ = pool_factory(parallel_extra={"devices": "auto", "reserve_devices": [0]})
    atoms = pool.build({}, n_configs=2)
    assert {cell.info["device"] for cell in atoms} == {"7"}
    assert len({cell.info["pid"] for cell in atoms}) == 1


def test_auto_uses_visible_allocation_when_no_devices_reserved(pool_factory):
    pool, _ = pool_factory(parallel_extra={"devices": "auto"})
    atoms = pool.build({"mode": "barrier"}, n_configs=2)
    assert {cell.info["device"] for cell in atoms} == {"3", "7"}


def test_auto_skips_calibration_when_one_job_per_gpu_is_enough(pool_factory):
    pool, root = pool_factory(parallel_extra={"workers_per_device": "auto"})
    atoms = pool.build({}, n_configs=2)
    assert len(atoms) == 2
    assert pool.autotune_report is None
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("reserve", [[0, 1], [2], [0, 0], [-1]])
def test_invalid_auto_reservations_rejected(pool_factory, reserve):
    with pytest.raises(ValueError):
        pool, _ = pool_factory(parallel_extra={"devices": "auto", "reserve_devices": reserve})
        pool.build({}, n_configs=1)


def test_manual_devices_cannot_also_request_auto_reservation(pool_factory):
    with pytest.raises(ValueError):
        pool, _ = pool_factory(parallel_extra={"reserve_devices": [0]})
        pool.build({}, n_configs=1)


def test_auto_calibrates_actual_work_once_and_excludes_probe_results(pool_factory):
    auto, root = pool_factory(
        cfg_extra={"test_delays": {"3": .15}},
        parallel_extra={"devices": [0], "workers_per_device": "auto", "min_speedup": 1.15},
    )
    serial, _ = pool_factory(parallel_extra={"devices": [0], "workers_per_device": 1})
    first = auto.build({}, n_configs=5, rng=np.random.default_rng(191))
    reference = serial.build({}, n_configs=5, rng=np.random.default_rng(191))
    assert [cell.info["draw"] for cell in first] == [cell.info["draw"] for cell in reference]
    assert len({cell.info["pid"] for cell in first}) == 2
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 10
    second = auto.build({}, n_configs=5)
    assert len(second) == 5
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == 15


def test_auto_pair_failure_falls_back_to_one_worker(pool_factory):
    pool, root = pool_factory(parallel_extra={"devices": [0], "workers_per_device": "auto"})
    atoms = pool.build({"mode": "pair_fail"}, n_configs=5)
    assert len(atoms) == 5
    assert len({cell.info["pid"] for cell in atoms}) == 1
    initializations = _initializations(root)
    assert len(initializations) == 2
    retired_pids = {row["pid"] for row in initializations} - {atoms[0].info["pid"]}
    _wait_stopped(retired_pids)


def test_auto_retires_calibrated_slot_not_needed_by_current_candidate(pool_factory):
    pool, root = pool_factory(
        cfg_extra={"test_delays": {"3": .15, "7": .15}},
        parallel_extra={"workers_per_device": "auto"},
    )
    atoms = pool.build({}, n_configs=3)
    assert len(atoms) == 3
    assert pool.autotune_report["counts"] == [2, 2]
    production_pids = {cell.info["pid"] for cell in atoms}
    assert len(production_pids) == 3
    initial_pids = {record["pid"] for record in _initializations(root)}
    assert len(initial_pids) == 4
    _wait_stopped(initial_pids - production_pids)
    assert all(_process_running(pid) for pid in production_pids)


def test_internal_plugin_optimization_error_is_not_silently_scored(pool_factory):
    pool, _ = pool_factory()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        pool.build({"mode": "internal_relax_oom"}, n_configs=2)


def test_serial_relaxation_keeps_legacy_exception_behavior():
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from rl_matdesign.utils.structure import relax_structure

    class FailedCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *args, **kwargs):
            raise RuntimeError("simulated serial optimizer failure")

    atoms = relax_structure(Atoms("H2"), calc=FailedCalculator(), relax_cell=False, steps=1)
    assert len(atoms) == 2


def test_rng_and_result_order_do_not_follow_completion_order(pool_factory):
    first, _ = pool_factory(cfg_extra={"test_delays": {"3": .1}})
    second, _ = pool_factory(cfg_extra={"test_delays": {"7": .1}})
    first_atoms = first.build({}, n_configs=5, rng=np.random.default_rng(77))
    second_atoms = second.build({}, n_configs=5, rng=np.random.default_rng(77))
    first_draws = [cell.info["draw"] for cell in first_atoms]
    second_draws = [cell.info["draw"] for cell in second_atoms]
    assert first_draws == second_draws
    assert len(set(first_draws)) == 5


def test_attached_calculator_is_not_serialized(pool_factory):
    pool, _ = pool_factory()
    atoms = pool.build({"mode": "calculator"}, n_configs=2)
    assert len(atoms) == 2
    assert all(cell.calc is None for cell in atoms)


def test_optional_relaxation_runs_in_workers_and_reuses_calculators(pool_factory, tmp_path):
    calculator_log = tmp_path / "calculator_initializations.jsonl"
    pool, _ = pool_factory(cfg_extra={"geo_opt": {
        "model": str(calculator_log), "head": "perovskite", "relax_cell": False,
        "fmax": .01, "steps": 2,
    }})
    assert pool.handles_geo_opt is True
    first = pool.build({}, n_configs=2)
    second = pool.build({}, n_configs=2)
    records = [json.loads(line) for line in calculator_log.read_text().splitlines()]
    assert len(records) == 2
    assert {record["device"] for record in records} == {"3", "7"}
    assert {record["head"] for record in records} == {"perovskite"}
    assert {record["pid"] for record in records} == {cell.info["pid"] for cell in first}
    assert all(cell.calc is None for cell in first + second)


def test_partial_failure_keeps_successful_configurations(pool_factory):
    pool, _ = pool_factory()
    atoms = pool.build({"mode": "one_fail"}, n_configs=2)
    assert len(atoms) == 1
    assert atoms[0].info["device"] == "7"
    # A failed candidate must not poison subsequent uses of its worker.
    assert len(pool.build({}, n_configs=2)) == 2


@pytest.mark.parametrize("mode", ["all_fail", "empty"])
def test_no_successful_configuration_raises(pool_factory, mode):
    pool, _ = pool_factory()
    with pytest.raises(RuntimeError):
        pool.build({"mode": mode}, n_configs=2)


def test_minimum_successful_configurations_is_enforced(pool_factory):
    pool, _ = pool_factory(parallel_extra={"min_successful_configs": 2})
    with pytest.raises(RuntimeError):
        pool.build({"mode": "one_fail"}, n_configs=2)


@pytest.mark.skipif(os.name != "posix", reason="worker process-group cleanup is POSIX")
def test_timeout_stops_worker_and_its_md_subprocess(pool_factory):
    pool, root = pool_factory(parallel_extra={"devices": [0], "timeout_s": 2})
    with pytest.raises((RuntimeError, TimeoutError)):
        pool.build({"mode": "hang"}, n_configs=1)
    child_path = root / "grandchild.pid"
    assert child_path.exists(), "the test must reach the nested MD process"
    pids = [row["pid"] for row in _initializations(root)]
    pids.append(int(child_path.read_text()))
    pool.close()
    _wait_stopped(pids)
    heartbeat = (root / "heartbeat").read_text()
    time.sleep(.1)
    assert (root / "heartbeat").read_text() == heartbeat


def test_startup_timeout_cleans_up_worker(pool_factory):
    pool, root = pool_factory(
        cfg_extra={"test_init_mode": "hang"},
        parallel_extra={"devices": [0], "startup_timeout_s": 2},
    )
    with pytest.raises((RuntimeError, TimeoutError)):
        pool.build({}, n_configs=1)
    pids = [row["pid"] for row in _initializations(root)]
    pool.close()
    _wait_stopped(pids)


def test_abrupt_worker_exit_cleans_pool_and_next_build_recovers(pool_factory):
    pool, root = pool_factory()
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        pool.build({"mode": "crash"}, n_configs=2)
    old_pids = {row["pid"] for row in _initializations(root)}
    _wait_stopped(old_pids)
    atoms = pool.build({}, n_configs=2)
    assert len(atoms) == 2
    assert not old_pids.intersection(cell.info["pid"] for cell in atoms)


@pytest.mark.parametrize("devices", [[], [0, 0], [-1], [True], [2]])
def test_invalid_gpu_device_selection_rejected(pool_factory, devices):
    with pytest.raises(ValueError):
        pool_factory(parallel_extra={"devices": devices})


@pytest.mark.parametrize("visible", ["", "-1", "3,3"])
def test_empty_or_duplicate_gpu_allocation_rejected(pool_factory, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        pool_factory()


def test_minimum_cannot_exceed_requested_configurations(pool_factory):
    pool, _ = pool_factory(parallel_extra={"min_successful_configs": 2})
    with pytest.raises(ValueError):
        pool.build({}, n_configs=1)


@pytest.mark.parametrize("setting,value", [
    ("threads_per_worker", 0),
    ("timeout_s", 0),
    ("startup_timeout_s", 0),
    ("min_successful_configs", 0),
])
def test_invalid_parallel_limits_rejected(pool_factory, setting, value):
    with pytest.raises(ValueError):
        pool_factory(parallel_extra={setting: value})


@pytest.mark.parametrize("key,value", [
    ("CUDA_VISIBLE_DEVICES", "7"), ("LOCAL_RANK", "1"), ("DEVICE", "cpu"),
])
def test_md_environment_cannot_override_gpu_isolation(pool_factory, key, value):
    with pytest.raises(ValueError):
        pool_factory(cfg_extra={"md": {"env": {key: value}}})


def test_worker_cuda_preflight_fails_before_expensive_builder(
    pool_factory, monkeypatch, tmp_path,
):
    monkeypatch.setenv("FAKE_CUDA_AVAILABLE", "0")
    trace = tmp_path / "cuda_imports.txt"
    monkeypatch.setenv("FAKE_CUDA_TRACE", str(trace))
    pool, root = pool_factory()
    with pytest.raises(RuntimeError):
        pool.build({}, n_configs=2)
    assert not (root / "initializations.jsonl").exists()
    pids = [int(line) for line in trace.read_text().splitlines()]
    assert pids
    pool.close()
    _wait_stopped(pids)


@pytest.mark.parametrize("share_structure", [True, False])
def test_predictor_builds_parallel_then_scores_gated_properties_and_closes(
    pool_factory, tmp_path, monkeypatch, share_structure,
):
    from rl_matdesign.predictors.structure_score import StructureScorePredictor
    import rl_matdesign.utils.dp_eval as dp_eval

    root = tmp_path / "predictor_builds"
    calculator_log = tmp_path / "predictor_calculators.jsonl"
    preload_marker = tmp_path / "preloaded_models.json"
    cfg = {
        "builder": "parallel_test_builder:Builder",
        "share_structure": share_structure,
        "test_root": str(root),
        "test_preload_marker": str(preload_marker),
        "n_random_configs": 5,
        "structure_parallel": {"devices": [0, 1], "workers_per_device": 1, "timeout_s": 15},
        "geo_opt": {"model": str(calculator_log), "relax_cell": False, "steps": 2},
        "properties": [
            {"name": "delta", "predictor": "dp_property", "models": ["delta.pt"],
             "direction": "max", "objective": "mean", "weight": 1,
             "test_root": str(root / "delta"), "test_preload_marker": str(preload_marker)},
            {"name": "pure", "predictor": "dp_classification", "models": ["pure.pt"],
             "direction": "max", "objective": "mean", "weight": 0, "output_index": 1,
             "test_root": str(root / "pure"), "test_preload_marker": str(preload_marker)},
        ],
        "reward": {
            "type": "gated",
            "gates": [{"name": "synthesizable", "score": {"terms": {"pure": 1}},
                       "min": .8}],
            "failure": {"reward": -10},
        },
    }
    seen = {"delta": [], "pure": []}

    def evaluate(name, value):
        def run(atoms, *args, **kwargs):
            seen[name].append(atoms)
            return np.full((1, len(atoms)), value)
        return run

    monkeypatch.setattr(dp_eval, "eval_property_ensemble", evaluate("delta", 5.0))
    monkeypatch.setattr(dp_eval, "eval_classification_ensemble", evaluate("pure", .9))
    predictor = StructureScorePredictor(cfg, seed=13)
    loaded = set()

    def load_models(prop):
        loaded.add(prop["name"])
        preload_marker.write_text(json.dumps(sorted(loaded)))
        return [object()], {"H": 0}

    monkeypatch.setattr(predictor, "_get_prop_models", load_models)

    def forbidden_parent_relax(atoms):
        raise AssertionError("parallel structures must not be optimized again in parent")

    monkeypatch.setattr(predictor, "_relax", forbidden_parent_relax)
    pids = set()
    try:
        candidate = {"A": {"H": 1.0}}
        assert predictor.predict(candidate) == pytest.approx((5.0, 0.0))
        assert predictor.predict(candidate) == pytest.approx((5.0, 0.0))
        assert predictor.composition_formula(candidate) == "H2"
        assert len(seen["delta"]) == len(seen["pure"]) == 1
        delta_atoms, pure_atoms = seen["delta"][0], seen["pure"][0]
        assert len(delta_atoms) == len(pure_atoms) == 5
        if share_structure:
            assert delta_atoms is pure_atoms
            assert len(_initializations(root)) == 2
        else:
            assert delta_atoms is not pure_atoms
            assert len(_initializations(root / "delta")) == 2
            assert len(_initializations(root / "pure")) == 2
        pids = {cell.info["pid"] for cell in delta_atoms + pure_atoms}
        assert len(pids) == (2 if share_structure else 4)
        records = [json.loads(line) for line in calculator_log.read_text().splitlines()]
        assert len(records) == len(pids)
    finally:
        predictor.close()
    _wait_stopped(pids)


@pytest.mark.parametrize("geo_opt_enabled", [False, True])
def test_absent_parallel_config_preserves_serial_batch_and_relaxation(
    monkeypatch, geo_opt_enabled,
):
    from ase import Atoms
    import rl_matdesign.registry as registry
    from rl_matdesign.predictors.structure_score import StructureScorePredictor

    calls = []
    relaxations = []

    class SerialBuilder:
        def build(self, candidate, *, n_configs, rng):
            calls.append(n_configs)
            return [Atoms("H") for _ in range(n_configs)]

    builder = SerialBuilder()
    monkeypatch.setattr(registry, "resolve_builder", lambda *args, **kwargs: builder)
    predictor = StructureScorePredictor({
        "n_random_configs": 5,
        "geo_opt": {"enabled": geo_opt_enabled, "model": "unused.pt"},
        "properties": [{"name": "delta", "predictor": "dp_property", "models": ["m.pt"],
                        "direction": "max", "objective": "mean"}],
    })

    def relax(atoms):
        relaxations.append(atoms)
        return atoms

    monkeypatch.setattr(predictor, "_relax", relax)
    monkeypatch.setattr(predictor, "_score", lambda prop, atoms: (float(len(atoms)), 0.0))
    assert predictor.predict({"H": 1.0}) == (5.0, 0.0)
    assert calls == [5]
    assert len(relaxations) == (5 if geo_opt_enabled else 0)
    predictor.close()


@pytest.mark.skipif(os.name != "posix", reason="tests POSIX SIGTERM and process groups")
def test_experiment_sigterm_cleans_worker_and_running_md_child(pool_factory, tmp_path):
    """Exercise the CLI signal wrapper around a real pool, not a mocked close."""
    module_dir = next(
        path for path in sys.path if (Path(path) / "parallel_test_builder.py").is_file()
    )
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    root = tmp_path / "terminated_experiment"
    runner = tmp_path / "signal_runner.py"
    runner.write_text('''
import sys
from pathlib import Path
from rl_matdesign import experiment
from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder

# experiment imported the real parent torch before the child shim path is added.
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[2])
def run(resources):
    pool = ParallelStructureBuilder("parallel_test_builder:Builder", {
        "test_root": str(root),
        "structure_parallel": {"devices": [0], "workers_per_device": 1,
                               "timeout_s": 30, "startup_timeout_s": 15},
    })
    resources.callback(pool.close)
    pool.build({"mode": "hang"}, n_configs=1)
experiment._run_experiment = run
experiment.main()
''')
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
    log_path = tmp_path / "signal_runner.log"
    worker_pids = []
    with log_path.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, str(runner), module_dir, str(root)], env=env,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            child_path = root / "grandchild.pid"
            while not child_path.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                time.sleep(.05)
            assert child_path.exists(), log_path.read_text()
            worker_pids = [row["pid"] for row in _initializations(root)]
            grandchild_pid = int(child_path.read_text())
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 128 + signal.SIGTERM, log_path.read_text()
            _wait_stopped(worker_pids + [grandchild_pid])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            # If an assertion fails, the test still owns and reaps its workers.
            for pid in worker_pids:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
