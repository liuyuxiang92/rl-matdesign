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
        if cfg.get("test_no_sample_protocol"):
            self.build_sample = None
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

    def sample_plan(self, candidate, *, n_configs, source_spec, eval_seed):
        from rl_matdesign.utils.structure_samples import sample_plan
        if candidate.get("mode") == "plan_fail":
            raise ValueError("sample plan failed for candidate")
        templates = self.cfg.get("test_templates")
        return sample_plan(candidate, n_configs=n_configs, source_spec=source_spec,
                           eval_seed=eval_seed, templates=templates)

    def build_sample(self, candidate, *, sample):
        structures = self.build(candidate, n_configs=1, rng=np.random.default_rng(sample.seed))
        if len(structures) != 1:
            return structures  # Deliberately malformed capability for contract test.
        atoms = structures[0]
        atoms.positions[1, 0] = float(np.random.default_rng(sample.seed).random())
        atoms.info["canonical_keys"] = list(candidate)
        return atoms

    def build(self, candidate, *, n_configs=1, rng=None):
        if not self.cfg.get("test_track_execution"):
            return self._build(candidate, n_configs=n_configs, rng=rng)
        lock = self.root / ("busy-" + self.device)
        lock.touch(exist_ok=False)  # A second task on this GPU must fail the test.
        def record(event):
            with (self.root / "execution_events.jsonl").open("a") as out:
                out.write(json.dumps({"event": event, "device": self.device,
                                      "pid": os.getpid(), "candidate_id": candidate.get("id")}) + "\\n")
        record("start")
        try:
            return self._build(candidate, n_configs=n_configs, rng=rng)
        finally:
            record("end")
            lock.unlink()

    def _build(self, candidate, *, n_configs=1, rng=None):
        assert n_configs == 1, "one MD+optimization task per dispatched configuration"
        self.calls += 1
        with (self.root / "build_calls.jsonl").open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "device": self.device,
                                  "call": self.calls,
                                  "candidate_id": candidate.get("id")}) + "\\n")
        (self.root / ("started-" + str(os.getpid()))).touch()
        mode = candidate.get("mode", self.cfg.get("test_mode", "normal"))
        if mode == "crash":
            os._exit(17)
        if mode == "barrier":
            # Serial execution cannot pass this rendezvous, regardless of speed.
            deadline = time.monotonic() + 8
            while len(list(self.root.glob("started-*"))) < candidate.get("barrier_count", 2):
                if time.monotonic() > deadline:
                    raise RuntimeError("second worker never reached the barrier")
                time.sleep(.01)
        if mode == "all_fail" or (mode == "one_fail" and self.device == "3"):
            raise RuntimeError("simulated MD failure on GPU " + self.device)
        if mode == "empty":
            return []
        if candidate.get("release"):
            (self.root / candidate["release"]).touch()
        if candidate.get("wait_for") and self.device == candidate.get("wait_for_device"):
            deadline = time.monotonic() + 8
            while not (self.root / candidate["wait_for"]).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("free GPU did not receive the next queued configuration")
                time.sleep(.01)
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
            candidate_id=candidate.get("id"),
            env={key: os.environ.get(key) for key in keys},
            import_env={key: getattr(torch, "IMPORT_ENV", {}).get(key) for key in keys},
            torch_threads=getattr(torch, "THREADS", None),
        )
        if mode == "calculator":
            atoms.calc = UnpickleableCalculator()
        return [atoms, atoms.copy()] if self.cfg.get("test_expanded") else [atoms]
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
        # Keep the model asset immutable, as a real calculator does. Diagnostics
        # belong in a separate sidecar, never in the checkpoint being validated.
        with Path(model).with_suffix(".jsonl").open("a") as out:
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
                "threads_per_worker": 1,
                "timeout_s": 15,
                "startup_timeout_s": 15,
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


def test_pipeline_formula_survives_external_wrapper_and_worker_ipc(pool_factory, tmp_path):
    """Formula planning crosses the generic FQN worker boundary without physics."""
    from ase import Atoms
    from pymatgen.core import Composition
    from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder
    path = tmp_path / "perovskite.vasp"
    Atoms(["Sr"]*4+["Fe"]*4+["O"]*12, positions=np.arange(60).reshape(-1,3),
          cell=[30,30,30], pbc=True).write(path, format="vasp")
    module = tmp_path / "fake_modules" / "formula_forwarding_builder.py"
    module.write_text("from rl_matdesign.predictors.builders.pipeline import PipelineBuilder\n"
        "class Builder:\n"
        "    def __init__(self, cfg, *, seed=None):\n"
        "        self._inner = PipelineBuilder({**cfg, 'builder':'pipeline'}, seed=seed)\n"
        "    def build(self, *args, **kwargs):\n"
        "        raise AssertionError('formula reporting must not run MD or optimization')\n"
        "    def composition_formula(self, candidate):\n"
        "        return self._inner.composition_formula(candidate)\n")
    pool = ParallelStructureBuilder("formula_forwarding_builder:Builder", {
        "base_poscar": str(path), "steps": [{"type":"substitute", "a_site_symbol":"Sr", "a_dopant_group":"A",
           "b_site_symbol":"Fe", "b_dopant_group":"B"}],
        "structure_parallel": {"devices":[0], "startup_timeout_s":15, "timeout_s":15}})
    pids=[]
    try:
        formula=pool.composition_formula({"A":{"Sr":.5,"Ba":.5}, "B":{"Fe":.75,"Co":.25}})
        assert Composition(formula).get_el_amt_dict() == {"Sr":2,"Ba":2,"Fe":3,"Co":1,"O":12}
        pids=[worker.process.pid for worker in pool._workers.values()]
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


def test_auto_discovers_visible_devices_and_reserves_logical_gpu(pool_factory):
    pool, _ = pool_factory(parallel_extra={"devices": "auto", "reserve_devices": [0]})
    atoms = pool.build({}, n_configs=2)
    assert {cell.info["device"] for cell in atoms} == {"7"}
    assert len({cell.info["pid"] for cell in atoms}) == 1


def test_auto_uses_visible_allocation_when_no_devices_reserved(pool_factory):
    pool, _ = pool_factory(parallel_extra={"devices": "auto"})
    atoms = pool.build({"mode": "barrier"}, n_configs=2)
    assert {cell.info["device"] for cell in atoms} == {"3", "7"}


@pytest.mark.parametrize("n_configs", [1, 2, 5])
def test_one_worker_per_gpu_runs_only_requested_configurations(pool_factory, n_configs):
    pool, root = pool_factory(cfg_extra={"structure_parallel": {"devices": [0, 1]}})
    atoms = pool.build({}, n_configs=n_configs)
    assert len(atoms) == n_configs
    assert len(_initializations(root)) == min(2, n_configs)
    assert len((root / "build_calls.jsonl").read_text().splitlines()) == n_configs
    assert len({(cell.info["device"], cell.info["pid"]) for cell in atoms}) == min(2, n_configs)


@pytest.mark.parametrize("setting,value", [
    ("workers_per_device", 1), ("workers_per_device", "auto"),
    ("workers_per_device", 2), ("min_speedup", 1.15),
    ("min_successful_configs", 1), ("min_successful_configs", 5),
])
def test_retired_parallel_options_rejected_before_gpu_discovery(pool_factory, monkeypatch, setting, value):
    import rl_matdesign.utils.parallel_structures as parallel
    monkeypatch.setattr(parallel, "_device_tokens", lambda *args: pytest.fail("GPU discovery must not run"))
    with pytest.raises(ValueError, match=f"Unknown structure_parallel options:.*{setting}"):
        pool_factory(parallel_extra={setting: value})


@pytest.mark.parametrize("reserve", [[0, 1], [2], [0, 0], [-1]])
def test_invalid_auto_reservations_rejected(pool_factory, reserve):
    with pytest.raises(ValueError):
        pool, _ = pool_factory(parallel_extra={"devices": "auto", "reserve_devices": reserve})
        pool.build({}, n_configs=1)


def test_manual_devices_cannot_also_request_auto_reservation(pool_factory):
    with pytest.raises(ValueError):
        pool, _ = pool_factory(parallel_extra={"reserve_devices": [0]})
        pool.build({}, n_configs=1)


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


def test_pipeline_relaxation_runs_in_workers_and_reuses_calculators(pool_factory, tmp_path):
    from ase import Atoms
    from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder
    calculator_log=tmp_path / "calculator_initializations.jsonl"
    model_path=calculator_log.with_suffix(".pt")
    model_path.write_bytes(b"fake DeepMD checkpoint")
    path=tmp_path / "POSCAR"
    Atoms("Ni2", positions=[[0,0,0],[2,2,2]], cell=[10,10,10],pbc=True).write(path,format="vasp")
    pool=ParallelStructureBuilder("pipeline", {"builder":"pipeline", "base_poscar":str(path),
        "steps":[{"type":"substitute", "a_site_symbol":"Ni", "a_dopant_group":"alloy"},
                 {"type":"geo_opt", "model":str(model_path), "head":"perovskite", "relax_cell":False,
                  "fmax":.01, "steps":2}],
        "structure_parallel":{"devices":[0,1], "timeout_s":15,"startup_timeout_s":15}})
    pids=[]
    try:
        candidate={"alloy":{"Ni":.5,"Co":.5}}
        first=pool.build(candidate,n_configs=2)
        second=pool.build(candidate,n_configs=2)
        records=[json.loads(line) for line in calculator_log.read_text().splitlines()]
        assert len(records)==2
        assert {row["device"] for row in records}=={"3","7"}
        assert {row["head"] for row in records}=={"perovskite"}
        pids=[worker.process.pid for worker in pool._workers.values()]
        assert {row["pid"] for row in records}==set(pids)
        assert all(cell.calc is None for cell in first+second)
    finally:
        pool.close()
    _wait_stopped(pids)




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


@pytest.mark.parametrize("setting,value", [
    ("threads_per_worker", 0),
    ("timeout_s", 0),
    ("startup_timeout_s", 0),
])
def test_invalid_parallel_limits_rejected(pool_factory, setting, value):
    with pytest.raises(ValueError):
        pool_factory(parallel_extra={setting: value})


@pytest.mark.parametrize("key,value", [
    ("CUDA_VISIBLE_DEVICES", "7"), ("LOCAL_RANK", "1"), ("DEVICE", "cpu"),
])
def test_md_environment_cannot_override_gpu_isolation(pool_factory, key, value):
    config = {"steps": [{"type": "md_average", "env": {key: value}}]}
    with pytest.raises(ValueError, match="GPU isolation"):
        pool_factory(cfg_extra=config)


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


def test_predictor_builds_parallel_then_scores_gated_properties_and_closes(pool_factory, tmp_path, monkeypatch):
    from rl_matdesign.predictors.structure_score import StructureScorePredictor
    import rl_matdesign.utils.dp_eval as dp_eval
    root=tmp_path / "predictor_builds"
    preload_marker=tmp_path / "preloaded_models.json"
    cfg={"structure":{"builder":"parallel_test_builder:Builder", "test_root":str(root),
            "test_preload_marker":str(preload_marker), "n_random_configs":5,
            "structure_parallel":{"devices":[0,1],"timeout_s":15}},
         "properties":[
            {"name":"delta","predictor":"dp_property","models":["delta.pt"],
             "direction":"max","objective":"mean"},
            {"name":"pure","predictor":"dp_classification","models":["pure.pt"],
             "direction":"max","objective":"mean","output_index":1}],
         "reward":{"type":"gated","objective":{"terms":{"delta":1}},"gates":[{"name":"synthesizable","score":{"terms":{"pure":1}},"min":.8}],
                   "failure":{"reward":-10}}}
    seen={"delta":[],"pure":[]}
    def evaluate(name,value):
        def run(atoms,*args,**kwargs):
            seen[name].append(atoms)
            return np.full((1,len(atoms)),value)
        return run
    monkeypatch.setattr(dp_eval,"eval_property_ensemble",evaluate("delta",5.))
    monkeypatch.setattr(dp_eval,"eval_classification_ensemble",evaluate("pure",.9))
    predictor=StructureScorePredictor(cfg,seed=13)
    loaded=set()
    def load_models(prop):
        loaded.add(prop["name"])
        preload_marker.write_text(json.dumps(sorted(loaded)))
        return [object()], {"H":0}
    monkeypatch.setattr(predictor,"_get_prop_models",load_models)
    pids=set()
    try:
        candidate={"A":{"H":1.}}
        assert predictor.predict(candidate)==(5.,None)
        assert predictor.predict(candidate)==(5.,None)
        assert predictor.composition_formula(candidate)=="H2"
        assert len(seen["delta"])==len(seen["pure"])==1
        delta,pure=seen["delta"][0],seen["pure"][0]
        assert len(delta)==len(pure)==5 and delta is pure
        assert len(_initializations(root))==2
        pids={cell.info["pid"] for cell in delta}
        assert len(pids)==2
    finally:
        predictor.close()
    _wait_stopped(pids)




def test_absent_parallel_config_preserves_native_serial_ensemble(monkeypatch):
    from ase import Atoms
    import rl_matdesign.registry as registry
    from rl_matdesign.predictors.structure_score import StructureScorePredictor
    calls=[]
    class SerialBuilder:
        def build(self,candidate,*,n_configs,rng):
            calls.append(n_configs)
            return [Atoms("H") for _ in range(n_configs)]
    monkeypatch.setattr(registry,"resolve_builder",lambda *args,**kwargs:SerialBuilder())
    predictor=StructureScorePredictor({"structure":{"builder":"external:Builder","n_random_configs":5},
        "properties":[{"name":"delta","predictor":"dp_property","models":["m.pt"],
                       "direction":"max","objective":"mean"}]})
    monkeypatch.setattr(predictor,"_score",lambda prop,atoms:(float(len(atoms)),0.))
    assert predictor.predict({"H":1.})==(5.,None)
    assert calls==[5]
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
        "structure_parallel": {"devices": [0],
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


def test_source_seeded_jobs_ignore_candidate_order_batching_and_worker_count(pool_factory):
    from rl_matdesign.structure_sources import StructureSpec
    from rl_matdesign.utils.structure_samples import SAMPLE_INFO_KEY

    source = StructureSpec({'builder': 'parallel_test_builder:Builder'})
    first, _ = pool_factory(cfg_extra={'test_delays': {'3': .02}})
    second, _ = pool_factory(parallel_extra={'devices': [0]})
    a, b = {'id': 'a', 'value': 1}, {'id': 'b', 'value': 2}
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=23)
    original = first.build_many_seeded([a, b], **kwargs)
    reordered = second.build_many_seeded([b, dict(reversed(list(a.items())))], **kwargs)
    scalar = first.build_seeded(a, **kwargs)

    def values(cells):
        return [(cell.info['draw'], cell.info[SAMPLE_INFO_KEY]) for cell in cells]
    assert values(original[0]) == values(reordered[1]) == values(scalar)
    assert values(original[1]) == values(reordered[0])
    for cell in original[0]:
        assert cell.info['draw'] == int(np.random.default_rng(cell.info[SAMPLE_INFO_KEY]['seed']).integers(0, 2**31))
    assert [record['status'] for record in first.last_sample_records[0]] == ['succeeded'] * 3


def test_pipeline_sample_plan_matches_real_serial_structures_on_one_and_two_workers(pool_factory):
    from contextlib import ExitStack
    from tests.test_seeded_pipeline import _builder_and_source, signature
    from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder

    # Install the child-only torch/CUDA shim, while building actual ASE structures.
    unused, root = pool_factory()
    root.mkdir(parents=True, exist_ok=True)
    builder, source = _builder_and_source(root)
    candidates = [{'alloy': {'Ni': .5, 'Co': .5}}, {'alloy': {'Co': .3, 'Ni': .7}}]
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=31)
    serial = builder.build_many_seeded(candidates, **kwargs)
    with ExitStack() as resources:
        for devices in ([0], [0, 1]):
            cfg = {**source.effective_config,
                   'structure_parallel': {'devices': devices, 'timeout_s': 15, 'startup_timeout_s': 15}}
            pool = ParallelStructureBuilder('pipeline', cfg)
            resources.callback(pool.close)
            parallel = pool.build_many_seeded(list(reversed(candidates)), **kwargs)
            assert signature(serial[0]) == signature(parallel[1])
            assert signature(serial[1]) == signature(parallel[0])
            assert [len(cells) for cells in parallel] == [5, 5]
            assert [record['template_id']['index'] for record in pool.last_sample_records[0]] == [0, 0, 1, 1, 1]


def test_seeded_partial_failure_preserves_failed_and_successful_sample_identity(pool_factory):
    from rl_matdesign.structure_sources import StructureSpec
    from rl_matdesign.utils.structure_samples import SAMPLE_INFO_KEY

    source = StructureSpec({'builder': 'parallel_test_builder:Builder'})
    pool, _ = pool_factory()
    cells = pool.build_seeded({'mode': 'one_fail'}, n_configs=2, source_spec=source, eval_seed=8)
    assert len(cells) == 1
    records = pool.last_sample_records[0]
    assert [record['status'] for record in records] == ['failed', 'succeeded']
    assert [record['sample_index'] for record in records] == [0, 1]
    assert all(record['source_id'] == 'structure' for record in records)
    assert 'simulated MD failure' in records[0]['error']
    assert cells[0].info[SAMPLE_INFO_KEY]['sample_index'] == 1


def test_external_sample_protocol_rejects_expanded_result(pool_factory):
    from rl_matdesign.structure_sources import StructureSpec

    source = StructureSpec({'builder': 'parallel_test_builder:Builder'})
    pool, _ = pool_factory(cfg_extra={'test_expanded': True})
    with pytest.raises(ValueError, match='exactly one ASE Atoms'):
        pool.build_seeded({'id': 'a'}, n_configs=2, source_spec=source, eval_seed=8)
    assert not pool._workers



def test_seeded_external_without_capability_rejected_before_scientific_build(pool_factory):
    from rl_matdesign.structure_sources import StructureSpec
    from rl_matdesign.utils.structure_samples import SamplingProtocolError

    source = StructureSpec({'builder': 'parallel_test_builder:Builder'})
    pool, root = pool_factory(cfg_extra={'test_no_sample_protocol': True})
    with pytest.raises(SamplingProtocolError, match='explicit build_sample'):
        pool.build_many_seeded([{'id': 'a'}, {'id': 'b'}], n_configs=2,
                               source_spec=source, eval_seed=8)
    assert not (root / 'build_calls.jsonl').exists()
    assert not pool._workers
    assert pool.last_sample_records == []
    # The historical, explicitly caller-owned RNG entry point remains usable.
    assert len(pool.build({'id': 'legacy'}, n_configs=1)) == 1


@pytest.mark.parametrize('templates', [None, [
    {'path': 'first', 'site_symbol': 'Fe', 'n_configs': 2},
    {'path': 'second', 'site_symbol': 'Ni', 'n_configs': 3},
]])
def test_explicit_external_protocol_serial_parallel_scalar_batch_and_order_parity(pool_factory, templates):
    import importlib
    from rl_matdesign.structure_sources import StructureSpec
    from rl_matdesign.utils.structure_samples import execute_many_seeded_samples, SAMPLE_INFO_KEY

    first, _ = pool_factory(cfg_extra={'test_templates': templates})
    second, second_root = pool_factory(cfg_extra={'test_templates': templates},
                                       parallel_extra={'devices': [0]})
    module = importlib.import_module('parallel_test_builder')
    serial = module.Builder({'test_root': str(second_root / 'serial'), 'test_templates': templates})
    source = StructureSpec({'builder': 'parallel_test_builder:Builder',
                                           'test_templates': templates})
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=17)
    a, b = {'z': 2, 'id': 'a'}, {'id': 'b', 'z': 3}
    original = execute_many_seeded_samples(serial, [a, b], **kwargs)
    reversed_parallel = first.build_many_seeded([b, dict(reversed(list(a.items())))], **kwargs)
    batched_parallel = second.build_many_seeded([a, b], **kwargs)
    scalar = first.build_seeded(a, **kwargs)

    def signature(cells):
        return [(cell.get_chemical_symbols(), cell.positions.tolist(), cell.info['draw'],
                 cell.info['canonical_keys'], cell.info[SAMPLE_INFO_KEY]) for cell in cells]
    assert signature(original[0]) == signature(reversed_parallel[1]) == signature(batched_parallel[0]) == signature(scalar)
    assert signature(original[1]) == signature(reversed_parallel[0]) == signature(batched_parallel[1])
    assert len(original[0]) == (3 if templates is None else 5)
    assert serial.last_sample_records[0] == first.last_sample_records[0]


def test_seeded_parallel_plan_failure_preserves_neighbor_results_and_diagnostics(pool_factory):
    from rl_matdesign.structure_sources import StructureSpec

    pool, root = pool_factory()
    source = StructureSpec({'builder': 'parallel_test_builder:Builder'})
    result = pool.build_many_seeded([{'id': 'a'}, {'mode': 'plan_fail'}, {'id': 'c'}],
                                    n_configs=2, source_spec=source, eval_seed=9)
    assert [len(result[0]), len(result[2])] == [2, 2]
    assert isinstance(result[1], RuntimeError)
    assert 'sample plan failed' in str(result[1])
    assert pool.last_sample_records[1][0]['status'] == 'plan_failed'
    assert 'sample_index' not in pool.last_sample_records[1][0]
    assert [row['config_attempted'] for row in pool.last_build_timings] == [2, 0, 2]
    assert len((root / 'build_calls.jsonl').read_text().splitlines()) == 4
    failed = pool.build_many_seeded([{'mode': 'plan_fail'}], n_configs=2,
                                   source_spec=source, eval_seed=9)
    assert isinstance(failed[0], RuntimeError)
    assert pool.last_sample_records[0][0]['status'] == 'plan_failed'
