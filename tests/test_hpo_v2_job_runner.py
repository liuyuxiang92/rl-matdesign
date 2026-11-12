"""Real job orchestration against deterministic fake worker artifacts."""
from pathlib import Path
import json

import pytest
import yaml

from rl_matdesign.config_schema import (
    GENERATION_PARAMETERS, load_yaml_config, resolve_config, selected_generation_parameter,
)
from rl_matdesign.hpo.artifacts import atomic_json, file_digest, read_json
from rl_matdesign.hpo.calibration import QualityCalibrationError
from rl_matdesign.hpo.job_runner import JobRunner
from rl_matdesign.hpo.spec import load_plan


def public_plan_path(tmp_path, generation_parameter="gen_temperature", threshold=0):
    scene = {
        "pg_warmup_eps": 0,
        "properties": [{"name": "energy", "output": "raw_model_energy",
                        "predictor": "dummy", "direction": "min", "unit": "eV"}],
        "reward": {"objective": {"terms": {"energy": 1}}},
    }
    # Exercise author-order selection from a scene containing all three keys.
    for parameter in GENERATION_PARAMETERS:
        if parameter != generation_parameter:
            scene[parameter] = .25
    scene[generation_parameter] = .5
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
    budget = {"mode": "absolute_unique", "min_unique": 4, "initial_unique": 8, "max_unique": 16}
    source = tmp_path / "hpo.yaml"
    source.write_text(yaml.safe_dump({
        "base_config": "scene.yaml", "method": "a2c", "random_seed": 42,
        "evaluation": {"threshold": threshold},
        "training": {"repeats": 2, "budget": budget,
                     "parameter_ranges": {"pg_batch_eps": {"type": "categorical", "choices": [2]}}},
        "generation": {"repeats": 2, "budget": budget, "initial_values": [0, .5, 1]},
    }, sort_keys=False))
    return source


class FakeWorker:
    def __init__(self, monkeypatch):
        self.calls = []
        self.values = [-4, -3, -2, -1, 0, 1, 2, 3]
        self.constraints = None
        self.archive_transform = None
        self.ledger_updates = {}
        owner = self

        class Process:
            returncode = 0

            def __init__(self, args, **kwargs):
                path = Path(args[args.index("--config") + 1])
                cfg = resolve_config(load_yaml_config(path)).config
                directory = Path(args[args.index("--out") + 1])
                owner.calls.append({"config": cfg, "args": args, "directory": directory, **kwargs})
                budget = cfg["hpo_target_unique"]
                atomic_json(directory / "candidate_budget.json", {
                    "status": "COMPLETED", "target_unique": budget,
                    "unique_evaluated": budget, "attempts_started": budget,
                    **owner.ledger_updates,
                })
                # Identical identities deliberately occur in both phases.
                rows = [{"identity": f"candidate-{index}",
                         "properties": {"energy": owner.values[index % len(owner.values)]},
                         "constraint_passed": (owner.constraints[index % len(owner.constraints)]
                                               if owner.constraints is not None else True),
                         "units": {"energy": "eV"}}
                        for index in range(budget)]
                if owner.archive_transform is not None:
                    rows = owner.archive_transform(rows)
                (directory / "candidate_records.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n")
                if cfg["hpo_phase"] == "train":
                    train_seed = args[args.index("--train-seed") + 1]
                    (directory / "policy.pt").write_text(f"frozen policy from seed {train_seed}")
                    (directory / "std_scaler.bin").write_text("frozen common scaler")

            def wait(self, timeout=None):
                return self.returncode

        monkeypatch.setattr("rl_matdesign.hpo.job_runner.subprocess.Popen", Process)


def test_training_calibrates_once_and_restores_same_scale_on_new_runner(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    plan = load_plan(source)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "campaign")
    first = runner({"pg_batch_eps": 2}, 8, 11, "train")
    assert first["eligible_for_selection"]
    assert first["raw_metrics"]["qualified_count"] == 5  # Equality at zero is included.
    assert first["quality_normalization"]["scale"] == 3.5
    calibration_path = runner.out / "quality_calibration.json"
    calibration = read_json(calibration_path)
    original_hash = file_digest(calibration_path)
    assert calibration["method"] == "reference_iqr"
    assert calibration["source_job_id"] == first["job_id"]
    assert calibration["source_records"] == first["records_path"]

    worker.values = [-400, -300, -200, -100, 0, 100, 200, 300]
    later = runner({"pg_batch_eps": 2}, 8, 29, "train")
    assert later["eligible_for_selection"]
    assert later["quality_normalization"]["scale"] == 3.5
    assert file_digest(calibration_path) == original_hash
    assert len(worker.calls) == 2

    fresh_plan = load_plan(source)
    assert fresh_plan["objective"]["normalization"]["scale"] is None
    restored = JobRunner(fresh_plan, runner.out)
    assert fresh_plan["objective"]["normalization"]["scale"] == 3.5
    assert restored({"pg_batch_eps": 2}, 8, 11, "train") == first
    assert restored({"pg_batch_eps": 2}, 8, 29, "train") == later
    assert len(worker.calls) == 2
    assert file_digest(calibration_path) == original_hash


@pytest.mark.parametrize("parameter", GENERATION_PARAMETERS)
@pytest.mark.parametrize("value", [0, .25])
def test_selected_generation_method_survives_zero_sorted_export_and_cache(
        tmp_path, monkeypatch, parameter, value):
    source = public_plan_path(tmp_path, parameter)
    plan = load_plan(source)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(plan, tmp_path / "campaign")
    trained = runner({"pg_batch_eps": 2}, 8, 11, "train")
    input_hashes = {path: file_digest(path) for path in trained["checkpoint_paths"].values()}
    generated = runner({"pg_batch_eps": 2, parameter: value}, 4, 401, "generation", trained)
    assert generated["eligible_for_selection"]
    assert len(worker.calls) == 2
    call = worker.calls[-1]
    cfg = call["config"]
    assert selected_generation_parameter(cfg) == parameter
    assert cfg[parameter] == value
    assert all(cfg[key] == 0 for key in GENERATION_PARAMETERS if key != parameter)
    assert cfg["exploration_gen_eps"] == 0
    assert cfg["hpo_phase"] == "generation"
    assert cfg["train_budget"] == 8
    assert cfg["num_gen_eps"] == 4
    assert "--only-generate" in call["args"]
    assert "--skip-generation" not in call["args"]
    assert call["args"][call["args"].index("--train-seed") + 1] == "11"
    assert call["args"][call["args"].index("--gen-seed") + 1] == "401"
    sorted_export = tmp_path / "sorted-runtime.yaml"
    sorted_export.write_text(yaml.safe_dump(cfg, sort_keys=True))
    reloaded = resolve_config(load_yaml_config(sorted_export)).config
    assert selected_generation_parameter(reloaded) == parameter
    assert reloaded[parameter] == value
    assert all(reloaded[key] == 0 for key in GENERATION_PARAMETERS if key != parameter)
    assert input_hashes == {path: file_digest(path) for path in trained["checkpoint_paths"].values()}
    assert runner({"pg_batch_eps": 2, parameter: value}, 4, 401, "generation", trained) == generated
    assert len(worker.calls) == 2


def test_training_seen_candidates_count_in_independent_generation_budget(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    trained = runner({"pg_batch_eps": 2}, 8, 11, "train")
    generated = runner({"pg_batch_eps": 2, "gen_temperature": 1}, 4, 401, "generation", trained)
    train_rows = [json.loads(line) for line in Path(trained["records_path"]).read_text().splitlines()]
    gen_rows = [json.loads(line) for line in Path(generated["records_path"]).read_text().splitlines()]
    assert {row["identity"] for row in gen_rows} <= {row["identity"] for row in train_rows}
    assert generated["ledger"]["unique_evaluated"] == generated["ledger"]["target_unique"] == 4
    assert generated["ledger"]["attempts_started"] == 4
    assert generated["raw_metrics"]["unique_evaluated"] == 4
    assert generated["raw_metrics"]["qualified_fraction"] == 1
    assert generated["eligible_for_selection"]
    assert generated["directory"] != trained["directory"]
    assert len(worker.calls) == 2


def test_degenerate_reference_records_and_cost_survive_calibration_failure(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path, threshold=0)
    worker = FakeWorker(monkeypatch)
    worker.values = [0]
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    failed = runner({"pg_batch_eps": 2}, 4, 11, "train")
    assert failed["status"] == "QUALITY_CALIBRATION_REQUIRED"
    assert not failed["eligible_for_selection"] and failed["score"] == -1
    assert "equal the threshold" in failed["error"]
    assert failed["raw_metrics"]["qualified_fraction"] == 1
    assert failed["ledger"]["attempts_started"] == failed["ledger"]["unique_evaluated"] == 4
    assert failed["ledger"]["status"] == "COMPLETED"
    directory = Path(failed["directory"])
    assert read_json(directory / "result.json") == failed
    assert read_json(directory / "execution.json")["status"] == "QUALITY_CALIBRATION_REQUIRED"
    assert len(Path(failed["records_path"]).read_text().splitlines()) == 4
    assert not (runner.out / "quality_calibration.json").exists()
    assert failed["checkpoint_paths"]
    assert len(worker.calls) == 1
    restored = JobRunner(load_plan(source), runner.out)
    assert restored({"pg_batch_eps": 2}, 4, 11, "train") == failed
    assert len(worker.calls) == 1


def test_generation_models_are_part_of_job_cache_identity(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    model_a = runner({"pg_batch_eps": 2}, 8, 11, "train")
    model_b = runner({"pg_batch_eps": 2}, 8, 29, "train")
    generation_a = runner({"pg_batch_eps": 2, "gen_temperature": 1}, 4, 401, "generation", model_a)
    generation_b = runner({"pg_batch_eps": 2, "gen_temperature": 1}, 4, 401, "generation", model_b)
    assert generation_a["job_id"] != generation_b["job_id"]
    assert generation_a["input_models"] != generation_b["input_models"]
    assert len(worker.calls) == 4


@pytest.mark.parametrize("next_seed", [11, 29])
def test_cached_or_new_job_cannot_replace_deleted_fixed_calibration(tmp_path, monkeypatch, next_seed):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    trained = runner({"pg_batch_eps": 2}, 8, 11, "train")
    (runner.out / "quality_calibration.json").unlink()
    with pytest.raises(QualityCalibrationError, match="calibration"):
        # Constructor validation blocks both cache lookup and a new proposal.
        restored = JobRunner(load_plan(source), runner.out)
        restored({"pg_batch_eps": 2}, 8, next_seed, "train")
    assert Path(trained["records_path"]).exists()
    assert len(worker.calls) == 1


@pytest.mark.parametrize("invalid_archive", ["ledger_mismatch", "invalid_constraint_flag"])
def test_first_invalid_archive_preserves_cost_without_freezing_quality_scale(
        tmp_path, monkeypatch, invalid_archive):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    if invalid_archive == "ledger_mismatch":
        worker.archive_transform = lambda rows: rows[:-1]
    else:
        worker.constraints = ["yes"]
    plan = load_plan(source)
    runner = JobRunner(plan, tmp_path / "campaign")
    result = runner({"pg_batch_eps": 2}, 8, 11, "train")
    assert result["status"] == "INVALID_RECORDS"
    assert not result["eligible_for_selection"]
    assert result["score"] == -1
    assert plan["objective"]["normalization"]["scale"] is None
    assert not (runner.out / "quality_calibration.json").exists()
    assert result["ledger"]["attempts_started"] == 8
    assert result["ledger"]["unique_evaluated"] == 8
    assert result["ledger"]["target_unique"] == 8
    assert Path(result["records_path"]).is_file()
    assert read_json(Path(result["directory"]) / "result.json") == result
    assert read_json(Path(result["directory"]) / "execution.json")["status"] == "INVALID_RECORDS"
    assert runner({"pg_batch_eps": 2}, 8, 11, "train") == result
    assert len(worker.calls) == 1


def test_incomplete_unique_count_cannot_become_calibration_reference(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    worker.archive_transform = lambda rows: rows[:-1]
    worker.ledger_updates = {"unique_evaluated": 7}
    plan = load_plan(source)
    runner = JobRunner(plan, tmp_path / "campaign")
    result = runner({"pg_batch_eps": 2}, 8, 11, "train")
    assert result["status"] == "INCOMPLETE"
    assert not result["eligible_for_selection"] and result["score"] == -1
    assert plan["objective"]["normalization"]["scale"] is None
    assert not (runner.out / "quality_calibration.json").exists()
    assert result["ledger"] == {"status": "COMPLETED", "target_unique": 8,
                                "unique_evaluated": 7, "attempts_started": 8}
    assert len(Path(result["records_path"]).read_text().splitlines()) == 7
    assert read_json(Path(result["directory"]) / "result.json") == result
    assert len(worker.calls) == 1


def test_terminal_worker_archive_can_finish_calibration_after_parent_interruption(tmp_path, monkeypatch):
    source = public_plan_path(tmp_path)
    worker = FakeWorker(monkeypatch)
    runner = JobRunner(load_plan(source), tmp_path / "campaign")
    original = runner({"pg_batch_eps": 2}, 8, 11, "train")
    # Recreate the interruption window after worker completion but before the
    # parent has persisted its first scale and scored result.
    (Path(original["directory"]) / "result.json").unlink()
    (runner.out / "quality_calibration.json").unlink()
    restored = JobRunner(load_plan(source), runner.out)
    recovered = restored({"pg_batch_eps": 2}, 8, 11, "train")
    assert recovered["eligible_for_selection"]
    assert recovered["recovered_from_terminal_artifacts"]
    assert recovered["quality_normalization"] == original["quality_normalization"]
    assert recovered["score"] == original["score"]
    assert recovered["ledger"] == original["ledger"]
    assert recovered["wall_seconds"] is None
    assert len(worker.calls) == 1
