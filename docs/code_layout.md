# Code layout and compatibility

## Entry points

- `rl-matdesign --config CONFIG --method METHOD --out DIR`: installed command.
- `python scripts/run_experiment.py ...`: source-checkout command with the same flags.
- `--validate-config`: resolve and export configuration before loading a model.
- `scripts/benchmark.py`: plan or execute matched-budget DQN-MC, DQN-bootstrap,
  A2C and exact-uniform random-search arms.
- `scripts/archive_experiment.py`: archive declared historical evidence without
  assigning today's code revision to an old experiment.

The source runner re-exports configuration helpers for existing imports. The
installed command uses package modules and does not require a sibling `scripts`
directory. BO/GA commands remain available; the new GlassNet benchmark's uniform
terminal-composition baseline is `scripts/baselines/run_random.py`.

## Responsibilities

| Location | Responsibility |
|---|---|
| `config_schema.py`, `configuration.py` | Core parameter ownership, YAML/CLI/HPO resolution and environment construction |
| `env.py`, `env_integer.py`, `env_multigroup.py`, `sequence.py` | Composition, accumulated tokens, labelled groups, explicit peptide positions and typed fields |
| `action_features.py`, `action_encoding.py` | Shared typed action encoding |
| `structure_sources.py`, `utils/structure_samples.py` | Source ownership, sample plans and candidate-local structure randomness |
| `predictors/outputs.py`, `predictors/structure_score.py` | Named scientific outputs, aligned sample statistics, shared model inference and reward integration |
| `predictors/glassnet.py` | Pinned composition-only GlassNet oracle and bounded cache |
| `training.py`, `model.py`, `experiment.py` | Algorithm updates, checkpoints and run orchestration |
| `registry.py` | Existing name/FQN plugin resolution |
| `utils/composition.py`, `utils/design_count.py` | Canonical candidate identity, exact counting and uniform sampling |
| `utils/model_contract.py`, `utils/execution_contract.py`, `utils/seeding.py` | Artifact/scaler compatibility, oracle/runtime identity and RNG continuation |
| `utils/timing.py`, `utils/provenance.py` | Evaluation ledger, timing and append-only invocation evidence |
| `src/abcde_ooh/` | OOH implementation and compatibility imports using the common contracts |
| `tests/` | Domain, scientific contract, runner and subprocess repeatability checks |

Candidates preserve labels and exact numeric distinctions; generic keys do not
round all fields to six decimals. Environment adapters own chemical zero removal
and integer GCD normalization. Evaluation identity additionally includes the
scientific model, source, preprocessing and conditions.

## Version 0.2 compatibility boundary

This refactor intentionally changes state/action features, some construction
MDPs, the sample seed protocol and scientific output statistics. Old weights,
scalers and replay cannot be accepted solely because shapes match. Incompatible
artifacts fail with an explicit contract error; historical experiments require
their actual archived implementation and assets. There is no OOH-only legacy path.

The mathematical updates of DQN-MC, bootstrap DQN and the existing Monte Carlo
A2C implementation remain distinct. Equivalent labelled assignments share a
state and final evaluation; a different partial amount or peptide position
remains distinguishable. Both algorithm parameter families remain in one YAML.

Source migration and output/uncertainty semantics are described in
[development contracts](development_contracts.md). Actual experiment YAMLs,
migration inventories, private model paths and detailed run evidence stay local.
This code release does not update remote GPU environments or historical runs.

## Validation

```bash
python -m pip install -r requirements.txt ase -e '.[dev]'
python -m ruff check --no-cache src scripts tests
PYTHONPATH=src python -m pytest
python -m compileall -q src scripts
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/rl-wheels .
```

CPU dummy runner tests, real pinned GlassNet CPU tests and remote DeepMD/GPU
validation establish different claims. The validation report records each
separately. A passing interface mock does not validate an external physical model.
