# Development contracts

The 0.2 refactor borrows configuration ownership, explicit plugin contracts and
automated checks from the DeepMD ecosystem; it does not copy its architecture or
claim compatibility with all of its models. Keep application-specific scientific
arguments in their adapter or source. Training code must not branch on a material
name. A new model adapter is appropriate when its API, preprocessing or physical
meaning differs; it should not require a new training interface.

## Extension path

1. Describe the labelled design variables and completion constraints in `groups`.
   A composition assignment and a peptide `(position, monomer)` assignment each
   consume one transition. Integer-token domains retain accumulation semantics.
2. Use a registry name or `package.module:Class` for a model. Prefer named
   `ScientificOutput` values with units, representation and actual uncertainty.
   Preserve raw model/configuration samples when a nonlinear inverse is needed.
3. Declare a structure source only if the model needs structures. Properties with
   one source share its sampled/relaxed ensemble. A parallel builder implements
   explicit `build_sample`; an opaque native ensemble cannot be split safely.
4. Separate scientific output from reward direction, scaling, weights and gates.
   Use `model_ref` when one model serves multiple outputs.
5. Test the domain, candidate identity, units, failure behavior and RNG isolation.
   Test doubles establish interface behavior; an opt-in real-model reference
   verifies scientific integration. Label missing assets and GPU checks honestly.

The core resolver owns defaults, CLI/HPO overrides and conflict validation.
Plugin schemas own their scientific parameters. Unknown plugin internals are not
claimed to be validated merely because their configuration can be loaded.

## Validation and release

```bash
python -m pip install -r requirements.txt ase -e '.[dev]'
python -m ruff check --no-cache src scripts tests
PYTHONPATH=src python -m pytest
python -m compileall -q src scripts
git diff --check
```

Ruff is pinned to `0.14.3` in the development extra and `pyproject.toml`.
The initial rule set is `E9`, `F63`, `F7`, and `F82`: parse/read errors, selected
invalid constructs and undefined names. It covers `src`, `scripts`, and `tests`
without formatting or automatic fixes. No broad file exclusions or rule ignores
are used to make the gate pass. Expand the rules deliberately in a separate
reviewed change. The settings use Ruff's documented
[project configuration](https://docs.astral.sh/ruff/configuration/).

Run these checks in the chosen local or CI environment. This code-only release
does not add a YAML workflow or publish experiment configurations. Optional
asset-dependent tests report skips; unit tests do not download property models.

Save each coherent stage after its targeted and integration checks. Never stage
unrelated work from another checkout. Model/scaler/replay contracts intentionally
reject incompatible artifacts; equal tensor shapes are insufficient. A paper
archive must include the actual generating revision, uncommitted configuration,
environment, weights/templates/scalers and seeds. A tag on today's code alone
does not identify the code that produced historical figures.

## Reproducible experiments

`eval_seed` controls evaluation samples; `train_seed` controls initialization,
rollouts and training; `gen_seed` controls generation. Global RNG isolation keeps
model loading/inference from consuming RL streams. External adapters with private
generators must implement their own stable candidate protocol. Batched inference
must explicitly declare candidate-seeded equivalence to use the batched path.

Checkpoints retain Python, NumPy, Torch CPU/CUDA states, optimizer/replay state,
budget history, the fitted scaler fingerprint and evaluation/learning contracts.
`--resume-training` with a full checkpoint continues toward the configured total
budget. Increasing a budget extends it; changing model assets, conditions, source
code, fitted scaler or learning settings requires a new run. Weight-only fallback
is a warm start and explicitly reports that it is not exact continuation.

The guarantee is conditional on recorded code, dependencies, assets, configuration
and hardware. It does not promise identical results across Torch/CUDA versions or
different GPU hardware, or identical trajectories to the pre-refactor code.

## Benchmark records

`train_budget` is the shared paid training budget, including DQN's rewarded
warmup. A2C's feature-only scaler warmup is timed but does not query the oracle.
With `train_count_duplicates: false`, only a newly evaluated successful candidate
spends a budget unit; duplicate rollouts can still train the network. PG's final
batch may be smaller than `pg_batch_eps`.

`evaluations.jsonl` retains every scoring request, candidate, physical outputs
when available, training best-so-far, failures and actual cache/backend telemetry.
Reporting reads are marked separately and do not add another training observation.
Training and generation have separate cost records; a generation discovery cannot
improve the primary training-budget curve. The unique ledger survives LRU eviction
and resume. Backend counters that are unavailable remain unknown; repeated requests
are not automatically cache hits. Resume reports both cumulative prediction time
and this invocation's prediction time.

Use `scripts/benchmark.py` to write a plan before `--run`. Every arm starts in a
fresh process, with an empty result cache. The random baseline samples uniformly
over unique composition maps using completion counts, not random action paths.
Its exact counter covers one fraction group with sum/bounds/menu constraints;
arbitrary chemical filters require an explicit counted-domain implementation.

The current A2C uses the existing Monte Carlo return target and value baseline;
this refactor does not replace it with TD-A2C. Keep the implementation definition,
query budget, seed split and HPO trial count fixed in any scientific comparison.

Exact resume also compares installed package versions, BLAS/OpenMP settings,
Torch thread/precision settings and available driver evidence. This conservative
check may reject an unrelated tool upgrade; preserve the recorded environment or
start a new run. Interpreter paths, CUDA initialization state and GPU UUIDs are
excluded. Version evidence does not replace a native-library/environment lock.
Checkpoint frequency and the duplicate guard are part of the learning contract:
they can change DQN prefetch-wave boundaries and therefore sampling trajectories.
Increasing the total budget extends saved state, but need not reproduce a run
that used that larger budget from the beginning, particularly when the original
last batch was partial. Elapsed wall time is reported separately per invocation;
stopped time between invocations is never guessed.
