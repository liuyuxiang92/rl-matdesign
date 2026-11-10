# Reinforce Learning for Materials Design

Config-driven RL runner for materials composition, peptide-position and typed-field design.
Supports multiple material systems (OOH catalyst, HEA, perovskite, oxides) via YAML configs.
Three RL algorithms: **DQN** (online, target-network, or MC targets), **REINFORCE**, and **A2C**.

Version 0.2 uses labelled assignments, explicit structure sources and named
scientific outputs. See the [migration and development contracts](docs/development_contracts.md)
before loading older checkpoints. Historical paper results require their actual
code and asset archive; a new-version run is not a replay of an old trajectory.

The [benchmark guide](docs/glassnet_benchmark.md) describes shared query budgets,
DQN-MC/DQN-bootstrap/A2C comparisons and uniform random search. Experiment YAMLs
remain local; supply your own domain files when generating a plan:

```bash
python scripts/benchmark.py --config /path/to/local/small.yaml \
  --config /path/to/local/large.yaml --out runs/benchmark --budget 1000
```

Add `--run` after preparing the scientific oracle and its assets. Stock GlassNet
prediction runs on CPU; the RL device is selected separately.

See [code layout and compatibility](docs/code_layout.md) for maintained entry
points, module responsibilities, and regression checks.


---

## Installation

Two conda environments are provided. Both pin `blas=*=openblas` so numpy/sklearn link against
OpenBLAS (pthreads-based) rather than MKL — eliminating the OpenMP runtime conflict with PyTorch
that otherwise causes segfaults on macOS and Linux.

**macOS / CPU (development):**
```bash
conda env create -f environment.yml
conda activate ooh-catalyst
```

**Linux + CUDA 12.x (HPC production):**
```bash
conda env create -f environment-gpu.yml
conda activate ooh-catalyst-gpu
```

Both commands do the full setup in one step: conda installs all base dependencies, then the
`pip:` section runs `pip install -e .` (editable install from `pyproject.toml`) and installs
DeepMD-kit. After this, `abcde_ooh` and `rl_matdesign` are importable as regular packages
from any directory — no `sys.path` manipulation needed.

PyTorch is installed via a pip wheel (not conda) to avoid BLAS/CUDA solver conflicts on HPC
clusters where CUDA is provided system-wide. The default targets CUDA 12.4. To change:
edit the `--index-url` in `environment-gpu.yml` (`cu124` → `cu121` / `cu118`) and change
`deepmd-kit[torch,cu12]` → `deepmd-kit[torch,cu11]` for CUDA 11.x.

---

## How it works

All hyperparameters, material system settings, and RL method defaults are stored in a **YAML config
file** under `configs/`. The CLI only takes the config path, output directory, method override, and
operational flags (seeds, resume, checkpoint). This means:

- Switching material systems = swap `--config`
- Reproducing a run = preserve the code, inputs, environment, effective config and seeds
- Tuning = edit the YAML, re-run

---

## Available configs

| Config | Material system | Default method |
|---|---|---|
| `configs/hea.yaml` | High-entropy alloy formation energy | reinforce |
| `configs/lips_sse.yaml` | Multi-sublattice LiPS solid electrolyte | Configured in YAML |
| `configs/oxides_sinter.yaml` | Oxide sintering temperature | reinforce |
| `configs/oxides_calcine.yaml` | Oxide calcination temperature | reinforce |
| `configs/test_dummy.yaml` | Fast smoke test (no real predictor) | dqn |
| `configs/test_multigroup_dummy.yaml` | Multi-sublattice smoke test (no real predictor) | CLI selection |

New and migrated scenario YAML files are maintained locally. This release does
not publish their contents or changes; previously published examples remain as-is. Examples below using those paths require your existing local configs
and model/structure files. They are not included in a fresh checkout.

---

## Quickstart

```bash
python scripts/run_experiment.py \
    --config configs/test_dummy.yaml \
    --method a2c \
    --out runs/smoke_a2c_seed0 \
    --eval-seed 321 --train-seed 123 --gen-seed 213
```

All hyperparameters (`pg_num_iters`, `pg_batch_eps`, `pg_entropy_coef`, `num_gen_eps`, etc.)
come from the YAML. Override them by editing the config or creating a copy for each experiment.

---

## Seeds

| Flag | Default | Purpose |
|---|---|---|
| `--eval-seed N` | YAML `eval_seed`, otherwise 0 | Property evaluation and random structures; fallback for the other two seeds. |
| `--train-seed N` | YAML `train_seed`, otherwise eval seed | Training random state. Deterministic Torch settings apply regardless of whether the seed is supplied by CLI or YAML. |
| `--gen-seed N` | YAML `gen_seed`, otherwise eval seed | Generation random state, reset independently of training. |

`--dp-seed` and YAML `dp_seed` remain migration aliases for `eval_seed`; they do
not create another random stream. CLI values override YAML. Fixed seeds support
repeatability within the recorded code/input/software/hardware environment;
they do not guarantee bitwise agreement across different environments.

Validate and print effective settings before loading a predictor:

```bash
python scripts/run_experiment.py --config configs/test_dummy.yaml \
    --method dqn --dqn-target-mode mc --validate-config
```

Shared defaults and validation now come from one [configuration contract](docs/configuration_schema.md).
A single YAML can keep both DQN and A2C settings. Duplicate keys, conflicting old/new
aliases, and unknown core parameters are rejected. Scientific plugin settings
remain owned by their plugins; this is not yet a full nested scientific schema.
Every invocation saves effective settings, their sources, and an append-only
provenance record. `--only-generate` writes `generation_log.csv` and preserves the
existing `training_log.csv`.

See [experiment archiving and reproduction](docs/reproducibility.md) and
[development contracts](docs/development_contracts.md).

---

## REINFORCE (default for OOH)

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --eval-seed 321 --train-seed 123 --gen-seed 213 \
    --save-checkpoint-freq 50
```

Key YAML fields (edit `configs/ooh.yaml`):

```yaml
pg_warmup_eps: 1000          # random episodes to fit the StandardScaler
pg_num_iters: 1000           # outer training iterations
pg_batch_eps: 21             # episodes per batch; one gradient step per batch
pg_entropy_coef: 0.15        # entropy bonus weight, in σ of batch return
pg_entropy_min: 0.3          # floor on normalised entropy H/ln|A|; 0 disables
pg_repeat_penalty_coef: 10   # penalises revisiting the same composition, in σ
gen_temperature: 3.0         # Boltzmann T for generation diversity
num_gen_eps: 2000            # unique compositions to generate
```

PG advantages are standardised per batch (no flag), so `pg_entropy_coef` and
`pg_repeat_penalty_coef` are in standard deviations of batch return rather than in
the property's own units. `pg_entropy_min` floors the policy's normalised entropy
so it keeps exploring for the whole budget instead of going deterministic partway
through — see [`docs/yaml_config_reference.md`](docs/yaml_config_reference.md).

When sweeping budgets, hold `pg_batch_eps` fixed and vary only `pg_num_iters`;
changing both confounds episode budget with update density.

---

## A2C

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method a2c \
    --out runs/ooh_a2c_seed0 \
    --eval-seed 0 --train-seed 42 --gen-seed 99
```

A2C trains an additional critic (`value_net.pt`) to reduce gradient variance.
Set `pg_lr_actor` and `pg_lr_critic` independently in the YAML.

---

## DQN

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method dqn \
    --out runs/ooh_dqn_seed0 \
    --eval-seed 0 --train-seed 42 --gen-seed 99
```

Key YAML fields:

```yaml
dqn_warmup_eps: 500           # warmup episodes to populate the initial buffer
dqn_num_train_eps: 20000      # total online training episodes
dqn_buffer_size: 50000        # replay buffer capacity (FIFO)
dqn_grad_steps_per_ep: 5      # gradient updates per training episode
dqn_target_update_freq: 100   # hard-copy Q-net → target-net every N episodes
dqn_eps_anneal_eps: 10000     # linear ε annealing: reaches dqn_eps_min after N episodes
dqn_eps_min: 0.05
dqn_lr: 0.001
dqn_batch_size: 256
```

Override the loss function from the CLI: `--dqn-loss smoothl1` (default) or `--dqn-loss mse`.

### Counting repeated candidates in the training budget

By default every completed training episode consumes one budget slot. To count
only candidates that have not been evaluated earlier in training, add these
top-level YAML fields:

```yaml
train_count_duplicates: false       # default: true
train_max_duplicate_attempts: 1000   # maximum consecutive completed duplicates
```

Repeated episodes still enter the DQN replay buffer and receive normal training
updates; REINFORCE/A2C also keep them in their training batches. Only the budget
counter changes. This works for every material system and both DQN target modes
(`mc` and `bootstrap`). Candidate identity uses the environment's final
composition key: action order does not matter, while sublattice assignments and
categorical choices remain distinct.

For DQN, `dqn_warmup_eps: 400` plus `dqn_num_train_eps: 1600` now requests 400
distinct warmup candidates followed by 1,600 **additional** distinct candidates.
The seen-candidate set is shared across both phases. Epsilon annealing, target
network updates, and checkpoint frequency still follow actual sampled online
episodes. For PG, the unique target is `pg_num_iters * pg_batch_eps`; duplicate
episodes can require additional batches and updates. Batches contain at most
`pg_batch_eps` sampled episodes, with a shorter final batch when the target is
reached. PG's scaler-only warmup does not enter the seen-candidate set.

Use `--no-train-count-duplicates` or `--train-count-duplicates` to override the
YAML, and `--train-max-duplicate-attempts N` to override the duplicate limit. A new
candidate resets the consecutive-duplicate counter. If the limit is reached,
training stops with an error instead of sampling indefinitely in an exhausted
search space. The limit applies only when duplicates do not consume budget.

The per-episode rows in `training_log.csv` report `budget_counted`,
`training_attempts`, `training_budget_used`, `training_unique_candidates`, and
`training_duplicate_episodes`. DQN totals include warmup. With duplicate counting
disabled, `episode` tracks actual samples and can exceed the configured budget.

---

## Checkpointing and resume

The following commands use a prepared model environment and a local
configuration (`/path/to/local/experiment.yaml` is a placeholder). Keep the same code, numerical environment, model/scaler assets,
training settings and checkpoint frequency when resuming.

```bash
# 1. Train toward 1,000 successful unique candidates.
python scripts/run_experiment.py \
    --config /path/to/local/experiment.yaml --method a2c \
    --out runs/a2c_repeat --train-budget 1000 \
    --eval-seed 321 --train-seed 123 --gen-seed 213 \
    --save-checkpoint-freq 50

# 2. Continue an interrupted run toward that same total.
python scripts/run_experiment.py \
    --config /path/to/local/experiment.yaml --method a2c \
    --out runs/a2c_repeat --train-budget 1000 \
    --eval-seed 321 --train-seed 123 --gen-seed 213 \
    --save-checkpoint-freq 50 --resume-training

# 3. Reproduce generation from the compatible saved model and scaler.
python scripts/run_experiment.py \
    --config /path/to/local/experiment.yaml --method a2c \
    --out runs/a2c_repeat \
    --eval-seed 321 --train-seed 123 --gen-seed 213 --only-generate

# 4. Extend the saved state to a larger total; omit extra generation.
python scripts/run_experiment.py \
    --config /path/to/local/experiment.yaml --method a2c \
    --out runs/a2c_repeat --train-budget 2000 \
    --eval-seed 321 --train-seed 123 --gen-seed 213 \
    --save-checkpoint-freq 50 --resume-training --skip-generation
```

Use `--load-policy`, `--load-qnet`, `--load-scaler` and `--load-value-net` for
non-default artifact paths. Full checkpoints preserve optimizer/replay, RNG,
scaler fingerprint, unique-candidate history and evaluation/learning contracts.
A budget extension continues saved state; it need not equal a run started with
that larger budget, because an earlier partial final batch may already have
updated the policy. Changing the duplicate guard or checkpoint frequency may
change sampling boundaries and is rejected as exact continuation.

Old checkpoints require their archived implementation and assets. Setting
`train_count_duplicates: true` does not bypass the new contracts. A compatible
current-version weights-only fallback is explicitly a warm start, lacking the
history and optimizer/RNG state needed for exact continuation.

---

## Custom material systems — three tiers

Most new systems need **only a YAML config** (Tier 1). For novel reward physics
you add a small predictor class and point at it from YAML via a fully-qualified
name — no edits to `run_experiment.py` (Tier 2). Complex domain-specific
predictors (OOH-style adsorbate physics) live in dedicated classes registered
under built-in short names (Tier 3).

### Tier 1 — pure YAML

For any system whose reward is "substitute composition onto a base POSCAR,
evaluate with one or more DeepMD models, aggregate to a scalar", use
a named `structure_sources` entry with `builder: substitute` and a property whose
`predictor` is `dp_energy`, `dp_property`, or `dp_classification`. The list
automatically selects the reward engine. Composition-only predictors can use
the same engine without a builder; see `configs/oxides_sinter_calcine.yaml` for
a multi-objective example. A minimal YAML:

```yaml
# Environment
groups:
  - name: alloy
    kind: composition
    species_set: [Ni, Fe]
    fraction_set: {min: 0.00, max: 1.00, step: 0.25}
    n_components: 2
    element_bounds:
      Ni: [0.25, 0.75]
      Fe: [0.25, 0.75]

# Predictor
structure_sources:
  bulk:
    builder: substitute                         # fixed-lattice element swap
    base_poscar: /path/to/template.POSCAR         # placeholder X on sub sites
    site_symbol: X
k: 1.0
properties:
  - name: energy
    source: bulk
    predictor: dp_energy                        # DP potential energy
    models: [/path/to/member1.pt, /path/to/member2.pt]
    direction: min                              # lower energy = better
    objective: mean_minus_kstd                   # reward = -mean - k*model_spread
    uncertainty: model_spread                    # requires at least two models

# Method + hyperparameters
method: a2c
pg_warmup_eps: 200
pg_num_iters: 500
# ...
```

Run:
```bash
python scripts/run_experiment.py \
    --config configs/your_system.yaml \
    --out runs/my_run --eval-seed 0
```

**Zero Python required.**

### Combining properties and synthesizability

The default reward remains the weighted sum of the configured properties.
Set top-level `reward.type: gated` to optimize an objective only after all
requirements pass. Both the objective and each gate can be weighted sums of
any named properties, including composition-only predictors; this is shared
across material systems.

This fragment assumes that the environment and structure builder have already
been configured, and that you supply the two model files:

```yaml
# Set n_random_configs: 10 inside the declared bulk source.
properties:
  - name: delta
    source: bulk
    predictor: dp_property
    models: [models/delta.pt]
    output_index: 0
    direction: max
    objective: mean
    scale: 1.0                       # delta stays in its original units
  - name: synthesizability_probability
    source: bulk
    predictor: dp_classification
    models: [models/synthesizability.pt]
    head: synthesizability
    output_index: [1, 2]             # nearly pure + pure; class 0 is impure
    direction: max
    objective: mean

reward:
  type: gated
  gates:
    - name: synthesis
      score:
        type: weighted_sum
        terms: {synthesizability_probability: 1.0}
      min: 0.8
      penalty_weight: 1.0
  objective:
    type: weighted_sum
    terms: {delta: 1.0}
  failure:
    reward: -10.0
```

The example accepts both nearly pure and pure structures. Use `output_index: 2`
to select only pure, or `output_index: 1` for a binary model trained with labels
`[impure, pure]`; class meanings must match the model's training labels. A list
must be nonempty, contain distinct nonnegative class indices, and use the
default `output_aggregator: index`. The classifier uses DeepMD's Python
`predict_proba()` directly, requiring a classification-enabled DeepMD
installation. No `dp test` output or probability conversion in
rl-matdesign is needed. With a list, selected class probabilities are summed
within each model/configuration first. The resulting probability is averaged
across the successfully prepared configurations, using the same structures for every
checkpoint. With several model files, the engine then computes the mean and
population standard deviation across those model averages (`model_spread`); with
one model that estimate is missing (`None`), rather than zero. All checkpoints
must use the same class order. This standard deviation is not `1 - p`.

Classification properties contribute `mean - model_spread` to gate scores when
at least two models are available, and their mean with one model. The coefficient
is `1`, independent of `k`; the gate does not fabricate a missing spread. Other property types still contribute their raw
mean. Here a synthesizability score `S = mean - std` of at least `0.8` gives
reward `delta`; otherwise, reward is `-10 - (0.8 - S)`.
`S` is a conservative score, not a calibrated probability. In `generated.csv`,
`obj_synthesizability_probability_mean` and
`obj_synthesizability_probability_std` retain the mean and selected spread
(default: model spread, blank when missing); named scientific outputs retain each
spread separately. `gate_synthesis_value` records `S`.
The failure value is configurable.
To guarantee that every passing candidate outranks every failing candidate,
choose it below the lowest possible feasible objective; a fixed value cannot
guarantee this when the objective has no known lower bound.

Omit `reward` or use `reward: {type: weighted_sum}` to keep the existing weighted
sum. No automatic property normalization is added. Start a new run or recompute
saved rewards after changing the reward definition; old replay-buffer targets
still encode the old reward. See the [YAML reference](docs/yaml_config_reference.md#reward-weighted-sum-or-gated)
for multiple gates, multi-property gate scores, and diagnostics.

### Parallel structure preparation on multiple GPUs

Add `structure_parallel` to a named source to prepare its configurations
concurrently. Each persistent worker executes one explicit sample and completes
its source-owned relaxation before returning it. Built-in `substitute` supports
this protocol, including multiple templates. An external builder must implement
`build_sample(candidate, *, sample)` returning one ASE `Atoms`; merely having
`build(..., n_configs=1)` is insufficient. Missing capability is a configuration
error before scientific jobs run. An MD builder must expose the same explicit
sample protocol before its MD-plus-optimization path can run in workers. Keep
`geo_opt: null` within that source if the builder already performs optimization.

```yaml
structure_sources:
  bulk:
    builder: substitute
    base_poscar: my.POSCAR
    site_symbol: X
    n_random_configs: 4
    structure_parallel: {}  # automatic GPUs; one task per GPU
# Each structure property refers to source: bulk.
```

On eight selected GPUs, the example collects two formulas with four configurations
each, so all eight tasks can run together. Extra tasks queue until a GPU becomes
free. Each formula keeps its own reward. Automatic batch
size is the GPU count divided by configurations per formula, rounded up; budgets
and training boundaries can shorten a batch. No worker-count or success-threshold
parameter is needed. Set `timeout_s` inside this block if the default one-hour
task limit is insufficient for a full MD-plus-optimization run.

MD failures and unconverged optimizations are excluded from each formula's
ensemble. Remaining usable configurations are averaged; a formula with none is
skipped without consuming training budget. The existing consecutive-failure
guard remains in place. These relaxation checks also apply to serial structure
scoring; external builders must report failures from their own optimizer.

Cross-formula DQN sampling collects a wave using the current policy before
processing its learning updates, which changes its sampling trajectory. An explicit
`candidate_batch_size: 1` restores single-formula collection. PG waves
stay within the existing learning batch. Cache hits reuse results and can leave
some GPUs idle; property inference and learning remain in the parent process.
Older `workers_per_device: auto` is accepted as `1` with a deprecation warning;
remove the now-unused `min_speedup` and `min_successful_configs`. A legacy success
threshold above one is ignored with a warning. Two workers per GPU are no longer supported.

By default `auto` includes GPU 0, which the parent also uses for prediction and
RL. To give workers the available GPUs without that contention, run the parent
on CPU:

```bash
DEVICE=cpu python scripts/run_experiment.py \
    --config configs/your_system.yaml --out runs/multigpu --device cpu
```

Workers explicitly select CUDA, independently of the parent's `DEVICE=cpu`.
CPU property prediction may cost more time, so measure it before choosing this
allocation. Alternatively, add `reserve_devices: [0]` to the block to leave visible
GPU 0 for the parent and automatically use the remaining GPUs. Parent DeepMD
scoring models are loaded before the first build. Their resident memory, external
GPU load and additional source-specific worker pools still affect available
memory.

Manual `devices: [1, 2]` is also supported. Indices are relative to the inherited
`CUDA_VISIBLE_DEVICES`: with `CUDA_VISIBLE_DEVICES=2,4,7`, this selects physical GPUs
4 and 7. Each worker sees its assigned GPU as local device 0. Keep LAMMPS's
single-GPU arguments, such as `-k on g 1`; do not override `CUDA_VISIBLE_DEVICES`,
`LOCAL_RANK`, or `DEVICE` through `md.env`. Start one experiment process; this
feature does not need `torchrun` or distributed RL training.

Property scoring and reward calculation wait for the completed structures and
use the same model-by-configuration statistics. Properties with the same
`source` reference reuse one prepared ensemble; distinct source IDs create
independent ensembles. `share_structure` is a legacy migration input, not the
recommended interface: old `true` can normalize to source `default`; old `false`
requires explicit source definitions. Omitting `structure_parallel` uses serial
execution. Source/sample seeds do not depend on candidate order, batch partition
or worker count; GPU arithmetic can still vary across hardware/backends. GPU execution requires the appropriate CUDA-enabled
DeepMD/LAMMPS installation; speedup depends on MD/optimization cost, CPU resources,
and the work that remains serial. See the [YAML reference](docs/yaml_config_reference.md#structure_parallel-gpu-workers-and-automatic-formula-batches)
for failure handling, supported builders, and reproducibility.

### Tier 2 — YAML + a custom scientific predictor

A custom adapter returns physical predictions; the property engine applies
reward signs, weights and penalties. Use `predict_outputs(candidate)` for a
composition model, or `predict_structure_outputs(structures)` for structures
prepared by a named source. This illustrative adapter exposes one checkpoint;
its `_bulk_modulus` method must implement the actual physics and return GPa:

```python
# my_pkg/elastic_predictor.py
import json
from rl_matdesign.predictors.outputs import OutputSchema, ModelSamples, aggregate_samples

class ElasticConstantPredictor:
    output_schema = {"bulk_modulus": OutputSchema("bulk_modulus", "GPa")}

    def __init__(self, cfg, *, seed=None):
        from deepmd.calculator import DP
        self.calc = DP(model=cfg["model"])

    def predict_structure_outputs(self, structures):
        # Source metadata identifies the same retained configurations in every model.
        ids = tuple(json.dumps(a.info["_rl_matdesign_sample"], sort_keys=True)
                    for a in structures)
        values = tuple(self._bulk_modulus(a, self.calc) for a in structures)
        row = ModelSamples("checkpoint", ids, values)
        output = aggregate_samples(self.output_schema["bulk_modulus"], [row])
        return {"bulk_modulus": output}

    def _bulk_modulus(self, atoms, calc):
        raise NotImplementedError("Implement the scientific calculation in GPa")
```

Declare the source and model once, then select its output:

```yaml
structure_sources:
  bulk:
    builder: substitute
    base_poscar: my.POSCAR
    site_symbol: X
    n_random_configs: 5
models:
  elastic:
    predictor: "my_pkg.elastic_predictor:ElasticConstantPredictor"
    model: my.pt
properties:
  - name: stiffness
    model_ref: elastic
    source: bulk
    output: bulk_modulus
    direction: max
    objective: mean
```

Properties referencing the same model, source and operating conditions share one
named-output inference. A singleton model has `model_spread=None`; configuration
spread remains separate. An explicit uncertainty penalty requires an available
statistic selected by `uncertainty`. Nonlinear inverse transforms act on retained
raw samples before statistics, never on an already folded mean/std pair.

Install your package or set `PYTHONPATH` so the FQN can be imported. Constructor
options belong to the model definition; output/reward/source settings belong to
the property. Existing inline properties remain supported. Legacy `score*` and
`predict*` interfaces have narrower compatibility rules; see the
[scientific output migration](docs/scientific_output_migration.md). Old adapters
that returned already-negated or penalized rewards must expose scientific values
before being used as properties.

### Tier 3 — specialized (OOH-like)

For complex domain physics (e.g. OOH slab and adsorbate construction before
the trained property prediction) or constraints that exceed the declarative `phase_pattern`
grammar (e.g. cross-element inequalities like `Fe ≤ max(Ni, Co)`), keep the
logic in a dedicated class:

- **Predictor**: write `src/rl_matdesign/predictors/<name>.py` and register
  `_make_<name>` in `src/rl_matdesign/registry.py:PREDICTORS`.
- **Constraint**: write `src/rl_matdesign/constraints/<name>.py` subclassing
  `ConstraintFilter`, register in `CONSTRAINTS`.

Use `predictor: <name>` in YAML. This is what the bundled `ooh` predictor
and `ooh_phase` constraint do.

### Plug-in surface reference

**Built-in predictor short names** — use as `properties[].predictor`;
registry leaves also support a flat `predictor: <name>`:

| Name | What it does |
|---|---|
| `structure_score` | Reward engine selected automatically by a non-empty `properties:` list; optional shared build/relax, then weighted-sum or gated reward |
| `dp_energy` / `dp_property` / `dp_classification` | DeepMD energy, regression, or class probabilities; use within `properties[].predictor` |
| `rf_magpie` | RandomForest on Magpie features (no DeepMD); select the target with `model` |
| `ooh` | OOH slab/adsorbate sampling and the trained DeepMD property; units require the model provenance |
| `dummy` | Random-noise predictor for smoke testing |

**Built-in constraint filters** — `filters: [{type: <name>, ...}]`:

| Name | What it does |
|---|---|
| `phase_pattern` | Declarative YAML rule set (required / forbidden / sum / ratio) |
| `ooh_phase` | OOH 5-phase rules (Ni / Co / NiFe / CoFe / NiFeCo) |
| `smact_charge` | SMACT-style charge-neutrality for ionic compositions |
| `last_step_element` | Forces a named element at the final step (oxide-style) |

**FQN form** for both predictors and constraints:
`predictor: "pkg.module:ClassName"` — no repo edit needed.

### Per-element fraction bounds and complete assignments

`element_bounds: {El: [min_frac, max_frac]}` intersects each species' exact menu
with a closed physical interval. Positive lower bounds imply required species.
Use `n_components` to choose a subset size, or assign the full declared list
when every label is a design variable. No Ti-specific last action is needed.

The deprecated `episode_style: fixed_order_amount` spelling means assign every
species once, in a freely chosen order. It no longer fixes the action order or
makes the final species a special remainder variable. One complete label/value
assignment consumes a transition; the feasible-action mask enforces the total.
Integer oxides retain accumulation and their explicit one-positive-O-token rule;
see [domain migration](docs/design_domain_migration.md).

### Flag-naming convention

The YAML/CLI follows a strict prefix convention so it's clear which method a
parameter belongs to:

- `dqn_*` — used only by `train_dqn_online`
- `pg_*` — used only by `train_pg` (REINFORCE / A2C)
- *(no prefix)* — shared across both methods

Examples (already correctly prefixed): `dqn_lr`, `dqn_batch_size`,
`dqn_warmup_eps`, `dqn_hidden_dim`, `pg_warmup_eps`, `pg_num_iters`,
`pg_batch_eps`, `pg_lr_actor`, `pg_lr_critic`. Shared: `method`, `gamma`,
`num_gen_eps`, `gen_temperature`, env keys, predictor keys.

**Deprecated → new** (old names still work with a `DeprecationWarning`):

| Deprecated | Use instead |
|---|---|
| `buffer_size` | `dqn_buffer_size` |
| `num_train_eps` | `dqn_num_train_eps` |
| `grad_steps_per_ep` | `dqn_grad_steps_per_ep` |
| `target_update_freq` | `dqn_target_update_freq` |
| `eps_anneal_eps` | `dqn_eps_anneal_eps` |
| `eps_min` | `dqn_eps_min` |
| `entropy_coef` | `pg_entropy_coef` |
| `repeat_penalty_coef` | `pg_repeat_penalty_coef` |
| `repeat_penalty_shape` | `pg_repeat_penalty_shape` |

To silence the warnings, rename the keys in your YAML — the values are unchanged.

---

## Output files

| File | DQN | PG | Description |
|---|---|---|---|
| `std_scaler.bin` | ✓ | ✓ | Fitted `StandardScaler` (joblib) |
| `qnet.pt` | ✓ | — | DQN Q-network weights |
| `policy.pt` | — | ✓ | Actor (PolicyNet) weights |
| `value_net.pt` | — | A2C | Critic (ValueNet) weights |
| `checkpoint.pt` | ✓ | ✓ | Latest mid-training checkpoint (symlink to numbered file) |
| `training_log.csv` | ✓ | ✓ | Per-episode: return, actor_loss, entropy, critic_loss, … |
| `training_samples.csv` | ✓ | ✓ | One material/property record per budget-consuming training episode, using generated.csv column names |
| `episode_timing.csv` | ✓ | ✓ | Incremental per-attempt wall time, MD/optimization task time, and RL update time |
| `timing.json` | ✓ | ✓ | Run/phase totals and the current invocation's episode timing totals |
| `generated.csv` | ✓ | ✓ | Candidates: formula, reward, dp_mean, dp_std |
| `run_config.json` | ✓ | ✓ | Effective configuration after aliases, CLI overrides, defaults and fallbacks |
| `config_sources.json` | ✓ | ✓ | Origin of each effective top-level configuration value |
| `provenance/*/manifest.json` | ✓ | ✓ | Per-invocation command, runtime flags, code/environment evidence and input hashes |

Training samples are saved automatically as episodes complete. DQN includes
random warmup and online training; REINFORCE/A2C include policy training, while
scaler warmup is outside their budget. Repeats are saved exactly when
`budget_counted` is true. Property values come from that episode's evaluation,
without another predictor call; unavailable scientific uncertainty stays blank.
`candidate_json` retains sequence, categorical and multi-group structure.

`--only-generate` preserves this file. Resume follows the saved checkpoint's
sample session, preserves superseded records in a sibling backup, and marks
missing sample history explicitly. Existing source-code and protocol checks
still apply to checkpoint loading. See [training samples](docs/training_samples.md)
for resume details and a training-versus-generation delta comparison. Use
`obj_<name>_mean` for physical-property distributions when `reward` or `dp_mean`
includes a synthesis-gate penalty. `generated_columns` controls the generated
report; the training export retains all available properties.

### Per-episode timing

Generation-only invocations preserve training logs/timings and instead write
`generation_log.csv`, `generation_episode_timing.csv` and `generation_timing.json`.

Timing is enabled automatically; no YAML option is required. `episode_timing.csv`
is written during training and generation, including failed evaluations and
repeated candidates. It remains available if a later episode stops the run.
Successful training rows also carry these fields in `training_log.csv`.

| Column | Meaning |
|---|---|
| `episode_wall_share_s` | Rollout time + the episode's share of evaluation wall time + its RL update time |
| `eval_wall_share_s` | The complete evaluation wave's elapsed time divided by its attempted episodes |
| `rollout_s` | Action selection/environment work, excluding terminal reward evaluation |
| `rl_update_s` | DQN replay insertion, gradient updates and target copy; for PG, a share of the actual actor/critic update batch |
| `md_task_s`, `opt_task_s` | Elapsed MD/optimization time summed over configurations, including unsuccessful work |
| `property_s` | Property model evaluation and per-objective statistics for that candidate |
| `structure_other_task_s` | Remaining structure-task time when MD and optimization are both instrumented |
| `config_attempted`, `config_succeeded` | Configurations attempted and successfully returned during this evaluation |

**Task seconds and wall seconds are different under parallel execution.** Do not
add `md_task_s` or `opt_task_s` to `episode_wall_share_s`. For example, a 100-second
evaluation wave containing two chemical formulas assigns 50 evaluation wall
seconds to each, while their configuration task times can sum to more than
100 seconds. Cached/duplicate candidates share the wave's wall interval but
report zero new MD/optimization work. A failed formula has `status=failed`, no
training-budget charge, and zero RL update time; its spent evaluation time is
still recorded. A blank stage value means unavailable, not zero.
Configuration counts can also be blank when a serial external builder does not
report its actual attempts; requested counts are not substituted for measurements.
Parallel workers report their dispatched attempts and returned successes directly.

`episode_wall_share_s` excludes global setup, scaler fitting, checkpoint/file
writes and generation's later output diagnostics. These remain in the run-level
timing/overhead. PG's `pg_train` row contains the whole batch's `rl_update_s`;
its `pg_episode` rows split the same time across the actual collected episodes,
so do not sum both kinds of training-log rows. The timing CSV contains episode
rows only. On resume it appends new rows with a new `timing_session` identifier;
`wall_elapsed_s` and wave IDs restart for that invocation. Identify a wave using
`timing_session`, `phase` and `evaluation_wave_id`.

For the external CLC perovskite builder, apply the included MD instrumentation
patch **from the CLC_workflow repository root**, using your rl-matdesign path:

```bash
git apply --check /path/to/rl-matdesign/integrations/clc_workflow_stage_timing.patch
git apply /path/to/rl-matdesign/integrations/clc_workflow_stage_timing.patch
```

Deploy that patched builder together with this rl-matdesign version. The patch
times the LAMMPS subprocess, including failed runs; the shared relaxation helper
times optimization. Without the external patch, MD duration remains unavailable
rather than being estimated from the whole structure-building time. Other
external builders can use the optional hooks in
[`stage_timing.py`](src/rl_matdesign/utils/stage_timing.py).
