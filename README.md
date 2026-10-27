# Reinforce Learning for Materials Design

Config-driven RL runner for materials composition discovery.
Supports multiple material systems (OOH catalyst, HEA, perovskite, oxides) via YAML configs.
Three RL algorithms: **DQN** (online, target-network, or MC targets), **REINFORCE**, and **A2C**.

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
- Reproducing a run = keep the same config + seeds
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

OOH, Ti-alloy, and perovskite scenario YAML files are intentionally local and
gitignored. Examples below using those paths require your existing local configs
and model/structure files. They are not included in a fresh checkout.

---

## Quickstart

```bash
python scripts/run_experiment.py \
    --config configs/test_dummy.yaml \
    --method a2c \
    --out runs/smoke_a2c_seed0 \
    --dp-seed 321 --train-seed 123 --gen-seed 213
```

All hyperparameters (`pg_num_iters`, `pg_batch_eps`, `pg_entropy_coef`, `num_gen_eps`, etc.)
come from the YAML. Override them by editing the config or creating a copy for each experiment.

---

## Seeds

| Flag | Default | Purpose |
|---|---|---|
| `--dp-seed N` | 0 | Predictor seed: controls random structure generation and alloy configs. Acts as fallback for the other two seeds. |
| `--train-seed N` | *(uses dp-seed)* | Training RNG. Also enables GPU deterministic mode (`cudnn.deterministic`, `use_deterministic_algorithms`). |
| `--gen-seed N` | *(uses dp-seed)* | Generation phase only — makes stochastic sampling reproducible independently of training. |

Fix all three for fully reproducible results. `--dp-seed` alone is sufficient for a quick sweep.

---

## REINFORCE (default for OOH)

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 --gen-seed 213 \
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
    --dp-seed 0 --train-seed 42 --gen-seed 99
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
    --dp-seed 0 --train-seed 42 --gen-seed 99
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

---

## Checkpointing and resume

```bash
# 1. Save checkpoints every 50 PG iterations (or DQN episodes)
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 \
    --save-checkpoint-freq 50

# 2. Resume after interruption (loads checkpoint.pt, appends training_log.csv)
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 \
    --resume-training

# 3. Generate candidates from a saved model without re-training
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --only-generate

# 4. Extend training without generating
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --resume-training --skip-generation
```

Use `--load-policy`, `--load-qnet`, `--load-scaler`, `--load-value-net` to load
model files from a non-default path.

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
a `properties:` list with `builder: substitute` and a property whose
`predictor` is `dp_energy`, `dp_property`, or `dp_classification`. The list
automatically selects the reward engine. Composition-only predictors can use
the same engine without a builder; see `configs/oxides_sinter_calcine.yaml` for
a multi-objective example. A minimal YAML:

```yaml
# Environment
species_set: [Ti, Al, V, Cr, Fe, Zr, Nb, Mo, Sn, Hf, Ta]
fraction_set: ["0.00", "0.01", ..., "0.90"]    # whatever step you want
total_units: 100                                # step = 1/total_units
n_components: 11
episode_style: fixed_order_amount               # each step picks an amount;
                                                # element is species_set[step]
element_bounds:                                 # per-element [min, max]
  Ti: [0.45, 0.90]
  Al: [0.00, 0.25]
  # ...

# Predictor
builder: substitute                             # fixed-lattice element swap
base_poscar: data/ti_alloy/FCC.POSCAR           # placeholder 'X' on sub sites
site_symbol: X
k: 1.0
properties:
  - name: energy
    predictor: dp_energy                        # DP potential energy
    models: [models/ti/m1.pt, models/ti/m2.pt]
    direction: min                              # lower energy = better
    objective: mean_minus_kstd                  # reward = -mean - k*std

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
    --out runs/my_run --dp-seed 0
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
n_random_configs: 10
properties:
  - name: delta
    predictor: dp_property
    models: [models/delta.pt]
    output_index: 0
    direction: max
    objective: mean
    scale: 1.0                       # delta stays in its original units
  - name: pure_probability
    predictor: dp_classification
    models: [models/synthesizability.pt]
    head: synthesizability
    output_index: 2                  # [impure, nearly pure, pure]
    direction: max
    objective: mean

reward:
  type: gated
  gates:
    - name: synthesis
      score:
        type: weighted_sum
        terms: {pure_probability: 1.0}
      min: 0.8
      penalty_weight: 1.0
  objective:
    type: weighted_sum
    terms: {delta: 1.0}
  failure:
    reward: -10.0
```

For a binary model trained with labels `[impure, pure]`, change `output_index`
to `1`; class meanings must match the model's training labels. The classifier
uses DeepMD's Python `predict_proba()` directly, requiring a classification-enabled
DeepMD installation. No `dp test` output or probability conversion in
rl-matdesign is needed. The selected probability is averaged across the
`n_random_configs` structures. With several model files, the engine then
computes mean and standard deviation across those model averages; one file
gives `std = 0`. This standard deviation is not `1 - p`.

Here a mean pure probability of at least `0.8` gives reward `delta`; otherwise,
reward is `-10 - (0.8 - pure_probability)`. The failure value is configurable.
To guarantee that every passing candidate outranks every failing candidate,
choose it below the lowest possible feasible objective; a fixed value cannot
guarantee this when the objective has no known lower bound.

Omit `reward` or use `reward: {type: weighted_sum}` to keep the existing weighted
sum. No automatic property normalization is added. Start a new run or recompute
saved rewards after changing the reward definition; old replay-buffer targets
still encode the old reward. See the [YAML reference](docs/yaml_config_reference.md#reward-weighted-sum-or-gated)
for multiple gates, multi-property gate scores, and diagnostics.

### Tier 2 — YAML + ~30-line custom predictor

For novel reward physics (custom property, bespoke aggregation, model from a
non-DeepMD framework), write a class implementing the `PropertyPredictor`
protocol and point at it from YAML via fully-qualified name. No edits to
`scripts/run_experiment.py`.

```python
# my_pkg/elastic_predictor.py
import numpy as np
from rl_matdesign.utils.structure import substitute_sites
from rl_matdesign.training import objective_from_mean_std

class ElasticConstantPredictor:
    def __init__(self, cfg, *, seed=None):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed or 0)
        from deepmd.calculator import DP
        self._calcs = [DP(model=p) for p in cfg["dp_models"]]

    def predict(self, composition):
        structs = substitute_sites(
            self.cfg["base_poscar"], composition,
            self.cfg.get("site_symbol", "X"), 5, self._rng,
        )
        vals = [self._bulk_modulus(a, c) for a in structs for c in self._calcs]
        m, s = float(np.mean(vals)), float(np.std(vals))
        reward = objective_from_mean_std(
            m, s, self.cfg.get("objective", "mean_minus_kstd"),
            float(self.cfg.get("k", 1.0)),
        )
        return reward, s

    def _bulk_modulus(self, atoms, calc): ...     # your physics
```

In the YAML:
```yaml
predictor: "my_pkg.elastic_predictor:ElasticConstantPredictor"
base_poscar: my.POSCAR
dp_models: [my.pt]
# ... any other keys your class reads from cfg
```

The runner uses `importlib` to load the class; install your package (or set
`PYTHONPATH`) so it can be found. Helpful error messages name the registered
built-ins if the FQN fails.

### Tier 3 — specialized (OOH-like)

For complex domain physics (e.g. OOH adsorbate placement → Sabatier
overpotential) or constraints that exceed the declarative `phase_pattern`
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
| `ooh` | OOH catalyst overpotential (adsorbate placement + Sabatier formula) |
| `dummy` | Random-noise predictor for smoke testing |

**Built-in constraint filters** — `constraint_filter: <name>`:

| Name | What it does |
|---|---|
| `phase_pattern` | Declarative YAML rule set (required / forbidden / sum / ratio) |
| `ooh_phase` | OOH 5-phase rules (Ni / Co / NiFe / CoFe / NiFeCo) |
| `smact_charge` | SMACT-style charge-neutrality for ionic compositions |
| `last_step_element` | Forces a named element at the final step (oxide-style) |

**FQN form** for both predictors and constraints:
`predictor: "pkg.module:ClassName"` — no repo edit needed.

### Per-element fraction bounds + `fixed_order_amount`

`CompositionEnv` accepts:

- `element_bounds: {El: [min_frac, max_frac]}` — per-element fractional range.
- `episode_style: fixed_order_amount` — every element in `species_set` appears
  in fixed order; the agent picks the amount per step; the last step's amount
  is forced to satisfy the sum constraint.

The two together enable the Ti-alloy-style use case. The default
`episode_style: element_then_amount` (with `element_bounds` omitted) preserves
the existing pick-element-then-pick-fraction behavior used by OOH/HEA/etc.

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
| `generated.csv` | ✓ | ✓ | Candidates: formula, reward, dp_mean, dp_std |
| `run_config.json` | ✓ | ✓ | Full config + argparse namespace for reproducibility |
