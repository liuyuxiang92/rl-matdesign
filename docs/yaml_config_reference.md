---
title: "rl-matdesign — YAML Configuration Reference"
subtitle: "Every config flag, what it does, and its choices"
date: "2026-06-10"
geometry: "margin=1.8cm"
fontsize: 10pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
colorlinks: true
---

# Overview

A run is configured by one YAML file plus a few CLI flags. The three pluggable
layers configure the search space, property prediction, and action masking.
A non-empty **`properties`** list selects the reward engine; each entry names
its **`predictor`**. A flat top-level `predictor` is also supported for a single
standalone predictor. The RL and generation knobs are shared across scenarios.

- The YAML is the base; a few CLI flags override matching keys.
- **Seeds are CLI-only**: `--dp-seed` (predictor RNG), `--train-seed` (training;
  defaults to `--dp-seed`), `--gen-seed` (generation; defaults to `--dp-seed`).
- The full merged config is written to `<out>/run_config.json`.

**CLI flags:** `--config` (required), `--out` (required), `--method`,
`--dp-seed`, `--train-seed`, `--gen-seed`, `--device` (`cpu`/`cuda`),
`--dqn-loss`, `--dqn-augment-permutations`, `--save-checkpoint-freq`,
`--resume-training`, `--only-generate`, `--skip-generation`, `--max-gen-attempts`,
`--load-qnet` / `--load-policy` / `--load-value-net` / `--load-scaler`.

\newpage

# 1. Environment

| Key | What it does | Choices / default |
|---|---|---|
| `env_type` | Selects the environment | `fraction` (default, `CompositionEnv`); `integer_ratio` (`IntegerRatioEnv`); `multi_group` (`MultiGroupEnv`) |

## `fraction` env keys

| Key | What it does | Choices / default |
|---|---|---|
| `species_set` | Candidate element symbols | **required** (list) |
| `fraction_set` | Allowed fraction strings (grid) | default = built-in `0.05 … 0.80` |
| `total_units` | Grid resolution; step = `1/total_units` | `20` (->0.05); use `100` for 0.01 |
| `n_components` | Distinct elements per episode | `5` |
| `anion_formula` | Fixed suffix appended to the formula | `""` (e.g. `O2H1`, `O3`) |
| `episode_style` | How each step picks | `element_then_amount` (default: choose element + amount); `fixed_order_amount` (order pinned to `species_set`, choose only amounts) |
| `element_bounds` | Per-element `{el: [min,max]}` fraction caps | none; **only with `fixed_order_amount`** |
| `constraint_filter` | Action-masking filter (see §3) | `null` |

## `integer_ratio` env keys

| Key | What it does | Choices / default |
|---|---|---|
| `species_set` | Candidate elements | **required** |
| `ratio_set` | Allowed integer-ratio digits | default `"0" … "9"` |
| `n_components` | Elements per episode | `5` |
| `constraint_filter` | Action-masking filter | `null` |

No sum-to-1 constraint; the composition is normalized afterward.

\newpage

# 2. `multi_group` env — `groups:`

`groups:` is an **ordered list** of sublattice **groups**, filled in order. A group
is the atomic unit; a single `fraction`/`integer_ratio` env is the one-group case.
The predictor receives the structured `{group_name: {element: value}}`.

**Common group keys**

| Key | What it does | Choices / default |
|---|---|---|
| `name` | Group label (appears in the structured terminal) | string |
| `kind` | Group type | `composition` (default) or `categorical` |
| `sites` | Sublattice size (atoms per formula unit) — bridges fraction<->count and assembles the chemical formula (`amount × sites`) | `1` |
| `constraint_filter` + its kwargs | Per-group filter (sees `prior_groups` for cross-group coupling) | (see §3) |

**`kind: composition`** — pick N elements with amounts summing to 1. Reuses the
`fraction`-env keys (`species_set`, `fraction_set`, `total_units`, `n_components`,
`episode_style`, `element_bounds`) plus friendly knobs:

| Key | What it does | Choices / default |
|---|---|---|
| `amount` | Generate the value grid instead of writing `fraction_set` | `{min, max, step}` or a list |
| `host` | A host element auto-takes the leftover (list only the dopants; wires a `host_complement` filter) | element symbol |

**`kind: categorical`** — pick discrete **real values**; no sum-to-1. Each slot is
one element with its own value list; the terminal returns the chosen values
unchanged (so a builder reads `Cl = 1.0` / `O = 1` directly).

| Key | What it does |
|---|---|
| `choices` | `[{element, values: [...]}, …]` — one slot per element |

A categorical filter may mask a slot by an earlier group (e.g. `sse_doping` masks
the O slot by the P-site metal's category).

# 3. Constraint filters (`constraint_filter:`)

| Value | What it does | Its keys |
|---|---|---|
| `null` | No constraint (default) | — |
| `smact_charge` | Keep only compositions that can be charge-neutral (SMACT oxidation states) | `smact_anions: [{symbol, charge, stoich}]` (or `smact_anion` / `smact_anion_charge` / `smact_anion_stoich`) |
| `last_step_element` | Force the last step to a required element | `required_elements`; `nonzero_ratio_at_last` (true); `reserve_for_last` (true) |
| `phase_pattern` | Keep compositions matching phase patterns | `phase_patterns` |
| `ooh_phase` | OOH oxyhydroxide phase screen | `target_phases` (default `[any]`) |
| `chain` | Apply several filters in sequence | `filters: [ {constraint_filter: …, …}, … ]` |
| `host_complement` | Dopants at a level, host takes the rest (wired automatically by a composition group's `host` knob) | `host_element`, `levels` |
| `sse_doping` | Mask the LiPS S-site O-form slot by the metal's category | `o_element` (`O`), `host_P` (`P`), `metal_only`, `oxide_only` |
| `"pkg.mod:Class"` | Your own filter (FQN) | your class's keys |

The `ooh_phase` sampler retains its five-cation rules: named primary elements
must be selected for the requested phase. `any` accepts the union of these phases.

The terminal OOH primary-phase checker counts every cation outside each primary
set toward the dopant limit, including species absent from the historical dopant
list. The dopant fraction must remain strictly below 25%.

\newpage

# 4. Property prediction and reward

A non-empty `properties:` list selects the reward engine automatically, for
one or many properties. Set `properties[].predictor` to one of the following:

| Value | What it does |
|---|---|
| `dp_energy` | DeepMD potential energy on built structures |
| `dp_property` | DeepMD regression head on built structures |
| `dp_classification` | DeepMD class probabilities on built structures |
| `rf_magpie` | RandomForest prediction from composition; `model` selects the checkpoint |
| `ooh` | OOH overpotential (adsorbate slabs) |
| `mgtransformer` | External MGTransformer predictor |
| `dummy` | Random score for smoke tests |
| `"pkg.mod:Class"` | Your own predictor (FQN) |

The three `dp_*` entries are structure-scoring branches of the reward engine,
not standalone top-level predictors. Registry leaves and FQNs can also be used
as a flat top-level `predictor`. `predictor: structure_score` remains an engine
alias but is unnecessary with a non-empty `properties` list.

## Reward engine keys

The pipeline is `build -> [relax, optional] -> score properties -> combine`.
Composition-only properties do not require a structure or a builder. The
combiner supports any property names, independent of the material system.

| Key | What it does | Default |
|---|---|---|
| `builder` | Builder name/FQN (`substitute` = fixed-lattice element swap; `sse` = doped supercell; or `pkg.mod:Class`) | `substitute` |
| `share_structure` | `true` = build+relax shared structures, score all structure properties on them; `false` = each property builds its own | `true` |
| `n_random_configs` | Random placements per candidate | `1` |
| `geo_opt` | Relaxation stage (see below); omit / `enabled: false` to skip | optional |
| `structure_parallel` | Persistent GPU workers for independent configurations; see below | omitted: serial execution |
| `properties` | Non-empty list of property specs (see below) | required |
| `reward` | Combination rule: `type: weighted_sum` or `type: gated` | weighted sum |
| `k` | Uncertainty coefficient for property objectives | `1.0` |
| `sweep` | Optional `{name, values}` — optimize a shared operating condition (e.g. temperature) per composition (see below) | optional |
| `base_poscar` / `site_symbol` | Builder knobs (for `substitute`); inherited by per-property builders when `share_structure: false` | — |

`geo_opt:` sub-keys — `model` (default `models/DPA-3.1-3M.pt`), `head` (user-defined),
`fmax` (`0.001`), `steps` (`1000`), `relax_cell` (`true`), `enabled` (`true`).

`properties[]:` sub-keys:

| Key | What it does | Default |
|---|---|---|
| `name` | Unique label (CSV column prefix) | `prop{i}` |
| `predictor` | `dp_energy`, `dp_property`, `dp_classification`, a registry leaf, or an FQN | **required** |
| `models` (or legacy `dp_models`) | DeepMD ensemble checkpoint list; registry leaves use their own keys such as `model` | required for `dp_*` |
| `head` | DeepMD branch name; explicitly select the classification head in multi-head checkpoints | none |
| `direction` | `max` / `min` / `target` (`dp_energy`: use `min` for "lower is better") | **required** |
| `target_value` | Value to hit, in the property's real units. Required with `direction: target`. **Not** the leaf's `model:`/checkpoint key. | — |
| `target_tolerance` | Deadband half-width; inside it the term is at its maximum (0) and stops competing with other objectives | `0.0` |
| `weight` / `scale` | Combine weight / unit scale | `1.0` / `1.0` |
| `objective` | `mean_minus_kstd` / `mean` / `mean_plus_kstd` | `mean_minus_kstd` |
| `energy_per_atom` *(`dp_energy`)* | Normalize by atom count | `true` |
| `output_index` / `output_aggregator` *(`dp_property`, `dp_classification`)* | Select a vector component, or collapse output channels using `mean` / `max` | aggregator `index`; index `0` for regression, **explicit index required for classification in `index` mode** |
| `transform` | `none`, or `exp` for regression heads predicting log-values; classification uses `none` | `none` |
| `fparam` / `aparam` *(`dp_property`, `dp_classification`)* | Frame / atomic parameters for heads trained with them. A `null` in `fparam` marks the slot filled by the `sweep` value (requires a top-level `sweep`) | none |

The default weighted-sum reward is `sum weight * value(prop) / scale`, where `value` is

- `max`/`min`: `objective_from_mean_std(direction*mean, std, objective, k)`
- `target`:    `objective_from_mean_std(-e, std, objective, k)` with
  `e = max(0, |mean - target_value| - target_tolerance)`

i.e. a target objective scores the (deadbanded) **distance** to `target_value`, negated so
closer is better. Routing it through the same helper keeps uncertainty handling identical:
`mean_minus_kstd` reads as "effectively farther from the target when less certain".

### What `mean` and `std` actually are

Two axes, folded separately and deliberately not pooled:

- **structures** — the `n_random_configs` random decorations of one composition. Their
  scatter is *configurational*: it is how the property is defined for a disordered
  composition, so it is **averaged away**.
- **models** — independent checkpoints of the same property. Their scatter is *epistemic
  uncertainty*, and is what `k*std` penalises.

So `mean` is the mean of the per-model averages and `std` is the spread **across models**.
A single model therefore gives `std == 0` exactly, whatever `n_random_configs` is, and
`mean_minus_kstd` is then identical to `mean` — no flag needed.
The same fold is used in both `share_structure` regimes.

### `structure_parallel`: per-configuration GPU workers

This optional block applies to the structure-scoring reward engine. It detects
available GPUs and distributes one candidate's configurations over persistent
subprocesses, each pinned to one GPU. It can measure whether one or two workers
per GPU improves throughput. Each task calls the configured builder with
`n_configs=1` and a task-local RNG, then applies top-level `geo_opt` if enabled.
For a builder that already runs
MD and optimization, such as `clc_workflow.rl_builder:MDAveragedOptBuilder`, both
stages run on the same worker; set `geo_opt: null` to avoid another relaxation.
The external builder itself needs no parallel implementation.

```yaml
share_structure: true
n_random_configs: 5
structure_parallel:
  devices: auto
  workers_per_device: auto
  reserve_devices: []
  threads_per_worker: 1
  timeout_s: 7200
  startup_timeout_s: 300
  min_successful_configs: 5
  min_speedup: 1.15
```

| Key | What it does | Default |
|---|---|---|
| `devices` | `auto` detects GPUs in the inherited allocation; alternatively, a non-empty list of distinct nonnegative visible GPU indices | `auto` |
| `workers_per_device` | `auto` measures one vs two workers per GPU when useful; `1` or `2` fixes the count and skips calibration | `auto` |
| `reserve_devices` | Visible GPU indices to leave out of worker allocation; requires `devices: auto`; `[0]` reserves GPU 0 for the parent | `[]` |
| `threads_per_worker` | CPU numerical-library threads per worker; positive integer | `1` |
| `timeout_s` | Maximum elapsed seconds for one configuration task, including build and optional relaxation | `3600` |
| `startup_timeout_s` | Maximum seconds allowed for worker initialization | `300` |
| `min_successful_configs` | Minimum completed configurations required to score this candidate; integer from 1 through `n_random_configs` | `1` |
| `min_speedup` | Required throughput ratio for auto mode to retain two workers instead of one; finite and at least 1 | `1.15` |

Concurrency is capped at the smaller of `n_random_configs` and the sum of the
selected worker counts across GPUs. Extra configurations wait for a free worker;
each worker runs only one task at a time. Four GPUs with two workers each can run
eight configurations concurrently only if at least eight configurations are
requested. Models cached by a builder or its optimizer remain loaded in that
worker across candidates. Extra slots used only for calibration are closed if
the requested ensemble does not need them. Results
are collected in configuration-index order, then scored in the parent process.
Property inference, reward combination, and RL updates are not distributed by
this option. A gated reward still evaluates after structure preparation and
property prediction; it does not avoid the MD cost of a failing candidate.

**Automatic concurrency selection.** With `workers_per_device: auto`, calibration
is needed only when `n_random_configs` exceeds the number of selected GPUs. For
example, five configurations on five GPUs use one worker per GPU without testing
two. Otherwise, the first candidate is used for the following calibration on each
tested GPU, using the same candidate and configuration seed:

1. Warm up the primary worker with one complete build/MD/optimization task.
2. Time a second complete task on that worker alone (`t_single`).
3. Start and warm up a second worker with the same task.
4. Time both workers running the task concurrently (`t_pair`, the longer completion time).

This costs **up to five additional full configuration executions per tested GPU**. The
calibration structures are excluded from the reward and do not count as additional
RL episodes, but their wall time is part of the run. The throughput ratio is
`2 * t_single / t_pair`; retain two workers only if it is at least `min_speedup`.
Reported task errors in the second-worker trial, including out-of-memory errors,
select one worker. A failure in the one-worker baseline fails the current candidate;
worker crashes and timeouts follow the pool-abort behavior described below.
The log reports `concurrency`, `reason`, `t_single`, `t_pair`, and `speedup` for each
GPU. `ParallelStructureBuilder.autotune_report` exposes the selected `counts` and
per-GPU `measurements` after calibration. This is an empirical choice for the first
candidate, not a guarantee that the same count is optimal for every later
composition. There is no automatic recalibration after a successful calibration.
Explicit `workers_per_device: 1` or `2` skips these trial executions.

**GPU allocation.** Indices are relative to the parent process's inherited
`CUDA_VISIBLE_DEVICES`, including allocations expressed as GPU UUIDs. If that
variable is `2,4,7`, index `1` means the allocation's second GPU, physical card 4.
Without that variable, indices select the CUDA device ordinals directly. A worker
exposes only its selected GPU, so its DeepMD and LAMMPS processes use local GPU 0.
Keep LAMMPS configured for one GPU per process (for example `-k on g 1`). Overrides
of `CUDA_VISIBLE_DEVICES`, `LOCAL_RANK`, and `DEVICE` in `md.env` are rejected so
they cannot bypass worker assignment.

By default `devices: auto` includes all visible GPUs, including GPU 0 normally
used by the parent. The parent DeepMD device is controlled by the DeepMD
environment; the experiment's `--device` controls RL. To give the worker pool the
available GPUs without sharing with parent models, launch with `DEVICE=cpu` and
`--device cpu`; workers still explicitly use CUDA. This trades GPU space for slower
CPU property inference. Alternatively, use `devices: auto` with
`reserve_devices: [0]` to leave GPU 0 for the parent, or supply a manual device list
such as `[1, 2, 3]`. At least one worker GPU must remain. The parent preloads its
persistent DeepMD energy, regression, and classification models before the first
parallel build, so calibration includes their resident memory. Later external GPU
load or additional per-property worker pools can still change available memory
and throughput. Start one experiment process, without `torchrun`.
The subprocess implementation targets POSIX systems; actual CUDA
execution requires a compatible Linux GPU environment and the installed builder,
DeepMD backend, and LAMMPS binary to support that environment.

**Failures and timeouts.** Ordinary task exceptions are logged. The predictor
scores the successful configurations if their count reaches
`min_successful_configs`; otherwise the candidate evaluation raises an error.
Workers propagate optimizer execution exceptions, including CUDA OOM, even when
an external builder calls the shared `relax_structure` helper internally. This
prevents failed optimizations from appearing as fast calibration successes.
Reaching the optimization step limit without convergence retains the existing
warning behavior; it is distinct from an execution exception.
The default of 1 permits a mean over fewer than the requested configurations;
set it to `n_random_configs` when every candidate must use the full sample count.
A worker crash or timeout aborts the current candidate and discards the worker
pool; the next prediction creates a new pool. The experiment's existing handling
of failed candidate evaluations still applies. Choose the timeout to allow a
complete MD-plus-optimization task, including first-use model initialization.
The CLI closes worker process groups on normal exit, exceptions, Ctrl+C, and
SIGTERM termination. SIGKILL cannot run Python cleanup and provides no such
guarantee.

**Builder contract and reproducibility.** The builder must return exactly one
ASE `Atoms` object in a sequence when called with `n_configs=1`; builders that
return multiple templates from that call are unsupported and raise an error.
Task seeds are assigned before scheduling, so builders that honor the supplied
RNG receive the same configuration seeds regardless of completion order or worker
count. They are not the same samples as the legacy serial path for an identical
predictor seed, and CUDA computations may still have numerical nondeterminism.
For the MD builder, configured `md_debug_dir` output is separated by run and worker
to prevent filename collisions.

With `share_structure: false`, each structure property inherits the top-level
`structure_parallel` block and may override it. Properties are still evaluated
one after another, with configurations parallel within each property. Independent
pools can retain separate model copies on the same GPUs; prefer
`share_structure: true` when properties can use the same structures.
Omit `structure_parallel` entirely to preserve the existing serial behavior.

### `dp_classification`: class probabilities

Use a classification-enabled DeepMD installation that exposes
`DeepClassification.predict_proba`. The engine loads each checkpoint using the
public `DeepEval(..., head=...)` factory and calls `predict_proba()` directly.
DeepMD returns a probability array of shape `(n_frames, num_classes)`.
The engine does not parse `dp test` output, consume logits, or implement softmax.
There is no `output_kind` setting.

```yaml
properties:
  - name: pure_probability
    predictor: dp_classification
    models: [models/synthesizability.pt]
    head: synthesizability
    output_aggregator: index
    output_index: 2                  # trained order: impure, nearly pure, pure
    direction: max
    objective: mean
```

For a binary model whose training labels are `[impure, pure]`, set
`output_index: 1`. The engine checks the index against the model's number of
classes; the user must supply its meaning from the training labels. Class
count alone cannot identify which class means "pure". For ensembles, all
checkpoints must use the same class order and meaning.

`output_aggregator` acts on the class channels **within one frame**:
`index` selects the requested class; `max` selects the most confident class
regardless of its identity; `mean` averages the classes and is always `1/K`
for `K` classes, so it is unsuitable as a synthesizability score. It does not
control averaging across configurations.

For class `k`, the composition score is the mean of `p[j,k]` across the
`n_random_configs` structures. Multiple checkpoints then follow the model
mean/std fold above. A single checkpoint gives `std = 0`, even when its
configuration probabilities differ. `1 - p_pure` means predicted probability
of a different class; it is not the model-ensemble standard deviation. The
probabilities themselves do not provide an independent uncertainty estimate.

### `reward`: weighted sum or gated

Omitting `reward` or writing `reward: {type: weighted_sum}` preserves the
existing combination and property weights. No automatic normalization is
introduced: use the default `scale: 1.0` to keep delta in its original units.

For a piecewise reward, choose `type: gated`. Every gate must pass before the
objective is used. Gate scores and the objective can each combine multiple
properties. For example, append this block to a configuration with properties
named `delta`, `conductivity`, `pure_probability`, and `stability`:

```yaml
reward:
  type: gated
  gates:
    - name: synthesis
      score:
        type: weighted_sum
        terms: {pure_probability: 1.0}
      min: 0.8
      penalty_weight: 1.0
    - name: combined_requirement
      score:
        type: weighted_sum
        terms: {stability: 0.7, pure_probability: 0.3}
      min: 0.5
      max: 1.0
      penalty_weight: 2.0
  objective:
    type: weighted_sum
    terms: {delta: 1.0, conductivity: 0.2}
  failure:
    reward: -10.0
```

The names and numeric thresholds above are illustrative; choose coefficients
and thresholds for your properties' units and meaning. No environment,
builder, predictor, or property name is hard-coded in the combination rule.

| Setting | Meaning |
|---|---|
| `reward.type` | `weighted_sum` (default) or `gated` |
| `reward.objective` | Optional `{type: weighted_sum, terms: {property_name: coefficient}}`; omitted means all properties with their configured `weight` |
| `reward.gates` | Non-empty list for `gated`; all gates must pass |
| `gates[].name` | Unique label used in diagnostics |
| `gates[].score` | `{type: weighted_sum, terms: {property_name: coefficient}}` |
| `gates[].min` / `max` | At least one bound required; both ends are inclusive |
| `gates[].penalty_weight` | Nonnegative coefficient multiplying this gate's violation; default `1.0` |
| `reward.failure.reward` | Required for `gated`; base reward for any failing candidate, chosen for the objective's range |

Gate scores use **raw property means**: property `direction`, `objective`,
`weight`, and `scale` are not applied. Thus a gate on `pure_probability` compares
the actual averaged probability to its threshold. Its score is
`g = sum(coefficient * property_mean)` and its violation is
`max(min - g, 0) + max(g - max, 0)`, omitting a term when that bound is absent.
This also permits multi-property requirements without assigning their signs
implicitly.

The optimization objective uses the same direction, target, uncertainty, and
optional scale handling as the legacy weighted sum. Explicit `terms`
coefficients **replace** each selected property's `weight`; they do not
multiply it again. Only properties named in `terms` contribute. An omitted
objective uses all property weights, so specify an objective explicitly when
some properties should only serve as requirements. An explicit objective is
also available with `type: weighted_sum`.

If every gate passes, reward equals the objective. Otherwise it equals
`failure.reward - sum(penalty_weight * violation)` over all gates. With only a
pure-probability gate at `0.8` and base `-10`, a candidate at `0.7` gets `-10.1`.
Penalties use the gate score's raw units. They do not rescale delta or the
objective. A fixed failure base guarantees that every passing candidate ranks
higher only when it is below a known lower bound of the feasible objective.
Choose it for the actual objective range; no such guarantee follows for an
objective with an unknown or unbounded lower range.

The same reward is used for training, generation, and operating-condition
sweeps. The predictor's `reward_details` exposes objective and gate diagnostics.
Generated candidate CSVs include `reward_objective`, `gate_passed`, and
`gate_<name>_value`, `gate_<name>_passed`, `gate_<name>_violation`; per-property
`obj_<name>_mean/std` retain their original meaning. `dp_mean` applies the same
reward rule with objective standard deviations set to zero, including the
same gates and failure branch; it is not an individual property's mean.
Changing the reward does not rewrite saved replay-buffer targets. Start a new run or recompute those targets
before training with the new definition.

### `sweep:` — per-composition operating-condition optimization

Some property heads depend on an external operating condition (e.g. temperature)
passed as a frame parameter. `sweep: {name, values}` turns that condition into an
**inner optimization**: for each composition the predictor scores every property
at each value and keeps the single shared value maximizing the **combined**
reward. Because the structure is built + relaxed **once** and reused, sweeping is
cheap (only the property inference repeats).

```yaml
sweep:
  name: temperature
  values: [460, 470, 480, 490]
properties:
  - name: conductivity
    predictor: dp_property
    fparam: [3.9568e-05, null, 6]      # null = the swept temperature slot
    direction: max
    ...
  - name: stability
    predictor: dp_property
    fparam: [3.9568e-05, null, 6, 4]
    direction: max
    ...
```

The chosen value is logged per candidate as `obj_<name>_mean` (e.g.
`obj_temperature_mean`) in `generated.csv`. A property with no `null` in its
`fparam` (or a `dp_energy` predictor) is condition-independent — scored once and
contributing equally at every sweep value.

**Migration cheatsheet** (old -> `properties` entries):

| Old predictor | Now |
|---|---|
| `dp_structure` | `builder: substitute`, one property `predictor: dp_energy`, `direction: min` |
| `dp_property` | `builder: substitute`, one property `predictor: dp_property` |
| `hea` / `perovskite` | as `dp_structure`, with `site_symbol: X` / `Fe` |
| `structure_pipeline` | `share_structure: true` (default), N `predictor: dp_property` properties |
| `composite` | `share_structure: false`, the `objectives:` list becomes `properties:` |

## `ooh` keys

`base_poscar`, `dp_models`, `objective`, `k`, `n_random_configs`, `ads_height`,
`ads_dz`, `geo_opt` (bool), `geo_opt_model`, `uncertainty`
(`models`/`configs`/`total`), `output_index`, `adsorbates`, `target_phases`.

`adsorbates` selects which intermediates are placed on each doped slab, in frame
order — default `[O, OH, OOH]`. An **empty list** means the bare parent slab (no
adsorbate atoms at all): one frame per random config instead of three, so ~3x
fewer DeepMD evaluations and ~3x fewer relaxations under `geo_opt: true`.
`ads_height` / `ads_dz` are unused in that case.

Bare is expressed as the empty list rather than as a member of the list because
all frames in one batch must have the same atom count (an adsorbate frame has
`nat_slab + 3` atoms, a bare one `nat_slab`).

The list order matters: the batch is frame-major and `output_index` indexes the
flattened output, so with the default list `output_index: 0` reads the **O\***
frame. Changing `adsorbates` or `output_index` invalidates a saved `dp_cache`
(they are folded into the cache key), so a resumed run pays a one-time recompute
instead of returning values that describe a different structure.

## `rf_magpie` keys

`model` is the joblib checkpoint path. Use separate named properties for
sintering and calcination, as in `configs/oxides_sinter_calcine.yaml`.

\newpage

## Writing your own predictor — the leaf contract

A **leaf represents exactly one model**; the reward engine owns all folding. That is what
makes "one model => no uncertainty" automatic instead of a per-scenario flag.

`model:` accepts one path **or a list of paths**. The engine builds one leaf instance per
path (each with a decorrelated seed), so an ensemble is a config change, never a code
change:

```yaml
model: ckpt/a.pt                    # 1 instance -> std == 0
model: [ckpt/a.pt, ckpt/b.pt]       # 2 instances -> std = their disagreement
```

Expose whichever of these fits; everything is auto-detected, with no YAML flag:

| method | returns | who folds | use when |
|---|---|---|---|
| `score_structures(atoms_list)` | one float **per structure** | engine | one model, geometry-dependent property |
| `score(composition)` | one float | engine | one model, stoichiometry-only property |
| `predict_structures(atoms_list)` | `(mean, std)` | leaf | leaf has a genuine *internal* ensemble |
| `predict(composition)` | `(mean, std)` | leaf | same, composition-based |

- **structure vs composition** — a leaf exposing either `*_structures` method receives the
  built (and optionally relaxed) cells; otherwise it receives the candidate composition.
- **engine-folded vs self-folding** — the `score*` form wins when both are present. The
  self-folding form is correct only when the leaf really does hold its own ensemble (e.g.
  `rf_magpie`'s random-forest tree variance); its `(mean, std)` is passed through untouched.

Prefer `score*` for anything wrapping a single checkpoint: fold your own structures and you
will report *configurational* scatter as if it were model uncertainty, which
`mean_minus_kstd` will then silently penalise.

Constructor is `__init__(self, cfg, *, seed=None)`, where `cfg` is the whole `properties[]`
entry. Add an optional `close()` to release external resources (subprocesses, file handles);
the engine calls it on every instance at the end of a run.

## `mgtransformer` keys

Bridge to an external MGTransformer checkout (`../MGTransformer`), one persistent
`serve.py` subprocess per instance.

| Key | Meaning | Default |
|---|---|---|
| `model` | Path to a finetuned checkpoint, forwarded as `serve.py --ckpt`. MGTransformer derives the target — and hence which calibration constants apply — from the path, so it is never named twice. | **required** |
| `mgt_repo` | Path to the MGTransformer checkout | **required** |
| `mgt_python` | Interpreter of MGTransformer's own env (its deps conflict with this repo's) | **required** |
| `device` | Forwarded to `serve.py --device` | `cpu` |
| `max_neighbors`, `cutoff`, `atom_features`, `triplet_endpoint`, `triplet_pad_mode` | Featurizer overrides forwarded to `serve.py` | see that repo |

Scores arrive in **real units** (eV, eV/atom): MGTransformer un-normalizes with the
train-split constants in its `mgt_calibration.json`. A target with no entry there falls back
to a raw z-score and warns once — ranking is unaffected either way, but a `direction: target`
objective would then be meaningless.

# 5. Builders (`builder:`)

A builder turns the agent's pick (a flat `{element: fraction}` or a structured
`{group: {…}}`) into ASE structures. Built-ins:

| `builder:` | What it does |
|---|---|
| `substitute` | Fixed-lattice element swap (no vacancies). **Single-sublattice**: fills `site_symbol` placeholder sites from a flat `{element: fraction}`. **Multi-sublattice**: add `site_map: {group: symbol}` and it fills each sublattice from a structured `{group: {element: fraction}}` candidate, one op per group. Keys: `base_poscar` (or `poscar`), `site_symbol` (default `X`), `site_map`. |
| `site_pick` | One element per site across N sublattices via `site_map` — the *categorical* counterpart of `substitute`'s multi-sublattice mode (no fractions). |
| `defect_site` | A/B-site doping plus a signed vacancy/interstitial defect axis. |
| `sse` | Doped-supercell recipe (P->metal, S->O/Cl/Br, charge-neutral Li vacancies). Keys below. |
| `"pkg.mod:Class"` | Your own builder (FQN) with `build(candidate, *, n_configs, rng)`. |

## `sse` builder keys

| Key | What it does | Default |
|---|---|---|
| `base_poscar` | Base supercell | required |
| `valences` | Charge table `{el: int \| {sulfide, oxide}}` (drives the Li-vacancy solve **and** oxide O count) | required |
| `halide_total` | Cl + Br per formula unit | `1.7` |
| `eligible_region` | Substitutable S region | `{symbol: S, take: last, count: 1000}` |
| `formula_units` | Formula units in the supercell — **inferred from the POSCAR** (host-P count / `p_site_per_fu`) unless set | inferred |
| `p_site_group` / `s_site_group` | Group names to read | `P_site` / `S_site` |
| `host` | `{P, S, Li}` host symbols | `{P:P, S:S, Li:Li}` |
| `p_site_per_fu` / `s_site_per_fu` / `li_per_fu` | Sites per formula unit | `1` / `6` / `6` |

The S-site reads the categorical values directly — O is a form flag (`0` metal /
`>0` oxide), Cl is a real per-f.u. count. `Br = halide_total − Cl` and the
charge-neutral Li vacancy are derived. (No `cl_map` / `o_off`.)

# 6. RL method & hyperparameters

| Key | What it does | Choices / default |
|---|---|---|
| `method` | Algorithm (CLI `--method` overrides) | `a2c` (default), `reinforce`, `dqn` |

## Policy gradient (`a2c` / `reinforce`)

| Key | Default | Key | Default |
|---|---|---|---|
| `pg_warmup_eps` | 200 | `pg_entropy_coef` | 0.01 |
| `pg_num_iters` | 1000 | `pg_entropy_min` | 0.3 |
| `pg_batch_eps` | 15 | `pg_repeat_penalty_coef` | 0.0 |
| `pg_lr_actor` | 0.001 | `pg_repeat_penalty_shape` | `log` (also `sqrt`) |
| `pg_lr_critic` | 0.001 (a2c only) | `gamma` (or `pg_gamma`) | 0.9 |

Advantages are **standardised across each batch** before the actor update. This is
unconditional and has no flag: without it the actor term scales with the raw
reward (sintering temperatures are 400–700, so |advantage| is O(hundreds)) while
the entropy bonus is at most `pg_entropy_coef · ln|A| ≈ 0.5`, which leaves the
entropy term ~2 orders of magnitude too small to matter. Two consequences:

- `pg_entropy_coef` and `pg_repeat_penalty_coef` are in **standard deviations of
  batch return**, so they mean the same thing in every scenario whatever units the
  property has. A `pg_repeat_penalty_coef` of 0.1 shifts the advantage by 0.1σ.
- In `training_log.csv`, `repeat_penalty` is now in σ, while `return_shaped`
  converts it back to the property's units so it stays comparable against
  `return_raw`. Logs written before this change match neither.

`pg_entropy_min` is a floor on **normalised** entropy `H / ln|A|` in `[0, 1]`, not
absolute nats — `|A|` is ~268 for the 80-element oxide env but far smaller for
OOH, so an absolute floor would not port. A proportional controller raises the
effective entropy weight while entropy sits below the floor and decays it back to
`pg_entropy_coef` above it; the base value is the lower clamp, so the controller
can only ever add exploration pressure. Set `pg_entropy_min: 0` to disable.

Watch `entropy_norm` and `entropy_coef_eff` in `training_log.csv`: `entropy_norm`
is directly comparable to `pg_entropy_min`, and a run whose `entropy_coef_eff` is
pinned at the ceiling is telling you the policy wants to collapse harder than the
controller can prevent — raise `pg_entropy_coef`.

## DQN

| Key | Default | Key | Default |
|---|---|---|---|
| `dqn_lr` | 0.001 | `dqn_target_update_freq` | 100 |
| `dqn_hidden_dim` | 256 | `dqn_eps_anneal_eps` | 10000 |
| `dqn_batch_size` | 256 | `dqn_eps_min` | 0.05 |
| `dqn_warmup_eps` | ->`pg_warmup_eps`/500 | `dqn_gamma` (or `gamma`) | 0.9 |
| `dqn_num_train_eps` | 20000 | `dqn_loss` | `smoothl1` (also `mse`) |
| `dqn_buffer_size` | 50000 | `dqn_augment_permutations` | 0 |
| `dqn_grad_steps_per_ep` | 5 | | |

## Generation (all methods)

| Key | What it does | Default |
|---|---|---|
| `num_gen_eps` | Final candidates to generate | 200 |
| `exploration_gen_eps` | Extra high-temperature candidates | 0 |
| `gen_temperature` | Sampling temperature (DQN Boltzmann / PG) | 1.0 |
| `gen_top_frac` | Restrict sampling to top fraction of actions | 0.0 |
| `gen_epsilon` | ε-random during generation | 0.0 |
| `k` | Uncertainty coefficient for reward folding | 1.0 |
