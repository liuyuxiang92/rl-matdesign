---
title: "rl-matdesign — YAML Configuration Reference"
subtitle: "Every config flag, what it does, and its choices"
date: "2026-11-06"
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
an inline **`predictor`** or a shared **`model_ref`**. A flat top-level `predictor` is also supported for a single
standalone predictor. The RL and generation knobs are shared across scenarios.

- The YAML is the base; a few CLI flags override matching keys.
- Seeds may be set in YAML: `eval_seed`, `train_seed`, `gen_seed`. Explicit CLI
  flags override them; training/generation seeds fall back to `eval_seed`.
  `--eval-seed` is canonical; `--dp-seed` is a deprecated CLI alias.
- The full merged config is written to `<out>/run_config.json`.

**CLI flags:** `--config` (required), `--out` (required), `--method`,
`--eval-seed` (legacy `--dp-seed`), `--train-seed`, `--gen-seed`, `--device` (`cpu`/`cuda`),
`--dqn-loss`, `--dqn-augment-permutations`, `--save-checkpoint-freq`,
`--train-count-duplicates` / `--no-train-count-duplicates`,
`--train-max-duplicate-attempts`, `--train-budget`,
`--resume-training`, `--only-generate`, `--skip-generation`, `--max-gen-attempts`,
`--load-qnet` / `--load-policy` / `--load-value-net` / `--load-scaler`.

\newpage

# 1. Environment — `groups:` is the public entry point

Every scenario declares one or more named `groups`. A lone composition/integer
group can use a flat internal environment; several groups preserve their names
in the candidate. `env_type` is an internal derived label, not a public selector.
See [configuration contract](configuration_schema.md) for executable core defaults
and validation, and retain experiment-specific YAMLs locally. Older published examples may
need an explicit migration before scientific use.

## `kind: composition` group keys

| Key | What it does | Choices / default |
|---|---|---|
| `name` | Group identity | explicit unique name |
| `kind` | Composition domain | `composition` |
| `species_set` | Allowed chemical species, including compound units where supported | required list |
| `fraction_set` | Exact allowed amounts: a list, `{min, max, step}`, or species-to-menu mapping | required |
| `n_components` | Component slots used by the domain | `5` |
| `element_bounds` | Per-species `[min, max]`; a positive minimum makes that species mandatory | optional |
| `sites` | Atoms/units per formula unit for chemical assembly | `1` |
| `filters` | Optional ordered action filters | omitted/empty = none |

`fraction_set` is the single grid authority. Integer resolution is derived from
its exact values; do not separately author `total_units`. Bounds intersect the
menus: minimums round up to the next allowed value and maximums round down. A
sparse menu stays sparse. A domain must admit an exact sum of one. Final candidate
identity omits absent zero-valued species and ignores assignment order.

```yaml
groups:
  - name: alloy
    kind: composition
    species_set: [Ti, Al, V]
    fraction_set: {min: 0, max: 1, step: 0.05}
    n_components: 3
    element_bounds: {Ti: [0.5, 0.9], Al: [0, 0.3], V: [0, 0.3]}
```

Old `episode_style: fixed_order_amount` is a compatibility spelling whose final
domain is compiled into the current free assignment protocol; it does not retain
an order-sensitive scientific identity. Historical `amount`, `host` and manually
matched `total_units` are not recommended inputs: write the grid in `fraction_set`
and put the host in `species_set` with its bounds.

## `kind: integer` group keys

| Key | What it does | Choices / default |
|---|---|---|
| `species_set` | Allowed element tokens | required |
| `ratio_set` | Nonnegative integer token amounts | default `0 … 9` |
| `n_components` | Number of token slots; species may repeat | `5` |
| `token_requirements` | Explicit count/positive-presence requirements | optional |
| `filters` | Optional ordered action filters | omitted/empty = none |

There is no sum-to-one constraint. Token multiplicities and amounts determine the
terminal material. For example, `token_requirements: {O: {count: 1, positive: true}}`
requires one positive O token at any step. The known historical reserved-positive
O-last filter migrates to that domain rule; it no longer forces an execution order.

# 2. Multiple groups, categorical fields and sequences

`groups` is an ordered list of subdomains. Group membership and sequence position
remain part of scientific identity, while assignment order inside an unordered
domain does not. A multi-group predictor receives `{group_name: group_payload}`;
a single composition/integer group may be unwrapped for a flat predictor.

`kind: categorical` assigns one typed value to each named slot, in any allowed
assignment order. Use `choices: [{name: temperature, values: [300, 350]}, ...]`
for generic fields. `values` also supports an exact `{min, max, step}` numerical
grid. Strings, booleans and numbers remain typed: `true`, `1` and `"1"` are distinct,
while equal numeric spellings such as `1` and `1.0` cannot be duplicated.

Chemical categorical slots use `element` (optionally a separate `name`) instead
of a generic name alone. Their chosen amounts remain chemical inputs: an SSE
builder can read `Cl = 1.0` and the O-form choice directly. Generic fields are not
converted into a chemical formula. Filters can couple slots to prior groups.

`kind: sequence` preserves positional order and has its own edit/length domain;
composition permutation rules do not apply. See [labelled assignments](labelled_assignments.md)
for positional semantics; external model chemistry and intrinsic constraints
must be validated through the actual plugin.

# 3. Constraint filters — `filters: [{type: ...}]`

Use the same ordered `filters` list at the applicable group/top-level boundary.
The framework creates the chain; do not author `type: chain`. Historical non-null
`constraint_filter` is rejected by current core configuration validation.

| `type` | What it does | Its keys |
|---|---|---|
| `smact_charge` | Keep charge-neutral compositions under the configured oxidation-state rules | `smact_anions: [{symbol, charge, stoich}]` or supported legacy single-anion keys |
| `last_step_element` | Legacy construction restriction; prefer a domain requirement when equivalent | `required_elements`, `nonzero_ratio_at_last`, `reserve_for_last` |
| `phase_pattern` | Keep compositions matching declared phase patterns | `phase_patterns` |
| `ooh_phase` | OOH oxyhydroxide phase screen | `target_phases` (default `[any]`) |
| `host_complement` | Legacy dopant/host construction filter | `host_element`, `levels`; explicit composition bounds are preferred |
| `sse_doping` | Restrict the LiPS S-site O-form by metal category | `o_element`, `host_P`, `metal_only`, `oxide_only` |
| `"pkg.mod:Class"` | External filter | adapter-specific settings |

The `ooh_phase` sampler retains its five-cation rules: named primary elements
must be selected for the requested phase. `any` accepts their union. Its terminal
checker counts every cation outside the primary set toward the dopant limit,
including species absent from the historical dopant list; the dopant fraction
must remain strictly below 25%.

\newpage

# 4. Property prediction and reward

A non-empty `properties:` list selects the reward engine automatically, for
one or many properties. Set an inline `properties[].predictor`, or declare that
predictor in `models.<id>` and use `properties[].model_ref`:

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
| `structure_sources` | Named source definitions owning builder, templates, samples and relaxation | absent for composition-only models |
| `models` | Named model definitions, selected by `properties[].model_ref` | optional; inline properties remain supported |
| `structure_sources.<id>.builder` | `substitute`, `sse`, another registered builder, or an FQN | `substitute` |
| `structure_sources.<id>.n_random_configs` | Random placements per candidate; template counts can override | `1` |
| `structure_sources.<id>.geo_opt` | Source-owned relaxation; omit / `enabled: false` to skip | optional |
| `structure_sources.<id>.structure_parallel` | Persistent GPU workers for explicit source samples | omitted: serial execution |
| `properties` | Non-empty list of property specs (see below) | required |
| `reward` | Combination rule: `type: weighted_sum` or `type: gated` | weighted sum |
| `k` | Uncertainty coefficient for property objectives | `1.0` |
| `sweep` | Optional `{name, values}` — optimize a shared operating condition (e.g. temperature) per composition (see below) | optional |
| `structure_sources.<id>.base_poscar` / `site_symbol` | Template and substitution-site settings for that source | — |

Properties sharing a `source` ID reuse the same built/relaxed ensemble. Distinct
source IDs are independent even when definitions match. `source` may be omitted
with exactly one declared source; several sources require explicit references.
Composition-only properties must not declare a structure source.

```yaml
structure_sources:
  bulk:
    builder: substitute
    base_poscar: bulk.POSCAR
    site_symbol: X
    n_random_configs: 4
models:
  potential:
    predictor: dp_energy
    models: [energy_a.pt, energy_b.pt]
properties:
  - name: energy
    model_ref: potential
    source: bulk
    direction: min
    objective: mean
```

A model definition owns its adapter/asset options. A property reference owns
source, output selection and reward options. Do not duplicate `predictor`,
`model`, `models`, `dp_models`, `head`, or another model-owned setting on a
reference. Unknown/unused references and conflicting ownership are errors.
One named model instance evaluates all its selected outputs once for the same
candidate, source and operating condition.

**Legacy migration only:** old top-level builder settings and
`share_structure: true` can normalize to source `default`. `share_structure: false`
is rejected with migration guidance: explicitly name the independent sources and
reference them. Do not combine new source definitions with old per-property
builder inheritance. See [source migration](structure_source_migration.md).

`geo_opt:` sub-keys — `model` (default `models/DPA-3.1-3M.pt`), `head` (user-defined),
`fmax` (`0.001`), `steps` (`1000`), `relax_cell` (`true`), `enabled` (`true`).

`properties[]:` sub-keys:

| Key | What it does | Default |
|---|---|---|
| `name` | Unique label (CSV column prefix) | `prop{i}` |
| `predictor` / `model_ref` | Inline adapter or reference to one top-level named model; mutually exclusive | one required |
| `source` | Named structure source; forbidden for composition-only leaves | inferred only when unambiguous |
| `output` | Named scientific output from an adapter | required if adapter returns multiple outputs |
| `models` (or legacy `dp_models`) | DeepMD ensemble checkpoint list; registry leaves use their own keys such as `model` | required for `dp_*` |
| `head` | DeepMD branch name; explicitly select the classification head in multi-head checkpoints | none |
| `direction` | `max` / `min` / `target` (`dp_energy`: use `min` for "lower is better") | **required** |
| `target_value` | Value to hit, in the property's real units. Required with `direction: target`. **Not** the leaf's `model:`/checkpoint key. | — |
| `target_tolerance` | Deadband half-width; inside it the term is at its maximum (0) and stops competing with other objectives | `0.0` |
| `weight` / `scale` | Combine weight / unit scale | `1.0` / `1.0` |
| `objective` | `mean` / `mean_minus_kstd` / `mean_plus_kstd` | `mean` |
| `uncertainty` | `model_spread`, `config_spread`, `native_uncertainty`, `pooled_spread` | model spread; legacy self-folded tuples use native uncertainty |
| `unit` / `representation` | Physical unit (`null` = unknown) and `physical`, `ln`, `log10`, `pIC50`; must agree with adapter schema | unknown unit; physical unless transform supplies a legacy log representation |
| `energy_per_atom` *(`dp_energy`)* | Normalize by atom count | `true` |
| `output_index` / `output_aggregator` *(`dp_property`, `dp_classification`)* | Select a vector component, or collapse output channels using `mean` / `max`. Classification also accepts a list of class indices to sum, with aggregator `index` | aggregator `index`; index `0` for regression, **explicit index or list required for classification in `index` mode** |
| `transform` | `none` / `identity`; `exp` for natural-log samples; `pow10` for log10 samples; classification uses identity | `none` |
| `fparam` / `aparam` *(`dp_property`, `dp_classification`)* | Frame / atomic parameters for heads trained with them. A `null` in `fparam` marks the slot filled by the `sweep` value (requires a top-level `sweep`) | none |

The default weighted-sum reward is `sum weight * value(prop) / scale`, where `value` is

- `max`/`min`: `objective_from_mean_std(direction*mean, std, objective, k)`
- `target`:    `objective_from_mean_std(-e, std, objective, k)` with
  `e = max(0, |mean - target_value| - target_tolerance)`

i.e. a target objective scores the (deadbanded) **distance** to `target_value`, negated so
closer is better. Routing it through the same helper keeps uncertainty handling identical:
`mean_minus_kstd` reads as "effectively farther from the target when less certain".

### Named scientific outputs and uncertainty

`predict_outputs` returns `{name: ScientificOutput}` with a versioned schema,
physical mean, optional retained samples and separate uncertainty quantities.
For a rectangular model-by-configuration matrix `Y[m,c]`:

- `mean = mean_m(mean_c(Y))`.
- `model_spread = std_m(mean_c(Y), ddof=0)`.
- `config_spread = std_c(mean_m(Y), ddof=0)`.
- `pooled_spread = std(Y flattened, ddof=0)`, derived only from retained samples.
- `native_uncertainty` preserves a backend's reported standard deviation.

All models must evaluate the same identified configurations in the same order.
An axis with one observation has no spread estimate (`None`); two identical
observations have measured spread zero. These descriptive quantities are not
interchangeable and are not automatically calibrated confidence intervals.

The compatibility `(mean, std)` property view uses the selected uncertainty.
`objective: mean` is the default. An explicit nonzero uncertainty penalty with a
missing selected statistic raises `EvaluationProtocolError`; it does not silently
turn into a mean objective. `predict` returns `(reward, None)` because unrelated
property spreads cannot be averaged into uncertainty of a combined reward.

`exp` and `pow10` act on each retained raw sample, then recompute mean and spreads.
They require the matching `ln` or `log10` representation, finite positive inverse
values, and no unresolved native uncertainty. Folded `(mean, std)` values alone
cannot be nonlinearly inverted. Declare the destination `unit` explicitly when
known; `null` means unknown. An adapter's unit/representation cannot be silently
overridden. `pIC50` has no automatic inverse conversion. Historical singleton
`std=0` and implicit `mean_minus_kstd` were old semantics; review those runs rather
than treating them as equivalent to the current contract.

### `structure_parallel`: GPU workers and automatic formula batches

This optional block applies to the structure-scoring reward engine. It detects
available GPUs and distributes configurations over persistent subprocesses,
each pinned to one GPU. It automatically evaluates several formulas together
when one formula has fewer configurations than the selected GPU count.
Each GPU runs exactly one structure task at a time, without concurrency
calibration or extra probe runs. The source's explicit sample plan supplies each
worker task, which calls `build_sample(candidate, *, sample)` and applies that
source's `geo_opt` if enabled. Built-in `substitute` supports this protocol,
including multiple templates with their own counts. An external MD/optimization
builder must implement the same sample capability; merely supporting
`build(..., n_configs=1)` is insufficient. Set source `geo_opt: null` if its
builder already optimizes each sample.

```yaml
structure_sources:
  bulk:
    builder: substitute
    base_poscar: bulk.POSCAR
    site_symbol: X
    n_random_configs: 4
    structure_parallel: {}  # automatic GPUs and formula batches
# Properties select source: bulk.
```

The following keys belong inside each source's `structure_parallel` block.
No scheduling or success-threshold parameters are required. The following
optional overrides are for device allocation, time limits and reproducing an
older single-formula sampling schedule.

| Key | What it does | Default |
|---|---|---|
| `devices` | `auto` detects GPUs in the inherited allocation; alternatively, a non-empty list of distinct nonnegative visible GPU indices | `auto` |
| `candidate_batch_size` | Maximum formulas collected for one reward-evaluation wave: positive integer, or `auto` to fill available GPUs using each formula's configuration count; `1` restores single-formula collection | `auto` |
| `reserve_devices` | Visible GPU indices to leave out of worker allocation; requires `devices: auto`; `[0]` reserves GPU 0 for the parent | `[]` |
| `threads_per_worker` | CPU numerical-library threads per worker; positive integer | `1` |
| `timeout_s` | Maximum elapsed seconds for one configuration task, including build and optional relaxation | `3600` |
| `startup_timeout_s` | Maximum seconds allowed for worker initialization | `300` |

Concurrency is capped at the smaller of the wave's total configuration count
and the selected GPU count. Extra configurations wait for a free worker; a worker
immediately takes the next queued configuration when its current task finishes.
For example, `n_random_configs: 4` with `candidate_batch_size: 2` supplies eight
configuration tasks from two formulas, enough to use eight GPUs. Increasing formula
concurrency does not increase `n_random_configs` for any formula.
Models cached by a builder or its optimizer remain loaded in that worker across
candidates. Results are collected in candidate/configuration order, then
scored in the parent process.
Property inference, reward combination, and RL updates are not distributed by
this option. A gated reward still evaluates after structure preparation and
property prediction; it does not avoid the MD cost of a failing candidate.

**Automatic formula batches.** The default `candidate_batch_size: auto` collects
enough formulas to fill the GPUs (selected GPU count divided by configurations
per formula, rounded up). An explicit positive integer fixes the maximum formulas
per wave; `1` disables cross-formula batching.

On eight GPUs this collects two formulas per wave, with four configurations each.
On six GPUs it also collects two formulas: six tasks start immediately, and the
remaining two start as workers become free. It does not wait for an entire formula
to finish before dispatching another queued task.
Cache hits and duplicate formulas in a wave reuse predictions instead of
submitting additional structure tasks. Consequently, a wave containing cached
formulas may leave some slots idle. Properties referring to the same named source
reuse its ensemble; distinct sources build their own ensembles. Source sharing is
independent of model reuse through `model_ref`.

The training and generation loops defer terminal reward evaluation until the
wave has been collected. DQN warmup, DQN online training, REINFORCE/A2C reward
evaluation, and generation use these waves. PG waves stay within the existing
`pg_batch_eps` learning batch, so no policy update occurs between episodes in
that batch. DQN holds the policy and epsilon fixed while collecting each wave,
then processes successful episodes and their updates in order. This changes
the DQN sampling schedule when the size exceeds `1`; identical seeds need not
produce the same training trajectory as single-formula execution. Set
`candidate_batch_size: 1` when the original episode-by-episode schedule is needed.
Waves are shortened at remaining episode-budget, duplicate-attempt,
evaluation-failure, and checkpoint
boundaries. A failed formula does not consume training budget or discard its
successful neighbors. Every formula with at least one usable configuration is
scored; a formula with no usable configurations is skipped.

**Migrating older configurations.** Remove `workers_per_device`, `min_speedup`
and `min_successful_configs`. For compatibility, the old `workers_per_device: auto`
means one worker per GPU, `min_speedup` is ignored, and a legacy success threshold
above one is ignored; these emit deprecation warnings. `workers_per_device: 1`
is also accepted, while `2` is rejected.
There are no additional calibration structures or timing trials. GPU discovery
and automatic formula-batch sizing remain available through `devices: auto` and
`candidate_batch_size: auto`.

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
parallel build. Their resident memory, external GPU load, or additional
per-source worker pools can still affect available memory
and throughput. Start one experiment process, without `torchrun`.
The subprocess implementation targets POSIX systems; actual CUDA
execution requires a compatible Linux GPU environment and the installed builder,
DeepMD backend, and LAMMPS binary to support that environment.

**Failures and timeouts.** MD errors, optimizer errors and optimization that
reaches its step limit without convergence are logged and the affected
configuration is skipped. The predictor averages the remaining configurations;
if none remain, that formula is skipped without consuming training budget.
Workers propagate optimizer execution exceptions, including CUDA OOM, even when
an external builder calls the shared `relax_structure` helper internally. This
prevents failed optimizations from being included in a successful ensemble.
The same strict relaxation checks apply to serial structure scoring. External
builders using their own optimizer must report failure/nonconvergence themselves;
in serial batch builds they must catch errors per configuration and return only
usable structures. No replacement configurations are generated for failed ones.
The existing consecutive-candidate-failure guard still stops training after 11
consecutive failed formulas, avoiding an endless run with no scores.
A worker crash or timeout aborts the current wave and discards the worker
pool; the next prediction creates a new pool. The experiment's existing handling
of failed candidate evaluations still applies. Choose the timeout to allow a
complete MD-plus-optimization task, including first-use model initialization.
The CLI closes worker process groups on normal exit, exceptions, Ctrl+C, and
SIGTERM termination. SIGKILL cannot run Python cleanup and provides no such
guarantee.

**Builder contract and reproducibility.** Seeded parallel execution requires
`build_sample(candidate, *, sample: StructureSample) -> Atoms`. Missing capability
raises `SamplingProtocolError` before scientific sample jobs; it is not a candidate
failure to retry. An optional `sample_plan(candidate, *, n_configs, source_spec,
eval_seed)` must return valid, identified samples with the source protocol's seeds.
Serial and worker execution consume the same plan. Multi-template plans preserve
each template's declared count and site symbol rather than averaging templates
equally. Source, template and sample identities survive partial preparation
failure; every scoring model receives the same remaining configurations.

Seeds derive from evaluation seed, source identity, canonical candidate, purpose,
template and sample index. They do not depend on candidate iteration order, batch
partition, worker count or completion order. This is a sampling contract, not a
promise of identical GPU floating-point arithmetic across backends/hardware.
An opaque legacy external `build(candidate, *, n_configs, rng)` can run serially
with one source/candidate ensemble seed; it cannot claim sample-by-sample parallel
parity. It must implement the explicit capability before parallel source use.
Debug output remains separate by run/worker where the builder supports it.

Independent named sources own their own pools and may retain separate model
copies on the same GPUs. Refer to one source where properties should evaluate the
same physical samples. Omit `structure_parallel` for serial execution. The old
`share_structure` boolean is only a migration input, not a pool-selection API.

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

To accept both nearly pure and pure with the class order shown above, use
`output_index: [1, 2]` and preferably name the property
`synthesizability_probability`. This list must be nonempty and contain distinct
nonnegative integers, each within the model's class range. Lists are supported
only for `dp_classification` with `output_aggregator: index` (the default);
regression selectors are unchanged.

`output_aggregator` acts on the class channels **within one frame**:
`index` selects one class or sums the listed classes; `max` selects the most
confident class regardless of its identity; `mean` averages the classes and is always `1/K`
for `K` classes, so it is unsuitable as a synthesizability score. It does not
control averaging across configurations.

For class `k`, each checkpoint's composition probability is the mean of
`p[j,k]` across the successfully prepared configurations. With a list such as
`[1, 2]`, sum `p[j,1] + p[j,2]` first, then average across configurations for
each model; do not combine separately computed class standard deviations.
All checkpoints evaluate the same surviving structures. Multiple checkpoints then follow the
model mean/std fold above, using the population standard deviation (`ddof=0`).
A single checkpoint has `model_spread=None`, even when its configuration
probabilities differ; those differences belong to `config_spread`. `1 - p_pure` means predicted probability
of a different class; it is not the model-ensemble standard deviation. The
probabilities themselves do not provide an independent uncertainty estimate.

When a `dp_classification` property contributes to a gate, its value is
automatically `mean - model_spread` with multiple models, or `mean` with one. Add checkpoints to the existing `models` list to
use this conservative ensemble score; no extra setting is needed. The standard
deviation coefficient is fixed at `1`, independent of `k` and the property's
`objective`. This score is not a calibrated probability or confidence bound,
and it can be negative; the raw predicted mean is kept separately. With a
single checkpoint the gate value remains the mean probability.
The conservative interpretation applies to lower bounds on desired-class
probabilities with positive gate coefficients; an upper bound still compares
against the same gate score; it does not switch to an upper confidence bound.

In `generated.csv`, `obj_<name>_mean` and `obj_<name>_std` report the mean and
selected spread of the probability (default: model spread, blank when missing).
Named output records retain every spread separately. For a gate
named `synthesis` with that property as its sole term of weight `1`,
`gate_synthesis_value` reports `mean - model_spread`, or `mean` with one model.

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
builder, or property name is hard-coded in the combination rule.

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

Gate scores use **`mean - model_spread` for multi-model `dp_classification`
properties**, their mean for a single model, and **raw means for other types**. Property `direction`, `objective`,
`weight`, `scale`, and `k` are not applied to these gate inputs. Thus a gate on
`pure_probability` compares its conservative ensemble score to its threshold.
Its combined score is `g = sum(coefficient * property_gate_value)` and its violation is
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
pure-probability gate at `0.8` and base `-10`, a candidate with
`mean - model_spread = 0.7` gets `-10.1` (or a single-model mean of `0.7`).
Penalties use the gate score's raw units. They do not rescale delta or the
objective. A fixed failure base guarantees that every passing candidate ranks
higher only when it is below a known lower bound of the feasible objective.
Choose it for the actual objective range; no such guarantee follows for an
objective with an unknown or unbounded lower range.

The same reward is used for training, generation, and operating-condition
sweeps. The predictor's `reward_details` exposes objective and gate diagnostics.
Generated candidate CSVs include `reward_objective`, `gate_passed`, and
`gate_<name>_value`, `gate_<name>_passed`, `gate_<name>_violation`; per-property
`obj_<name>_mean/std` retain the scientific mean and selected spread (missing
spread is blank). The separate named outputs preserve all spread types. Gate
values always use the classification model spread, independently of the spread
selected for a property objective; they do not overwrite scientific statistics.
For a single-property gate with coefficient `1`, `gate_<name>_value` records
that conservative score; for a mixed gate it records the combined weighted score.
`dp_mean` applies the same
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
| `dp_structure` | Named substitute source; one `dp_energy` property with `source` and `direction: min` |
| `dp_property` | Named substitute source; one `dp_property` property referring to it |
| `hea` / `perovskite` | As above, source `site_symbol: X` / `Fe` |
| `structure_pipeline` / legacy `share_structure: true` | One source ID referenced by all relevant properties |
| `composite` / legacy `share_structure: false` | Explicit independent sources; migrate `objectives:` to `properties:` and attach references |

## `ooh` keys

`base_poscar`, `dp_models`, `objective`, `k`, `n_random_configs`, `ads_height`,
`ads_dz`, `geo_opt` (bool), `geo_opt_model`, `uncertainty`
(`models`/`configs`/`total` in the flat legacy adapter), `output_index`, `adsorbates`, `target_phases`.

The named scientific output is `overpotential`, positive in its original direction,
with unit explicitly unknown (`None`). In the property engine choose
`output: overpotential`, `direction: min` and normally `objective: mean`; engine
uncertainty selectors are `model_spread`, `config_spread`, `pooled_spread`, not the
legacy spellings. The flat adapter maps `models`/`configs`/`total` to those three
spreads and defaults to objective `mean`. `total` is pooled population spread,
not native uncertainty. Singleton estimates remain missing.

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
frame. Changing `adsorbates` or `output_index` defines a different oracle and invalidates
its cached scientific outputs. Start a new run for that change; exact resume must
not silently switch the oracle while retaining old replay targets. Model/template
assets and the evaluation protocol are part of the recorded identity.

## `rf_magpie` keys

`model` is the joblib checkpoint path. Use separate named properties for
sintering and calcination, as in `configs/oxides_sinter_calcine.yaml`.

\newpage

## Writing your own predictor — the leaf contract

Prefer the named scientific protocol; a leaf can expose one or several outputs
and can retain its own explicitly identified model/configuration samples.

| Method | Returns | Meaning |
|---|---|---|
| `predict_structure_outputs(atoms_list)` | `{name: ScientificOutput}` | Geometry-dependent scientific outputs on source-prepared structures |
| `predict_outputs(candidate)` | `{name: ScientificOutput}` | Composition/sequence/field scientific outputs |
| `score_structures(atoms_list)` | one finite float per structure | Legacy single-model raw samples, folded by engine |
| `score(candidate)` | one finite float | Legacy single-model point sample, folded by engine |
| `predict_structures(atoms_list)` / `predict(candidate)` | `(mean, std_or_None)` | Legacy physical estimate with native uncertainty; no raw distribution |

An optional `output_schema` mapping must exactly match named output names and
schemas. Named methods take priority, then `score*`, then legacy `predict*`.
None of these property interfaces should apply reward direction, scale, weight or
`k`: the engine does that once. A flat historical adapter that returns a reward
needs a separate scientific method before composition into `properties`.

`model: ckpt/a.pt` creates one registry/FQN instance; a list creates one per
checkpoint. One sampled model has missing model spread, while two sampled models
can estimate disagreement. Several self-folding legacy tuple outputs cannot be
combined by discarding or averaging native uncertainties. Named ensembles must
provide compatible schemas, aligned raw samples and no unresolved native
uncertainty. A point-only named output without raw samples is not enough for this
ensemble path. Use `ScientificOutput`/`ModelSamples`/`aggregate_samples` from
`rl_matdesign.predictors.outputs`; do not invent zero spread for a singleton.

The constructor is `__init__(self, cfg, *, seed=None)`. Inline properties retain
compatibility constructor options; a named `model_ref` resolves the model-owned
options. Add optional `close()` for external resources; each resource is closed
once. Shared model inference applies only to the same candidate, source and
operating conditions. Do not silently ignore declared `fparam`/`aparam`.

See [scientific output migration](scientific_output_migration.md) for schema
versioning, units, raw sample alignment, nonlinear transforms and cache records.

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
| `"pkg.mod:Class"` | Native serial `build(candidate, *, n_configs, rng)`; seeded serial/parallel parity requires explicit `build_sample(candidate, *, sample) -> Atoms` (see above). |

For `defect_site`, the `formula` column in `generated.csv` contains the full
elemental composition, including the surviving host atoms and untouched template
elements such as oxygen. Counts are normalized by the **original number of
A-site atoms** in `base_poscar`. A template with equal A/B counts and three O per
A gives, for example, `Sr0.5Ba0.5Fe0.75Co0.25O3`; an A-site vacancy can give
`Sr0.875FeO3`, and an interstitial can give `Sr1.125FeO3`. Oxygen stoichiometry is
read from the template, not fixed to three. The label follows the actual integer
substitutions and defect clamping, preserves fractions such as `0.03125`, and
combines an element present on multiple sublattices. External MD wrappers that
delegate `composition_formula()` to this builder, including
`clc_workflow.rl_builder:MDAveragedOptBuilder`, receive the same complete formula
in serial and parallel execution, without an additional MD or optimization run.
When the optional generation charge-neutrality filter is enabled, it also checks
this complete formula rather than a dopant-only label.

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

## Training budget and duplicate episodes (all methods)

| Key | What it does | Choices / default |
|---|---|---|
| `train_budget` | Shared total paid training budget; derives the active algorithm's episode limits | Positive integer or `null` (legacy per-algorithm limits) |
| `train_count_duplicates` | Whether a completed episode revisiting a seen final candidate consumes budget | YAML boolean; `true` |
| `train_max_duplicate_attempts` | Stop with an error after this many consecutive completed duplicate episodes when `train_count_duplicates: false` | Positive integer; `1000` |

These keys are top-level and apply to all environments and material systems.
`--train-count-duplicates` / `--no-train-count-duplicates` override the YAML
boolean; `--train-max-duplicate-attempts N` overrides the limit. Use an unquoted
YAML `true` or `false`; strings such as `"false"` are rejected.

When `train_budget: B` is set, DQN's additional online budget is
`B - dqn_warmup_eps`, and PG derives `ceil(B / pg_batch_eps)` nominal iterations
while stopping at exactly `B` counted candidates. For example, budget 1000 with
batch size 16 derives 63 iterations and an eight-candidate final batch when no
duplicates occur; the budget is 1000, not 1008. Conflicting manually supplied
`dqn_num_train_eps` or `pg_num_iters` values are rejected. Omit those derived
limits when using the shared budget. HPO recomputes them after sampling the batch
size. DQN's rewarded warmup is included; PG's feature-only scaler warmup is not.
`--train-budget B` overrides the shared YAML budget.

With the default `true`, training budgets retain their existing meaning: one
completed episode consumes one slot, including repeated candidates. With
`false`, only the first successfully completed episode for a final candidate key
consumes a slot. Duplicate episodes still train normally: DQN stores their
transitions in replay, and REINFORCE/A2C include them in gradient batches. Failed
candidate evaluations continue to use the separate evaluation-retry mechanism
and do not consume budget.

Identity comes from `env.terminal_comp_key()`, independently of the predictor
cache. Different action orders leading to the same final candidate are
duplicates. Multi-group keys preserve group assignments, integer ratios, and
categorical values, so these distinctions are not lost by flattening a formula.
The configurations sampled to evaluate one candidate are not separate episodes.

With `train_count_duplicates: false`, the phase budgets mean:

- **DQN (`mc` and `bootstrap`):** `dqn_warmup_eps` counts distinct warmup
  candidates; `dqn_num_train_eps` counts additional distinct online candidates.
  One seen-candidate set spans both phases, so warmup candidates cannot consume
  the online budget again. For example, `400 + 1600` requests 2,000 distinct
  evaluated candidates. Epsilon annealing, target-network updates, and periodic
  checkpoints follow actual sampled online episodes, including duplicates.
- **REINFORCE/A2C:** the unique target is `train_budget` when set, otherwise
  `pg_num_iters * pg_batch_eps`.
  Each update uses up to `pg_batch_eps` sampled episodes, including duplicates;
  additional batches are collected until the target is reached. The final batch
  may be shorter to stop at the exact target. The scaler-only `pg_warmup_eps`
  phase does not populate the seen-candidate set.

When unique accounting is enabled, a new candidate resets the consecutive
duplicate counter. Reaching `train_max_duplicate_attempts` stops training with
an error, which bounds stalls when exploration collapses or the candidate space
is exhausted. A larger unique budget can therefore require more gradient steps
and wall time than an equally sized episode budget.

Per-episode `training_log.csv` columns distinguish these quantities:

| Column | Meaning |
|---|---|
| `budget_counted` | Whether this completed episode consumed a slot |
| `training_attempts` | Cumulative successfully completed episodes, including duplicates |
| `training_budget_used` | Cumulative slots consumed under the selected mode |
| `training_unique_candidates` | Cumulative distinct final candidate keys |
| `training_duplicate_episodes` | Cumulative completed episodes that revisited a key |
| `training_budget_history_complete` | Whether the counters and seen keys cover the entire training history |

DQN cumulative columns include warmup, while its `episode` column is the actual
attempt number within the current phase. PG episode counts exclude scaler-only
warmup. New checkpoints preserve counters, candidate keys, and accounting mode.
Changing `train_count_duplicates` while resuming a nonempty checkpoint is
rejected. Exact continuation also requires compatible state/action schemas,
model/source assets, fitted scaler, learning settings and numerical runtime.
Setting duplicate counting to `true` does not make an incompatible historical
checkpoint resumable. Use its archived generating version for historical work.
Weight-only fallback, when otherwise compatible, is a warm start with incomplete
history, not exact continuation; it cannot establish a unique-budget comparison.
See [reproducibility contracts](development_contracts.md#reproducible-experiments).

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
