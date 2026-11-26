# Configuration contract

The experiment core now has one lightweight configuration contract in
`rl_matdesign.config_schema`. Loading and validation do not import PyTorch,
matminer, DeepMD, or external model packages. This first migration covers shared
training, generation, and seed settings. It does **not** replace the scientific
schemas owned by environments, predictors, builders, or constraints.

## Resolution and provenance

1. `load_yaml_config(path)` accepts a YAML mapping and rejects duplicate keys at
   every nesting level, reporting their locations. Standard YAML merge defaults
   may still be overridden explicitly. Empty files and non-mapping roots fail.
2. `validate_config(raw)` checks canonical core keys without inserting defaults;
   this is suitable before HPO modifies the raw configuration. Retired spellings
   are rejected.
3. `resolve_config(raw, overrides)` applies explicit overrides over YAML,
   validates the combined configuration, derives seed fallbacks, and fills defaults. It
   returns `ResolvedConfig(config, sources)`. A `None` override is absent; `False`
   and `0` are real overrides.

The `sources` mapping records every effective top-level value as `yaml`,
`override`, `default`, or `derived:<dependency>`.
Dependency origins remain available under the dependency's own key. Nested
plugin options retain their complete mappings and the origin of that mapping;
per-leaf plugin provenance is not implemented in this migration.

Both DQN and policy-gradient settings stay in the same effective configuration.
Changing `method` does not delete inactive algorithm settings. Invalid shared
parameter values are rejected even if their algorithm is inactive.

`gamma` is the shared discount factor for every method, defaulting to 0.9.
DQN and PG warmup budgets are independent: `dqn_warmup_eps` defaults to 500,
and `pg_warmup_eps` defaults to 200. HPO applies sampled/fixed values before
materializing defaults.

`train_seed` and `gen_seed` fall back to `eval_seed`. Seeds are integers in `[0, 2**32 - 1]`, compatible with NumPy's seed range.
The run CLI additionally accepts a nonempty list of those integers in any of the
three seed fields, and a positive integer `parallel_jobs` (default 1).
`resolve_run_configs(raw, overrides)` expands these operational settings into
`ResolvedRunConfigs(runs, parallel_jobs, seed_list_keys)`. Lists must have equal
length and are paired by index; scalars broadcast and null/missing train/gen
seeds inherit the corresponding evaluation seed. Scalar CLI overrides apply
before expansion. `runs` contains ordinary `ResolvedConfig` objects, each with
scalar seeds and no `parallel_jobs` field. This keeps scheduling out of scientific
execution identities. `resolve_config` and `validate_config` remain scalar-only
contracts for individual experiments and HPO jobs.

`rl-matdesign run --config ... --out ...` schedules list entries in independent
subprocesses, bounded by `parallel_jobs` or `--parallel-jobs`. An explicit
one-item list still uses a batch output directory. Validation prints every
expanded run without constructing a predictor or writing outputs. See the
[seed execution examples](../README.md#seeds) for output paths and per-seed resume.

This naming change does not itself change a predictor's seed-to-structure
algorithm. Runtime determinism and complete experiment archives are separate
requirements; configuration resolution alone does not certify repeatability.

## Extension boundary

The core validates shared parameter types, finite numbers, ranges and choices.
It accepts existing top-level fields declared in `DELEGATED_KEYS` according to
ownership (groups, reward, structure builders, and property predictors).
These declarations define ownership boundaries, **not** a claim that every
nested option has a complete schema. `groups`, `properties`, and group `filters` have
basic container validation; their scientific values are validated by owners.

The singular `structure` block owns the builder, templates, sample count and
ordered operations. Built-in structure declarations use `builder: pipeline`;
peptide assembly uses `builder: peptide`. All structure-consuming properties
share the resulting ensemble. `resolve_structure_plan` checks ownership and
rejects unused structures or properties that require a missing structure without
loading model weights or templates. Per-property sources and standalone builder
inheritance are not supported.

Invocation provenance uses the same `resolve_source_assets` function as structure
evaluation. It records and hashes list/dictionary templates, explicit models in MD and relaxation operations. A relative path owned by
an unknown external builder stays unresolved; it cannot silently become a cwd
file merely because a file of that name exists there. Supply resolved explicit
assets for such plugins when packaging an experiment.

Every scenario declares a nonempty `properties` list. Each entry owns its
`predictor`, optional checkpoint `models: [...]`, model options and output/reward
selection. There is no top-level predictor shortcut, top-level model registry,
`model_ref`, or property `model`/`dp_models` alias. A one-checkpoint model uses a
one-item `models` list. Adapters such as GlassNet and installed peptide plugins
may obtain their pretrained assets internally and do not require an explicit list.

Identical model declarations are reused automatically. Different operating
conditions such as `fparam` remain distinct inference requests; properties selecting
different outputs of the same candidate and model request share the raw outputs.
This keeps YAML explicit without loading the same ensemble repeatedly.

Authored scenario files place sections in this order: `groups`, optional
`structure`, `properties`, `reward`, then training and generation controls.
The section order improves review; YAML key order does not change evaluation.

Reward coefficients are declared only under `reward`: objective terms, gate
score terms, or maximum-alternative terms. Property `weight` is a retired key and
is rejected rather than migrated at runtime. A single-property weighted sum can
omit objective terms and uses coefficient 1. Multiple-property weighted sums and
gated rewards require explicit objective terms, which identify exactly which
properties contribute. A property used only by a gate or retained for diagnostics
is not passed through adjusted objective arithmetic.

`representation` is `physical`, `ln`, `log10` or `pIC50`; `transform` is `none`,
`identity`, `exp` or `pow10`. Transform validity requires the predictor's output
schema and is checked by the evaluation owner. `unit: null` means unknown, not
dimensionless. Available uncertainty names are `model_spread`, `config_spread`,
`native_uncertainty` and `pooled_spread`; they are distinct statistics. The default
property objective is `mean`. An uncertainty objective needs an available selected
statistic; a missing estimate is never fabricated as zero. These changes require
explicit review of older results that relied on implicit uncertainty penalties.

An unknown top-level key is an error, with spelling
suggestions. Predictor-specific options should normally live in that predictor's
`properties` entry. Those entries may contain arbitrary options owned by the
selected external predictor; a property-level FQN does not open the experiment
root to arbitrary keys.

External predictors use `properties[].predictor: module:Class`; their options
stay inside that property. Custom builder options stay inside `structure`; custom
filter options stay inside the applicable group's filter entry. No plugin opens
the experiment root to arbitrary keys. No plugin import is needed to load the
configuration; the plugin validates its own scientific options.

Chemical groups require `species_set`, `amount_set`, and `n_steps`; categorical
groups require explicit targets, available values, and `n_steps`. Only these two
group kinds are supported. Final totals, uniqueness and bounds are intrinsic
domain rules. Optional scientific checks use `filters: [{type: ...}]` on the
applicable group. Listed filters always apply during warmup, training and generation;
omitting the list (or using `filters: []`) skips optional scientific filters.
Removed group fields, source registries, sweep declarations and
standalone builder settings are rejected, including empty placeholders.

## Runner and HPO integration

`--validate-config` prints the effective config and source map before constructing
an environment or predictor. `run_config.json` stores all effective core values;
`config_sources.json` stores their origins. Invocation evidence also retains the
original YAML, exact command and operational flags. Generation-only runs preserve
training logs. All seed sources enable the same deterministic Torch settings.
Use `--eval-seed` to set the scientific evaluation seed.

HPO trials use this same contract after fixed and sampled overrides. Budget
scaling reads effective defaults, accepts only episode/iteration-budget fields,
and rejects invalid multipliers. It materializes only the changed fields, leaving
other derived defaults free to follow later overrides. Stage-two confirmation
replays `frac_of` through the same sampler as stage one before integer validation.
The HPO control file itself retains its existing separate search-space schema.

`train_budget` is an optional shared limit on paid training candidates, counted
according to `train_count_duplicates`. DQN spends `dqn_warmup_eps` from that
budget and derives its remaining `dqn_num_train_eps`. Policy-gradient training
derives `pg_num_iters = ceil(train_budget / pg_batch_eps)` and uses an exact final
remainder batch. Its descriptor-only scaler warmup is outside paid training.
With `value_normalization: popart`, the 100-candidate reward-calibration phase
precedes training and is additional overhead for every method. Its candidates
are not injected into DQN warmup, replay or the training seen-candidate set.
`gradient_clipping: clipping` uses these same 100 candidates to estimate initial
gradient caps without optimizer updates. With raw rewards it collects a separate
100-candidate calibration pilot instead. Both uses remain outside the training
allowance. The default `gradient_clipping: none` leaves the existing unclipped
training path unchanged; no fixed threshold or numerical clipping controls are
public configuration fields. Adaptive controllers keep separate Q/actor/critic
histories, which participate in exact resume together with the resolved internal
profile. See [gradient clipping](yaml_config_reference.md#gradient-clipping).
DQN warmup has no calibration-imposed minimum of 100. Fresh DQN training requires
at least one warmup candidate to fit its state scaler; the configured warmup plus
online training consumes `train_budget`, while A2C spends that budget on policy
samples. `reward_calibration_budget.json` records calibration overhead separately.
The default `null` keeps legacy algorithm-specific limits. Warmup above the shared
budget or contradictory explicitly authored totals are errors; remove redundant
totals instead of maintaining two versions of the same budget.

HPO retains authored settings until after sampled/fixed overrides and budget
scaling. Sampling `pg_batch_eps` therefore changes update grouping, not the paid
candidate count. When `train_budget` is set, scale it rather than derived
`pg_num_iters` or `dqn_num_train_eps`; warmup can be scaled separately but must
still fit the resulting budget. Configuration limits do not themselves prove
equal actual model work: record requests, unique candidates, failures, cache hits,
batch calls and wall time alongside budget curves.

## Core parameters

Integer and boolean fields do not accept strings or interchange booleans with
integers. Float fields accept finite numbers and numerical strings, including
`1e-3` which PyYAML's YAML 1.1 resolver otherwise reads as text. Zero episode
budgets are allowed because generation-only and reduced-budget runs use them;
backend-specific prerequisites are checked when that operation is requested.

The following table is generated by `schema_markdown()` from the executable
parameter definitions. A test checks it for drift.

<!-- schema:start -->
| Parameter | Type | Default / fallback | Constraints | Meaning |
|---|---|---|---|---|
| `method` | str | 'a2c' | dqn, reinforce, a2c | Training algorithm; inactive algorithm settings are retained. |
| `value_normalization` | str | 'none' | none, popart | Use raw RL rewards, or automatic reward calibration and PopArt value targets for DQN and A2C. |
| `value_normalization_profile` | str | 'popart_v1' | popart_v1 | Versioned PopArt settings shared by DQN and A2C. |
| `gradient_clipping` | str | 'none' | none, clipping | Disable clipping, or adapt the gradient norm cap from an evaluation-only 100-candidate pilot and subsequent learning gradients. |
| `eval_seed` | int | 0 | >= 0; <= 4294967295 | Random seed for property evaluation and structure sampling. |
| `train_seed` | int | eval_seed | >= 0; <= 4294967295 | Training random seed. |
| `gen_seed` | int | eval_seed | >= 0; <= 4294967295 | Candidate-generation random seed. |
| `hpo_recording` | bool | False | — | Enable strict candidate accounting for an isolated HPO job. |
| `hpo_resume_generation` | bool | False | — | Restore an exactly synchronized HPO generation checkpoint. |
| `hpo_phase` | str | 'train' | train, generation | HPO job phase. |
| `hpo_target_unique` | int | None | >= 1 | HPO target successfully evaluated distinct candidates. |
| `hpo_ratio_limit` | float | 1.5 | > 1 | Strict HPO attempts/unique upper bound. |
| `train_count_duplicates` | bool | False | — | Count repeated terminal candidates toward training budgets. |
| `train_budget` | int | None | >= 1 | Training candidates including DQN warmup, excluding the separate 100-candidate reward/gradient calibration pilot; derives algorithm episode limits when set. |
| `train_max_duplicate_attempts` | int | 1000 | >= 1 | Maximum consecutive duplicates in a unique-candidate budget. |
| `exploration_target` | float | 0.7 | >= 0; <= 1 | Initial normalized action-entropy target for training. |
| `gen_exploration_target` | float | 0.7 | >= 0; <= 1 | Initial normalized action-entropy target for candidate generation. |
| `exploration_adaptive` | bool | True | — | Use Step3 adaptive entropy exploration; false restores historical epsilon/entropy controls. |
| `dqn_eps_anneal_eps` | int | 10000 | >= 1 | Legacy DQN epsilon annealing episodes; requires exploration_adaptive=false. |
| `dqn_eps_min` | float | 0.05 | >= 0; <= 1 | Legacy DQN minimum random-action probability. |
| `pg_entropy_coef` | float | 0.01 | >= 0 | Legacy policy entropy coefficient. |
| `pg_entropy_min` | float | 0.3 | >= 0; <= 1 | Legacy normalized policy entropy floor. |
| `pg_repeat_penalty_coef` | float | 0.0 | >= 0 | Legacy repeated-candidate policy penalty coefficient. |
| `pg_repeat_penalty_shape` | str | 'log' | log, sqrt, linear | Legacy repeated-candidate penalty growth. |
| `gen_temperature` | float | 1.0 | >= 0 | Legacy generation softmax temperature; zero selects greedy behavior. |
| `gen_top_frac` | float | 0.0 | >= 0; <= 1 | Legacy generation top-action fraction. |
| `gen_epsilon` | float | 0.0 | >= 0; <= 1 | Legacy generation random-action probability. |
| `gamma` | float | 0.9 | >= 0; <= 1 | Discount factor used by DQN, REINFORCE and A2C. |
| `dqn_hidden_dim` | int | 256 | >= 1 | DQN hidden-layer width. |
| `dqn_lr` | float | 0.001 | > 0 | DQN optimizer learning rate. |
| `dqn_buffer_size` | int | 50000 | >= 1 | DQN replay capacity in transitions. |
| `dqn_num_train_eps` | int | 20000 | >= 0 | DQN training episode budget. |
| `dqn_warmup_eps` | int | 500 | >= 0 | DQN random warmup episodes. |
| `dqn_batch_size` | int | 256 | >= 1 | DQN replay minibatch size. |
| `dqn_grad_steps_per_ep` | int | 5 | >= 1 | DQN gradient updates per training episode. |
| `dqn_target_update_freq` | int | 100 | >= 1 | DQN target-network update interval. |
| `dqn_loss` | str | 'smoothl1' | mse, smoothl1 | DQN regression loss. |
| `dqn_target_mode` | str | 'bootstrap' | bootstrap, mc | DQN regression target definition. |
| `dqn_augment_permutations` | int | 0 | >= 0 | DQN replay trajectory permutation count. |
| `pg_warmup_eps` | int | 200 | >= 0 | Policy-gradient scaler warmup episodes. |
| `pg_num_iters` | int | 1000 | >= 0 | Policy-gradient training iterations. |
| `pg_batch_eps` | int | 15 | >= 1 | Episodes per policy-gradient update. |
| `pg_lr_actor` | float | 0.001 | > 0 | Actor optimizer learning rate. |
| `pg_lr_critic` | float | 0.001 | > 0 | Critic optimizer learning rate. |
| `num_gen_eps` | int | 200 | >= 0 | Requested distinct generated candidates. |
| `max_gen_attempts` | int | 10 * num_gen_eps | >= 0 | Maximum candidate-generation attempts. |
| `k` | float | 1.0 | >= 0 | Default uncertainty coefficient passed to property evaluation. |
<!-- schema:end -->

## Updating an authored design

Use the current [group and pipeline guide](design_groups.md). Choose numerical
increments and their final constraints explicitly; write each operating
condition as a named property and combine those properties in `reward`.
Historical domain/build/structure parameter normalization has been removed.
Training and seed parameters also require canonical names; no alias conversion
is performed.

Audit local files with `scripts/audit_scenarios.py --domains --inventory <local-report.json>`.
The inventory is a local JSON object with `entries: [{"path": "relative/config.yaml"}]`.
This checks configuration and initial legal actions without running a model.
It cannot establish compatibility of old RL weights, replay, or physical models.
