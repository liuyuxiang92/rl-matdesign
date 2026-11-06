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
2. `normalize_aliases(raw)` returns a copy using canonical spellings. Old and new
   spellings with conflicting values in one source fail instead of silently
   choosing one. Equal values are accepted with a `ConfigDeprecationWarning`.
3. `validate_config(raw)` checks canonical core keys without inserting defaults;
   this is suitable before HPO modifies the raw configuration.
4. `resolve_config(raw, overrides)` normalizes each source separately, applies
   explicit overrides over YAML, derives fallbacks, and fills defaults. It returns
   `ResolvedConfig(config, sources)`. A `None` override is absent; `False` and `0`
   are real overrides. Overrides may legitimately supersede an old YAML alias.

The `sources` mapping records every effective top-level value as `yaml`,
`override`, `default`, `yaml:alias:<old_name>`, or `derived:<dependency>`.
Dependency origins remain available under the dependency's own key. Nested
plugin options retain their complete mappings and the origin of that mapping;
per-leaf plugin provenance is not implemented in this migration.

Both DQN and policy-gradient settings stay in the same effective configuration.
Changing `method` does not delete inactive algorithm settings. Invalid shared
parameter values are rejected even if their algorithm is inactive.

Two historical fallback rules need care:

- If `dqn_warmup_eps` is absent, an **explicit** `pg_warmup_eps` supplies its value.
  If both are absent, DQN uses 500 and PG uses 200. Resolving the PG default must
  not accidentally change DQN to 200.
- Each method's explicit gamma overrides shared `gamma`; shared gamma overrides
  the default 0.9. HPO must apply sampled/fixed values before materializing these
  derived defaults.

`eval_seed` replaces the name `dp_seed`; `train_seed` and `gen_seed` fall back to
it. Seeds are integers in `[0, 2**32 - 1]`, compatible with NumPy's seed range.
This naming change does not itself change a predictor's seed-to-structure
algorithm. Runtime determinism and complete experiment archives are separate
requirements; configuration resolution alone does not certify repeatability.

## Extension boundary

The core validates shared parameter types, finite numbers, ranges and choices.
It accepts existing top-level fields declared in `DELEGATED_KEYS` according to
ownership (environment, reward, builders, constraints, and flat predictors).
These declarations are compatibility boundaries, **not** a claim that every
nested option has a complete schema. `groups`, `properties`, and `filters` have
basic container validation; their scientific values are validated by owners.

`structure_sources` is owned by the lightweight structure-source schema. Core
validation checks each named source definition and the type of
`properties[].source`, without loading templates or models, changing the authored
configuration, or guessing external predictor capabilities. After adapters are
resolved, `normalize_structure_sources` checks source references, unused sources,
composition-only properties and conflicting legacy builder settings. Named sources
are independent identities even when their definitions match.

Invocation provenance uses the same `resolve_source_assets` function as source
evaluation. It records and hashes list/dictionary templates, explicit relaxation
models and the default relaxation model when enabled. A relative path owned by
an unknown external builder stays unresolved; it cannot silently become a cwd
file merely because a file of that name exists there. Supply resolved explicit
assets for such plugins when packaging an experiment.

Reusable model instances are declared in a top-level `models` mapping. A property
selects one with `model_ref` and chooses its named `output`. Model definitions own
adapter options; properties own `direction`, `weight`, `scale`, `objective`,
`source`, `output`, `unit`, `representation`, `transform`, and `uncertainty`.
Repeating a model-defined setting at a reference is an error even when equal.
Inline property models remain supported. A legacy top-level `models` list retains
its flat-adapter meaning and is not a named model table.

`representation` is `physical`, `ln`, `log10` or `pIC50`; `transform` is `none`,
`identity`, `exp` or `pow10`. Transform validity requires the predictor's output
schema and is checked by the evaluation owner. `unit: null` means unknown, not
dimensionless. Available uncertainty names are `model_spread`, `config_spread`,
`native_uncertainty` and `pooled_spread`; they are distinct statistics. The default
property objective is `mean`. An uncertainty objective needs an available selected
statistic; a missing estimate is never fabricated as zero. These changes require
explicit review of older results that relied on implicit uncertainty penalties.

An unknown top-level key is an error for built-in applications, with spelling
suggestions. Predictor-specific options should normally live in that predictor's
`properties` entry. Those entries may contain arbitrary options owned by the
selected external predictor; a property-level FQN does not open the experiment
root to arbitrary keys.

Existing **flat** custom predictors and builders named `module:Class`, and FQN
filters inheriting the top-level configuration, may still own additional flat
keys. Core prefixes (`dqn_`, `pg_`, `train_`, `gen_`, `eval_`) and clear misspellings
of core names remain reserved. No import of the plugin is needed to load this
configuration. The plugin remains responsible for validating its own options.

`constraint_filter: null` is tolerated while old files migrate; a non-null value
is rejected because the current runtime requires `filters: [{type: ...}]`.
Legacy shared structure settings are handled by the source migration owner;
they are not reinterpreted by the core parameter defaults.

## Runner and HPO integration

`--validate-config` prints the effective config and source map before constructing
an environment or predictor. `run_config.json` stores all effective core values;
`config_sources.json` stores their origins. Invocation evidence also retains the
original YAML, exact command and operational flags. Generation-only runs preserve
training logs. All seed sources enable the same deterministic Torch settings;
`--dp-seed` is retained as a CLI migration alias.

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
| `eval_seed` | int | 0 | >= 0; <= 4294967295 | Random seed for property evaluation and structure sampling. |
| `train_seed` | int | eval_seed | >= 0; <= 4294967295 | Training random seed. |
| `gen_seed` | int | eval_seed | >= 0; <= 4294967295 | Candidate-generation random seed. |
| `constrain_training` | bool | True | — | Apply configured constraints during training. |
| `train_count_duplicates` | bool | True | — | Count repeated terminal candidates toward training budgets. |
| `train_budget` | int | None | >= 1 | Total paid training candidates; derives algorithm episode limits when set. |
| `train_max_duplicate_attempts` | int | 1000 | >= 1 | Maximum consecutive duplicates in a unique-candidate budget. |
| `gamma` | float | 0.9 | >= 0; <= 1 | Shared discount fallback for both algorithms. |
| `dqn_gamma` | float | gamma; otherwise 0.9 | >= 0; <= 1 | DQN discount, overriding the shared fallback. |
| `pg_gamma` | float | gamma; otherwise 0.9 | >= 0; <= 1 | Policy-gradient discount, overriding the shared fallback. |
| `dqn_hidden_dim` | int | 256 | >= 1 | DQN hidden-layer width. |
| `dqn_lr` | float | 0.001 | > 0 | DQN optimizer learning rate. |
| `dqn_buffer_size` | int | 50000 | >= 1 | DQN replay capacity in transitions. |
| `dqn_eps_anneal_eps` | int | 10000 | >= 1 | DQN epsilon annealing episodes. |
| `dqn_eps_min` | float | 0.05 | >= 0; <= 1 | DQN minimum exploration probability. |
| `dqn_num_train_eps` | int | 20000 | >= 0 | DQN training episode budget. |
| `dqn_warmup_eps` | int | explicit pg_warmup_eps; otherwise 500 | >= 0 | DQN warmup; explicit pg_warmup_eps is a legacy fallback. |
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
| `pg_entropy_coef` | float | 0.01 | >= 0 | Normalized-entropy penalty coefficient. |
| `pg_entropy_min` | float | 0.3 | >= 0; <= 1 | Normalized policy entropy floor. |
| `pg_repeat_penalty_coef` | float | 0.0 | >= 0 | Repeated-candidate policy penalty coefficient. |
| `pg_repeat_penalty_shape` | str | 'log' | log, sqrt, linear | Repeated-candidate penalty growth. |
| `num_gen_eps` | int | 200 | >= 0 | Requested exploitation candidates. |
| `exploration_gen_eps` | int | 0 | >= 0 | Additional random-exploration generation episodes. |
| `max_gen_attempts` | int | 10 * num_gen_eps | >= 0 | Maximum candidate-generation attempts. |
| `gen_temperature` | float | 1.0 | >= 0 | Generation softmax temperature; zero selects greedy behavior. |
| `gen_top_frac` | float | 0.0 | >= 0; <= 1 | Generation top-action fraction; zero disables top-fraction sampling. |
| `gen_epsilon` | float | 0.0 | >= 0; <= 1 | Generation random-action probability. |
| `k` | float | 1.0 | >= 0 | Default uncertainty coefficient passed to property evaluation. |
<!-- schema:end -->

## Deprecated spellings

| Old spelling | Canonical spelling |
|---|---|
| `dp_seed` | `eval_seed` |
| `buffer_size` | `dqn_buffer_size` |
| `num_train_eps` | `dqn_num_train_eps` |
| `grad_steps_per_ep` | `dqn_grad_steps_per_ep` |
| `target_update_freq` | `dqn_target_update_freq` |
| `eps_anneal_eps` | `dqn_eps_anneal_eps` |
| `eps_min` | `dqn_eps_min` |
| `entropy_coef` | `pg_entropy_coef` |
| `repeat_penalty_coef` | `pg_repeat_penalty_coef` |
| `repeat_penalty_shape` | `pg_repeat_penalty_shape` |

Legacy aliases are a migration aid, not additional independent settings.

## Migrating authored pre-0.2 files

Safe aliases are centralized; historical scientific defaults are not silently
restored throughout the runtime. `scripts/migrate_config.py` prints a one-time
conversion diff and can write a distinct new local file. Scientific decisions
must be resolved explicitly; it never imports models or guesses physical units.
Audit local files with `scripts/audit_scenarios.py --domains --inventory <local-report.json>`.
The inventory is a local JSON object with `entries: [{"path": "relative/config.yaml"}]`.
It contains paths, not embedded configuration bodies. Old YAML acceptance is
distinct from compatibility of old RL weights or buffers.
