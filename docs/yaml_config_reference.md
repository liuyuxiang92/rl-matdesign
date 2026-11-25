# YAML configuration reference

A scenario declares `groups`, optional `structure`, `properties`, and `reward`,
followed by training and generation settings. See [core parameters](configuration_schema.md)
for the executable table of shared training, generation, seed and parallelism
options, and [design groups](design_groups.md) for application examples.

The current interface has two group kinds: `composition` and `categorical`.
Old fractional/integer/sequence schemas, standalone scientific builders, named
structure sources, property `source` selectors and `sweep` are removed. They are
not compatibility interfaces. Update an old scenario explicitly and retrain;
old artifacts cannot be resumed through a schema rename.

## Chemical groups

```yaml
groups:
  - name: alloy
    kind: composition
    species_set: [Fe, Ni, Co]
    amount_set: [0.25, 0.5, 0.75]
    n_steps: 2
    total: 1
    unique: true
    element_bounds: {Fe: [0.25, 0.75]}
```

`amount_set` supplies increments, either a list, `{min, max, step}`, or one menu
per species. `n_steps` is the exact action count. Repeated labels accumulate
unless `unique: true`. `total` is optional and constrains the final sum of the
raw amounts. `element_bounds` constrains final amounts of the listed choices,
with absent choices treated as zero. All restrictions apply together.

For oxide counts, use integer amounts and omit `total`; require oxygen with
`element_bounds: {O: [1, 9]}`. Oxygen may be selected at any step. For GlassNet,
compound labels such as `SiO2` carry formula-unit amounts. A bound on `SiO2`
applies to that recipe label rather than to elemental oxygen from every oxide.

Optional scientific `filters` belong to each group. Every listed filter must
pass. Supported built-ins are `smact_charge`, `pauling_en`, and `ooh_phase`; their
scaffold/context is explicit. Listed filters apply during warmup, training and
generation. Omit `filters` or use `filters: []` to skip optional filters. Intrinsic
domain feasibility remains active in either case. There are no `fraction_set`, `ratio_set`, `n_components`,
`sites`, `token_requirements`, or chemical `anion_formula` parameters.

## Categorical groups

```yaml
groups:
  - name: conditions
    kind: categorical
    n_steps: 2
    choices:
      - name: temperature
        values: [300, 350, 400]
      - name: linker
        values: [linker_a, linker_b]
```

For shared replacement menus use `labels`, `values`, and `n_steps` instead of
`choices`. One action assigns one value to one target; a target can be assigned
once. Assignment order is irrelevant, while labels retain their identity.
Numerical fields retain physical magnitudes; symbolic values use typed identity
features. Parent sequences and peptide chemistry belong to the peptide assembler
or predictor plugin, not generic group options. See [labelled assignments](labelled_assignments.md).

## Structure pipelines

Include each desired operation in `structure.steps`; omit an operation to skip it.
LiPS standalone evaluation uses this same declared pipeline without a separate
geometry-optimization enable/disable flag.

```yaml
structure:
  builder: pipeline
  base_poscar: POSCAR
  n_random_configs: 10
  steps:
    - type: substitute
      a_site_symbol: Fe
      a_dopant_group: alloy
    - type: geo_opt
      model: models/relax.pt
      fmax: 0.05
      steps: 100
```

Use only the operations required by the application. Substitution alone builds
an ASE/POSCAR structure; MD and geometry optimization are optional. Every later
operation consumes the previous output. All structure properties share this one
sample ensemble. No `application` wrapper or named source registry is needed.

Supported operations are `substitute`, `sse_charge_balance`, `ooh_adsorbates`,
`md_average`, and `geo_opt`. Symbol/group pairs run alphabetically from a to z
and resolve original POSCAR header blocks by symbol occurrence. For `Sr Fe Fe`,
a=Fe selects the first Fe block, b=Fe the second, and c=Sr selects Sr.

Signed perovskite defects are paired with substitution through
`a_defect_group` (and analogous letters). LiPS uses `amount_basis:
cation_fraction` on substitution, followed by `sse_charge_balance`. OOH uses
substitution followed by `ooh_adsorbates`. External builders may use a fully
qualified `module:Class`; their options belong to the plugin.

The peptide assembler uses `structure: {builder: peptide, length: 11}` for a new
sequence, or a `parent` list for replacement. It assembles categorical residue
stages and does not require POSCAR operations.

## Properties, models and reward

Every scenario uses a nonempty `properties` list. Each output has a unique `name`,
its own `predictor`, and any model files in a `models` list. One checkpoint still
uses a list. Model paths, heads, conditioning parameters and scientific feature
construction are predictor-owned. The top-level `predictor`/`models` forms and
property `model_ref`, `model`, and `dp_models` aliases are not supported.

```yaml
properties:
  - name: energy_T300
    predictor: dp_property
    models: [models/one.pt, models/two.pt]
    fparam: [300]
    direction: min
    scale: 1
    objective: mean
    uncertainty: model_spread
reward:
  type: weighted_sum
```

Reward coefficients have one owner: `reward`. `properties[].weight` is removed
and rejected. For a single property, `reward: {type: weighted_sum}` uses coefficient
1 and needs no `objective.terms`. For multiple properties, `weighted_sum` and
`gated` require explicit `reward.objective.terms`. For example:

```yaml
reward:
  type: weighted_sum
  objective:
    terms: {bandgap: 1.0, formation_energy: 0.5}
```

Only properties named in the objective contribute to that objective. Coefficients
multiply their scores after direction, uncertainty objective, and scale have been
applied. Gates have independent coefficients in `gates[].score.terms`; `max`
alternatives have coefficients in `alternatives[].terms`. Do not duplicate these
coefficients in property or model declarations.

ActMOF has one property `q`, with `direction: max`, `objective: mean`, and
`scale: 100000`. Its concise `reward: {type: weighted_sum}` gives `reward = q / 100000`.
An explicit term is needed only to choose a different coefficient.

In the perovskite gate, `name: synthesis` is a user-chosen diagnostic label; it
names saved gate fields such as `gate_synthesis_value`. `score` combines raw
property statistics using its `terms`. Here `pure_probability: 1.0` uses that
statistic unchanged and `min: 0.8` is its required threshold. With one classifier
checkpoint the statistic is mean pure probability; with multiple checkpoints it
is mean minus model spread. Gate scores do not apply the property's direction,
scale, or uncertainty objective. Passing the gate enables the separate delta
objective, whose three `0.33333` coefficients occur only in `reward.objective.terms`.
The classifier is omitted from that objective and contributes only to the gate.

OOH uses `predictor: ooh_frames` and `models` in its `overpotential` property.
There is no `predictor: ooh` shortcut. Its direction is `min`, while its selected
`mean_plus_kstd` objective rewards higher model spread after sign conversion.
The retained `configs/ooh.yaml` uses 29 available cations and DQN Monte Carlo.

GlassNet declares `predictor: glassnet` separately on each output property; its
adapter loads its own pretrained assets. Identical model declarations are reused
automatically, so selecting `T3`, `CTEbelowTg`, and `Tg` shares model loading and
raw predictions. Constructor options belong to the property, not a model registry.
Structure-operation `model` settings remain local to `md_average` or `geo_opt`;
they choose a simulation potential and are not property prediction ensembles.

Declare different experimental conditions as separate named properties.
LiPS declares conductivity and stability at each of 460, 470, 480 and 490 K.
Its `reward: {type: max, alternatives: [...]}` compares weighted property pairs
at the same temperature; all eight predictions remain available. There is no
condition sweep or separate source selection. Scalar and batch evaluation use
the same reward and sample identities.

The supported reward operators and scientific uncertainty contract are described
in [scientific outputs](scientific_output_migration.md). Geometric randomness and model
spread are distinct; unavailable uncertainty must not be represented as zero.

### Automatic reward calibration

The top-level `value_normalization` setting controls both reward calibration
and value-target normalization for DQN-bootstrap, DQN-MC and A2C. Use this
setting for a fresh pilot of exactly 100 distinct, successfully evaluated candidates.

```yaml
value_normalization: popart
```

The reward type selects the profile. A weighted sum uses
`auto_weighted_sum_v1`. The pilot estimates the mean and population standard
deviation of each nonzero objective term before combining the standardized
terms with the user’s weights. Oxide sintering rewards lower predicted
temperatures. [Oxide sintering configuration](../configs/oxides_sinter.yaml)

A gated reward uses `auto_gated_softplus_v1`. In GlassNet, the pilot fits a
reference and scale for T3 and a separate violation scale for each gate.
Gate checks still use the original CTE and Tg bounds. Passing candidates
receive `1 + softplus(U)`, where `U` is the weighted standardized quality.
Failing candidates receive the negative sum of normalized violations, with a
default priority of 1 per gate. [GlassNet coarse configuration](../configs/glassnet_coarse_2pct.yaml)

The perovskite example uses `quality_mode: aggregate` to calibrate the combined
uncertainty-adjusted delta objective. Its synthesizability probability supplies
the gate. [Perovskite configuration](../configs/perovskite_calibrated.example.yaml)

The runner freezes the fitted scales for training and generation. Gate failures
with valid predictions count toward the pilot. Duplicate candidates and failed
evaluations do not fill pilot slots. These 100 candidates are separate calibration
overhead: they do not consume training or generation budget, populate DQN replay,
or enter the training seen-candidate set. DQN collects its configured warmup
after calibration and spends the rest of its training budget on online samples.
Fresh DQN training requires `dqn_warmup_eps >= 1` to fit the state scaler; there
is no calibration-imposed minimum of 100. A2C uses its full training budget for
policy samples; its descriptor-only scaler warmup remains separate. A
calibration candidate encountered later can count toward training.

For a fresh run with `train_budget: 115`, DQN-MC therefore collects 100 calibration
candidates and then 115 budgeted training candidates, including its configured
warmup. The calibration observations are saved in
`reward_calibration_observations.json`, not `training_samples.csv`. They are also
excluded from HPO training scores and attempt-efficiency accounting. Predictor
work can include retries, failures and cache reuse. The independent
`reward_calibration_budget.json` records this overhead with
`accounting: outside_training_budget`:

| Field | Meaning |
| --- | --- |
| `target_candidates` | Required distinct valid candidates: 100 |
| `valid_candidates` | Distinct valid calibration observations collected so far |
| `attempts_started` | Calibration sampling attempts, including duplicates and failed attempts |
| `evaluations` | Calls to obtain calibration observations, including failed evaluations; cache reuse can reduce backend work |
| `status` | `RUNNING` until the calibration is frozen, then `COMPLETED` |

The counters are saved at settled calibration checkpoints. Loading an existing
compatible calibration artifact skips new calibration sampling.

With `value_normalization: none` (the default when omitted), training uses the
configured raw reward and learns values in raw reward units. It skips calibration
sampling unless `gradient_clipping: clipping` requests the evaluation-only pilot
described below.
A passing 1500 K GlassNet candidate earns 1500. A gate failure starts at
`failure.reward` and subtracts the weighted violations in native units.
In the oxide example a predicted sintering temperature of 850 earns −850.
The `max` reward operator uses this mode to compare its declared alternatives.

When migrating a calibrated YAML, delete the entire `reward.calibration`
section, property `normalization` and gate `violation_normalization`. Keep
scientific weights and thresholds. Gate `priority` applies in PopArt mode.
The raw gated reward uses `failure.reward` and each gate's `penalty_weight`.
For an aggregate gated objective, put `quality_mode: aggregate` under `reward.objective`.

In PopArt mode the runner stores pilot state in `calibration_checkpoint.pt` and writes the
frozen `reward_calibration.json` before training. `--resume-training`
continues from the last settled pilot checkpoint when a learner checkpoint
has yet to be created. The frozen JSON is required for generation and learner
resume. Use `--load-reward-calibration /path/to/reward_calibration.json` to
reuse compatible calibration on another grid without evaluating another pilot.
Exact resume requires the same calibration accounting protocol and source version;
checkpoints made when calibration consumed training budget need the archived code
or a new run directory.

`value_normalization: popart` enables adaptive value-target normalization for
DQN or A2C under the shared `value_normalization_profile: popart_v1`. PopArt
adapts to the learning targets after the scientific reward is computed.
See the [Step 1 design](reusable_training_step1_plan.md) for the scale rules,
failed-candidate examples and experiment comparisons.

### Gradient clipping

There is one public clipping setting, in YAML or as `--gradient-clipping`:

```yaml
gradient_clipping: clipping   # choices: none, clipping; default: none
```

`none` retains the existing training behavior without gradient clipping. It does
not introduce another calibration phase. `clipping` enables adaptive global L2
gradient clipping for DQN-MC, DQN-bootstrap and policy-gradient learning. DQN clips
the online Q-network; A2C clips its actor and critic separately. Target networks
receive model copies and have no gradient-clipping controller.

The initial cap is measured from gradients evaluated on 100 distinct, successfully
scored pilot candidates. With PopArt, these are the same candidates used for reward
calibration, so enabling clipping does not request a second 100-candidate pilot.
With raw rewards, the pilot keeps the configured scientific reward unchanged.
Gradient evaluation performs forward/backward calculations without an optimizer
step. The pilot does not train the RL networks, populate replay or training sample
files, or spend the training/generation allowance. Its evaluation and computation
cost are real overhead and must be reported separately. `gradient_calibration.json`
records the per-network pilot norms, initial caps, resolved profile, target
statistics and zero optimizer updates.

The `adaptive_grad_clip_v1` recipe sets each network's initial cap to the 90th
percentile of its positive pilot gradient norms. Each norm comes from the mean
loss over one candidate trajectory on an unchanged disposable network copy.
PopArt coordinates on that copy are updated once using all pilot targets; the
live learner is untouched. Actor advantages use one pilot-wide standardization.

After calibration, each network keeps its own initial cap until its last 200
successful learning updates contain at least 20 positive gradient norms. It then
uses the 90th percentile of those positive, unclipped norms. A gradient does not
set its own adaptive threshold: only prior successful updates determine the cap.
These percentile, window and startup rules are versioned implementation settings;
there are no fixed-threshold, percentile or window options in scenario YAML. An
all-zero pilot is recorded as unresolved and the first later positive gradient
can establish a provisional cap without fabricating a fixed numerical threshold.

Clipping occurs after backpropagation and before Adam. It limits the gradient
norm supplied to Adam, not the optimizer's resulting parameter displacement.
Q, actor and critic histories and their profile are saved in learner checkpoints.
Exact resume restores them instead of recalibrating; generation does not update
them. A frozen reward-calibration JSON alone does not contain the scored
trajectories needed to evaluate initial gradients.

## Running and validation

```bash
python scripts/audit_scenarios.py --config configs/lips_sse.yaml
python scripts/run_experiment.py --config configs/test_dummy.yaml --out runs/demo
```

YAML is the base; explicit CLI flags override matching values. Only canonical
training names are accepted. Use shared `gamma` for every method; DQN and PG
warmup budgets are independent. `num_gen_eps` is the entire generation quota.
Training counts distinct successful candidates by default; set
`train_count_duplicates: true` or pass `--train-count-duplicates` to count
repeats too. The [coarse GlassNet scenario](../configs/glassnet_coarse_2pct.yaml) gives
`train_budget`, `num_gen_eps` and `gen_temperature` a value for each method.
Each of these three fields also accepts a single value shared by all methods.
For DQN the key is `dqn-bootstrap` or `dqn-mc`, selected by `dqn_target_mode`.
For policy-gradient runs the key is the method name. The runner applies explicit
CLI overrides after selecting the method's values. Resolved run configs store scalars.

Scientific output columns and the 200,000-record prediction cache
limit are fixed by the implementation.
The fully resolved config and scientific contracts are stored with run artifacts.
A metadata-only audit does not validate external model weights or template files.
