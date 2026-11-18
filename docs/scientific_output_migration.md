# Scientific output and shared model migration

Stage 4 separates physical predictions from rewards. `StructureScorePredictor`
returns `(reward, None)` from both `predict` and `predict_raw`: standard deviations
of conductivity, energy and probability cannot be averaged into an uncertainty
for their combined reward. Physical means and uncertainty quantities are available
through `predict_outputs(candidate)` as immutable `ScientificOutput` records.
`per_objective_stats` retains a compatibility view `(mean, selected_spread)`, with
`None` when the selected quantity is unavailable.

## Model reuse

Every property declares its own predictor and optional `models` checkpoint list.
The engine reuses identical model declarations without a public registry:

```yaml
properties:
  - name: expansion
    predictor: glassnet
    output: CTEbelowTg
    direction: min
  - name: transition_temperature
    predictor: glassnet
    output: Tg
    direction: max
```

The adapter's output catalog and model assets still need validation before a
scientific run. Identical predictor construction settings share model instances.
One `predict_outputs` call serves properties using the same candidate, shared
structure ensemble and operating conditions. Different conditions retain their
distinct inference. DeepMD property models reuse their complete raw output matrix
before individual `output_index`/`output_aggregator` selections.

There is one supported YAML shape: `properties[].predictor` and, when explicit
checkpoint paths are needed, `properties[].models`. Top-level `predictor`,
top-level `models`, `model_ref`, and property `model`/`dp_models` aliases are rejected.
Duplicate checkpoints within an ensemble are also rejected. Output selection and
reward parameters do not create redundant copies of the underlying model.

Adapters should implement `predict_outputs(candidate)` (or
`predict_structure_outputs(structures)` for engine-built structures), returning
`{output_name: ScientificOutput}`. An optional `output_schema` mapping must agree
exactly with the returned names and schemas. Named outputs take priority over a
legacy `predict` method, which may already contain a reward direction or penalty.
A multi-output adapter requires an explicit `output` selector.

## Axes, units and transformations

Retained samples form a rectangular model-by-configuration matrix, with explicit
identities for each model and evaluated configuration. Every model must predict
the same surviving configurations in the same order. Failed structure preparation
is diagnosed by its source/sample identity before scoring; an adapter may not
silently drop a model row or return a shortened row.

For a matrix `Y[m,c]`:

- Mean: average configurations within each model, then average models.
- `model_spread`: population standard deviation of model means.
- `config_spread`: population standard deviation of configuration means.
- `pooled_spread`: population standard deviation of all retained values.
- `native_uncertainty`: the backend's own reported standard deviation, whose
  original method is not relabeled as model or configuration spread.

A singleton axis has no spread estimate (`None`). Two equal model means have a
measured spread of zero. These are different statements. The axis summaries are
not a variance decomposition, and none is automatically a calibrated confidence
interval.

Units default to unknown (`None`), not dimensionless. Output representations are
`physical`, `ln`, `log10` or `pIC50`. `transform: exp` requires natural-log samples;
`transform: pow10` requires log10 samples. Both act on every raw sample, then
recompute all statistics. For log samples `[0, ln(9)]`, the physical mean is 5 and
configuration spread is 4; exponentiating the log mean would incorrectly return
3. An explicit `unit` on a transformed property declares the destination unit.
An adapter's existing unit or representation cannot be silently overridden.
There is no generic inverse conversion for pIC50.

Legacy `score`/`score_structures` values are retained as raw model samples. Legacy
`predict`/`predict_structures` returns `(mean, std)` as a point estimate with native
uncertainty. Without its raw distribution, a nonlinear inverse is rejected.
Several self-folding legacy models cannot have their native uncertainties averaged;
they must expose raw samples through the scientific output protocol.

## Rewards and classification

The default property objective is now `mean`. Existing explicit objectives remain
explicit. `mean_minus_kstd` and `mean_plus_kstd` require an available selected
uncertainty. Select one with `uncertainty: model_spread`, `config_spread`,
`native_uncertainty` or `pooled_spread`. With no selection, the legacy self-folding
interface uses native uncertainty; other protocols default to model spread.
For a property participating in the reward objective, an unavailable statistic
with an explicit penalty is an `EvaluationProtocolError`,
including a single-model point predictor with no model-spread estimate. The engine
does not quietly fabricate zero, change the uncertainty type, or disable a penalty.

Reward coefficients belong exclusively to `reward.objective.terms`,
`reward.gates[].score.terms`, or `reward.alternatives[].terms`. Property `weight`
is rejected. A single-property weighted sum defaults to coefficient 1; multiple
properties require explicit objective terms. Properties absent from those terms
remain available as scientific outputs and gate inputs, without applying adjusted
objective arithmetic to them. Objective terms consume direction/uncertainty/scale
adjusted values; gate terms consume raw gate statistics.

For ActMOF, the only property is `q`, with a maximizing mean objective and scale
100000. The concise `reward: {type: weighted_sum}` produces `q / 100000`.
In the supplied perovskite configuration, `name: synthesis` labels diagnostics,
not a model or application class. Its `score.terms: {pure_probability: 1.0}`
passes the classifier's raw gate statistic to the `min: 0.8` test. The three delta
coefficients remain `0.33333` only in the objective terms; the classifier has no
coefficient in that objective.

Classification gates preserve their existing mean-minus-model-spread rule with
multiple models. With one model, the gate uses its mean and the saved model spread
remains missing. Selecting configuration spread for a property's reward does not
change the gate's separate definition. Reward scales, signs, thresholds, weights
and target-distance objectives otherwise retain their semantics.

## Evaluation records and accounting

One successful record contains named scientific outputs, compatibility statistics,
leaf metadata and structure diagnostics. Explicit properties at different
conditions retain all outputs, including those outside the selected reward
alternative. Metadata is captured immediately after the corresponding inference;
reusing the same model for another condition does not replace an earlier result's
metadata. `cached_outputs(candidate)` returns an existing record or `None`, without
causing inference. Batch records remain readable after persistent LRU eviction.

`oracle_identity()` describes physical model/source declarations, output selection
and frozen asset evidence. Known weights are hashed once at initialization and
checked with file metadata before reuse. Unknown external dependencies are stated
as unverified. Reward-only changes are excluded. External model construction and
inference are isolated from RL RNG streams; candidate/model/source/condition seeds
are stable across candidate query order, cache eviction and scalar/batch calls.
This cannot reset an adapter's private generator unless that adapter implements
its own stable candidate protocol.

`cost_counters()` reads each unique leaf instance's reported counters once, even
when several properties reference it. Missing backend counters remain unknown.
For built-in DeepMD paths, the engine reports attempted ensemble helper calls and
requested model/configuration pairs. These are not network forward calls: the
backend may batch or retry internally, so `network_forwards` remains unknown.

The existing `predict_cache_size` bounds persistent candidate records, but each
record now includes raw sample matrices and diagnostics; memory use therefore also
scales with model count, configurations and property count. Batch records temporarily
retain the current wave for consistent diagnostics. No additional configuration
switch has been introduced for this contract.

## Verification boundary

Tests exercise independent numerical examples, mocked named models and DeepMD
matrices, actual ASE structures, stochastic external leaves, shared-instance
lifecycle, source isolation, and generated CSV missing-value semantics. These tests
establish framework behavior. They do not establish accuracy, throughput or GPU
numerical reproducibility for an external scientific checkpoint. GlassNet, OOH,
peptide and other research assets still require their respective integration tests
in the target environment.
