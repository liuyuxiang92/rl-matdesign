# Structure source migration: stage 3 design and acceptance gates

Status: runtime source normalization, ownership, deterministic sampling, paired
sample diagnostics and resource cleanup are implemented on 2026-11-06. The
original design audit below remains the acceptance contract. The final section
records actual local verification and remaining limits; real GPU/model parity
is not implied by the local test suite.

The aim is to replace `share_structure` with one explicit relationship: a
property refers to a named source of structures. The source can produce an
ensemble from several templates. Properties referring to that source evaluate
the same built and relaxed samples. Independent sources remain independent even
when their configuration dictionaries happen to be equal.

## Current behavior and concrete risks

| Area | Verified current implementation | Migration requirement |
|---|---|---|
| Single candidate | `StructureScorePredictor._materialize_structures` uses one shared builder or one builder per property | Group materialization by source identity; build and relax once per candidate/source |
| Batch | `_materialize_structures_many` groups by the same boolean, uses `build_many` when available, and isolates failed candidates | Keep input order, successful neighbors, duplicate coalescing and atomic successful-stat caching |
| Parallel | `ParallelStructureBuilder` owns persistent workers and reports `handles_geo_opt`; the parent skips worker-performed relaxation | Source owns the pool; preserve this ownership signal and never re-relax returned structures |
| Seeds | One advancing predictor `_rng` feeds all builders; batch worker seeds depend on the flattened batch length/order | Derive source/candidate/sample streams independently of query order, cache hits, property order and scheduling |
| False branch | Per-property builders inherit only `_SHARED_BUILDER_KEYS`; `geo_opt` is still a global predictor setting | Compute the old effective configuration explicitly before migration; do not invent inheritance that never occurred |
| Model instances | FQN/registry leaves instantiate once per property/model; DeepMD caches are keyed by property name | Source sharing does not itself reuse model instances; model-resource identity needs its own contract |
| Ensemble statistics | Average structures for each model, then mean/std across model means, with population std | Source extraction must preserve both axes and the exact reduction |
| Multiple templates | Substitute/SSE builders concatenate their samples; template-specific counts override the fallback count | Preserve returned counts, sample order and count weighting; do not replace them with equal-template averages |
| Plugin mutation | Built-in energy scoring copies each ASE object, but FQN structure leaves receive the shared mutable list | Isolate plugin mutations at the scoring boundary without rebuilding or relaxing again |
| Lifetime | Statistics have an LRU; structures live only in the evaluation/batch or diagnostic result; `close` visits every resource | Keep bounded evaluation lifetime, clean partial initialization, and close pooled objects once |

`_fold_models_structures` currently evaluates, for model predictions `y[m,s]`,
`u[m] = mean_s(y[m,s])`, then `(mean_m(u), std_m(u, ddof=0))`. One model therefore
has zero ensemble-model uncertainty even with several configurations. DeepMD
`transform: exp` is applied to each prediction before this reduction. A leaf
with one self-folded `(mean, std)` retains its own uncertainty. None of these
formulas belongs to the source refactor; changing them here would mix two
scientific changes. The existing FQN transform inconsistency remains a separate
stage 4 item, not a reason to silently change it during source extraction.

The parallel worker currently calls `builder.build(..., n_configs=1)` and
requires exactly one returned cell. A multi-template native builder can return
more than one cell for that call, and the worker explicitly rejects it. Merely
splitting a native ensemble into one-config calls is therefore not a valid
general implementation of source sampling.

## One source definition, explicit references

Proposed public form:

```yaml
structure_sources:
  bulk:
    builder: substitute
    base_poscar:
      - {path: phase_a.POSCAR, n_configs: 3}
      - {path: phase_b.POSCAR, n_configs: 2}
    site_symbol: Ti
    n_random_configs: 5
    geo_opt:
      model: relaxation.pt
      fmax: 0.01
      steps: 1000
      relax_cell: true

properties:
  - name: stability
    predictor: dp_energy
    models: [energy.pt]
    source: bulk
    direction: min
  - name: conductivity
    predictor: dp_property
    models: [property.pt]
    source: bulk
    direction: max
```

`source` may be omitted when exactly one source is declared. With several
sources a structure property must name one. Unknown names and unused/misplaced
source-specific settings produce actionable configuration errors. Composition
and sequence predictors need no source and must not construct a builder, import
DeepMD, or start workers. Supplying a source to a predictor that only accepts a
composition is a configuration error, rather than a silently ignored option.

The source owns templates, substitution/site rules, counts, builder MD or
optimization protocol and the optional post-build relaxation. These are moved
from their existing locations, not duplicated. Execution settings such as
`structure_parallel` can be inherited once by the source runtime; they are not
part of its physical protocol fingerprint. Reward weights, output indices and
property directions do not belong to a source. New-form configurations cannot
simultaneously define contradictory top-level/per-property builder settings.

Internal objects can remain small: a normalized immutable `StructureSourceSpec`,
a runtime source owning its builder and relaxer, and a property-to-source map.
No general workflow graph or new ordered/joint/two-step configuration switch is
needed. A source identifier is stable within an effective configuration and is
stored in provenance alongside its physical protocol and input hashes.

Two entries named `bulk_a` and `bulk_b` are **two sources**, including when all
their fields match. Only references to the same source identifier share samples.
Do not use equality of settings, paths, or protocol hashes to deduplicate these
entries. Hashes establish provenance and cache validity; declared identifiers
establish ownership and independent sampling namespaces.

## Migrating the old boolean without hiding a sampling change

| Legacy input | Explicit migration | Scientific effect |
|---|---|---|
| No boolean or `true`, with structure properties | One source named `default`, all structure properties reference it | Same intended shared-ensemble semantics; the new RNG protocol changes historical sample sequences |
| Pure composition properties | No source | No structure work introduced |
| `false`, properties intended to use the same physical sample source | One named source, all those properties reference it; remove the boolean and repeated builder fields | Deliberate replacement of separate sampled realizations by a common realization ensemble |
| `false`, genuinely different templates/protocols/conditions | Separate explicitly named sources and references | Keeps distinct physical evaluations |
| Ambiguous `false` in an external configuration | Migration report lists effective builders and independent legacy sampling; require an explicit source mapping before running the new form | No unannounced merge based on dictionary equality |

Sharing one ensemble changes the correlation between property estimates and can
change rewards and learned policies. Make that decision explicitly and archive
the old effective configuration with historical results. Do not infer shared
ownership merely from equal template paths or settings.

For generic false configurations, reconstruct the exact prior effective builder
configuration from `_SHARED_BUILDER_KEYS` plus property overrides, and the global
`geo_opt`. For example, a top-level builder-specific setting outside that key
list was not necessarily inherited. An audit should flag that discrepancy;
silently moving every top-level value into a new source would change the old
false behavior without explanation. Conflicting per-property `n_random_configs`
also cannot be discarded when several properties are assigned one source.

Compatibility parsing may accept the unambiguous old shared form and normalize
it immediately, with a migration note in effective configuration/provenance.
It must not keep a second legacy execution branch. Explicitly selected mappings
for old false scenarios become normal source declarations before evaluation.

## Sampling, templates and reproducibility

A versioned stable seed derivation must encode the recorded `eval_seed`, typed
canonical candidate identity, declared source identifier, physical protocol,
template/sample identity, and a purpose such as decoration or MD. Use a stable
hash/serialization, not Python `hash()`, property index or an advancing shared
generator. Different purpose streams prevent adding one stochastic operation
from consuming another operation's random draws. A missing seed must be resolved
and recorded once if reproducible replay is claimed.

Changing reward weights, property order, number of workers, GPU selection, batch
partitioning, cache size or unrelated intervening candidates must not change a
source's sample identities. Changing physical templates, sampling rules or
relaxation protocol must be visible in the evaluation protocol and provenance.
Renaming a source changes its independent random namespace; the effective names
are therefore part of the run record, not cosmetic metadata.

There are two separate implementation levels:

1. A candidate/source-local RNG around the existing native ensemble builder can
   establish query-order and cache-eviction independence while preserving its
   native multi-template call. Batch APIs must receive aligned candidate RNGs or
   sample plans; they cannot reuse one advancing RNG across the batch.
2. For serial/worker sampling parity, supported builders must expose a common
   internal sample plan. Each task identifies template and sample index; serial
   and worker execution consume the same task seed and return results in the
   plan's order. Built-in substitute/SSE plans must retain template count
   overrides and physical builder parameters. External builders without this
   capability retain their native seeded serial ensemble path; reject an
   unsupported parallel request clearly instead of silently splitting or
   changing it. This is capability detection, not another YAML mode switch.

The second level is necessary before claiming serial/parallel sampling parity.
Exact relaxed coordinates and model outputs additionally depend on the model,
software stack, deterministic GPU operations and convergence behavior. Local
dummy tests establish framework semantics, not real GPU numerical equivalence.

For template counts 3 and 2, the current property mean gives the templates
weights 3/5 and 2/5 when all structures succeed. Preserve this sample-count
weighting. Nonempty surviving ensembles are currently accepted after individual
relaxation failures; preserve that policy and expose surviving template/sample
identities and counts in diagnostics. Do not add a new template-weight setting
merely to reproduce the existing reduction.

`score_breakdown` intentionally returns inspected structures and bypasses the
scalar-stat cache. Candidate/source-local sampling should regenerate the same
sample identities if diagnostics follow a cached prediction. It need not retain
all ASE objects indefinitely. The sweep still reuses one materialized ensemble
across scoring conditions; a condition that actually changes the source's build
physics must be represented in its source protocol rather than silently reused.

## Execution, models, mutation and accounting

For each unique uncached candidate, materialize every referenced source once.
Hold its ensemble in an evaluation-local map, then give each property its source
payload. Batch materialization operates on aligned candidate/source work items;
failed candidates leave the active set while successful neighbors continue.
Return exceptions in the original input positions and cache statistics only
after every required property succeeds. Duplicate candidates launch no duplicate
physical work and receive no duplicate physical task charges.

Preserve the current parent-before-worker model preloading so workers do not
occupy all available accelerator memory before property models are loaded.
Top-level source relaxation is performed in the worker when the builder reports
that ownership; the parent consumes the relaxed result directly. Builder-native
MD/optimization and the explicit source post-build relaxation are different
declared operations and should not be mistaken for duplicate calls merely
because both involve an optimizer.

One source shared by two properties does not justify sharing arbitrary model
instances. At present identical assets under two property names can be loaded
twice. A separate resource cache can later reuse supported DeepMD resources by
backend/factory, asset identity, head and actual runtime construction arguments;
property output selection, direction and reward scale remain evaluation settings.
Do not merge FQN leaves from equal dictionaries: their private state and model
ownership are unknown without a plugin resource contract. Stage 3 must at least
avoid reconstructing models per candidate/source and keep the existing lazy
caches. Full generalized model-adapter pooling belongs to stage 4.

The built-in energy path already copies ASE objects before attaching a
calculator. Built-in property inference reads arrays. FQN structure scorers can
mutate positions, arrays, metadata or calculators in the shared list; isolate
them with detached copies at their scoring boundary (including each ensemble
leaf), or a formally enforced read-only adapter contract. Copying is not another
build or relaxation. Tests should assert identical scientific sample content,
not require identical Python object identity across external scorers.

Keep the bounded statistics cache; do not add a persistent cache of millions of
heavy structures. A source runtime may cache immutable templates and model
resources. Batch-local ensembles are released after scoring. Freeze source
specifications so editing a caller-owned configuration dictionary cannot change
physics while stale statistics remain cached.

Sum build/MD/relax task time and attempted/successful sample counts by actual
source execution, not by the number of consuming properties. Unknown work after
worker failure stays unknown (`None` plus incomplete status), not zero. Property
scoring time remains separate. Align worker records with stable task identities,
including partially successful batches.

Use exception-safe ownership during construction and teardown: a source or model
created before a later constructor failure must close immediately. Close each
actual pooled resource once by object identity; one close failure must not skip
other sources. Preserve worker process-group termination, timeout handling,
connection cleanup and atexit unregistration. Run-level `ExitStack` alone cannot
clean resources from a constructor that fails before returning.

`composition_formula` currently takes the shared or first per-property builder.
That is ambiguous for explicit sources with different formulas. Preserve the
existing single-source label; for multiple sources use the environment's
candidate label and put source-specific formulas in diagnostics, rather than
silently selecting the first source as the universal physical structure.

## Minimal implementation sequence and tests

| Increment | Minimal production surface | Required acceptance tests |
|---|---|---|
| 3A: normalize and own sources | New `structure_sources.py` (spec/runtime or split when justified), `predictors/structure_score.py`, current configuration/schema and provenance asset-discovery paths | One-source inference; explicit multi-source references; unknown/missing references; two equal-but-separately-named sources remain independent; pure composition loads no builder; legacy true normalization; explicit false migration notes and conflicts; input dict mutation cannot alter runtime |
| 3B: evaluate and account once | Existing single/batch materialization, source relaxer ownership, `score_breakdown`, lifecycle methods | Two properties and all sweep values build/relax once; distinct sources twice; fixed model-by-structure matrix gives unchanged mean/std; per-sample exp unchanged; nonempty survivor policy; multi-template counts 3+2 retain weighting; duplicate and cached candidates charge zero physical tasks; partial failure leaves good neighbors and no partial success cache; diagnostic structures match predictions |
| 3C: deterministic samples and workers | Source seed helper, `utils/parallel_structures.py`, `utils/structure_worker.py`, built-in builder sample-plan support | Candidate permutation/order/intervening-query/cache-eviction invariance; split vs joined batch; property reorder and weights unchanged; source namespaces independent; serial/worker same sample tasks; multi-template stable counts/order; unsupported plugin parallel fails before launching work; worker-relaxed samples never re-relaxed by parent |
| 3D: resource and scenario gates | Resource ownership integration, migrated tracked configs and local-scenario fixtures, docs | Constructor-failure cleanup; close idempotence; one bad close cannot leak another pool; worker crash/timeout process cleanup; mutation of one property input cannot contaminate another; model construction counts stable; parent preload order; real-asset integration marked separately from dummy tests |

Existing regression suites to retain and extend are `test_structure_score.py`,
`test_cross_formula_predictor.py`, `test_parallel_structures.py`,
`test_predictor_stage_timing.py`, `test_parallel_stage_timing.py`,
`test_cross_formula_integration.py`, `test_parallel_nonconvergence.py`,
`test_substitute_builder.py`, `test_sse_builder.py`, plus configuration/provenance
and resource-lifecycle tests. Focused new tests should live in
`test_structure_sources.py` and `test_source_sampling.py` when implementation
begins. Update old false/shared-object expectations to test declared source
semantics and scientific payloads; do not weaken them merely to make the rewrite
pass.

The tracked HEA and LiPS-SSE examples need explicit source migrations. Oxide RF,
GlassNet and peptide composition/sequence prediction must continue without
structure machinery. Local studies select their physical sharing protocol
explicitly. OOH owns structure generation inside its own predictor; moving
that work behind the shared source contract requires an adapter with OOH's
adsorbate/frame/sample semantics and is not completed by renaming a seed. The
stage 3 acceptance record must state which source adapters are integrated and
which real models/GPU checks remain pending; an inventory is coverage scope, not evidence that every external application
has run successfully.


## Implemented runtime and verification record (2026-11-06)

`StructureScorePredictor` now owns one builder per normalized named source and
routes each property through `source_plan.property_sources`. There is one
materialization path for shared and independent sources. Legacy shared inputs
normalize to `default`; ambiguous `share_structure: false` raises
`LegacySourceMigrationRequired` with an explicit migration report. Tests of
independent ensembles now declare independent sources; production does not
silently make that decision.

The runtime calls `build_seeded` / `build_many_seeded` when the builder supports
the source protocol. A native external serial builder receives one canonical,
detached candidate and one source/candidate RNG for its whole ensemble. Its
legacy batch API accepting a single advancing RNG is not used. Native returned
samples receive stable source/index identities and an explicit
`seed_scope: native_ensemble`; these are returned-ensemble indices, not a claim
that the plugin exposed every attempted task or template. Built-in substitute
sampling has explicit template/sample identities, including template count
weights. Parallel capability limitations are discussed below.

Each source owns its post-build relaxation settings and lazy calculator. The
parent respects `handles_geo_opt` and does not repeat worker relaxation. A
source's ensemble is reused across properties and sweep conditions. Every
external structure leaf receives detached ASE arrays, nested metadata and a
fresh list; an attached calculator is not copied to another leaf. The statistical
fold and DeepMD memberwise transforms are unchanged. Source sharing does not
merge model instances under equal property dictionaries.

`source_diagnostics(candidate)` returns diagnostics stored in the same evaluation
record as its property statistics and winning metadata. It records source/sample
identities, native versus explicit sample plans, build results, parent-relaxation
failures and surviving samples. `score_breakdown` also returns these source
diagnostics. Heavy structures remain local to an evaluation; the persistent
cache stores only statistics and diagnostics. Cost records contain a per-source
breakdown and charge physical work once, independently of property count.

`source_identity()` returns the resolved evaluation seed, seed protocol version,
frozen source schemas, property references, migration notes and asset-resolution
evidence. `resolve_source_assets(spec, base_dir=None)` hashes known template and
model files, including template-list dictionaries and the default post-build
relaxation model. Missing files are explicitly marked `missing`; an external
builder's relative path is `unresolved` unless its ownership is known. Only
resolved contents enter the asset fingerprint. Known file metadata is checked
before reuse, and changed assets require a new predictor. This is a guard
against ordinary accidental file replacement; it is not a filesystem snapshot
or a proof about every external plugin dependency.

Known built-in physical fingerprints exclude search-domain settings they do not
read. Substitute, site-pick and defect-site builders do not hash `groups`; SSE
hashes only the S-site's ordered slot names and element identities. Changing a
search grid or action filter therefore does not change the physical oracle for
the same candidate. Unknown builders keep their declared context because the
framework cannot prove that it is irrelevant to their physics.

Resources are registered immediately after construction. Failed initialization
closes earlier leaves/builders; failed lazy ensemble loading closes partially
loaded models. `close()` is idempotent and deduplicates resource identities, and
one close failure does not skip the remaining resources.

Final stage 3 targeted validation: 410 tests passed across 20 source, structure,
parallel, timing, failure, reward, training-integration and lifecycle files.
Additional runtime tests verify asset changes and the search-domain/physics
boundary. ASE structures and actual worker subprocesses are exercised with fake
model/CUDA adapters. These tests do not run scientific DeepMD, external MD,
GlassNet or peptide assets on the user's GPU machine.

Remaining boundaries: an unknown external builder must not be assumed to have
serial/parallel sample parity merely because `build(n_configs=1)` returns one
cell. Its native ensemble and independently seeded one-cell calls can represent
different protocols. Parallel source execution requires a declared common sample
capability; unsupported plugins must keep the native serial path or implement
that protocol. SSE native serial sampling retains its whole-ensemble behavior;
explicit SSE per-template worker parity requires an SSE sample adapter. Real
convergence, GPU numerical reproducibility and all external research assets
remain separate integration checks.
