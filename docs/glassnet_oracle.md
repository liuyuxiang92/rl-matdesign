# GlassNet composition oracle

`rl_matdesign.predictors.glassnet.GlassNetPredictor` is a composition-only
scientific predictor. It uses one persistent stock GlassNet instance and returns
three named physical quantities. It does not create structures, choose reward
weights, decide feasibility, or implement the training query budget. The `glassnet` registry entry integrates it with named models; experiment
configuration owns the research domain and reward.

## Model and output contract

The implementation targets GlassPy 0.6.0 at
[`ddc06240152749d14d01e0c4f7cec9e4e79c706d`](https://github.com/drcassar/glasspy/tree/ddc06240152749d14d01e0c4f7cec9e4e79c706d),
the published [GlassNet model](https://doi.org/10.1016/j.ceramint.2023.08.281), and
the author's [model archive](https://zenodo.org/records/13625981).

| Output name | Adapter unit | Stock model unit | Selected network |
|---|---|---|---|
| `CTEbelowTg` | K^-1 | log10(K^-1) | GlassNetMH |
| `Tg` | K | K | Tg ST-NN |
| `T3` | K | K | GlassNetMH |

`T3` is the temperature at viscosity 10^3 Pa.s. These definitions are present in
the pinned [`translators.py`](https://github.com/drcassar/glasspy/blob/ddc06240152749d14d01e0c4f7cec9e4e79c706d/glasspy/data/translators.py#L58)
and its [`prop_units`](https://github.com/drcassar/glasspy/blob/ddc06240152749d14d01e0c4f7cec9e4e79c706d/glasspy/data/translators.py#L644).
The original SciGlass column `T4` is renamed to model output `T3`; do not confuse
the original dataset column name with the model output name.

The adapter computes `10 ** CTEbelowTg` exactly once. Temperatures already use
kelvin; no Celsius conversion is applied. `output_schema` maps each output name
to an `OutputSchema` with its physical unit and representation. Source units,
transforms and selected heads remain in `identity["output_metadata"]`.
`predict_outputs` returns immutable `ScientificOutput` records; their `.mean`
contains the physical point prediction. All spreads, native uncertainty and raw
sample distributions are explicitly `None`: these pretrained point predictions
do not supply a calibrated uncertainty. There is no implicit standard deviation
of zero and no legacy `(reward, std)` `predict` method.

[`GlassNet(st_models=['Tg'])`](https://github.com/drcassar/glasspy/blob/ddc06240152749d14d01e0c4f7cec9e4e79c706d/glasspy/predict/models.py#L356)
retains the same three selected outputs as the default hybrid model while
avoiding its 24 unused specialist networks. One stock `predict` call still
contains the MH and Tg network forwards; it is not one neural forward. The
author's 98-feature construction, scaler transforms and output inverse scaling
are unchanged. Stock prediction creates CPU float32 tensors; this adapter does
not call `.cuda()` or claim GPU oracle support. An RL agent may use a GPU
independently.

## Named oxide inputs and usage

```python
from rl_matdesign.predictors.glassnet import GlassNetPredictor

# Toy input for the API, not a recommended glass formulation.
# In integration, derive oxide_basis from the existing design-domain labels.
oracle = GlassNetPredictor(oxide_basis=["SiO2", "Na2O"])
composition = {"SiO2": 1, "Na2O": 1}
outputs = oracle.predict_outputs(composition)
values = {name: output.mean for name, output in outputs.items()}
batch = oracle.batch_predict_outputs([composition, composition])
manifest = oracle.identity
schema = oracle.output_schema
work_counts = oracle.cache_info()
```

Amounts are oxide **mole ratios**. Mol.% values, fractions summing to one and
proportional integer mole units represent the same composition. They are not
mass fractions or elemental atomic fractions. Missing allowed oxides mean zero;
negative, nonfinite, nonnumeric and all-zero inputs are rejected. Unknown labels
are rejected even when their amounts are zero. The default basis includes SiO2,
B2O3, Al2O3, Na2O, K2O, CaO and MgO. An explicit `oxide_basis`, as in this
example, may describe another set of binary oxides.
This is input validation, not a claim of predictive accuracy for every oxide or
concentration. Research-domain bounds remain the environment's responsibility.

The adapter sorts named labels and canonicalizes exact decimal mole ratios.
Changing mapping insertion order or the order of prior composition assignments
does not change candidate identity. Permuting numbers without their oxide names
changes the material. Explicit zeros and omitted zeros share an identity. The
adapter never rounds fine-grid compositions together. It does not replace the
environment's authoritative integer-grid candidate identity.

## Assets, preparation and reproducibility

Importing this adapter is lightweight and does not import GlassPy. First access
to `identity` or first prediction checks the local GlassPy source, cache version
record, required model files and prepared x/y scalers before importing GlassPy.
This avoids the stock package's import-time cache deletion/download path when
versions differ, and avoids an unexpected scaler fit during a benchmark.

Prepare the pinned optional environment and assets before running an experiment.
Use `platformdirs.user_data_dir('GlassPy')` to locate the actual stock cache.
On Linux the standard `XDG_DATA_HOME` convention can select a persistent cache
before process startup; this adapter does not monkeypatch process-global cache
paths. Missing or incompatible assets fail with an actionable exception. Do not
edit `versions.json` merely to silence a dependency mismatch.

The source digest covers all Python files and chemical-feature CSV files under
the GlassPy package, in sorted relative-path order, with NUL separators before
and after each file body. It must equal
`fcaff450cef398c4955ee36d00591e27bb750de904ece37c4145e495ac4a599d`.
Changing the supported source requires a new audit and adapter update.

| Asset under `models/` | Required SHA256 |
|---|---|
| `GlassNetMH.pth` | `3a5d19fe43e6d97350b5f65151b2705c6cfe49c111cb45cf28933eeea2ee9a0c` |
| `st-nn/Tg.pth` | `1c9f881c6dc0eed4a3767f4c169edfb0abc53033e84b1e4f1a7f35d2976a41d4` |
| `GlassNetMH_lc.p` | `dd89a5f2dbed5c5d9144a83023385f3f5f697772ca2d822c5825685032c452e7` |
| `st-nn/Tg_lc.p` | `22d7146b6da3f53e79db1dc3ce4e3f92be6283603a911ab99fac441f09ba1134` |

Stock construction also loads learning-curve pickles, so these are checked even
though they do not determine inference. Only trusted prepared artifacts should
be loaded. Scalers are derived artifacts rather than an author-distributed fixed
scaler release. Their actual SHA256 hashes and dependency versions are included
in `identity`; the adapter records whether they match the previously audited
scalers but permits independently prepared compatible scalers. A benchmark must
freeze and compare the complete identity across runs, including both scaler
hashes, rather than checking only the neural weights or the GlassPy version.
The previously audited scaler hashes are:

- x: `da48e50966b684175e8b0d9b3e010efed4c71c914ea44834fcbee8faa526614e`
- y: `e5178856e031ff8215743d6220b464cb2bfd33b77d44779dbea543e47b527d34`

The adapter checks assets again immediately before its first model load, so an
identity recorded earlier cannot silently describe different loaded scalers or
a changed cache directory. Input basis and batching/cache settings are immutable
for the instance lifetime. After loading, the model and scalers stay in memory;
editing files on disk does not hot-reload the oracle or invalidate its existing
predictions. Start a new instance to use new assets and compare its identity.

The earlier scaler preparation used the official split and preprocessing with
an independently checked equivalent row-filter optimization; it did not train
neural weights. This adapter does not prepare or alter those caches.

Batch size, numerical library versions, hardware and thread settings belong in
the run manifest. Float32 neural inference can vary slightly with batch shape
and numerical backend. Compare predictions using a declared tolerance; do not
promise bitwise identity across every platform or batching schedule. Decide how
to treat values within that tolerance of scientific feasibility thresholds.

## Cache and accounting

The per-instance LRU holds at most 4096 named scalar records by default. Generic
runtime settings may supply a different `cache_size` and `batch_size`; they do
not need additional glass-specific scientific flags. A model loads once per
instance. Calls are serialized within that instance; separate worker processes
have separate models and caches. Reuse one oracle for all three properties.

Batch calls coalesce repeated missing candidates, preserve request order and
process at most `batch_size` request rows at a time. A failed prediction batch is
not cached as a penalty or a partial success. Previously completed chunks remain
cached. Caller input/output storage is proportional to the requested batch, not
the entire design space. Returned dictionaries cannot mutate the cache.

`cache_info()` distinguishes requests, existing-cache hits, coalesced duplicates,
model-batch attempts and model-candidate attempts. These are work counters, not
the training unique-candidate ledger. For example, an evicted candidate can incur
another real model evaluation without becoming a new unique training candidate.
Failures consume work and time even when they do not consume a successful-query
budget. Three outputs from one candidate are not three candidate queries.

## Software verification

Run `python -m pytest tests/test_glassnet.py -q` for the adapter contract tests.
They use artificial inputs and outputs to check units, canonicalization, cache
behavior, asset validation and named outputs. They do not establish material
prediction accuracy. The real-model test is opt-in with
`RL_MATDESIGN_GLASSNET_REAL=1` and requires prepared, pinned GlassPy assets.

Measure cold-start latency, steady-state single-candidate latency and batch
throughput separately on the target machine. Preserve the oracle identity,
numerical environment and batch/cache settings with local measurements. This
release does not prescribe research compositions or publish local-run results.

## Local research protocols

Choose composition bounds, grids, objectives and feasibility gates in local
experiment configurations. They are separate from the adapter contract and are
not distributed here. Use [the benchmark runner guide](glassnet_benchmark.md)
for budget accounting and comparisons; software tests do not establish physical
validity or accuracy over a proposed research domain.
