# Named fields and external scores

Experimental conditions use the existing `categorical` group. A choice with a
`name` and no `element` is a generic field. No new group kind, trainer, or
top-level parameter is required:

```yaml
groups:
  - name: conditions
    kind: categorical
    choices:
      - name: temperature
        values: {min: 100, max: 200, step: 5}
      - name: solvent
        values: [water, ethanol]
```

Values can be an explicit nonempty list or an inclusive numeric grid. The maximum
must lie on the grid. Each episode fills each field once in its declared order.
That order is a construction convention; the resulting candidate is a mapping,
for example `{"conditions": {"temperature": 150, "solvent": "water"}}`.
Mapping order does not affect its identity or cache key. Sequence groups retain
their position semantics.

The environment represents selected field values explicitly instead of treating
field names as chemical elements. Numeric action codes are normalized to the
field's range; the predictor receives original values. String choices use their
declared indices as codes. This reuses the existing scalar action input, so it
does not guarantee that the learned policy is invariant to a reordering of
category labels or configuration fields. Keep the schema fixed within a run.
`design_schema.json` beside checkpoints guards against reloading incompatible
field definitions. Set the existing `dqn_augment_permutations` to `0` for fields.

## External prediction

Use the existing fully qualified predictor interface under `properties`:

```yaml
properties:
  - name: q
    predictor: my_package.api:ConditionPredictor
    model: path/to/model
    direction: max
    objective: mean
```

The class implements `__init__(self, cfg, *, seed=None)` and
`predict(self, candidate) -> (mean, std)`. Read field values by their names, call
the external model, and return its raw prediction. Return `0.0` for `std` if the
model has no uncertainty estimate. Existing reward direction, scaling, and
uncertainty objectives remain available. The same configuration works with
`--method dqn`, `--method reinforce`, or `--method a2c`.

For a fixed Cartesian score table, the supplied
`rl_matdesign.predictors.lookup:LookupPredictor` loads an NPZ specified by
`model`. The archive contains `field_names`, one `axis_i` array per field, and
`mean` with the corresponding Cartesian shape. Optional `std` is a scalar or
array of the same shape; omitted `std` becomes zero. Lookup requires an exact
grid match and does not interpolate or train on the table. The RL trainer only
receives scores for candidates it evaluates.

## ActMOF example through the normal runner

From the repository root:

```bash
python scripts/prepare_actmof_lookup.py --download --out data/actmof
python scripts/run_experiment.py --config configs/actmof.yaml \
  --method dqn --out runs/actmof_dqn_seed0 \
  --dp-seed 0 --train-seed 0 --gen-seed 0
```

Alternatively, preparation accepts `--source /path/to/ActMOF` instead of
`--download`. It verifies the seven published data files against a fixed
[ActMOF source revision](https://github.com/deep-synthesis/ActMOF/tree/f1733f05bf8f18a585d50088b8aae3f86a2c5078),
checks grid completeness and uniqueness, and writes `lookup.npz`, a provenance
`manifest.json`, and the source `LICENSE`. Scores are the authors' `q_final`;
no uncertainty is supplied. The five configuration values are the authors'
encoded coordinates, not physical units. This reproduces access to their
published score landscape, not the fitting of the landscape from experiments.

The example requests 200 generated candidates after training. Its training
settings are initial defaults, not tuned results. Use the existing YAML fields
to choose training and generation sizes. This is the normal RL workflow; it
does not impose the paper's 120-query batch protocol or run comparison methods.

## Output and later comparison

The output remains `generated.csv`, produced by the shared existing writer and
sorted by descending `reward`. For fields, `formula` contains the condition
mapping as JSON, and the existing `candidate_json` column retains that mapping.
Existing materials and peptide output behavior is unchanged.

With `configs/actmof.yaml`, `obj_q_mean` stores raw q and `obj_q_std` is zero.
The existing `properties` output convention puts the scaled combined score in
`dp_mean`; `reward` is `q / 100000`. Scaling improves numerical conditioning
without changing the ranking. The per-property standard deviation is in
`obj_q_std`, as for other `properties` configurations. Direct predictors without
`properties` continue to use `dp_mean` and `dp_std`.

Send `generated.csv` for later candidate-quality comparisons. Keep the existing
`run_config.json`, `training_log.csv`, and `timing.json` as well: generated rows
are sorted final candidates, not a chronological record of every training query.
The file alone cannot establish sample efficiency. Training evaluations and
generation evaluations both cost budget; repeated candidates may hit the cache.
Strict best-so-far comparisons by unique query require a separate complete query
history, which this change deliberately does not add.
