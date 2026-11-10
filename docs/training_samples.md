# Training material samples

Every training run through `scripts/run_experiment.py` writes
`<out>/training_samples.csv` automatically. Each row describes one terminal
material whose episode consumed the configured training budget. It uses the
same material/property column names as `generated.csv`, so training and
generation distributions can be compared directly.

```bash
python scripts/run_experiment.py \
  --config /path/to/perovskite.yaml --method dqn --out runs/perovskite
```

## Which episodes are saved

| Training method | Saved phases | Budget |
| --- | --- | --- |
| DQN | `dqn_warmup`, `dqn_train` | Random warmup and online training episodes |
| REINFORCE, A2C | `pg_episode` | Policy training episodes |

The export follows the existing `budget_counted` decision in `training_log.csv`:

- With `train_count_duplicates: true` (the default), repeated materials are
  retained as separate rows when their episodes consume budget.
- With `train_count_duplicates: false` or `--no-train-count-duplicates`, only
  the first budgeted occurrence is saved. Repeated episodes can still update
  the model and remain visible in `training_log.csv`.
- Failed evaluations that are retried without consuming training budget are
  absent. A successfully evaluated material that fails a reward gate is still
  saved when its episode consumes budget.
- PG scaler warmup, replay-buffer reuse, DQN permutation augmentation,
  per-iteration metric summaries, and generation are not training-budget
  episodes and do not add rows.

Rows are written as episodes complete; an interrupted run retains samples
already written. `--only-generate` leaves `training_samples.csv` unchanged.
`--resume-training` continues the export from the resumed training state.
If the CSV contains samples newer than the restored checkpoint, the original
file is preserved as `training_samples.superseded-<id>.csv` and the active
file follows the restored checkpoint's history. A fresh training run in an
existing output directory also preserves the previous file this way.

The columns `training_session_id` and `training_samples_history_complete`
make resume history explicit. Loading weights without a full training state
starts a separate session with incomplete history. Checkpoints created before
sample export was added cannot reconstruct missing property predictions;
their newly exported samples are also marked as incomplete. Compare samples
within the intended session and inspect these flags before treating a CSV as
the full training history.

## Columns and property meaning

Common material columns include `formula`, `reward`, `dp_mean`, `dp_std`, and
`dp_mean_minus_std` when those quantities are defined for the predictor.
Multi-property scenarios instead expose `obj_<name>_mean` and
`obj_<name>_std`. Applicable gate and phase columns retain the names used by
`generated.csv`, such as `gate_passed`, `gate_<name>_value`, `primary_ok`, and
`primary_label`.

Training metadata includes `phase`, `episode`, `budget_counted`, and
`training_budget_used`. The budget position is the cumulative count across
the training run; phase-specific episode numbers are not always globally
unique.

`reward` is the terminal training reward. It can include objective transforms,
uncertainty terms, or gate penalties, so it is not necessarily a physical
property. For perovskite delta comparisons, use the actual property columns,
for example `obj_delta_T400_mean`, `obj_delta_T500_mean`, and
`obj_delta_T600_mean`, in both CSV files. A failed synthesis gate can replace
the composite reward or `dp_mean` with a penalty while these property columns
still contain the predicted delta values. Compare gate pass rates separately
or apply the same gate filter to both populations.

Sample export reuses the prediction made for the episode; it does not launch
an additional structure realization or property inference. Built-in structure
and OOH predictors provide a read-only prediction-record hook. A custom
predict-only predictor uses its returned mean/std. If a custom predictor has
a separate `predict_raw` method but no read-only `prediction_record` hook,
the export retains `predict_mean`/`predict_std` and records
`prediction_record_error` instead of running `predict_raw` again or labeling
its reward-space output as raw properties.

## Comparing distributions

```python
import pandas as pd

training = pd.read_csv("runs/perovskite/training_samples.csv")
generated = pd.read_csv("runs/perovskite/generated.csv")
columns = ["obj_delta_T400_mean", "obj_delta_T500_mean", "obj_delta_T600_mean"]

summary = pd.concat(
    {"training": training[columns].describe(),
     "generated": generated[columns].describe()},
    names=["phase", "statistic"],
)
print(summary)
```

Training rows represent budget expenditure and can include repeats;
`generated.csv` normally contains deduplicated accepted candidates. State
whether a comparison uses all budgeted episodes or unique training materials,
and keep the same property definitions and filters on both sides.
