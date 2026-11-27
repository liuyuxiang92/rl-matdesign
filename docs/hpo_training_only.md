# Training-only HPO

Use `--stage training` to run range diagnosis, training-budget evaluation, and
hyperparameter search, including validation of the selected training parameters.
The campaign stops after writing `best_training.yaml` and `frozen_models.json`.
It does not run candidate generation.

```bash
rl-matdesign hpo validate --hpo-config experiment.yaml --stage training
rl-matdesign hpo run --hpo-config experiment.yaml --out runs/experiment --stage training
```

The `generation` section can be omitted from a training-only HPO YAML. Neither
generation budgets nor generation sampling parameters need to be authored.
`plan --stage training` reports only training budget schedules and stages.
An unknown exact design-space size does not prevent an absolute-budget training
campaign merely because the unused generation defaults use fractional budgets.

To keep the training candidate budget fixed while tuning learner parameters,
set all three budget values to the same count:

```yaml
training:
  budget:
    mode: absolute_unique
    min_unique: 10000
    initial_unique: 10000
    max_unique: 10000
  search:
    max_rounds: 1
```

These counts apply per training job, including its DQN random warmup. Search
trials, range probes, and repeated/validation seeds each have their own budget.
A singleton budget grid cannot expand the count. Runtime feasibility and
sampling-efficiency checks still apply.

`resume` and `restart` retain the saved training-only stage scope. They recompute
the relevant domain information and preserve the existing manifest checks.
The default `--stage all` continues to include generation. The existing
`--stage train-search` selects only that individual stage and still requires its
earlier stages to have completed; use `--stage training` for a fresh campaign.
