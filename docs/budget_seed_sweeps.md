# Training budgets crossed with paired seeds

The ordinary `rl-matdesign run` CLI accepts a list of training budgets in YAML:

```yaml
parallel_jobs: 10
train_seed: [2026112601, 2026112602, 2026112603]
train_budget: [100, 200, 300]
num_gen_eps: 0
```

This example runs nine independent experiments: every budget with every seed.
Budgets are expanded first, then the paired seed rows. Multiple seed lists
(`eval_seed`, `train_seed`, `gen_seed`) still pair by position; scalar seeds
broadcast across those rows. A scalar `--train-budget` replaces the YAML budget
list, and scalar seed overrides retain their existing behavior.

```bash
rl-matdesign run --config config.yaml --validate-config
rl-matdesign run --config config.yaml --out runs/budget_sweep --skip-generation
```

`parallel_jobs` limits concurrent child processes across the whole grid.
Each child has a separate scalar config, log, model, and output directory under
`budget_<count>/`. `seed_runs.json` records every requested seed/budget pair and
its outcome. Existing seed-only directory names remain unchanged. Existing
results are never overwritten. To resume an individual run, use that child's
saved scalar config and output directory, as with seed-only sweeps.

The runs start independently. Smaller-budget outputs are not automatically
reused or resumed when running larger budgets. All selected learner parameters
remain fixed; warmup counts, epsilon schedules, target-update frequencies, and
duplicate-attempt guards are not rescaled implicitly.

When `train_count_duplicates: false`, the budget counts distinct training
candidates. DQN warmup candidates are included. A2C's scaler warmup and the
separate PopArt reward-calibration pilot retain their existing accounting;
the 100-candidate PopArt pilot is outside `train_budget`. A2C budget feasibility
does not depend on inactive DQN warmup settings.

Lists are an ordinary-run orchestration feature. Individual resolved runtime
configs and HPO trial configs still have scalar budgets.
