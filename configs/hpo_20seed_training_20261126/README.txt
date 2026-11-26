NINE FULL HPO CONFIGURATIONS — DIRECT CLI RUNS
Prepared 26 November 2026

Each config.yaml is a complete standalone experiment config: composition domain,
property definitions, physical gates, reward, learner settings and training budget.
Each config runs 20 paired seeds. These are full configs, not best_training.yaml
selection summaries. No separate campaign launcher is required.

FILES

Condition       A2C                          DQN-MC                       DQN-bootstrap
baseline        baseline/a2c/config.yaml     baseline/dqn_mc/config.yaml  baseline/dqn_bootstrap/config.yaml
step1           step1/a2c/config.yaml        step1/dqn_mc/config.yaml     step1/dqn_bootstrap/config.yaml
step1_step2     step1_step2/a2c/config.yaml  step1_step2/dqn_mc/config.yaml  step1_step2/dqn_bootstrap/config.yaml

baseline uses value_normalization: none and gradient_clipping: none.
step1 uses value_normalization: popart and gradient_clipping: none.
step1_step2 uses value_normalization: popart and gradient_clipping: clipping.
All configurations set exploration_adaptive: false and preserve their historical
HPO-selected exploration settings. Each condition retains its own best HPO
learner settings; comparisons are between independently tuned configurations.

RUN DIRECTLY

Use the prepared framework with built-in paired seed-list and parallel_jobs
support. This support was added after commit 8a9b1304963008739c1458fe9613ffdbf9664281;
that published base alone cannot run these lists. The rl-matdesign installation
must use the updated implementation. The same GlassPy/GlassNet environment and
pretrained assets used by the HPO are required; model weights are not bundled.

Install the UPDATED framework checkout into the training environment first:

  python -m pip install -e .

Run that installation command from the updated framework directory. An editable
installation pointing at another checkout will keep using that other source.
The checkout-local script below works without changing the installed entry point.

From the updated framework checkout, validate one full config without loading predictors:

  rl-matdesign run --config configs/hpo_20seed_training_20261126/baseline/a2c/config.yaml --validate-config

Run its 20 seeds, with at most 4 runs executing simultaneously:

  rl-matdesign run --config configs/hpo_20seed_training_20261126/baseline/a2c/config.yaml --out runs/hpo_20seed_training_20261126/baseline/a2c --skip-generation --save-checkpoint-freq 1

Replace baseline/a2c with any of the other eight condition/method pairs. For an
extracted ZIP, pass its full config path. Each YAML has parallel_jobs: 4 near the
top; edit that integer for your machine. It is a per-command concurrency limit.
num_gen_eps: 0 also disables generation in the config itself.

The checkout-local entry point can be used with the same arguments:

  python scripts/run_experiment.py --config configs/hpo_20seed_training_20261126/baseline/a2c/config.yaml --out runs/hpo_20seed_training_20261126/baseline/a2c --skip-generation --save-checkpoint-freq 1

SEEDS AND OUTPUTS

train_seed and gen_seed each contain the 20 values 2026112601 through 2026112620.
They pair by list index: the first run uses the first training and generation
seeds, the second uses the second pair, and so on. This is 20 runs per config,
not a Cartesian product. Scalar eval_seed: 0 is shared by all 20 runs to retain the
same scientific evaluation realization used for the T3 comparison. The same 20
training seeds pair across all nine configs, for 180 total runs.

The parent output directory contains seed_runs.json. Each child directory has
a scalar config.yaml, run.log and the normal training artifacts, for example:

  <out>/run_001_eval_0_train_2026112601_gen_2026112601/
  <out>/run_002_eval_0_train_2026112602_gen_2026112602/

SCIENTIFIC SETTINGS

Training budgets are 916 distinct candidates for A2C, 458 for DQN-bootstrap and 115
for DQN-MC. DQN random warmup is included in that budget and keeps the historical
count for each selected HPO winner. PopArt or clipping's 100-candidate calibration
pilot is separate overhead. Strict HPO attempt/unique accounting is preserved.

All configs retain the same seven-component, 2%-grid GlassNet domain and gates:
  0 <= forming temperature T3 <= 2000 K
  0 <= thermal expansion CTEbelowTg <= 4.9e-6 K^-1
  glass transition Tg ≥ 838.15 K
The baseline retains forming_temperature.scale: 1000, giving a passing scalar
reward of -T3/1000. PopArt configurations retain their calibrated reward recipe.
Physical T3 remains in kelvin for reporting and comparison.

provenance.json records source-template and historical-selection hashes, the
selected HPO settings and exact resolved exploration schedules. All scientific,
learner and budget fields match the previously verified full templates. Only
the training/generation seed lists and parallel_jobs differ. The preserved
algorithm-specific defaults can remain in the full YAML; inactive exploration
controls and entropy targets are omitted.

Preparing and validating this set does not start scientific training.
