# Code layout and compatibility

## Entry points

- `rl-matdesign --config CONFIG --method METHOD --out DIR`: installed command.
- `python scripts/run_experiment.py ...`: source-checkout command, with the same flags.
- `rl_matdesign.configuration`: YAML aliases, predictor/filter construction, and
  `build_env`, shared by RL, BO, GA, and the invariance checker.
- `rl_matdesign.experiment`: argument parsing and experiment orchestration.
- `scripts/baselines/run_bo.py` and `run_ga.py`: baseline commands.

The source script re-exports its existing config helpers for scripts importing
`run_experiment`. The installed command uses package modules directly; it no
longer assumes a sibling `scripts/` directory exists in a wheel installation.

## Responsibilities

| Location | Responsibility |
|---|---|
| `src/rl_matdesign/env*.py` | Flat composition, integer-ratio, and multi-group environments |
| `src/rl_matdesign/training.py` | DQN, REINFORCE, A2C, checkpoint handling, candidate generation |
| `src/rl_matdesign/model.py` | Existing model architectures and state-dict names |
| `src/rl_matdesign/registry.py` | Predictor, builder, and constraint plugin resolution |
| `src/rl_matdesign/predictors/` | Reward evaluation and structure builders |
| `src/rl_matdesign/utils/composition.py` | Canonical numeric/categorical composition keys |
| `src/abcde_ooh/` | OOH predictor/environment support and compatibility imports |
| `scripts/` | CLI tools and analysis workflows |
| `tests/` | Regression tests and subprocess smoke runs |

`abcde_ooh` remains in place: the generic OOH predictor still depends on its
DeepMD implementation and constraints. Its encoding functions delegate to the
identical shared implementation in `rl_matdesign.encoding`.

Structure-score cache keys preserve full float precision. Baseline caches and
predictor cost accounting keep their existing six-decimal rounding. Categorical
labels and group names remain part of the key.

## Compatibility boundary

The cleanup preserves the existing feature branch's algorithms, reward formulas,
RNG order, YAML aliases, CLI flags, model tensor names/shapes, scaler files, and
output schemas. It does not change scientific settings to improve test results.

Before this cleanup, `feat/multi-sublattice-lips` had already adopted `groups:`
and removed several older predictor adapters. Configs or commands from much
older branches (including `run_ABCDEOOH_experiment.py`) are not automatically
restored by this cleanup. Audit those separately before merging to `main`.

Private scenario configs, checkpoints, datasets, results, and untracked analysis
scripts in the original checkout are retained. The cleanup is developed in an
isolated worktree, based on feature-branch commit `08bb24a`.

## Validation

```bash
python -m pytest
python scripts/run_experiment.py --config configs/test_dummy.yaml --method a2c --out /tmp/rl-smoke
python scripts/run_experiment.py --config configs/test_multigroup_dummy.yaml --method dqn --out /tmp/rl-multi-smoke
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/rl-wheels .
```

The full suite needs the scientific dependencies used by the tested features,
including ASE and SMACT. Dummy-predictor smoke runs do not require real DeepMD
models. Optional tests using a local `perovskite.vasp` skip when it is absent;
synthetic unit-cell and supercell tests always cover the same substitution rules.

For numerical compatibility, compare fixed-seed CPU runs before and after edits:
generated CSV bytes, every saved model tensor, scaler arrays, non-timing training
log fields, generation from old checkpoints, and resumed training. Cover flat
and multi-group configs with DQN bootstrap, DQN MC, REINFORCE, and A2C. Real
DeepMD/MGTransformer model runs require separate validation with their original
external files and runtime.
