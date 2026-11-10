# Benchmark execution and reporting

The benchmark runner compares DQN-MC, DQN-bootstrap, A2C and uniform random
search using experiment configurations supplied locally. Research domains,
property gates, model paths and study-specific parameter values are not bundled
with this release. Keep one configuration per search-space scale and preserve
both DQN and A2C settings so the CLI can select the algorithm.

Use exact composition menus and bounds. Before comparing scales, verify their
terminal counts and nesting; report action branching and horizon as well as the
number of unique candidates. Candidate identity excludes construction-order
permutations. `count_compositions` and uniform terminal sampling use integer
dynamic programming instead of enumerating a dense property table.

Choose a shared successful-unique-candidate budget, matched seed sets, the same
scientific oracle protocol and separate output directories. Uniform random
search samples terminal compositions; uniform random actions generally induce
a different distribution. Record HPO resources separately from evaluation runs.

## Plan and execute

Supply local files; the following paths are placeholders:

```bash
python scripts/benchmark.py \
  --config /path/to/local/small.yaml \
  --config /path/to/local/medium.yaml \
  --config /path/to/local/large.yaml \
  --out runs/benchmark --budget 1000 --seeds 11 23 37
```

Without `--run`, this writes a plan without invoking a scientific predictor.
Add `--run` after preparing the environment and model assets. Each arm starts a
fresh process/cache and skips extra generation. The runner verifies the actual
completed unique budget. A query returning several properties is one candidate
query; repeated requests and failures still consume work/time and are recorded.

For GlassNet, see [its input/output and asset contract](glassnet_oracle.md).
The stock oracle runs on CPU independently of the RL network's selected device.
The model produces point estimates, not an uncertainty ensemble. Physical output
conversion is performed by the adapter; do not apply it again in a reward.

## Report completed runs

```bash
python scripts/summarize_benchmark.py runs/benchmark
```

The `summary/` directory contains:

- `curves.csv`: successful unique training observations and best-so-far reward;
  duplicate, diagnostic and generation queries do not improve this curve.
- `summary.json`: actual budget/completion checks, oracle work, optimizer updates,
  feasibility, best-found candidates and evidence issues.
- `aggregate.csv`: medians and observed ranges over completed seeds, with sample
  and excluded-arm counts. Observed ranges are not confidence intervals.

Actual algorithm, target mode, seed and budget must match the plan. Completion
records and the unique ledger must agree; successful process exit is insufficient.
Incomplete arms remain visible diagnostically but are excluded from aggregates.
Different scientific protocols are not pooled. Use recorded oracle identities
to compare source/model/scaler assets across machines.

Feasibility replays configured gates from named physical outputs, independently
of reward sign. Missing or unsupported evidence stays unknown; zero feasible
candidates yield no best-feasible value. Property gates do not prove experimental
material feasibility. A best-found reference is not a known global optimum.

Invocation wall time and cumulative prediction time have different scopes.
Do not add restored cumulative histories repeatedly or invent interruption time.
Missing historical timings remain missing. Archive the actual source, local
configurations, dependencies and assets with scientific results; distribute such
archives separately after checking what they contain.
