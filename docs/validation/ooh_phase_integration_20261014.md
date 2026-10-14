# OOH phase fixes integrated into the cleanup branch — 2026-10-14

`codex/cleanup-integration` already inherited both local feature commits
`9c918608` and `08bb24ae` through its cleanup commit `1231b055`.
No duplicate cherry-pick was needed. This update ports the three OOH core-file
changes from the original feature checkout without changing their contents.

## Behavior and scope

- `require_primary_presence` defaults to `true`, preserving strict sampling.
  Setting it to `false` permits absent named primary elements while retaining
  five positive cations, the total composition budget and all phase limits.
- The terminal phase checker counts all outside-primary cations as dopants,
  including species absent from the historical fixed list.
- Portable core tests replace the analysis-dependent phase test setup with inline
  species/grids and a stub featurizer. They cover registry configuration,
  missing-primary prefixes, rejection boundaries and public five-step rollouts.
- No analysis scripts, historical results, private configuration, local links or
  unrelated budget-sweep changes are included. The original checkout is retained.

## Validation

Run from a clean tracked-source worktree using `/opt/anaconda3/bin/python`:

```bash
python -m pytest tests/test_ooh_primary_phase.py tests/test_ooh_phase_sampler.py
python -m pytest
git diff --check
```

- Targeted tests: **24 passed**.
- Full suite: **372 passed, 4 skipped**, one existing SMACT deprecation warning.
- Skips: two missing optional RF model files and two tests requiring the optional
  local `perovskite.vasp`. Synthetic builder tests remain in the passing suite.
- Independent review found no blocking issue. A temporary reviewer check covered
  44,016 terminal/phase combinations: default behavior matched the committed
  feature version; optional behavior matched independent integer rules. That
  auxiliary script was not retained; the portable regression tests are committed.
- The original feature checkout's HEAD, status and three core-file hashes were
  checked unchanged. Both feature commits remain ancestors of the cleanup branch.

Counts, skip reasons and code hashes are in
[ooh_phase_integration_20261014.json](ooh_phase_integration_20261014.json).
The September numerical-equivalence evidence remains historical; this update
runs the present core suite and does not repeat real-model long experiments or
validate migration of older `main` configuration schemas. No main merge or push.
