# Withdraw optional-primary extension and revalidate — 2026-10-15

The user rejected adding a scenario-specific presence toggle during cleanup.
This follow-up withdraws that extension from `27af53ef` and retains only the
complete dopant-accounting fix in the OOH core source.

## Final scope

- `phase_sampler.py` and `registry.py` are byte-identical to cleanup commit
  `1231b055`; the presence parameter and optional-sampling branch are removed.
- `primary_phase.py` retains its tested correction: all cations outside the
  phase's primary set contribute to the dopant sum. No new parameter is required.
- Core phase tests use inline inputs and verify named-primary requirements,
  dopant/Fe/ratio bounds and full public rollouts for all six target settings.
- Active configuration documentation describes strict sampling. The October 14
  validation is marked historical, preserving its original evidence.
- Training, networks, configuration construction and runner code are unchanged
  from `1231b055`. The two original unpushed feature commits remain inherited.

This preserves the legacy distinction between the action sampler and the
standalone terminal checker; it does not redesign or unify their presence rules.
The original feature checkout and historical analysis artifacts remain intact.

## Fresh validation

Using `/opt/anaconda3/bin/python` in the cleanup worktree:

```bash
python -m pytest --junitxml=/private/tmp/cleanup-revert-full-20261015.xml
git diff --check
```

**364 passed, 4 skipped**, one existing SMACT deprecation warning. This includes
all **16** retained OOH checker/sampler cases. The four skips require two optional
RF model files and the optional `perovskite.vasp`; synthetic builder tests pass.
No real DeepMD/MGTransformer long experiment was repeated.

## Entry-point, checkpoint and main-config audit

A temporary export of `main` commit `39227da88a3450c126dfa52e094e308eb27fbb79`
trained a small CPU dummy-predictor DQN run using its original config schema.
It used 4 warmup and 4 training episodes, hidden dimension 128, batch size 4,
one gradient step per episode, predictor/train/generation seeds 11/12/13,
and a saved checkpoint every 2 episodes.

The same composition space was explicitly expressed with the feature branch's
`groups:` schema for the cleanup-version checks:

- The package CLI loaded the old `qnet.pt` and scaler with `--only-generate`
  and produced a nonempty generated CSV.
- The script CLI resumed a copy of the old full checkpoint at episode 4 and
  completed the configured 6 training episodes, then generated candidates.
- All files in the original baseline run retained their SHA256 hashes.

These are checkpoint/entry-point smoke checks, not a claim of identical
scientific results across the older main and feature algorithms.

**Remaining main-merge compatibility issue:** the unmodified old-main
`configs/test_dummy.yaml` is rejected with `Config has no 'groups:'`. The new
schema requirement already existed before this OOH integration. Old configurations
must be migrated or a separately designed compatibility path must be provided
before claiming old-main commands work unchanged. This task does not merge main.

Machine-readable counts, skips, hashes and smoke commands are in
[ooh_phase_revert_20261015.json](ooh_phase_revert_20261015.json).
Temporary reproduction driver/logs are `/private/tmp/verify_cleanup_compatibility_20261015.py`
and `/private/tmp/cleanup-compatibility-20261015/`.
