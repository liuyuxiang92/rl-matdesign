# DQN/HPO performance v2 validation — 2026-11-15

This update continues intact, cooperatively stopped jobs from `ba6bc1cf` and
`b947208`, including each release's exact six historical predictor-file variant.
It does not repair disk-full or incomplete checkpoint/accounting state.

## Changes and invariants

- Bootstrap computes legal next-action maxima in bounded batches, with duplicate
  next states shared within a minibatch. Target equations, reward/gamma handling,
  replay sampling, update counts and synchronization schedules are retained.
  Matrix shapes change; old/new floating-point trajectories are **not** promised
  bitwise equal. No cross-update transient cache affects resumed chunk shapes.
- HPO keeps its checkpoint cadence. Growing containers use content-defined chunks
  in `checkpoint_data.sqlite3`; commit blobs, publish/fsync `.pt`, then reclaim
  stale records. The loader verifies referenced content and restores original
  container values. Ordinary non-HPO checkpoints remain standalone.
- Validation contains only the current round's three best distinct configurations.
  Completed old experiments stay cached, but do not enlarge the new panel.
- At a settled sampling wave, stop when even perfect remaining samples cannot meet
  the strict final attempts/budget limit. Paid results remain recorded, with no
  subsequent gradient step or forced completion of a partial A2C batch.
- Unchanged ledger flushes avoid duplicate writes; changed state remains durable.

## Native continuation coverage

The integration fixtures create actual jobs with archived release source, send
SIGTERM to the controller, and run the new CLI without editing checkpoints,
manifests or saved hashes. Both DQN methods cover both releases, clean and with
all six byte-exact historical files. A2C covers the same four source variants.

Checks include read-only preflight, retained paid prefixes, complete optimizer and
RNG state, terminal candidate/accounting equivalence, and cooperative stopping a
second time after upgrade. Bootstrap compares continuous and interrupted **new**
continuations from the same old checkpoint; it does not incorrectly demand old
scalar-target and new batched-target bit parity. MC retains the old continuous
control comparison.

The compatibility inventory also verifies an actual target source copy containing
those six files. Unknown source, changed scientific configuration/assets/runtime,
and damaged backing records remain rejected.

## Test results

- Full repository run: 2,596 passed, six skipped, one failed because a standalone
  configuration-test subprocess lacked this checkout's `PYTHONPATH`. Re-running
  that test with the correct source path passed; no product source change was
  needed. The six skips require absent GlassPy/structure/model assets or CUDA.
- Final review added the backing database to completed-result artifact hashes.
  On the final sealed source, 213 affected controller/runner/checkpoint/restart/
  efficiency/config tests passed, plus 23 store tests and 122 compatibility tests.
- Native old-to-new CLI matrix on the final sealed source: **12 passed** — three
  algorithms × two historical releases × clean/historical-extra-files source.
  Each bootstrap/A2C case includes an additional stop/restart under the new code.
- A fresh wheel was built and installed outside the checkout. Its source hashes
  match the final sealed inventory, all four historical origin variants are
  accepted, and its installed CLI validates the dummy bootstrap HPO configuration.
- `git diff --check` passed. Ruff was unavailable in the local interpreter; no
  lint success is claimed. No physical GlassNet/H20 training benchmark was run.

## Local storage benchmark

Synthetic state: 20,000 replay rows plus 60,000 historical seen records. Perform
129 saves, each evicting/adding seven replay rows and adding history, crossing two
garbage-collection cycles. Both implementations include serialization and atomic
publication. These are local CPU/storage measurements, **not** GlassNet/H20/NAS
end-to-end speedup claims.

| Measurement | Incremental store | v1 full packed checkpoint |
|---|---:|---:|
| First save | 0.808 s | 0.579 s |
| Subsequent median save | 0.357 s | 0.568 s |
| Initial total size | 22.42 MB | 22.94 MB |
| Total after 129 saves | 29.36 MB | 22.94 MB |
| One load | 0.480 s | 0.149 s |

Typical incremental saves added approximately 119 KB backing data and wrote a
21 KB `.pt`, versus rewriting approximately 23 MB. SQLite reused freed pages after
the first GC cycle. The trade-offs are initial construction cost, retained slack
between GC cycles, and a slower one-time restart load. Copy whole job directories,
not just `.pt` files. Total campaign storage still includes all observations and
all jobs; this change is not a global output-retention policy.

## Operational verification

Stop the original controller once and wait for workers to exit. Keep the same
YAML, model assets, environment and output directory, update to the reviewed
release, then run:

```bash
rl-matdesign hpo restart --out ./runs/dqn_bootstrap_medium --check-only
rl-matdesign hpo restart --out ./runs/dqn_bootstrap_medium
```

Use the corresponding original directory for MC/A2C. No additional YAML control
is needed. Actual restart writes a separate
`source_upgrades/dqn-batched-targets-persistence-v2.json` receipt and preserves any
previous v1 receipt. Read-only preflight writes nothing.
