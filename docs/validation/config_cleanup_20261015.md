# Cleanup-branch configuration preparation — 2026-10-15

Prepared 29 configuration files in the cleanup worktree: 23 runtime scenarios
and 6 HPO definitions. Existing private configs were copied from the original
feature checkout, preserving the originals. The 13 private copies remain ignored;
only public usage documentation, ignore rules and this audit enter Git.

## Changes

- Removed the withdrawn `require_primary_presence` field from the OOH runtime
  copy. The cleanup branch uses the original strict sampler.
- Removed `env_type` from three Cs2AgBiCl6 scenario copies and `total_units`
  from their twelve composition groups. Environment type and grid units are
  already derived from `groups` and `fraction_set`; normalized groups agree.
- Updated two Cs2AgBiCl6 HPO copies to resolve `base_config` relative to their
  current `configs/hpo/` location, rather than a different machine's directory.
- Corrected HEA and perovskite usage comments to use existing `--dp-seed`,
  `--train-seed` and `--gen-seed` flags instead of unsupported `--seed`.
- Added ignore entries for the two private Cs2AgBiCl6 HPO files.

The parsed scientific settings and training budgets are unchanged except for
removing the specified inactive fields. Supported legacy aliases and parameters
for other methods are retained. Historical analysis configs
`ooh_design_space_29.yaml` and `ooh_design_space_34.yaml` stay in their original
analysis scope; they were not copied or rewritten as runtime scenarios.

## Validation and local persistence

- All 29 YAML files parse and differ from their sources only by the listed edits.
- All 23 runtime configs build an environment and complete one legal-action
  episode with a constant reward stub, including the terminal reward callback.
- All 6 HPO configs pass the existing HPO validator; their base paths resolve
  inside this configuration set and their search spaces validate.
- Every private copy is ignored by Git. Original config hashes are unchanged.
- No runtime source changed, so the previously passing full regression suite
  was not rerun for this config-only preparation.

Detailed checks and hashes: [config_cleanup_20261015.json](config_cleanup_20261015.json).
A complete ignored snapshot is retained in the **original repository checkout**
at `runs/cleanup_integration_configs_20261015/`, with manifest, validation record
and the verification driver. Private configs do not travel with a Git branch;
restore them separately at the same relative paths when using another checkout.

Model and structure paths were preserved, including any machine-specific paths
or placeholders. Real inference, relaxation, HPO trials and training were not run;
those resources still require verification before a production screen.
