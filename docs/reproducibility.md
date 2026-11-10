# Reproducing and archiving experiments

A Git tag fixes tracked code. It does not preserve an uncommitted working tree,
model weights, a Conda environment, or ignored run directories. Paper evidence
must associate these inputs with the runs that actually produced the results.
The tools below capture that association without publishing or creating tags.

## New runs: append-only invocation evidence

The experiment runner writes a new `provenance/<UTC>-<unique-id>/manifest.json`
for every training, resume, or generation invocation. Reusing an output directory
does not overwrite previous invocation manifests. Generation-only execution also
preserves `training_log.csv`, `episode_timing.csv`, and `timing.json`, writing
`generation_log.csv`, `generation_episode_timing.csv`, and `generation_timing.json`
instead. Each record includes:

- The exact argument vector, effective configuration, configuration sources,
  runtime arguments, seeds, working directory and phase.
- Git commit, tracked dirty status, a binary patch against that commit, and the
  number of untracked files. Untracked files are never silently added.
- Python/platform and installed package versions. If Torch is already imported,
  its build CUDA version and deterministic settings are recorded without
  initializing CUDA. TF32/cuDNN settings and a capped, three-second-timeout
  `nvidia-smi` GPU/driver query are recorded when available. Only four environment
  variables are captured: `CUBLAS_WORKSPACE_CONFIG`, `CUDA_VISIBLE_DEVICES`,
  `OMP_NUM_THREADS`, and `PYTHONHASHSEED`; no general environment dump is made.
- SHA256 hashes and sizes for known configuration file references and explicitly
  supplied input files. The original YAML is copied when available.

Most built-in file references (including RF/DeepMD models and templates) are
resolved against the application's **runtime working directory**, not the YAML
directory. MGTransformer is an explicit exception: its `mgt_repo` resolves against
the runtime directory, and relative `model`/`models` and optional `config` paths
resolve against `mgt_repo`, where its server runs. Absolute paths are preserved.
Invocation records include the applied path rule and resolved input paths.

Unknown plugins own their path semantics. Relative file-valued settings under an
unknown predictor are recorded as unresolved instead of being incorrectly hashed
from the current directory. Declare their actual resolved files via `input_files`
or `--asset`, and retain the plugin's own configuration/provenance needed to
explain that mapping. Implicit caches, downloaded pretrained weights, and unknown
file-valued keys also require explicit declaration. Missing or unresolved
references remain visible; they are not silently treated as valid inputs.

`write_run_manifest` is also available to custom runners:

```python
from rl_matdesign.utils.provenance import write_run_manifest

manifest = write_run_manifest(
    output_dir, effective_config,
    argv=argv, config_path=config_path,
    seeds={"eval": eval_seed, "train": train_seed, "gen": gen_seed},
    phase="train", runtime_args=vars(args),
    config_sources=configuration_sources,
    input_files=[model_path, template_path],
)
```

It uses OS randomness for its unique directory name and does not consume the
Python, NumPy, or Torch random streams used by an experiment.

## Bundle an actual run on the machine containing its assets

Stop writers to the selected run directories first. Create environment records
inside the run directory (or provide them separately with `--asset`):

```bash
conda env export > runs/paper_run/environment.yml
conda list --explicit > runs/paper_run/conda-explicit.txt
python -m pip freeze > runs/paper_run/pip-freeze.txt
nvidia-smi > runs/paper_run/nvidia-smi.txt
```

The last command applies to an NVIDIA GPU machine. Inspect exported environment
files before sharing: package source URLs or local paths may need redaction.
Package versions alone do not guarantee that a fresh environment can be rebuilt;
validate the environment export on a clean installation.

From the repository and original runtime working directory:

```bash
python scripts/archive_experiment.py create \
  --run-dir runs/paper_run \
  --config configs/paper.yaml \
  --asset /absolute/path/to/model.pt \
  --asset /absolute/path/to/POSCAR \
  --out /absolute/path/to/archives/paper-ooh-v1

python scripts/archive_experiment.py verify \
  /absolute/path/to/archives/paper-ooh-v1
```

Repeat `--run-dir`, `--config` and `--asset` as needed. Assets are explicit files;
run directories are copied recursively. The output must be a new directory
outside all selected runs. Symlinks inside copied run/source trees are skipped
and reported, so an unrelated external directory is never recursively copied.
An explicitly supplied asset file may be a symlink: its target's bytes are copied.

The bundle contains:

| Path | Meaning |
| --- | --- |
| `bundle.json` | Run/asset path mapping, evidence gaps and source provenance |
| `checksums.json` | SHA256 for every archived file except this checksum manifest |
| `runs/` | Copies of selected results, including all prior invocation evidence |
| `assets/` | Input/model/config files with recorded original paths |
| `source/head.tar` | Tracked HEAD at packaging time |
| `source/working_tree/` | Tracked files as they existed at packaging time |
| `source/git.patch` | Tracked changes against packaging HEAD |
| `source/historical/<commit>.tar` | Recorded run revisions available in local Git |

The source snapshot does not automatically include ignored or untracked scripts.
Required untracked files must be reviewed and explicitly archived; the report
marks their omission as unresolved. A historical dirty revision can be rebuilt
from its archived commit plus the `git.patch` stored with that invocation.

**A legacy run with no invocation record does not acquire the current checkout's
commit as its historical version.** The bundle reports its original version as
unknown. Its `run_config.json` can help identify assets, but relative paths must
be resolved against the original working directory, which legacy runs may not
have recorded. Additional explicit assets can preserve the bytes; recovering the
actual historical source version still requires contemporaneous evidence.

An explicit environment export/lock file must be present at each run's top level
or passed with `--asset`. Recognized names include `environment.yml`,
`conda-explicit.txt`, `pip-freeze.txt`, `conda-lock.yml`, `requirements-lock.txt`,
`uv.lock`, and `poetry.lock`. Installed-package metadata alone does not satisfy
this requirement. An export's presence does not prove it can rebuild the run;
`reproduction_validated` stays false until a separate reproduction is performed.

The CLI returns `0` only when checksums pass and no recorded evidence gaps remain.
It returns `1` for an intact but incomplete bundle (such as an old run with an
unknown source revision), as well as for checksum failures. It returns `2` for
invalid arguments or unusable inputs. An incomplete bundle is still written so
that recoverable evidence is preserved and its deficiencies can be inspected.

Checksums detect accidental file changes; they are not signatures or proof of
scientific reproducibility. The `complete` field means no known collection gaps,
not that implicit plugin assets, all scientific assumptions, or GPU determinism
have been independently validated. File integrity and evidence completeness are
reported separately.

## Reproduce before assigning the paper release

Use a new directory and recreate the recorded environment. Extract the recorded
source revision and apply its recorded patch if necessary. Use `bundle.json` to
map original model/template paths to copied assets; configurations are preserved
verbatim rather than rewritten silently.

Check these separately:

1. Fixed evaluation seed and candidate produce the same sampled structures and
   predictions. Evaluation order must not change per-candidate samples.
2. Fixed training and generation seeds reproduce the trained parameters and
   generated candidates under the supported software/hardware environment.
3. Archived raw results reproduce the paper's figures using archived analysis
   scripts. Save those scripts and the figure-to-run mapping as explicit assets.

Record comparison tolerances and hardware/software limits. Bitwise equality
across different GPUs, library versions or nondeterministic kernels is not
promised by saving seeds. Reproducibility from a fresh start and reproducibility
of checkpoint continuation are separate tests; checkpoint files alone do not
prove that all random stream states were restored.

After validating the actual paper results, create an annotated tag and release
for that source revision. Protect the published tag from later changes and
archive the evidence bundle in a durable repository. A historical release can
remain reproducible while the main branch
adopts the common state/action/evaluation design. This workflow does not add an
OOH-specific compatibility path to the new implementation.
