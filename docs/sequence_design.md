# Ordered sequence design and external GP models

For the inspected user-provided model and its actual 138-monomer catalog, see
[gp_physchem integration](gp_physchem_integration.md). The sections below describe
the reusable framework and the initial generic template.

The same DQN, REINFORCE and A2C runner now supports ordered token sequences in
addition to material compositions. `kind: sequence` is a generic group: tokens
can represent monomers, building blocks or process operations. Sequence length,
vocabulary, repetition and token-count bounds come from YAML. No peptide
chemistry, amino-acid names or GP implementation is hardcoded in the RL core.

## Run the synthetic integration tests

```bash
python -m pytest tests/test_sequence_integration.py -q
```

The tests create temporary configurations and run DQN, REINFORCE and A2C;
no local scenario YAML is required or committed. They use a
**synthetic random predictor**, a three-token vocabulary,
11 positions and two occurrences of token C. C is only a test label here.
This is a software smoke test, not a trained GP or a peptide optimization result.

## Configure a monomer catalog and an N-terminal action

The following is a template, not a description of an inspected model or CSV.
Replace the ID column, annotation values, N-terminal choices and predictor
module with the real model's contract. CSV paths are relative to the YAML file;
existing predictor-specific path conventions are unchanged.

```yaml
groups:
  - name: n_terminal
    kind: categorical
    choices:
      - name: treatment
        element: treatment
        values: [free, acetyl]  # EXAMPLES: use only model-supported choices
  - name: peptide
    kind: sequence
    length: 11
    tokens:
      csv: ../data/new_monomers.csv
      id_column: monomer_id    # replace with actual stable ID column
    allow_repeats: true
    count_constraints:
      - where:
          column: has_free_thiol
          equals: "true"      # exact CSV text; quote strings in YAML
        min: 2
        max: 2

properties:
  - name: gp_property
    predictor: "my_gp.rl_adapter:PeptidePredictor"
    # Additional keys here are forwarded to the adapter constructor.
    model: /absolute/path/to/model_bundle
    direction: max           # max or min, according to the real target
    objective: mean_minus_kstd
k: 1.0

method: a2c
pg_warmup_eps: 100
pg_num_iters: 100
pg_batch_eps: 16
gen_temperature: 1.0
num_gen_eps: 100
exploration_gen_eps: 0
```

The budgets above illustrate the config shape; they have not been tuned against
the user's GP. Repetition is allowed by default: choosing the same monomer at
two different positions is valid. Use `allow_repeats: false` only if the design
requires distinct monomer IDs.

The catalog must have unique nonempty IDs. IDs remain strings, including leading
zeros; CSV order sets the action vocabulary. A list is also supported:
`tokens: [A, B, C]`. Count membership can instead be explicit:
`{tokens: [C, D], min: 2, max: 2}`. Multiple count bounds may overlap; action
masking checks their joint feasibility and rejects impossible configurations.
Hard count constraints remain active with `constrain_training: false`.

For peptides, the GP-side preprocessing should supply or audit the annotation
for a **usable free thiol before disulfide formation**. Do not infer it from
elemental sulfur or from token names. Protected groups and monomers with
multiple reactive sites need explicit application-side treatment. Counting two
marked residues only expresses the supplied annotation; it does not validate
chemical assembly. A catalog without this column can use a separately verified
explicit list of IDs, without modifying the original CSV.

## GP-side adapter contract

Keep domain-specific work in the GP repository or an installable adapter module.
The existing plugin loader calls:

```python
class PeptidePredictor:
    def __init__(self, cfg, *, seed=None):
        # Load the trained GP AND its exact training-time preprocessing here.
        ...

    def predict(self, candidate):
        # candidate["peptide"] is an ORDERED list of 11 monomer IDs.
        # candidate["n_terminal"]["treatment"] is the selected treatment.
        # Map IDs to structures/descriptors using the GP's own catalog.
        # Locate the two verified thiol-bearing residues; pass their positions
        # to the GP in its actual indexing convention and expected pair format.
        # Apply the same assembly, encoding, scaling and target transform as
        # training, then return one scalar mean and one STANDARD DEVIATION.
        return mean, std
```

For example, with zero-based internal indexing:

```python
sequence = candidate["peptide"]
positions = [i for i, token in enumerate(sequence) if token in verified_thiol_ids]
if len(positions) != 2:
    raise ValueError("Expected exactly two eligible residues")
# Only if this GP expects one-based positions:
gp_positions = [i + 1 for i in positions]
```

The two selected residues determine the single pair; a separate bond-pair action
is unnecessary under this constraint. Their positions remain part of the GP
input. The adapter should validate terminal treatment compatibility and all
additional chemistry rules before prediction. If such rules reject many
candidates, add a reusable feasibility interface to the RL environment rather
than relying on repeated terminal rejection. Arbitrary chemistry-dependent or
cross-group compatibility masks are not included in this first extension.

The adapter must own these specifics:

- Stable monomer-ID lookup, SMILES/structure handling and training vocabulary.
- N-terminal encoding and the GP's supported choices.
- Disulfide-position indexing (zero/one based), pair ordering and cyclization.
- Molecular/sequence featurization and all fitted preprocessing objects.
- Whether the GP returns variance or standard deviation, latent or observation
  uncertainty, and the output target's scale. Convert variance to standard
  deviation and keep mean and standard deviation on the same target scale.
- Finite scalar output and explicit errors for unsupported inputs.

Expose `predict`, without an unrelated sklearn-style `score` method on this
adapter: the reward engine interprets `score(candidate)` as a point predictor.
Use **one GP adapter instance** when preserving its posterior uncertainty;
passing a list of separate models invokes the engine's existing ensemble-mean
spread semantics instead. FQN modules must be importable in the runner's Python
environment. If the GP has incompatible dependencies, the adapter may call a
separate process/service without changing the RL algorithms.

## Reward, identity and outputs

The existing `properties` engine maximizes a scalar reward. Let `d=+1` for
`direction: max` and `d=-1` for `direction: min`:

| objective | Reward before property weight/scale |
|---|---|
| `mean` | `d * mean` |
| `mean_minus_kstd` | `d * mean - k * std` (conservative) |
| `mean_plus_kstd` | `d * mean + k * std` (uncertainty seeking) |

The adapter returns the raw mean and standard deviation; it should not negate
the mean or apply another uncertainty penalty. The engine rejects nonfinite,
nonscalar predictions and negative standard deviations. A GP's uncertainty is
used as reported; software tests do not establish its calibration or accuracy.

State features preserve every occupied sequence position and previous
categorical choices. Actions use token identity features. Different sequence
orders and N-terminal treatments have distinct deduplication and cache keys.
`dqn_augment_permutations` must be zero: reordering a sequence changes the
candidate and cannot reuse its reward.

For sequence runs, `generated.csv` includes `candidate_json`, containing the
complete ordered sequence and categorical groups. The legacy `formula` column
contains a readable JSON candidate label, not a molecular formula.
With the property name above, `obj_gp_property_mean` and `obj_gp_property_std`
contain the raw GP statistics; `reward` contains the optimization score.
Legacy `dp_mean` is the engine's signed/weighted mean aggregate and must not be
mistaken for the raw GP target. The adapter can derive the exact two positions
again from the saved sequence and verified catalog.

`sequence_schema.json` is saved alongside weights and the scaler. Keep it when
moving checkpoints: generate-only/resume validates the vocabulary order,
sequence length, terminal choices and resolved count bounds before loading.
It guards RL input semantics; model-version and chemistry-preprocessing
provenance remain the GP adapter's responsibility. Existing sequence-free
material encodings and checkpoints retain their previous path.

## Current validation boundary

Initial generic validation on 2026-10-15: **447 passed, 2 skipped** for the
full repository suite. The two skips need an unavailable `perovskite.vasp` fixture. Sequence
tests include actual CPU DQN/REINFORCE/A2C training and generation, DQN reload,
A2C critic-schema rejection, joint count feasibility and uncertainty scoring.
An additional exhaustive audit matched 1,976 prefix masks in 200 small random
search spaces. These checks establish software behavior, not GP accuracy.

The current GP path directly imports the installed GP package in the same Python
process and reads its original monomer CSV. Application-specific chemistry lives
in that package, as documented in [gp_physchem integration](gp_physchem_integration.md).
After removing the former process bridge, the full generic suite passed 458 tests
with 2 missing-material-fixture skips; external GP API checks run separately when
that package is installed. This extension covers the RL runner; BO/GA baselines
are not adapted to sequence search.
