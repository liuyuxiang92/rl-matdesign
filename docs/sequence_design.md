# Ordered sequence design and external GP models

For the inspected user-provided model and its chemistry definitions, see
[gp_physchem integration](gp_physchem_integration.md). The sections below describe
the reusable framework and the initial generic template.

The same DQN, REINFORCE and A2C runner now supports ordered token sequences in
addition to material compositions. `kind: sequence` is a generic group: tokens
can represent monomers, building blocks or process operations. Sequence length,
vocabulary, repetition and token-count bounds come from YAML. Sequences can be
built from scratch or edited from a parent. No peptide
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

## Configure candidate names and an N-terminal action

The following is a generic template. Replace the token names, counted subset,
N-terminal choices and predictor module with the real model's contract. RL only
reads an explicit list of names; the predictor owns their domain-specific meaning.

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
    tokens: [A, B, C]
    allow_repeats: true
    count_constraints:
      - tokens: [C]
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

The token list must have unique nonempty names. Quote numeric-looking IDs to
preserve leading zeros. List order sets the action vocabulary. Count membership
is an explicit set, for example `{tokens: [C, D], min: 2, max: 2}`. Multiple count bounds may overlap; action
masking checks their joint feasibility and rejects impossible configurations.
Hard count constraints remain active with `constrain_training: false`.

For peptides, the GP-side preprocessing should supply or audit the classification
for a **usable free thiol before disulfide formation**. Do not infer it from
elemental sulfur or from token names. Protected groups and monomers with
multiple reactive sites need explicit application-side treatment. Counting two
marked residues only expresses the supplied annotation; it does not validate
chemical assembly. A predictor can supply these sets automatically through
`sequence_constraints(group_name, tokens)`; the environment applies the returned
hard count constraints in addition to any configured bounds. The names passed to
this optional hook include the replacement choices and any parent-only names.
The GP adapter can therefore validate all names before training begins.

### Replace exactly n positions of a parent

Add only `parent` and `num_mutations` to a sequence group:

```yaml
- name: sequence
  kind: sequence
  length: 4
  tokens: [A, B, C]
  parent: [A, B, A, C]
  num_mutations: 2
```

Each episode starts from the same complete parent. Each mutation has two
decisions: choose an unmodified position, then choose its replacement from
`tokens`. Replacing a token with itself is excluded and a position can be edited
only once. The result consequently differs from the parent at exactly
`num_mutations` positions. Replacement choices are made independently at each
edit; different sites may receive the same token when `allow_repeats` is true.
Names present only in the parent can remain in the sequence but are not offered
as replacements. The predictor must understand those names too.

Edits are counted by token identity. Use one stable name per building block;
chemical aliases for the same monomer should not be separate choices.

With n mutations there are 2n sequence decisions, plus any categorical decisions
such as an N-terminal treatment. Both position and replacement use the existing
RL policy and exploration settings; replacement is not forced to be uniformly
random throughout training. Without `parent` and `num_mutations`, the existing
left-to-right construction behavior is unchanged.

The state contains the complete current sequence, edited positions and the
pending position between the two decisions. Hard count bounds apply to the
final sequence. A temporary count violation is allowed only when the remaining
edits can repair it. For example, moving a thiol residue can temporarily create
one or three thiol-bearing residues while still finishing with exactly two.
No partial candidate is sent to the predictor: only the final sequence and
categorical choices are scored. Deduplication uses that final candidate rather
than the order of edits.

### Migrating old CSV inputs

`tokens: {csv: ..., id_column: ...}` and CSV-column `where` constraints are no
longer accepted. Replace the former with `tokens: [name1, name2, ...]`, and use
explicit count sets or a predictor-supplied constraint hook. GP owns and loads
its existing `examples/new_monomers.csv` relative to its package directory;
the RL configuration needs no chemistry-file path. RL and HPO use the inline
names to define the replacement choices.

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

    def sequence_constraints(self, group_name, tokens):
        # Optional: resolve names using this predictor's domain knowledge.
        if group_name != "peptide":
            return []
        thiols = [name for name in tokens if self.has_free_thiol(name)]
        return [{"tokens": thiols, "min": 2, "max": 2}]
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
categorical choices. Parent editing additionally records the selected position
and edit history. Actions distinguish position choices from token choices. Different sequence
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
sequence length, parent/edit budget, terminal choices and resolved count bounds before loading.
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
process. Its package-owned monomer CSV is loaded by default and provides chemistry
definitions; RL receives the chosen names directly in YAML. Application-specific
chemistry lives in that package, as documented in [gp_physchem integration](gp_physchem_integration.md).
After removing the former process bridge, the full generic suite passed 458 tests
with 2 missing-material-fixture skips; external GP API checks run separately when
that package is installed. This extension covers the RL runner; BO/GA baselines
are not adapted to sequence search.
