# Labelled assignments across material domains

One environment action assigns one value to one explicit label. A pair of
one-hot action indices remains the internal transport; it does not mean two
independent decisions or require a particular label to be chosen first.

| Domain | Label | Value | Completion |
|---|---|---|---|
| Fractional composition | Species | Absolute fraction | Exact number of assignments and total fraction, within bounds |
| Labelled sublattices | Group and species | Fraction within that group | Declared group order retains cross-group dependencies |
| Peptide from blanks | Explicit residue position | Monomer ID | Every position filled once |
| Parent peptide | Explicit residue position | Replacement monomer ID | One through K distinct changed positions; legal STOP after a valid partial edit |
| Conditions/fields | Named variable | Physical number or categorical identity | All declared fields filled once |
| Integer composition | Species | Added integer token | Existing token budget and accumulation; this is not a one-time assignment |

For a peptide, assigning `3:C, 1:A, 2:B` and `1:A, 2:B, 3:C` creates the same
sequence `A,B,C`. Assigning `1:C, 2:B, 3:A` creates a different sequence. Parent
mutation and blank construction use the same position labels and monomer values.
A parent position cannot be edited twice, and replacing it with its unchanged
original monomer is not an edit. `num_mutations` is the maximum, not an exact
mutation count. STOP is a control action, not a monomer; it receives a separate
identity and terminates the group without padding the trajectory to its maximum.
Training returns, bootstrapping and generation use the actual completed path.

Example group specifications:

```yaml
groups:
  - name: peptide
    kind: sequence
    tokens: [A, B, C]
    length: 3
```

For parent mutation replace `length: 3` with `parent: [A, B, C]` and
`num_mutations: 2`. Token IDs are explicit strings, including quoted numerical
IDs. A predictor may supply intrinsic token-count constraints through
`sequence_constraints(group_name, tokens)`; resolved constraints enter the
saved design contract. User design constraints remain in `count_constraints`.
Legacy CSV/`where` selectors must be resolved to an explicit token list before
running, so the design space is not secretly altered by catalog row order.

```yaml
groups:
  - name: conditions
    kind: categorical
    choices:
      - name: temperature
        values: [300, 350, 400]
      - name: linker
        values: [linker_a, linker_b]
```

Pure numeric values retain physical magnitudes. Categorical and mixed catalogs
use identity channels. `true`, `1` and `"1"` are distinct identities; redundant
numeric entries `1` and `1.0` in the same domain are rejected. Scientific cache
keys treat finite equal numbers as the same physical value without coercing
booleans or strings. Mapping order is irrelevant; sequence position is retained.
Numeric fields are never normalized into chemical fractions. Existing
categorical choices with `element:` still expose chemical count semantics to
structure builders, while `name:` fields remain structured values.

RL inputs now include explicit label identity plus applicable material
features, and typed value features. All three benchmark methods consume the
same tables. Artifacts record these meanings, not only tensor shapes, and
reject older incompatible models. Generated records contain `candidate_json`
and `metadata_json`; scalar metadata also appears as `meta_*`, without replacing
formula, reward or property columns. An optional `generated_columns` mapping
selects/renames export columns while the full run log retains recorded values.

The standalone lookup adapter and ActMOF preparation script are available as
an example of numeric field design. Their tests use synthetic lookup data;
external peptide GP models, actual ActMOF grid data and GPU inference require
separate model assets and validation.
