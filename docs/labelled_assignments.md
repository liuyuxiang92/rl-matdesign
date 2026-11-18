# Labelled actions across material domains

One action selects one label and one value. The pair of action indices is one
complete decision. Chemical labels accumulate amounts; categorical labels name
immutable targets whose values are assigned once.

| Domain | Label | Value | Completion |
|---|---|---|---|
| Chemical recipe | Element or compound | Numeric increment | Exact `n_steps`, optional `total`, final bounds |
| Labelled sublattices | Group and species | Numeric increment in that group | Each group's declared budget and constraints |
| Novel peptide | Residue position | Monomer ID | All positions assigned by the residue stages |
| Parent peptide | Original residue position | Replacement monomer ID | Exact budget, distinct edited positions |
| Experimental conditions | Named variable | Number or categorical value | Exact declared action count |

Chemical groups have no scientific assignment order: Fe then Ni and Ni then Fe
produce the same accumulated recipe. Repeated labels are allowed unless
`unique: true`. Site groups remain distinct even when they contain the same
element. Two precursor recipes can expand to the same elemental composition
while retaining distinct recipe identities for structure construction.

```yaml
groups:
  - name: alloy
    kind: composition
    species_set: [Fe, Ni]
    amount_set: [0.25, 0.5, 0.75]
    n_steps: 2
    total: 1
```

Categorical position labels retain order in the resulting sequence. Assigning
`3:C, 1:A, 2:B` and `1:A, 2:B, 3:C` gives the same sequence `[A, B, C]`.
Assigning `1:C, 2:B, 3:A` gives a different sequence. There is no execution-order
flag: target identities express the distinction directly.

```yaml
groups:
  - name: n_terminal
    kind: categorical
    n_steps: 1
    choices:
      - name: treatment
        values: [H, Ac]
  - name: peptide
    kind: categorical
    labels: [1, 2, 3]
    values: [A, B, C]
    n_steps: 3
structure:
  builder: peptide
  length: 3
```

For parent replacement, put `parent: [A, B, C]` and optional
`require_change: [peptide]` in `structure`, replacing `length`. Group labels may
refer to immutable parent residue occurrences or explicit numeric positions.
The peptide assembler resolves them before rollout. There is no generic group
`parent`, sequence kind, mutation-count shortcut, or STOP action.

Novel ordinary and thiol stages can share one final sequence. The thiol group's
`labels: {unassigned_from: peptide}` selects the positions left by the ordinary
stage. Plugin-declared token-count restrictions remain intrinsic. The predictor
receives one complete ordered peptide, including unchanged parent positions.

```yaml
groups:
  - name: conditions
    kind: categorical
    n_steps: 2
    choices:
      - name: temperature
        values: [300, 350, 400]
      - name: linker
        values: [linker_a, linker_b]
```

Pure numeric values retain magnitudes; symbolic and mixed catalogs use typed
identity channels. `true`, `1`, and `"1"` are distinct identities. Redundant
numeric entries `1` and `1.0` in one menu are rejected. Scientific cache keys
ignore mapping insertion order while preserving positional sequence order.
Chemical count fields use composition groups, never a categorical `element` key.

RL states include action-relevant assignments, masks, budgets and initialized
context. Parent states retain the whole original peptide. Artifacts record
feature meaning as well as tensor shape and reject incompatible old models.
Generated records retain `candidate_json`, `metadata_json`, scalar metadata and
scientific property columns; export selection does not alter the candidate.
