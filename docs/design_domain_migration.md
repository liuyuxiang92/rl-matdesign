# Updating an old design to the current domain contract

The current runtime supports only chemical `composition` and `categorical`
groups. The old domains and parameter handlers have been removed at the user's
request. There is no automatic legacy-domain converter or fallback runtime.
This guide describes explicit scientific choices when rewriting a scenario;
it does not authorize resuming old trained artifacts under a new schema.

| Previous concept | Current declaration |
|---|---|
| Fractional species picks | `composition`, `amount_set`, `n_steps`, `total: 1` |
| Integer token counts | `composition`, integer `amount_set`, `n_steps`, no total |
| Distinct species picks | `unique: true` |
| Mandatory oxygen token | Final `element_bounds: {O: [1, upper]}` |
| Fixed host fraction | Include host in `species_set` and constrain its final amount |
| Sequence position selection | `categorical` labels and values |
| Parent sequence/mutations | Peptide `structure.parent` and exact categorical edit budget |
| Categorical metal/oxide flag | Explicit compound choices in a composition group |
| Group `sites` | Original POSCAR symbol/group pairs in a substitution step |

Numeric actions are increments. Do not assume renaming a parameter preserves an
old overwrite-based domain: choose `unique: true` when each species may occur
once. If the old domain required exactly five nonzero species, use five unique
actions with an all-positive amount menu. Zero-valued unique actions still
consume a label and an action.

Final bounds are on raw amounts of listed choices before normalization or
empirical reduction. Missing choices are zero. A zero final bound prevents
positive terminal presence but does not prohibit a zero action, or intermediate
signed edits that cancel. A positive oxygen minimum ensures terminal oxygen
without forcing it to be the last action.

Ordinary compound amounts represent formula units. LiPS explicitly selects
cation-fraction interpretation in its structure substitution operation and
balances charge in the next operation. Signed defect groups retain their raw
signed edits and use the structure plan to resolve scientific composition.

Categorical label/value actions use immutable targets. Parent peptide occurrence
selectors resolve once; a later edit cannot change which original residue a
label identifies. Novel peptide thiols select unfilled positions. The parent,
sequence assembly, and chemistry-specific restrictions remain plugin-owned.

After rewriting, run the metadata audit and application-specific validation.
Retrain because state schemas, actions and candidate semantics have changed.
The historical sealed-release tests retain their archived input only to verify
that current code rejects unauthorized exact resume; they are not a supported
old-schema execution path.

See [complete current examples](design_groups.md) and the
[singular structure plan](structure_source_migration.md).
