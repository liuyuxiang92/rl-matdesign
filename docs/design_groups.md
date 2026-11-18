# Chemical amounts, categorical assignments, and structure pipelines

New configs use two group kinds. `composition` adds numeric amounts to chemical
choices; `categorical` assigns values to immutable targets. Both count one complete
label/value pair as one action. Removed group kinds, parameter spellings, and
standalone builders are rejected; there is no compatibility execution path.
Checkpoint compatibility includes the state/action schema and structure
declaration: start a new run when changing a design.
Unused legacy fields and unknown built-in group/structure options are errors.
Declare scientific filters per group, with any fixed scaffold inside that filter;
multigroup designs cannot silently inherit a top-level filter.

## Chemical amounts

| Parameter | Meaning | Application |
|---|---|---|
| `name` | Group identity, also referenced by structure steps | `A_dopant`, `P_site`, `glass` |
| `kind: composition` | Add numerical amounts to chemical labels | Elements, oxide choices, signed site edits |
| `species_set` | Available elements or compound labels | `Fe`; `SiO2`; LiPS `Ni2O3` |
| `amount_set` | Per-action increments: list, `{min,max,step}`, or one menu per species | Glass molar fractions; oxide integer counts |
| `n_steps` | Exact number of label/amount actions | Five OOH selections |
| `total` | Optional final accumulated sum | `1` for fractions; omitted for integer counts |
| `unique` | Default false; true uses each species label at most once | OOH/HEA: five positive actions give five distinct elements |
| `element_bounds` | Inclusive final raw amount bounds for each listed choice | Oxides `O: [1,9]`; LiPS `P: [0.92,0.98]` |
| `filters` | Optional scientific filters beyond intrinsic amount feasibility | OOH phase; charge neutrality; Pauling electronegativity |

Absent choices have amount zero. Bounds of `[0,0]` ban positive final presence;
they do not prohibit a zero-valued action. `unique` consumes a label even when its
selected amount is zero. Repeated additions accumulate when `unique` is false.
Bounds and action menus apply together. Bounds are on listed choices: a bound on
`Ni2O3` does not also constrain bare `Ni` or every oxygen atom from other compounds.
Signed edits may cancel, so a final zero bound does not prohibit intermediate edits.

Completion masks use exact decimal arithmetic and the remaining step budget.
For an oxide, `element_bounds: {O: [1,9]}` requires oxygen at any action position;
there is no last oxygen token. Raw bounds apply before any empirical formula
reduction. Standalone integer chemical recipes use reduced empirical identities;
structure-bound groups retain their physical raw amounts and group identities.

Ordinary compound amounts denote formula units: `SiO2: 0.6` contributes
`Si: 0.6, O: 1.2`. LiPS's substitution operation explicitly uses
`amount_basis: cation_fraction`: `Ni2O3: 0.04` contributes `Ni: 0.04, O: 0.06`.
The `sse_charge_balance` operation also forbids two positive forms of the same
metal. This intrinsic rule applies even when no optional scientific filters are listed.

## Categorical assignments

Use `choices: [{name: ..., values: [...]}]` for different named fields, or
`labels: [...]`, `values: [...]`, `n_steps: ...` for shared replacement menus.
Each target can be selected once per group; replacement values may repeat across
different targets. Numerical fields retain numerical features; symbolic catalogs
use typed categorical identity features. A label such as temperature is distinct
from a label such as time, irrespective of assignment order.

Peptide initialization belongs in `structure`, not generic group parameters:

```yaml
structure:
  builder: peptide
  parent: [A, HCys, F, K, V, W, W, A, P, R, b-HCys]
  require_change: [peptide]
```

Parent ordinary labels `[A,F,K,V,W,W,A,P,R]` resolve once to positions
`[1,3,4,5,6,7,8,9,10]`. The second original W always means position 7, even after
another position is changed to W. A single `A` selects the first original A;
numeric label `8` selects the second. Mixing strings and integer selectors,
excess occurrences, and overlapping stage targets are errors. Five ordinary
actions make five genuine edits. Thiol labels `[HCys,b-HCys]` stay at positions
2 and 11; retaining `b-HCys` is allowed only at its own initialized position.

For a novel peptide, `structure: {builder: peptide, length: 11}` initializes an
empty sequence. Nine ordinary assignments select positions from 1 through 11.
The thiol stage uses `labels: {unassigned_from: peptide}`, assigning the two
remaining positions. All 55 position pairs are possible. Cap choice + nine
ordinary assignments + two thiols gives 12 complete actions. The parent design
uses 8 actions (cap + five ordinary + two thiols).

Predictors receive exactly `{n_terminal: {treatment: H}, peptide: [11 ordered IDs]}`.
The assembler merges residue stages before scalar/batch evaluation, caching,
metadata, and CSV export. Existing GP plugins retain molecular assembly and their
training-reference parent. When a plugin declares intrinsic token count bounds,
the compiled catalogs are checked before rollout; the generic core never guesses
thiol chemistry from residue names.

Assignment order does not affect identity. Position order does: `AG` and `GA`
are different peptides. Named ActMOF fields also keep their own identities.

## One structure, optional operations

```yaml
structure:
  builder: pipeline
  base_poscar: POSCAR
  n_random_configs: 10
  steps:
    - type: substitute
      a_site_symbol: Co
      a_dopant_group: ooh
    - type: ooh_adsorbates
      adsorbates: [O*, OH*, OOH*]
      ads_height: 1.9
      ads_dz: 1.0
```

The pipeline caches the original template, copies it per sample, and passes each
operation's result to the next. There is no application class or public source
registry. All structure properties share the resulting sample ensemble. Simulation
is optional: substitution alone builds a POSCAR-compatible ASE structure; add
`md_average` and/or `geo_opt` only when required.

`a_site_symbol` through `z_site_symbol` pair with the matching `*_dopant_group`.
Pairs resolve alphabetically against the original POSCAR header blocks. For
`Sr Fe Fe`, a=Fe uses the first Fe block, b=Fe the second, and c=Sr the Sr block.
Header order between different symbols is irrelevant. Unselected atoms remain
fixed. Missing/exhausted symbols are errors. Repeated blocks are captured before
ASE combines species; there is no `site_map` or `group.sites` parameter.

Optional `*_defect_group` applies signed edits to a substituted block. New species
can be inserted; excess vacancy requests clamp by default or error with
`vacancy_policy: error`. Atom identities persist across deletions and additions.

`sse_charge_balance` uses original P/S/Li populations, independently rounded
dopant counts, oxide-derived oxygen counts, Cl/Br choices, and Li deletion for
charge neutrality. It consumes the substituted structure. `geo_opt` consumes its
output. OOH intermediate frames retain mode labels, inactive-atom masks, and the
parent sample identity. The OOH adapter evaluates each parent's frame batch once
per model, preserving the existing DeepProperty scalar contract; frames are not
independent random configurations.

MD uses the optional CLC LAMMPS I/O/averaging helpers on the current structure,
not the old builder that reloads a template and always performs MD plus relaxation.
It requires an importable CLC package, LAMMPS, and a compatible **frozen** model.
The perovskite example uses the supplied frozen `compressed_dpa4c.pt2` for MD
and `dpa4.ckpt.pt` for relaxation. It retains two random configurations and the
user's model/head/executable paths, which must exist in the target environment.

## LiPS temperatures and shared models

`configs/lips_sse.yaml` contains eight properties: conductivity and stability at
460, 470, 480, and 490. Each property has explicit `predictor`, `models`, and
`fparam`. The two repeated ensemble declarations are reused automatically; there
is no top-level model registry, `model_ref`, or `sweep`. Distinct conditions require
distinct inference, while structure building and relaxation happen once per sample.

Generic `reward: {type: max, alternatives: [...]}` selects the best weighted
conductivity/stability pair at the **same** temperature. Ties select the first
declared alternative. All property outputs remain available, with
`selected_alternative` and alternative scores in prediction records. This retains
the previous best-shared-temperature objective; it does not average temperatures
or independently maximize the two properties.

## Property definitions and reward coefficients

`properties` owns predictor settings, output selection, direction, uncertainty
objective, and scale. All coefficients belong to the `reward` block. There is no
property `weight`. Multi-property weighted sums and gated rewards explicitly name
the objective's properties in `reward.objective.terms`; gate and maximum-alternative
coefficients stay in their own terms. A single-property weighted sum defaults to
coefficient 1, so ActMOF's `q` property with scale 100000 yields `q / 100000` with
only `reward: {type: weighted_sum}`.

The perovskite gate named `synthesis` records diagnostics under that label.
Its score is `1 * pure_probability` using the classifier's gate statistic, with a
minimum of 0.8. One classifier checkpoint supplies its mean; an ensemble supplies
mean minus model spread. Passing enables the separate three-temperature delta
objective, which retains its `0.33333` coefficients solely in the reward terms.

## Descriptors

Chemical state layout is `[chemical descriptor, raw amounts, pick counts,
remaining total (when configured), remaining steps]`, separately for each group.
The usual chemical descriptor has 145 features. For species Fe/Ni with 0.5 each,
both selected once, total1 and two completed steps, the numeric tail is
`[0.5, 0.5, 1, 1, 0, 0]`; the full group vector has 151 features. Explicit
amounts distinguish absolute counts that normalized composition features cannot.
SSE uses its cation basis resolver. Signed defect groups use the current dopant
composition plus edits, with signed raw deltas retained separately.

Categorical states encode each target's current value, initialized/filled/edited
masks, eligible targets, and remaining actions. Peptides additionally include the
full ordered sequence, including untouched parent residues. There is no single
universal dimension: catalogs and targets determine the width. Predictor input
features (GlassNet, Magpie RF, peptide GP, or structures) remain predictor-owned.

## Complete scenario configs

All 25 full local configs live under `configs/`: one 29-species OOH design using
DQN Monte Carlo, HEA, Ti alloy, three Cs2AgBiCl6 levels, the supplied binary-gated
perovskite design, LiPS SSE, three default oxide objectives (calcination, sintering,
and both), ActMOF, parent/novel peptides for both GP variants, and nine GlassNet
grid/method combinations. Each follows `groups`, optional `structure`,
`properties`, `reward`, then training controls. OOH and all other properties use
inline `predictor` and optional `models`; retired shortcuts are not supported.
The perovskite design retains the supplied delta conditions and binary probability
gate; its binary classifier checkpoint remains an explicit placeholder. Model/data/template paths are relative to the run working directory,
as before; missing assets and existing placeholders have not been fabricated.
Existing `.gitignore` rules for proprietary OOH, Ti, perovskite, and halide
configs are preserved. Those local files are not automatically added to a public
source distribution. Tests of private configs skip when the files are absent;
generic domain and pipeline tests remain independent of those assets.
