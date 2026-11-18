# One structure plan per scenario

A scenario owns one optional `structure` declaration. It describes construction
and sampling; all structure properties consume the resulting sample ensemble.
There is no named structure registry, property source selector, source alias,
shared-structure override, or old-source migration converter.

```yaml
structure:
  builder: pipeline
  base_poscar: POSCAR
  n_random_configs: 10
  steps:
    - type: substitute
      a_site_symbol: Fe
      a_dopant_group: A_dopant
      b_site_symbol: Fe
      b_dopant_group: B_dopant
      c_site_symbol: Sr
      c_dopant_group: C_dopant
    - type: geo_opt
      model: models/relax.pt
      fmax: 0.05
      steps: 100
```

The example resolves `Sr Fe Fe` by symbol occurrence: A receives the first Fe
block, B the second Fe block and C the Sr block. Stable original atom identities
survive deletion and insertion. Public group site multipliers and site maps are
not needed. Templates load once; each sample starts from a copy and each step
consumes its predecessor's result.

The built-in pipeline supports substitution, signed defects within substitution,
SSE charge balancing, OOH adsorbate frames, MD averaging and geometry optimization.
An external fully qualified builder may implement another construction method.
Peptide assembly uses its own `builder: peptide` declaration with parent or length.
Operation options stay local to the operation that consumes them.

Substitution alone is sufficient to build a POSCAR. Add simulations only when
required. LiPS chains cation substitution, anion/charge balancing and optional
relaxation. Perovskites can chain substitution, MD and relaxation. OOH chains
substitution and adsorbate preparation; labelled intermediate frames remain
children of their original random sample rather than extra samples.

## Runtime ownership

`resolve_structure_plan(cfg, needs_structure=...)` validates the single plan.
`StructureSpec` holds builder configuration and asset hashes with a fixed internal
identity `structure`; `StructurePlan` holds the spec and property names. Derived
internal mappings only route shared evaluator work and are not configurable
registries. Relaxation is an explicit pipeline step, not a property or source flag.

Sample identity includes the candidate, seed, template and sample index.
Scientific output records preserve those identities. Shared model instances do
not merge different conditioning parameters: distinct named properties still
perform their required inference. Scoring returns direct statistics, outputs,
reward details and structures, without sweep rows or a selected-row wrapper.

Declare temperatures or other experimental conditions as separate properties.
Use a reward expression to combine or select them. LiPS maximizes weighted
conductivity/stability pairs at a common temperature while keeping every output.

## Updating old configurations

Move template, builder and sample-count options into `structure`; express old
standalone builder actions as ordered pipeline steps. Remove every property
`source`, top-level source registry and `sweep`. Move relaxation settings into a
`geo_opt` step. This requires inspecting the scientific workflow rather than
mechanically renaming a source identifier. The old converter is removed because
it produced the retired source-registry schema.

Use [design groups](design_groups.md) for current complete application examples.
Historical validation reports describe their original revision and must not be
used as an API reference for the current runtime.
