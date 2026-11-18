# Constraint filter contract

Every filter returns a subset of the actions it receives. An empty list means
that this partial candidate has no legal continuation. It remains empty;
restoring rejected actions would admit a candidate excluded by a scientific
constraint. Rollout code handles infeasible episodes with bounded retries.
Incomplete candidates must not reach the oracle.

A group's `filters` list applies every scientific check in declaration order:

```yaml
filters:
  - type: smact_charge
  - type: pauling_en
```

Actions must pass both checks. `apply_filters` checks each result against that
filter's input and rejects newly introduced or duplicate actions and malformed
results with `ValueError`. These are plugin programming errors, not scientific
infeasibility. A filter may prune its own list in place without modifying the
caller's list. The registered built-ins are `smact_charge`, `pauling_en`, and
`ooh_phase`; custom classes use `type: package.module:ClassName`.

Intrinsic chemical amount feasibility runs independently of optional filters.
It enforces exact menus, action budgets, optional total, uniqueness and final
raw-amount bounds. Oxygen presence is a final bound and has no reserved last
position. LiPS's one-positive-precursor-form-per-metal rule comes from its
structure plan and remains active whether optional scientific filters are present
or absent.
Neither a last-step element filter nor a categorical oxygen-form choice is part
of the current interface.

SMACT charge and Pauling filters receive raw accumulated chemical amounts and
explicit completed-group/scaffold context. Categorical fields are generic
assignments and do not implicitly become elemental counts. Structure operations
resolve compound bases and signed edits rather than asking filters to infer them.

The charge screen retains its existing unknown-chemistry policy:

- Missing oxidation-state data remains permissive.
- Missing electronegativity data skips the Pauling check; known-state charge
  balance is still required.
- Exceeding the oxidation-state combination cap warns and remains permissive.
- A Pauling backend exception falls back to charge-neutrality acceptance.
- `allow_alloys` controls its documented alloy exception.

A successful screen means the configured data policy did not reject the design;
it is not proof of every physical validity condition. Known-invalid results
remain rejected when every action fails. The OOH phase filter also returns an
empty action set for infeasible states. Its scientific phase rules remain
independently tested.

The filter list is the sole switch: configured filters apply during warmup,
training, generation and baseline searches. Omit `filters` or use `filters: []`
to run without optional scientific filters. Intrinsic group feasibility and
reward gates still apply. Configured charge/Pauling checks additionally screen
completed generated formulas. There is no separate training enable/disable flag.
