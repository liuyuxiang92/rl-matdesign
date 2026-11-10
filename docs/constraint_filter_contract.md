# Constraint filter contract

Every filter returns a subset of the actions it receives. An empty list means
that this partial candidate has no legal continuation. It must remain empty;
restoring the input would admit a candidate already rejected by a scientific
constraint. Rollout code handles this condition as an infeasible episode and
uses bounded retries. An incomplete candidate must not be sent to the oracle.

`ChainConstraintFilter` checks each child against that child's immediate input.
It rejects newly introduced actions, actions removed by an earlier child,
duplicate actions, and malformed return types with `ValueError`. These are
plug-in programming errors, not infeasible episodes to retry. Children may prune
their own list in place; the caller's original action list is protected.

This change affects the legacy last-step element rule, SMACT charge and Pauling
filters (both composition and categorical paths), and SSE metal/oxide form
selection. The last-step rule tests numeric zero, so `0`, `0.00`, and `-0e0`
have the same meaning. It uses `code_to_value` when provided. Numeric nonzero
values retain their meaning; this filter does not introduce a positive lower
bound. The generic domain defines nonnegative amounts separately.

The change does not replace the charge check's existing policy for unknown
chemistry. `constraints/charge.py` is unchanged:

- Missing oxidation-state data remains permissive.
- Missing electronegativity data skips the Pauling check; charge balance is
  still required when oxidation states are known.
- Exceeding the oxidation-state combination cap emits its existing warning and
  remains permissive.
- A Pauling backend exception falls back to charge-neutrality acceptance.
- The existing `allow_alloys` option and scaffold/amount assembly are unchanged.

A successful screen therefore means "not rejected under the configured data
policy", not a proof that every physical validity condition has been verified.
Known-invalid results now remain rejected even when every action fails.

OOH phase and generic phase-pattern filters already return empty lists for
infeasible states. Their criteria, primary-phase intervals and required elements
are unchanged; the chain now preserves these rejections. Regression coverage
includes valid OOH rollouts for all six target phase settings, explicit
phase-pattern rejection, zero spellings, chain subset enforcement, SSE sorted
dopant pairing, and the unchanged unknown-data behavior above.
