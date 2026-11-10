# Design-domain migration: exact grids and free construction order

This document describes domain semantics implemented by the general environment.
Actual experiment YAMLs and their migration records remain local. Use the
migration tool to review a particular old configuration rather than assuming
that every legacy last-step restriction is interchangeable with a final bound.

## Invariants and ownership

Keep three concepts separate:

1. The domain defines which completed materials are permitted.
2. Construction rules define which trajectories can produce those materials.
3. Chemical screening defines additional scientific acceptance criteria.

Freeing construction order deliberately changes the MDP and trajectory count.
The migration must demonstrate equality of the intended terminal domain; it
does not promise identical learned weights, sampling distributions or rewards
at the same training iteration. Correcting an existing fail-open filter is a
separate recorded behavior change, not evidence that its invalid outputs were
part of the intended scientific domain.

Hard arithmetic, token-budget and required-species rules belong to the domain.
They remain active when `constrain_training` disables optional chemical
screening. The same rules apply to random, DQN, policy-gradient, generation,
augmentation and baseline callers. A terminal candidate identity excludes its
construction history, but the RL state includes remaining slots and any
selection counts needed to determine future legal actions.

## Exact composition grids

Wire the existing public helper names in `env.py` to
`utils/composition_domain.py`. Keep decimal menu codes and their order for
compatible existing configurations. Regular ranges contain only
`min + i * step <= max`; the upper limit is not rounded outward or inserted
when it is not on the grid. Explicit menus retain their gaps.

Physical element bounds are intersected with the actual menu. A positive
physical lower bound cannot become zero after integer conversion. Use exact
decimal/rational arithmetic, not a floating-point `ceil` or an arbitrary
epsilon. Reject an empty mandatory menu or an impossible total before rollout.

Completion checks use `(actual_menu, mandatory)` for each remaining species.
For a fixed-order composition, every remaining slot is mandatory, including
slots whose selected amount is zero. For free species choice, mandatory comes
from the declared positive lower bound; it must not be inferred merely from a
positive minimum in the shared menu. Selecting zero consumes a slot; omitting
an optional species does not. Preserve these distinct cases in state features.

The exact completion helper prevents the old false-positive combination of a
sum-of-bounds interval and a bound-agnostic sum table. For example, two remaining
species with legal amounts `{2,5,8}` cannot total 8, even though 8 is within
their `[4,16]` sum window and is reachable as `0+8` on a broader global menu.

## Integer oxides: retain the terminal domain while freeing O order

The reference domain is five token selections:

- Exactly one O token, whose digit is in the original positive digit menu.
- Exactly four non-O tokens, whose digits use the original menu including zero.
- Non-O elements may repeat; repeated picks accumulate counts.
- A zero token still occupies one selection slot.
- The terminal material key drops zero amounts and reduces integer counts by
  their greatest common divisor, as before.

The existing last-step filter bundles these domain rules with the requirement
that O be selected last. Remove only that order requirement in the migrated
scenario. O may be selected first, in the middle or last. The integer update
remains `counts[element] += digit`; it must not become the composition domain's
one-time assignment of an absolute fraction.

Requiring only terminal `O > 0` would expand the domain. Five selections
`Ba1, Ti8, Zn0, O8, O9` would produce `BaTi8O17`, which cannot be represented by
the original one-O-digit rule. The formula also admits the simple oxidation
assignment Ba(II), Ti(IV), O(-II), so a neutrality screen does not fix this
domain expansion.

Compile the domain into an internal immutable token requirement, for example:

```text
symbol=O, minimum_picks=1, maximum_picks=1, positive_values_only=true
total_picks=5
```

These are scientific domain data, not a new user switch for choosing whether
correct validation occurs. Their authored spelling should be owned by the
shared configuration schema. Migration of the existing oxide configuration
can derive the record from its singleton reserved, nonzero last-step rule;
the normalized configuration records the new free-order semantics explicitly.
Do not infer this record for arbitrary external filters or ambiguous legacy
settings.

`IntegerRatioEnv` needs a selection-count map separate from accumulated atom
counts. After tentatively choosing a token, check each maximum, forbid zero
for a positive-only required token, and ensure that the remaining slots can
still meet all minimum token counts. With the oxide record, an omitted O
becomes forced only when one slot remains. After O has been chosen, all future
O actions are forbidden. Ordinary repeated cation picks remain legal.

Initialization checks that required species exist, their value menus contain
an allowed positive value and their minimum token counts fit the horizon.
Both `allowed_actions` and `step` enforce the same domain; `step` rejects an
externally supplied illegal action before changing state or calling a model.

## Composition: required presence belongs to the domain

A positive lower bound makes a species mandatory regardless of when its amount
is chosen. Remove an obsolete final-step restriction only when the terminal
bounds independently enforce the same requirement. Preserve the number of slots,
zero-amount semantics and the exact final composition set. A structure builder's
placeholder element does not determine an RL action position.

## Legacy last-step rules

Retain `LastStepElementFilter` as an explicit compatibility construction rule
for users intentionally studying fixed ordering. Correct its empty-result
fallback and numerical zero test. A code such as `"0.00"` is zero even though
it is not the string `"0"`.

Do not globally translate `required_elements: [A,B]` into requiring both A and
B: its current final-step meaning is “choose one of these elements.” Likewise,
`reserve_for_last: false`, a nonpositive token allowance or composition bounds
that do not independently ensure presence require their own migration report.
Known singleton-token and positive-bound cases can be migrated explicitly;
other legacy forms retain their declared order semantics until a concrete
equivalent domain has been established.

The DQN augmentation code detects the legacy last-position pin. A migrated
free-order domain has no pin, so its O/Ti assignment may be permuted with the
other assignments, subject to replaying and checking every resulting action.
Keep legacy pin detection for unmigrated configurations.

## Chemical screening and composite constraints

The current implementation is named `ChainConstraintFilter`; there is no
separate `CompositeConstraint` class to introduce. It must compose predicates
by intersection. Every child receives the preceding subset, must return a
subset of that input, and may return an empty list. An empty result stays empty;
the chain must not restore its original action list. Reject invented actions
from a plugin rather than silently enlarging the domain.

The removed fallbacks include `LastStepElementFilter`, both SMACT paths and
`SSEDopingFilter`. `PhasePatternFilter` and OOH phase constraints already kept
empty results; their existing criteria are retained and covered by regression
tests. See [constraint_filter_contract.md](constraint_filter_contract.md) for
the remaining unknown-chemistry policy.

Apply hard domain rules first, then optional chemistry screens to that legal
action set. When screening is enabled, a failed screen cannot restore an
action. When it is disabled for the existing training ablation, hard domain
rules still hold. This deliberately changes the historical
`no_train_filter` variants, which previously disabled O presence along with
chemistry; record the new meaning and do not compare their logs as unchanged
continuations of the old MDP.

SMACT already tests the complete accumulated composition at the final token,
not specifically the O action. Thus moving O earlier requires no scientific
assumption that the last action is oxygen. For example, if O was selected
first and the final action adds Fe, screen `prior_counts + final_Fe`, with O
included once. `ElectronegativityFilter` uses the same assembly and adds its
Pauling test. Test this path directly, rather than just checking that the
constraint object is present.

Preserve the distinction between a known-invalid composition and an unavailable
screen. `charge.py` currently allows missing oxidation-state data, exceeded
combination caps and some failed Pauling checks. Removing action-list fallback
does not resolve those cases or prove certified neutrality. This stage retains
that existing data policy. Passing the screen means “not rejected under its
available-data policy”, not certified neutrality; no new per-candidate
unknown-data certification is claimed. Changing whether such candidates remain
eligible is a separate scientific-policy change.

## Dead ends and training integration

Exact domain completion avoids arithmetic/token dead ends. A final chemical
screen can still reject all continuations of an earlier prefix; exhaustive
lookahead through 80 elements and all oxidation states is not a reasonable
hidden replacement for the current screen.

Represent this as a specific infeasible-rollout outcome. Do not assign an
invented physical reward, restore illegal actions, or send an incomplete
candidate to the predictor. The training owner should connect it to bounded
retry handling for random/DQN/PG rollouts, scaler-only warmup, generation and
batched/deferred evaluation. Distinguish it from `CandidateEvaluationFailed`,
which means a completed valid candidate could not be scored. Do not catch all
`RuntimeError` or programming errors as ordinary infeasibility.

Discard the incomplete path before replay/actor updates. Log the rejected
attempt and elapsed time, but do not charge the successful-unique-candidate
budget or call the property model. A bounded consecutive-failure guard must
stop an impossible or persistently blocked configuration with the last
constraint context. Generation counts rejected attempts against its existing
attempt cap. A terminal `allowed_actions=[]` remains a normal completed state;
an identical result before termination is an infeasible rollout.

## Acceptance matrix

| Area | Required acceptance |
|---|---|
| Exact range | `{min:0,max:0.15,step:0.02}` ends at 0.14; nonzero origins and scientific notation retain exact values. |
| Sparse completion | Every action in an exhaustively enumerated small bounded domain has a valid completion; invalid sparse-window prefixes are pruned. |
| Oxide equal domain | Enumerate a small menu for old reserved-last trajectories and new free-order token trajectories; reduced terminal keys match exactly, while O-first/middle/last are reachable. |
| Oxide counts | Repeated non-O picks accumulate; zero non-O tokens consume slots; O-zero and repeated O are rejected; `BaTi8O17` counterexample remains outside the domain. |
| Legacy order | Intentional last-step configurations remain pinned; numeric zero codes are rejected; ambiguous multi-element rules are not silently reinterpreted. |
| Chain | Empty first/middle/final child stays empty; a later identity filter cannot resurrect actions; plugin-added actions are rejected. |
| SMACT/Pauling | O-first and O-last representations of the same terminal composition produce the same screening result; final non-O addition is included exactly once. |
| No training filter | Optional chemistry can be disabled; O/Ti presence, token quotas, exact bounds and sum rules remain active. |
| Dead-end accounting | Scalar/batched random, DQN, A2C, PG warmup and generation handle rejection without model call, invalid training rows or unique-budget consumption; retry caps terminate. |
| OOH/LiPS/perovskite | Existing valid fixed candidates retain their scientific mappings; impossible filter combinations fail closed; O_a/O_b keep dopant dependencies. |
| Augmentation | New free-order token paths allow legal O/Ti permutations; legacy pins remain pinned; no invalid permutation enters replay. |
| Artifacts | Resolved domain/feature/action contracts identify the migration; old incompatible checkpoint/scaler artifacts fail explicitly. |

## Implemented interfaces and validation

- Fraction groups accept a shared menu or a mapping of species to sparse menus.
  Bounds intersect each menu exactly, with ceil/floor on integer units.
- Integer groups use `token_requirements: {O: {count: 1, positive: true}}`.
  The known legacy singleton reserved-positive O rule normalizes to this domain;
  ambiguous last-step rules retain their meaning.
- Integer network state contains descriptor, absolute counts, per-species pick
  counts and remaining tokens. A zero token consumes a pick. An all-zero terminal
  is invalid. MultiGroup integer payloads are positive GCD-reduced empirical
  counts; a group’s site multiplier acts on that empirical unit. External plugins
  expecting unreduced absolute integer counts must migrate explicitly.
- Reward adapters with `evaluate_candidate` receive a detached canonical mapping
  in immediate and batched paths. Ordinary formula callbacks remain supported.
  Compound labels such as SiO2 render as `(SiO2)fraction` for descriptors.
- `InfeasibleEpisode` denotes an empty nonterminal action set. Retry is bounded;
  incomplete candidates never receive a physical reward or enter replay.

Independent review compared 1,200 fractional domains (10,333 partial states)
against exact enumeration and 100 integer domains (1,403 partial states) against
an independent valid-token enumeration. The stage-wide test gate and its final
result belong with the local validation record for the actual experiment.
