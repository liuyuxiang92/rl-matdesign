# Step 1 normalization upgrade review plan

Status: implementation on `codex/step1-normalization`, starting from `codex/unified-material-groups` at `c76614665`. The GlassNet task maximizes T3 with expansion and glass-transition gates. This document records the design and evaluation protocol.

Step 1 has two independently switchable parts. **1A adds adaptive PopArt value normalization under a fixed selected reward definition. 1B adds optional automatic initialization of fixed property calibration when property scales are unknown.** Implement and test 1A first, then 1B, so their effects can be distinguished. For GlassNet, every scientific comparison arm uses the newly requested maximization task and only the two remaining gates. Existing explicit property scales remain supported.

The review used local GlassNet and perovskite YAMLs. This branch supplies the migrated GlassNet examples and a portable perovskite template. Capture the configuration and model fingerprints in each comparison manifest.

## 1 Scope and expected result

| Included in Step 1 | Deferred to subsequent work |
|---|---|
| Adaptive normalization of Q/value learning targets | Gradient clipping and automatic clipping thresholds |
| Output preservation when PopArt statistics change | Replacing actor advantage standardization |
| Fixed scientific reward semantics during a run | Loss replacement, including HL-Gauss |
| Optional initial property calibration with persisted provenance | Exploration and entropy-controller upgrades |
| Correct training, checkpoint, resume and generation integration | Budget estimation, allocation and stopping |
| Numerical, integration and controlled scale-stress tests | Pretrained-policy transfer |

For 1A, retain the selected DQN loss and the current A2C critic squared-error loss. Moving a SmoothL1 loss into normalized units changes the physical-unit meaning of its transition threshold; that is part of target normalization, not an independent switch to another loss. Keep the existing optimizers, learning rates, replay schedule, target-copy cadence, actor loss and sampling rules.

The engineering guarantee sought is precise: changing the internal normalization statistics alone must not change decoded predictions or scientific rewards. Better cross-scenario learning is a separate empirical hypothesis. Step 1 does not by itself establish that HPO is unnecessary.

## 2 The two different scales

Scientific reward calibration determines what the user considers valuable. Its references, property scales and user coefficients stay fixed after resolution. Learning normalization determines the coordinates in which Q/value regression occurs; its mean and spread adapt during training.

For two simple maximization properties, an optional calibrated reward is:

\[
R=w_1(P_1-m_{1,0})/s_{1,0}+w_2(P_2-m_{2,0})/s_{2,0}.
\]

If the resolved scales are 10 and 0.1, and the weights are 2 and 1, improvements of 10 and 0.1 contribute 2 and 1 reward units respectively. Later observations cannot redefine that exchange rate. Values outside the calibration range are allowed and are not clipped.

The implementation does not need continuously changing per-property scores followed by compensating coefficients: their compensated result would equal this fixed reward. PopArt supplies the adaptive numerical treatment after reward construction.

## 3 What the current implementation actually does

The current [model definitions](../src/rl_matdesign/model.py) use scalar output layers for Q and V. [The shared DQN update](../src/rl_matdesign/training.py) selects either a stored MC return or a bootstrap target, then fits it in raw return units. [A2C](../src/rl_matdesign/training.py) fits a raw critic return and calculates the actor advantage as return minus detached raw V.

Scientific objective scaling already occurs in [the composite predictor](../src/rl_matdesign/predictors/structure_score.py). The [reward combiner](../src/rl_matdesign/rewards.py) owns coefficients, gates and alternative selection, and intentionally performs no normalization.

The historical GlassNet configuration minimized forming temperature and constrained T3 to 0..2000 K. The [coarse bootstrap example](../configs/glassnet_coarse_2pct_dqn_bootstrap.yaml) now maximizes T3 with expansion and glass transition as its two gates. All comparison arms use this revised task to isolate the effects of calibration and PopArt.

### Continuous objectives versus gates

The distinction concerns the reward rule, not whether a material or property is complicated. A continuous weighted sum rewards improvements in every objective and permits one improvement to compensate for another property's deterioration. A gate checks an acceptance requirement; its property need not contribute to the passing objective at all. A property can have both roles.

For the revised GlassNet task, thermal expansion must be between 0 and `5e-6 K^-1`, and glass-transition temperature must be at least `773 K`. T3 is exclusively a maximization objective; it contributes no gate decision or violation penalty. To establish the comparison using the existing reward combiner, use passing reward `+T3 / 1000` and failure reward `-10` minus the existing CTE/Tg native-unit penalties. This is the revised-task baseline, not the newly calibrated reward. Illustrative inputs, not evaluated materials:

| Forming temperature (K) | Expansion (K^-1) | Glass transition (K) | Result |
|---:|---:|---:|---|
| 1200 | 4e-6 | 800 | Pass; baseline reward 1.2 |
| 2100 | 4e-6 | 800 | Pass; baseline reward 2.1; no upper T3 gate |
| 1200 | 6e-6 | 800 | Fail expansion; reward -11 |
| 1000 | 4e-6 | 760 | Fail glass transition; reward -10.013 |

The generic legacy combiner provides no global passing-versus-failing ordering guarantee for arbitrary finite T3 outputs. The new profile below guarantees that ordering without assuming a T3 lower or upper bound. Do not silently restore the removed gate through clipping, a validity filter, or a calibration fallback.

The legacy perovskite recipe and the [calibrated example](../configs/perovskite_calibrated.example.yaml) both gate synthesizability at `0.8` and, on passing, combines three delta objectives with weights `0.33333` each. Those delta terms use mean plus model spread with `k=1`. The classifier gate generally uses mean probability minus classifier model spread. This YAML supplies one classifier, so spread is unavailable and no margin is subtracted; keep unavailable uncertainty recorded as missing, not as measured zero uncertainty. For example, delta means `0.1, 0.2, 0.3` with zero delta-model spreads give passing reward `0.199998` at probability `0.9`; probability `0.7` gives failure reward `-10.1`. An ensemble fixture with probability mean `0.85` and spread `0.10` fails with reward `-10.05`. The configured classifier path is a placeholder; these examples test reward arithmetic without asserting model readiness.

### Gate handling in Step 1A

Use this order: native property predictions -> unchanged gate checks and objective/failure combination -> raw discounted learning targets -> PopArt. The Q-network or critic uses one common normalizer for its passing and failing targets. Do not use separate passing/failing coordinates or fit a PopArt normalizer to a probability threshold. Online and frozen target Q-networks retain their own saved normalizer states as specified below.

The classifier's model spread and PopArt's target spread are different: the first participates in a scientific acceptance rule; the second describes numbers the learner is fitting. Only the second adapts through PopArt. No gate-specific network head, reward clipping or new loss is required in Step 1.

Rare passing candidates still pose a learning problem. The return spread includes the pass/fail gap, which can dwarf quality differences among passing candidates. PopArt addresses numerical scale; it does not guarantee discovery of rare passing candidates or sufficient resolution of their quality differences. Test these cases explicitly rather than claiming gate support proves exploration quality.

## 4 Step 1A shared PopArt component

Introduce one reusable component, proposed as `src/rl_matdesign/value_normalization.py` on the implementation branch. Q and V use the same implementation; PolicyNet keeps its current output.

For a raw target y, learn its normalized representation n against `(y - mean) / scale`. Decode predictions as `raw_value = scale * n + mean`. When statistics change from `(mean, scale)` to `(mean_new, scale_new)`, compensate the final linear layer:

\[
W'=\frac{scale}{scale_{new}}W,\qquad
b'=\frac{scale\,b+mean-mean_{new}}{scale_{new}}.
\]

This preserves decoded outputs before the next gradient update. It is the central PopArt mechanism, rather than ordinary target standardization without correction. [PopArt reference](https://arxiv.org/html/1602.07714v2)

### Statistics and lifecycle

Use one versioned shared profile, proposed as `popart_v1`. Its initial candidate adaptation weight is 0.001 per optimizer minibatch, corresponding to a half-life of approximately 693 learner updates. This is a development default to validate across scenarios, not a proven optimal value or a new per-scenario HPO dimension.

Maintain mean, regularized variance estimate, effective scale, initialization status and update/target counters as saved state. Accumulate statistics in float64. Initialize from actual finite targets using the conservative startup rule below. For subsequent minibatches with mean m_b and population variance v_b:

\[
m'=(1-\beta)m+\beta m_b,
\qquad
v'=(1-\beta)v+\beta v_b+\beta(1-\beta)(m_b-m)^2.
\]

The update uses the learner's actual target batches. Repeated replay targets therefore participate in the learner-target distribution; this is different from counting repeated material evaluations as new property calibration observations.

Update statistics once before constructing the loss graph. Do not update them during action scoring, generation, validation-only forwards or checkpoint loading. Reject nonfinite targets before changing the statistics or weights.

A usable divisor must remain positive. The numerical profile explicitly covers constant targets and all-zero startup. Proposed handling is:

- Until a batch has a nonzero RMS, retain identity normalization and mark scale calibration unresolved. Zero targets remain valid training targets.
- On the first nonzero finite target batch, persist its RMS as an anchor in return units, set the mean to its batch mean, and initialize the variance state to `anchor ** 2`.
- This initial variance is a deliberately conservative prior, not the empirical variance. It handles constant nonzero targets without an abrupt switch to a tiny divisor.
- Thereafter apply the EMA formula above on every learner update and use `max(sqrt(variance_state), 1e-6 * anchor)` as the effective scale. For constant targets the prior decays gradually, rather than discontinuously switching to the numerical floor.

These are proposed initialization rules to exercise in tests, not canonical or proven-optimal PopArt settings. The first nonzero calibration can still substantially rescale a randomly initialized head when targets are extremely small; test that transition explicitly. The stronger statistical floor discussed for suppressing tiny noisy learning signals, and actor-specific safeguards, belong to Step 2. A numerical floor alone does not establish training stability.

### Public model interface

`QRegressor.forward()` and `ValueNet.forward()` continue to return decoded values in original return units. Add an explicit `forward_normalized()` used only by their regression losses. Keep the existing scalar final layer and initialization in mode `none`; optional PopArt buffers must not consume extra random numbers in that mode.

This boundary prevents internal normalization from silently changing Boltzmann temperatures, bootstrap arithmetic or actor advantages.

### Optimizer handling

Keep existing Adam behavior and retain its moments when compensating the output layer. Do not reset Adam at each normalization update. Save and restore optimizer state exactly.

The algebra guarantees output preservation at a statistics update; it does not guarantee invariant subsequent Adam trajectories. A new optimizer-coordinate transformation would require separate derivation and validation and is not included here.

## 5 Integration for each algorithm

| Network | Target before normalization | Where PopArt is used | Required invariant |
|---|---|---|---|
| DQN-bootstrap Q | `r + gamma * max(raw target Q)`; terminal transitions use r | Online Q regression loss | Never add raw reward to normalized next-Q |
| DQN-MC Q | Stored discounted episode return | Online Q regression loss | Replay rewards and MC returns remain raw |
| A2C critic | Current implementation's observed return G | Critic regression loss | Actor still uses `G - detached raw V` |
| A2C actor | Existing advantage and policy-gradient calculation | No new actor normalization in 1A | Existing actor standardization, entropy and repeat penalties remain unchanged |

For DQN, construct detached raw targets first, update online statistics and compensate its head, then compute normalized predictions and loss. The target network owns its own frozen statistics. At the existing target-copy event, copy weights and statistics together and clear the TD cache.

The [TD cache](../src/rl_matdesign/utils/td_target_cache.py) stores raw next-state maxima. Updating online statistics alone must not invalidate an unchanged frozen-target cache. Cached and uncached target arithmetic must agree even when online and target statistics differ.

For A2C, update critic statistics once after collecting the real batch returns and before `_episode_pg_terms` builds critic autograd graphs. Do not change the head halfway through constructing those graphs.

A2C's existing feature-scaler warmup replaces rewards with zero and performs no predictor calls. It cannot initialize reward statistics. PopArt initialization uses the first actual training targets, without adding a material-evaluation calibration phase to 1A.

## 6 Step 1B automatic fixed property calibration

This is a separately enabled scientific reward feature, addressing unknown and unequal property ranges. It is not necessary to run PopArt on existing configured rewards.

### Initial supported contract

The revised proposal includes continuous `weighted_sum` calibration and an opt-in `auto_gated_softplus_v1` profile, using GlassNet first. The gated profile supports continuous min/max mean objectives without known bounds. It resolves their references/scales and gate violation scales from physical metadata and an initial pilot. It does not treat pilot minima/maxima as global bounds. Existing reward modes remain supported by Step 1A with their selected scientific definition unchanged by PopArt.

Automatic calibration must select the appropriate complete reward profile. Do not silently center only the passing branch while retaining an unrelated failure constant, or independently transform competing max alternatives. Target-distance and uncertainty-aware property utilities need explicit calibration semantics; missing objective bounds alone are no longer a reason to reject gated calibration.

For a maximization counterexample, naively replacing only the passing score with `(T3 - 1500) / 50` makes a passing 900 K candidate score -12, below a slightly failing candidate near -10. These illustrative calibration values show why the passing and failure branches must be designed together.

The GlassNet profile below defines both branches together. T3 needs quality calibration only; CTE and Tg need violation scaling only. Gate thresholds remain scientific requirements. The fixed revised-task baseline plus PopArt remains one comparison arm alongside the new calibrated reward.

### GlassNet automatic gated profile: proposed design

This is an explicit scientific reward change, separate from PopArt. Retain three rules: passing candidates outrank failing candidates; **higher T3** is better among passes; smaller total normalized violation is better among failures. These rules follow the feasibility-comparison principle in [Deb (2000)](https://www.sciencedirect.com/science/article/abs/pii/S0045782599003898). The scalar mapping and calibration estimator below are our proposed adaptation, not a published RL performance guarantee.

**Quality calibration.** Evaluate a declared initial sample and retain raw T3 outputs from all valid, distinct candidates, including gate failures. Fit a fixed reference `m = population_mean(T3)` and scale `s = population_std(T3)`. An optional declared physical difference scale/resolution is an always-active floor. Do not invent a T3 floor from 2000 K, absolute temperature, the pilot maximum or a numerical epsilon. The calibration sample need not contain any passing candidate.

For the single maximization objective, the standardized quality is `(T3 - m) / s`. For several continuous mean objectives, use:

```
z_i = direction_i * (property_i - reference_i) / scale_i
U = sum(weight_i * z_i) / sum(weight_i)
```

`direction_i` is +1 for maximize and -1 for minimize. Require finite nonnegative weights with a positive total, finite references and positive finite scales. Scaling all objective weights by the same positive factor changes nothing. Freeze these references, scales and relative weights before learning. The convention assigns priorities per initial calibrated property spread; it does not infer scientific priorities from data. Multiplying one property weight deliberately changes its priority.

**Passing branch.** Apply softplus once to the combined standardized utility:

```
softplus(U) = log(1 + exp(U))
passing_reward = 1 + softplus(U)
```

Softplus is positive and increasing, so every passing reward is above 1 in real arithmetic. It has no upper ceiling. For large positive U, it is approximately U, retaining nearly linear incentives for further T3 improvements. This is preferable here to a sigmoid/softsign upper band, which would compress the high-quality differences the revised task seeks. Apply the transform after combining properties, not separately to each property; the resulting candidate rankings and equal-utility trade-off contours match the frozen weighted U.

Compute with a stable library softplus, or `max(U, 0) + log1p(exp(-abs(U)))`, using beta=1 as part of the shared versioned profile. Validate finite calibrated U and combined reward; finite raw inputs do not guarantee that arithmetic cannot overflow. In floating point, very negative U can round passing reward to 1 and erase distinctions among very poor passes; log compression diagnostics and do not claim strict numerical ranking in that tail. Do not hard-clip U or T3 to the pilot range. [Stable softplus implementation reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html)

An unbounded upper T3 range alone does not rule out affine scoring: a certified lower bound could support it. Here the user removed both range bounds, so the recipe assumes neither and uses a positive transform valid for arbitrary finite U. Do not replace a missing lower bound with the lowest observed pilot value.

**Failing branch.** For each remaining gate score `g_j` and available bounds `L_j, H_j`, calculate native violation `v_j = max(0, L_j - g_j, g_j - H_j)`, omitting missing bounds. Apply the same scale to both sides of an interval. GlassNet retains the lower/upper CTE checks and lower Tg check. T3 supplies no gate, violation, or penalty. Define:

```
total_violation = sum(priority_j * violation_j / violation_scale_j)
failing_reward = -total_violation
```

Dimensionless gate priorities default to 1 and may be declared by the user. Nonnegative priorities preserve the ordering guarantee; a zero priority leaves that gate mandatory but removes graded feedback for its violations. Passing rewards are at least 1 numerically, while failures are at most 0, so every completed passing candidate outranks every failed candidate. Increasing only T3 cannot improve the reward of a failed candidate. These constants are one shared proposed recipe, not per-scenario HPO choices or a claim of optimal learning.

Do not multiply normalized violations by existing native-unit `penalty_weight` coefficients. That would normalize units twice. The new profile owns the failure rule and dimensionless priorities; ambiguous mixtures with legacy `failure.reward` or `penalty_weight` must be rejected or explicitly migrated. Legacy mode keeps its exact existing behavior. The new profile intentionally changes relative penalties among failed candidates and the balance between feasibility and quality; it does not preserve the entire legacy utility.

**Automatic gate scales.** Collect all valid, distinct pilot observations, including every candidate failing a scientific gate. A failed gate is a useful observation; a predictor exception or nonfinite output is a different event and cannot supply a fabricated property value.

For each gate and pilot score `g_ij`, let `d_ij` be the absolute distance to its nearest finite gate boundary, whether the candidate passes or fails. Let `B_j` be a declared physical scale floor: use the width of an existing finite acceptance interval, or a certified domain width of the actual gate score when applicable; a declared physical resolution/scale can also supply a floor. Never invent a domain from a pilot minimum/maximum or from the magnitude of a threshold. With no such metadata, `B_j=0` and the pilot must identify a positive usable scale.

```
violation_scale_j = max(
    population_std(all pilot gate scores),
    RMS(all pilot distances to the nearest boundary),
    B_j,
)
```

Use stable float64 arithmetic. Physical width/resolution is an always-active floor, not an exactly-zero fallback. For GlassNet, the CTE interval supplies floor `5e-6 K^-1`. Tg has only a lower threshold; its typical pilot distance from 773 K supplies the empirical violation scale. A constant Tg of 750 K gives scale 23 K; a constant Tg of 800 K gives 27 K. Neither case requires any fully passing candidate. There is no T3 violation scale.

If every Tg equals 773 K without a physical fallback, or every T3 is identical without a declared quality difference scale, calibration remains unresolved. Near-zero quality spread and near-boundary gate data without meaningful physical resolution can also produce small scales; this rule is not a universal noise safeguard. Do not use machine epsilon, absolute temperature, or the removed T3 bounds to fabricate missing information, and do not silently expand the evaluation allowance. Finite extreme predictor outputs can enlarge scales and soften penalties or quality differences; log quality reference/spread and gate spread/RMS distance/floor, extremes and final scales. The estimators express empirical conventions, not scientifically proven tolerances.

**Worked all-failed pilot.** Two rows illustrate arithmetic, not the recommended pilot size:

| Candidate | T3 (K) | CTE (K^-1) | Tg (K) | Failure |
|---|---:|---:|---:|---|
| A | 1200 | 4e-6 | 753 | Tg |
| B | 1400 | 6e-6 | 793 | CTE |

Despite both candidates failing, the T3 quality reference is `m_T3=1300 K` and its population spread is `s_T3=100 K`. The gate violation scales are `s_CTE=5e-6` and `s_Tg=20 K`. With equal gate priorities, subsequent candidates have:

| T3 (K) | CTE (K^-1) | Tg (K) | New reward |
|---:|---:|---:|---:|
| 1200 | 4e-6 | 800 | 1 + softplus(-1) = 1.313262 |
| 1400 | 4e-6 | 800 | 1 + softplus(1) = 2.313262 |
| 2100 | 4e-6 | 800 | 1 + softplus(8) = 9.000335; passes |
| 1200 | 6e-6 | 800 | -0.2 |
| 1200 | 4e-6 | 763 | -0.5 |
| 1200 | 6e-6 | 763 | -0.7 |

Freeze scales and priorities before any optimizer updates. Later discoveries, including the first passing candidate, do not change them. Raw properties, gates and violations remain visible alongside the new reward. PopArt subsequently adapts to the resulting learning targets, including failed outcomes.

**Guarantees and limits.** Under a consistent property unit change `P' = a*P+b`, with `a>0`, the quality reference becomes `a*m+b` and its scale becomes `a*s`, preserving standardized U. Converting gate values/bounds similarly multiplies violations, spreads, distances and physical floors by a, preserving normalized penalties. The new reward is therefore unchanged, including under Kelvin/Celsius conversion.

Softplus is nonlinear and convex: it preserves candidate ordering under the fixed U but changes preferences over stochastic outcomes. It can favor variable quality, and rare very-high-quality successes can outweigh frequent modest successes in expected return. Candidate-level feasibility dominance does not mean maximizing success probability first. Low-U passing differences are compressed; PopArt cannot restore differences altered by reward construction. This is an explicit proposed utility choice, not a policy-invariant normalization or a theorem that HPO is unnecessary. [Reward-transformation theory](https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/)

The same calibration procedure applies to coarse/medium/fine spaces without enumerating them. Independent pilots may produce different quality references/scales and violation scales; for comparisons requiring identical physical preferences, reuse one resolved calibration artifact across compatible grids and algorithms. New RL weights are still trained separately. Persist objective direction, remaining gate set and transform identity so artifacts from the earlier minimizing/bounded proposal cannot be silently reused.

### Proposed configuration

The GlassNet configuration uses this shape. See the runnable configuration examples under `configs/glassnet_*.yaml` for complete design grids. The profile owns its passing/failing reward rules; there is no `forming_temperature_protocol_range` gate, legacy `scale: 1000`, `failure.reward`, or native-unit `penalty_weight` in this new-profile example:

```yaml
value_normalization: popart
value_normalization_profile: popart_v1

properties:
  - name: forming_temperature
    predictor: glassnet
    output: T3
    unit: K
    representation: physical
    direction: max
    objective: mean
    normalization:
      mode: auto_fixed
  - name: thermal_expansion
    predictor: glassnet
    output: CTEbelowTg
    unit: K^-1
    representation: physical
    direction: min
    objective: mean
  - name: glass_transition
    predictor: glassnet
    output: Tg
    unit: K
    representation: physical
    direction: max
    objective: mean

reward:
  type: gated
  calibration:
    profile: auto_gated_softplus_v1
    num_candidates: 100
  objective:
    terms:
      forming_temperature: 1.0
  gates:
    - name: low_expansion
      score:
        terms:
          thermal_expansion: 1.0
      min: 0.0
      max: 5.0e-6
      priority: 1.0
      violation_normalization: auto_fixed
    - name: sufficient_transition_temperature
      score:
        terms:
          glass_transition: 1.0
      min: 773.0
      priority: 1.0
      violation_normalization: auto_fixed
```

Keep legacy `scale: number` semantics, including reference zero, unchanged. Reject simultaneous legacy `scale` and new `normalization` declarations. New calibrated scales must be finite and strictly positive. User coefficients stay in `reward.objective.terms`.

Retain the separate simple `weighted_sum` automatic-calibration option, where the calibrated signed terms are summed without softplus. Do not silently change legacy coefficient semantics: normalized relative objective weights and softplus belong specifically to the new gated profile. Calibrate and orient each objective exactly once before combining, while gate evaluation continues to receive native physical scores.

Remove the new reward-only configuration from physical leaf-predictor construction and model identity inputs, as is already done for the existing scale. Otherwise a normalization declaration could accidentally change a predictor key or leak into its constructor.

### Calibration procedure

1. Collect a predeclared number of fresh, distinct, successfully evaluated initial candidates. Expose a declared calibration count with proposed shared default 100; it is a calibration protocol, not an estimated sufficient search budget. Tiny finite test fixtures may explicitly select a smaller test profile. Production runs must not silently lower the count. Record all evaluation costs and discoveries normally.
2. Persist their selected physical property outputs and candidate identities immediately. Do not rely on an LRU cache retaining all calibration candidates.
3. For both profiles, fit automatic quality references as sample means and scales as population standard deviations, with any declared physical difference scale as an always-active floor. Apply directions once. For the gated profile, combine normalized relative objective weights, apply softplus once, and separately resolve the remaining gate violation scales. Include gate-failing candidates with valid physical outputs; do not fit calibration from the final failure reward.
4. Freeze the calibration before the first optimizer update. An insufficient valid sample count or a scale unresolved by the selected profile must be reported explicitly rather than fabricated with epsilon. Predictor failures consume their normal allowance but do not count as valid calibration observations. User-supplied meaningful fixed scales can resolve missing scale information.
5. Score the calibration episodes using the resolved reward. For DQN, insert them into replay only after reward resolution and compute MC returns with the existing discounted-return helper. This avoids mixing provisional reward labels with final ones.
6. For A2C, use random calibration episodes only for calibration and discovery accounting. Subsequent actor training uses fresh episodes sampled from the current policy, not those random episodes.
7. Use the frozen calibration for all later training and generation. Never refit it from a generation batch or after resume.

DQN can reuse its actually evaluated initial random candidates. If its configured warmup contains fewer than the declared calibration count, calibration consumes the necessary initial portion of the existing total training allowance and is recorded as such. A2C needs real property evaluations for 1B; its feature-only warmup does not supply them. Recompute the remaining learning allowance after counting calibration: the current DQN derivation subtracts configured warmup, while PG derives iteration counts from the full allowance. Neither may accidentally allocate the calibration cost a second time. Carry the same seen-candidate identities and budget ledger into learning, and explicitly preserve the existing meanings of episode, unique-candidate and optimizer-update clocks.

A known insufficient allowance or finite space should fail preflight rather than quietly expand the allowance or use fewer samples. Otherwise, collection can still exhaust its attempt/duplicate limit before gathering enough successful distinct candidates; report unresolved calibration and stop that run cleanly.

Add a mandatory physical-output collection path for calibration. Existing optional prediction-record logging and cache-only lookup are useful building blocks, but unresolved calibration must never cause provisional rewards to be reported as final scientific results.

### Calibration phase and interrupted runs

Introduce the explicit lifecycle `collecting_calibration -> frozen -> learning`. Calibration is a pre-model phase; existing learner checkpoint factories alone do not cover it. A calibration-progress checkpoint must include the declared calibration specification and evaluator identity, raw observations, pending episode trajectories, seen identities, progress count, RNG, predictor state and the shared accounting ledger. This contract exists before a final calibration hash can be assigned.

Before freezing, label physical observation records with a pending reward status. Do not insert provisional scalar rewards into replay, reward-bearing summaries or HPO selection. Freeze the calibration artifact and finalize episode rewards/MC labels once, with a recoverable transition that cannot double-charge candidates or finalize them twice. Resume during collection continues the same calibration sample without reevaluating completed candidates. Resume after freezing restores the artifact without refitting. Generation requires frozen calibration.

### Calibration artifact

Persist `reward_calibration.json` with schema version, immutable content hash, selected reward profile, quality-transform version/parameters, property/output identities, units, representations, directions, coefficients and their normalization convention, quality references/scales, remaining gate boundaries/priorities, physical floors, counts, fitting rule, calibration candidate identities, evaluator fingerprints and source-domain provenance. Quality bounds are not required. Keep raw observations in a companion calibration record. Reject previous-task artifacts with minimizing T3, the removed gate, or the earlier bounded-affine transform; this is a changed scientific contract, not exact resume.

Resume and generation validate the artifact rather than refit it. When the same physical objective is deliberately used across coarse/medium/fine grids, reuse the same calibration. A different grid is recorded as provenance; incompatible property definitions, units or evaluators invalidate reuse. This shares scientific units, not pretrained RL weights.

## 7 Configuration and saved state

Default to `value_normalization: none` for existing configurations. `popart` enables the new path. Save the full resolved profile, including numerical constants, even when users select it through one profile name. Reject enabling critic normalization for an algorithm without a critic instead of silently ignoring it.

The implementation updates the [central parameter registry](../src/rl_matdesign/config_schema.py), CLI/override forwarding in [experiment.py](../src/rl_matdesign/experiment.py), and generated configuration documentation. New normalization keys must enter the [learning execution contract](../src/rl_matdesign/utils/execution_contract.py) explicitly; the current prefix-based selection would miss a generic shared key.

Every full and partial checkpoint must save online/target normalizer state, initialization/fallback flags, profile version, counters, optimizer state and applicable property-calibration identity. Generation-only loading must construct the correct model representation before strict loading. A2C generation loads its actor, but still needs the resolved scientific reward contract for evaluation.

Use a separate normalization/model metadata version rather than unnecessarily changing the shared state/action and feature-scaler schema. A PopArt artifact missing its statistics must fail clearly; never hide this with `strict=False`.

The current repository rejects unversioned artifacts and hashes package source into execution identity. Default-off numerical compatibility does not automatically make historical checkpoints cross-version compatible. Preserve existing source/oracle checks. Historical import, if needed, requires a narrowly audited compatibility rule or explicit weights-only migration; it is not automatically exact resume. Exact continuation within the new version must restore all statistics and must reject changed normalization settings.

## 8 Diagnostics

Log normalizer mean, estimated spread, used scale, initialization/floor status, update count, normalized loss and raw-unit prediction error. Record online and target statistics separately where applicable. Track these at existing learner logging points rather than add new evaluator calls.

For property calibration, report each reference, scale and coefficient, standardized combined U, passing softplus contribution, and each native/normalized gate violation. The local passing-reward response to a physical improvement depends on U through softplus; do not report it as one constant slope. Report the fixed physical exchange rates in U separately. Keep native property columns and the selected-task feasibility diagnostics unchanged. Track the fraction of very negative U, floating-point ties at the passing floor, and unusually large positive U so compression or extreme-target behavior is visible.

Lower normalized loss is not itself evidence of improved material discovery. Compare candidate properties and a fixed scientific evaluation criterion.

## 9 Verification and acceptance

### Engineering acceptance

Add focused tests proposed as `tests/test_value_normalization.py`, `tests/test_calibration_run.py` and `tests/test_reward_calibration.py`, and extend the existing checkpoint, reward and TD-cache tests. Tests should exercise actual model/trainer helpers rather than duplicate their implementation in the test.

| Test | Acceptance requirement |
|---|---|
| PopArt output preservation | Mean-only, spread-only and combined changes preserve raw Q/V before any optimizer step; repeated changes and floor transitions included |
| Numerical tolerance | Well-conditioned float64 fixtures use approximately `1e-10` relative/absolute tolerance; float32 uses approximately `1e-5` relative tolerance and scale-aware absolute tolerance |
| Target arithmetic | Raw MC recursion and raw masked Bellman targets are correct, including terminal states and different online/target statistics |
| Cache | Cached and uncached targets match; online-stat changes do not affect frozen-target cache; target copies include stats and clear cache |
| A2C | Critic loss is normalized, actor advantage uses raw V, statistics update precedes graph construction |
| Statistics | Once-per-learner-update behavior; no inference mutation; constant/singleton/all-zero/abrupt-scale/nonfinite cases handled explicitly |
| Scientific reward | 1A leaves identical candidates' rewards, gates, failure penalties, alternative choice and physical records unchanged |
| Property calibration | 100-fold property scale difference, fixed contribution ratios, unit conversion, values beyond calibration range, unsupported modes and zero spread |
| Persistence | Saved/loaded predictions and rewards agree; cooperative and ordinary resume restore all required state |
| Calibration phase | Interrupt before, during and after freezing; resume preserves the same sample, never recharges evaluations, never exposes pending rewards as final, and refuses generation before freezing |
| Calibration accounting | DQN extended warmup and A2C added calibration consume the original total allowance; seen identities persist; insufficient space, allowance or successful distinct observations fail explicitly |
| Exact continuation | New-version uninterrupted versus resumed runs agree in the deterministic CPU fixture, including normalization, optimizer, RNG, candidate and accounting state |
| Mode off | Existing behavior, state keys and initialization RNG consumption remain unchanged within the declared artifact-compatibility boundary |

Extend [exact resume tests](../tests/test_exact_resume.py), [TD cache tests](../tests/test_td_target_cache.py), [model contract tests](../tests/test_model_contract.py), [reward tests](../tests/test_rewards.py), and the current end-to-end smoke tests for all three algorithms.

### Required gated-reward cases

Extend the [existing all-method gated integration tests](../tests/test_reward_integration.py) and [classification gate tests](../tests/test_reward_integration.py). Use recorded/synthetic predictor outputs; no MD or trained material predictor is necessary for these engineering checks.

| Case | Required result |
|---|---|
| GlassNet examples above | Exact hand-calculated branch, reward and penalties; unchanged with PopArt off/on |
| Each threshold, just below/equal/above | Existing inclusive-bound behavior; one failure and simultaneous failures combine correctly |
| Perovskite single classifier and ensemble | Missing spread remains missing; mean alone can pass while mean minus available spread fails; objective `k` does not alter the classifier gate |
| Every candidate fails, then first pass, then increasingly frequent passes | Finite statistics/normalized targets; no scale reset by outcome class; statistics-only changes preserve decoded Q/V |
| Constant failures and a rare passing outlier | Exercise regularized startup and minimum-scale transitions; report scale and normalized target extremes without promising rare-success learning |
| All three algorithms | DQN-MC and A2C use correctly discounted complete rewards; DQN-bootstrap uses raw target-Q values, masks terminal future value and handles differing online/target statistics |
| Interrupt immediately before/after first pass | Resume restores the same statistics, target network, optimizer and continuation |
| Equivalent CTE unit conversion | Multiply values/bounds by `1e6` and divide violation penalty weight by `1e6`; gate decisions, penalty cost and total reward stay unchanged |
| Passing-versus-failing ordering | New softplus profile gives passing rewards at least 1 and failing rewards at most 0; legacy combiner retains its own semantics |
| Accidental passing-only calibration | Require the complete gated profile and reject ambiguous mixtures with legacy penalties; lack of quality bounds is supported |

For the new profile, test the all-failed pilot and six new reward examples above, constant failing/passing Tg, exactly-at-threshold unresolved gate calibration, constant T3 quality calibration with/without a declared scale, active physical floors, zero valid predictions, and fixed calibration after the first success. Test unit conversion of both quality references/scales and gate penalties, including Kelvin/Celsius. Verify the two remaining gate labels match the revised-task baseline even though reward values differ. Exercise pending-pilot relabeling, checkpoint and generation behavior with all three methods.

Add explicit removed-bound tests: T3=2100 and 3000 remain passing when CTE/Tg pass; increasing T3 improves their reward; a synthetic negative T3 must not fail an absent range gate. Changing only T3 while CTE or Tg fails leaves the failure reward unchanged. Test later values beyond pilot extrema without clipping/refitting. Test multiobjective fixed indifference contours and invariance to multiplying all objective weights by a positive constant. Check stable softplus at large finite positive/negative U, monotonicity where floating-point resolution permits it, documented lower-tail ties, and approximately linear increments at high U. Reject nonfinite computed utilities/rewards. Test rejection of old minimizing/three-gate calibration identities.

The unit-conversion test is distinct from a PopArt statistics update. It changes the entire physical-unit declaration consistently; ordinary PopArt never changes a physical property, threshold or penalty coefficient.

### Scale robustness experiment

First use cheap finite synthetic/lookup environments. Compare normalization off/on with whole learning-reward multipliers 0.01, 1 and 100. Apply this experimental multiplier only after scientific evaluation, persist it in the test-run learning contract, and distinguish scientific reward from learning reward in records. Keep scientific property outputs and evaluation utility unchanged. Multiply the entire learning reward consistently, including any failure reward and penalties; scaling only one property would change priorities.

Include three fixture families: a two-property weighted sum with a 100-fold physical scale difference, a GlassNet-style continuous gate, and a perovskite-style classifier gate with several quality terms. Within the gated fixtures, include both common and rare feasible candidates with known optima. Cross all three learners, PopArt off/on and all three whole-reward multipliers. Use a proposed ten paired seeds per cheap fixture for an initial screen, with identical starting data and declared allowances. Deterministic unit tests establish correctness; these stochastic runs establish only empirical robustness.

Do not expect identical learned candidates or exactly identical Adam trajectories. Exact invariance is required for the algebraic statistics-update tests; learned outcome distributions are the empirical comparison.

There is a specific sampling confound: DQN generation currently uses `exp(raw_Q / temperature)`. Scaling rewards by 100 changes that distribution even for a perfectly scaled Q predictor. Use the existing rank-based/epsilon-greedy sampling path for scale-stress tests, or explicitly decode the experimental multiplier back to common scientific units before temperature sampling. Do not mistake this effect for a PopArt failure or introduce a new exploration controller into Step 1.

### Scientific validation

Use the revised GlassNet maximization task on coarse/medium/fine grids as the first scientific example. Every arm uses only the CTE/Tg gates. Perovskite's existing scientific definition can separately test PopArt; retain a simple two-objective fixture for weighted-sum calibration. Softplus does not require bounded perovskite delta properties. The perovskite example uses aggregate quality calibration. It applies the existing uncertainty-adjusted delta utilities and weights before fitting the combined reference and scale. Scientific runs also require the appropriate model and structure assets. Candidate count affects search difficulty, but neither normalization procedure enumerates the whole design space.

For GlassNet, use four arms: revised-task baseline without/with PopArt, and new calibrated softplus reward without/with PopArt. The baseline means `+T3/1000` on pass, otherwise existing -10 and CTE/Tg native penalties; it does not mean the historical minimizing task or the removed T3 gate. Give paired arms the same initial evaluated candidates and charge calibration within their common allowance. Resolve one calibration per paired seed and reuse it across both new-reward arms. This separates reward-construction effects from PopArt. Evaluate all arms using the same two raw gates, feasible counts and highest-T3 metrics; their training reward numbers are not comparable quality measures.

For a given fixed reward definition, compare PopArt off/on with matched training/generation allowances, evaluation protocol, sampling rules and paired seeds. A proposed initial real-task screen uses five paired seeds; report uncertainty and treat it as screening rather than strong proof. A larger confirmatory experiment and exact allowances belong in a reviewable run manifest before launching costly evaluations.

For a calibration comparison, either give both arms the same resolved calibration or evaluate using common physical criteria. A legacy reward and a newly calibrated reward can express different scientific priorities; their scalar rewards and losses are not directly comparable.

Report qualifying unique candidates, relevant physical-property quality, instability/failure rate and outcome variation across reward multipliers. Separate engineering completion from demonstrated improvement. Freeze shared recipe settings before testing an untouched design setting or task family. No scenario-by-scenario tuning is used to make a failing transfer appear successful.

For each gated task, report feasibility rate and distinct feasible count separately from quality among feasible candidates. For revised GlassNet, best means highest T3, and top-k sorts T3 descending. Predeclare k; if fewer qualify, report the count instead of filling with failures. If none qualify, report zero feasible candidates and undefined feasible quality. Keep native properties and gate margins alongside the passing utility. In finite fixtures, report the gap to the known highest feasible objective. A higher overall average reward alone is insufficient evidence of better scientific discovery. Include a rare-high-quality-success fixture to expose the expected-return trade-off introduced by the unbounded passing reward.

Before confirmatory runs, declare the primary feasible-discovery and quality metrics and a scientifically meaningful acceptable regression margin. Report paired-seed differences and uncertainty, the worst outcome across reward multipliers and sensitivity to those multipliers. Treat a wide uncertainty interval as inconclusive, not as evidence of equivalence. No arbitrary universal percentage or per-task HPO is substituted for this criterion; exact real-task margins and allowances belong in the reviewable run manifest.

## 10 Implementation and validation order

1. Create the isolated branch from the reviewed baseline and capture resolved contracts, including a reviewed GlassNet overlay that maximizes T3 and removes its range gate in every experiment arm.
2. Implement PopArt statistics/head preservation and direct numerical tests.
3. Integrate the shared DQN and A2C critic paths, raw-unit interfaces, target copies/cache behavior and default-off mode.
4. Complete configuration, all load/save/resume paths and diagnostics; run the meaningful unit/integration/resume checks.
5. Add the separate optional fixed property-calibration path, artifact and reward-contract tests, with its stated first-version scope.
6. Run the cheap scale-stress matrix and prepare a report separating algebraic correctness from learning behavior.
7. Prepare the real-task comparison manifest with explicit allowances before material-evaluation runs. Review its results before declaring Step 1 scientifically successful or proceeding to Step 2.

The production implementation is on the isolated branch. Synthetic checks exercise all three methods and reward-scale multipliers. Comparative runs with real GlassNet or perovskite assets are a separate scientific evaluation.

The [synthetic scale benchmark](../scripts/bench_step1_reward_scale.py) runs
900 CPU cases across five fixtures, three learners, two normalization modes,
three reward multipliers, and ten paired seeds. All losses were finite and
generation reported unscaled scientific rewards. Across the gated cases, each
normalization mode produced at least one feasible candidate in 272 of 360
runs. Variation in passing counts across reward multipliers did not decrease
with PopArt in these fixtures. This verifies execution across scales and does
not establish a discovery-quality improvement.
