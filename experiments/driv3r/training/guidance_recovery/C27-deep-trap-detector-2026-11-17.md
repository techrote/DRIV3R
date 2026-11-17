# C27 — Deep-trap recovery: separate mistake detection and cue acceptance

**Research disposition:** REJECT both C27 candidates. Retain frozen C12 as the trained reference and C18-A0 as the untrained attested-cue control. This is an external synthetic-host experiment, **not** native CyberSand qualification. DRIV3R PR #1 remains Draft/HOLD.

## Hypothesis and bounded experiment

C26's short induced prefixes often produced an immediate distance improvement and failed to test multi-step recovery. C27 constructs protected-cell corridors where the shortest **host-legal** route requires temporarily increasing Manhattan distance. The training fixture has a 26-step legal route for a six-cell Manhattan separation (4.33x). Withheld geometry has a 28-step legal route for an eight-cell Manhattan separation (3.50x). In both cases, the first shortest-path step *increases* Manhattan distance. An offline oracle verified that legal BACKTRACK operations can reach both goals in 26 and 28 steps; the oracle is not a model or a source of inference features.

The C27-JOINT and C27-DETECT models are frozen C18-A0 plus 12→16→1 bounded, active-guidance-only BACKTRACK residual heads. C27-DETECT separately trains a 4→8→1 mistake detector on earlier **executed** nonprogress events before combining it multiplicatively with a cue-acceptance residual. C27-JOINT has no detector gate. Both preserve all non-BACKTRACK candidate scores and no-cue scores exactly. They use stochastic action choice and the host's legal candidate mask. They cannot authorise protected-material operations. The historical `d_common` and other research harness dependencies are loaded as fixtures without editing them.

Training used 464 host-legal observed menus, 928 executed forced-action counterfactual continuations, 112 CPU AdamW detector steps and 56 CPU AdamW steps per correction head (**224 optimizer updates total**). Detector cross-entropy fell from 0.6131 to 0.4661, but that objective distinguishes the synthetic disturbance generator's *badturn/neutral* labels; it does not establish correct online mistake recognition. Head objectives fell as well. Independent repetitions from the same seed produced byte-identical checkpoint files after deterministic metadata serialisation.

Evaluation used **1,600 matched stochastic synthetic trajectories** across two disjoint evaluation seeds, 90 decisions maximum, no guidance and weak/strong BACKTRACK/CONTINUE arms, two personalities, two hint weights, and no-guidance C12, C18-A0 and historical C26-TRIGGER controls. Training used the two-lane family; withheld evaluation used a rotated three-lane family with independent seeds. This is synthetic topology generalisation, not a native gameplay test. The four targets are `reverse`, `ahead`, `shortcut` and `bridge`.

## Held-out results

| Policy | Cohort A guided goals | Cohort B guided goals | Deep-reverse goals (A+B) |
|---|---:|---:|---:|
| Frozen C12 | 49/128 | 38/128 | 0/64 |
| C18-A0 | **52/128** | **40/128** | 0/64 |
| C26-TRIGGER | **52/128** | 39/128 | 0/64 |
| C27-JOINT | **52/128** | **40/128** | 0/64 |
| C27-DETECT | **52/128** | **40/128** | 0/64 |

Neither trained head changed a held-out goal-completion count relative to C18-A0 in any tested condition. Both pass a *numerical* strict nonregression gate but do not improve gameplay; zero deep-reverse completions means this gate is not evidence of adequacy. For each C27 model, all 64 no-cue paired action and painting hashes in both cohorts agree with C12. No synthetic host-rejected action, illegally highest-scoring raw action, or protected-cell edit was recorded.

A separate deliberately **non-promotable** STOP-score ablation tested whether the strong tendency to terminate was the whole explanation. On 64 matched deep-reverse guided episodes, C18-A0 stopped before the horizon in 64/64 and reached **0/64** goals. Subtracting 3 from the STOP score reduced early stops to **8/64**; subtracting 6 reduced them to **0/64**. Nevertheless, **both ablations still completed 0/64** goals. These are causal probes, **not trained candidate policies**; they intentionally change cue-off scores, violate the no-guidance preservation constraint, and must never be promoted. They show that removing premature termination alone does not solve the required multi-step route.

## What this means

The original one-step metric is actively misleading on these fixtures: individual post-cue moves can lower distance immediately, yet the worm still cannot complete the blocked detour. The failure is a long-horizon action-selection/planning failure, not just a short recovery-delay issue. The current BACKTRACK-only residual architecture is inadequate for multi-step traps; further optimisation of a local margin without expanding the policy's temporal competence has low expected value.

Observed two-sided personality-preferred painting opportunities remained very scarce, despite longer trials. Consequently, **personality preservation under consequential recovery remains unestablished**, not passed. The detector's lower classification loss is not a substitute for the absent goal successes. The test uses preloaded historical paths, which are **not attested executed trails** from live CyberSand; this is a limitation of this synthetic fixture. The oracle shortest-path audit sees hidden geometry but **no model** does.

## Safeguards, source and remaining qualification

- Training and inference are isolated to `experiments/driv3r/training/guidance_recovery/**` in `techrote/DRIV3R`. No CyberSand, Neurounner, native, Workbench or shared host-fixture modification.
- Sixteen hundred evaluation paths, two training receipts sets, two checkpoint pairs, an additional 192 diagnostic traces, SVG gallery and SHA-256 manifest are retained in the local evidence archive. Model binaries are **not committed to Git**. Immutable GitHub Actions artifact custody remains outstanding; a local downloadable archive is *interim* custody only.
- Targeted tests verify geometric solvability, host-only legal BACKTRACK feasibility, exact no-cue score parity and unchanged non-BACKTRACK scores. Seven targeted C26/C27 regression tests passed. Neither native CyberSand tests nor physical Windows gameplay tests ran.
- No new PR, issue, branch, workflow, paid API call or GPU charge. PR #1 remains **Draft/HOLD, unmerged**.

**Next useful experiment:** Use multi-step commitment/option policies on the same solvable detours, with action legality enforced each step and explicit conservative fallback on wrong cues. Make deep-reverse completion nonzero on previously unseen corridors **before** treating personality gains as promotable. Track actual steps to goal, not only first distance decrease.
