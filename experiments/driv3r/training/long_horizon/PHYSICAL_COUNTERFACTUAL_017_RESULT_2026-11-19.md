# PHYSICAL-COUNTERFACTUAL-017 — offscreen material damage is not identifiable without observation

**19 November 2026.** Based on long-horizon Draft/HOLD branch at `d1c3fb8a95188b016590f4c1d83045eece9b2c51`. Read-only source pin `physical_inspection_016.js` Git blob `45a426f893203f93fa9a3972d63bd4a110ceaa3d`.

## Scientific question
Can a learned controller selectively revisit previously painted terrain after an offscreen material erasure **when the visible action history and local material observations remain unchanged**? We created synthetic paired counterfactuals by taking the exact same host-executed prefix and applying an explicitly hidden intervention at its endpoint: reset painted target A to correct, and either erase it or preserve it. Natural prior A erasure is superseded at the identical boundary in both arms. Both counterfactuals have identical public observations, known walls, move budget, legal menu, location, recent actions, and prefix receipts; hidden damage is never an input to a deployable policy.

80 TRAIN world clusters × four horizon/cap conditions × two outcomes yielded **640** identical-feature paired labels. A real 1200-step logistic optimizer fitted the causal features. Goal-first TRAIN calibration then chose whether to skip inspection or use the physical feasibility guard.

60 disjoint DEVELOPMENT world clusters × four conditions × two outcomes × five strategies yielded **2400 full sequential synthetic policy rollouts** and **116999 replayed action receipts**, under a fresh instance of the same host with the experimental intervention reconstructed exactly at the causal boundary. Both forged-action and forged-position receipts were rejected. The entire scientific result reproduced identically across two complete V8 campaigns, excluding variable runtime.

## Results

| Controller | Goals / 480 | Mean utility | Planner calls | Checks |
|---|---:|---:|---:|---:|
| never | 209 | 329.192 | 1800 | 0 |
| late | 370 | 627.725 | 2286 | 292 |
| guard | 370 | 636.846 | 2100 | 292 |
| learner | 370 | 636.846 | 2100 | 292 |
| prefix_oracle | 355 | 619.204 | 1950 | 146 |

The fitted damage probability was exactly **0.5** for all paired inputs (balanced labels with no distinguishing observable information); TRAIN calibration selected cutoff **0.25**. The learner consequently replicated the deterministic feasibility guard, **not a meaningful learned damage predictor**. A privileged *prefix* oracle sees whether A was erased at this one time, but does not see a later second 128-horizon erasure and is not a deployable policy.

| Horizon / cap | Never | Late | Guard | Learner | Privileged prefix oracle |
|---|---:|---:|---:|---:|---:|
| 36/36 | 60 | 86 | 86 | 86 | 86 |
| 72/64 | 60 | 120 | 120 | 120 | 120 |
| 128/64 | 44 | 44 | 44 | 44 | 44 |
| 128/128 | 45 | 120 | 120 | 120 | 105 |

**Important negative result:** the privileged prefix oracle achieves fewer total goals than the feasibility guard because 128/128 worlds can suffer a *later* independent erasure. Knowing the initial erasure alone is insufficient for safe selective revisiting. The 128/64 logical-action-censored cohort is distinct from 128/128 shadow.

## Provenance, artifacts and limits
The JSON ledger contains every seed/cap/hidden-state/policy row, action distributions, receipt hash and paired exceptions; the SVG shows side-elevation synthetic trajectories. The exact-host-source SHA-1 Git blob pin is checked by the standalone Node runner. The optimizer is *not* scheduled on push; use explicit `node experiments/driv3r/training/long_horizon/physical_counterfactual_017.js /absolute/output/dir`.

The V8 run qualifies source-replay only, not a Node machine benchmark: per-inference p95, Node peak RSS, actual SLM checkpoint, independent implementation and native CyberSand acceptance remain outstanding. The Node runner reports RSS when invoked explicitly. No actual game-side observable notification or ABI claim is made. Structural HELDOUT unopened; no GPU/API charges.

**Decision:** continue with *observable* local material checks, feasible return budgets, and multi-pulse recovery; arbitrary risk learning over an exactly unobservable outcome cannot work.
