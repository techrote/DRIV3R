# DRIV3R TRANSFER-STRESS-019 — frozen inspection timing under shifted geometry

**19 November 2026.** Explicit bounded opt-in synthetic experiment stacked on long-horizon Draft/HOLD. Compared 018's unchanged TRAIN-reconstructed softmax selector and cap-aware timing against conservative feasibility, physical early/late checks, never recheck, and a rule using known route distances. No training/threshold changes using evaluation worlds.

## Results

### Fresh unsealed development

| Controller | Terminal goals / 960 | Material goals | Planner calls | Mean utility |
|---|---:|---:|---:|---:|
| feasible | 668 | 672 | 4558 | 551.14 |
| censor_aware | 655 | 669 | 4898 | 543.05 |
| student | 669 | 673 | 4562 | 552.165 |
| route_budget | 608 | 669 | 4992 | 540.106 |
| early | 448 | 509 | 5012 | 382.169 |
| late | 546 | 672 | 4952 | 542.629 |
| never | 315 | 319 | 4000 | 209.85 |

### Shifted obstacles

| Controller | Terminal goals / 960 | Material goals | Planner calls | Mean utility |
|---|---:|---:|---:|---:|
| feasible | 675 | 676 | 4566 | 556.083 |
| censor_aware | 667 | 687 | 4900 | 562.469 |
| student | 693 | 694 | 4638 | 574.533 |
| route_budget | 625 | 687 | 4984 | 559.698 |
| early | 465 | 527 | 4998 | 401.748 |
| late | 556 | 676 | 4958 | 547.438 |
| never | 317 | 318 | 4000 | 209.642 |

**Complete cohort:** 160 disjoint synthetic DEVELOPMENT world seeds in two groups (80 standard, 80 with three fixed additional blockers); 1920 first/second material pulse conditions, 13440 policy rollouts and 753494 independently reconstructed synthetic-host receipts. 173 full paired counterexamples retained. All 2 corrupted-receipt checks rejected tampering. Counts for 36/36, 72/64, 128/64 and 128/128 are separate in JSON evidence; 128/128 is a shadow, **not** native 64-move admission. Strict experimental terminal goal requires correct material and a STOP receipt, not a verified game ABI invariant.

## Scientific interpretation and limits

Frozen 018 TRAIN label counts: never 86, early 124, feasible 70, both 0 (280 prefixes); 1100 true softmax update steps. Initial physically invisible erasure is forced 0/1; for 128 horizons the later pulse is forced 0/1. Counterfactual arms share identical public observations at each fork; policies never see intervention labels. The 'route_budget' rule uses static known walls and an optimistic future route estimate, not hidden damage or privileged outcome.

Every synthetic action receipt was replayed through a fresh instance of the pinned source host, including reconstructed offscreen interventions and end-state checks. This is not independent native CyberSand replay; the host is the same source and the late pulse is an experimental intervention. Source-smoke CI does not rerun science. Batch of 100000 isolated softmax predictions: 234 ms in V8 (not p95); process RSS is **not qualified** in this run, although the explicit Node runner records RSS. No new SLM checkpoint, paid compute, gameplay change or structural HELDOUT exposure. Use full paired seed records and trajectory SVG before any promotion.

## Reproduction

From exact source, manually run `node experiments/driv3r/training/long_horizon/transfer_stress_019.js /absolute/output/path`. The runner pins the Git blobs of 016 and 018 and rejects drift. No training runs on import or routine CI.
