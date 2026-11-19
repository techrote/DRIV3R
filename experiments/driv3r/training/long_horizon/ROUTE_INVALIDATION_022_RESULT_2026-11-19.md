# ROUTE-INVALIDATION-022 v2 — corrected goal-safe obstruction trial

19 November 2026 · DRIV3R long-horizon Draft/HOLD · synthetic DEVELOPMENT evidence.

## Critical scientific correction

V1 (`5993624ec20c4f5077dc842cc7f26768b631b475`) selected an obstacle at mandatory goal cell (11,4) in numerous cases. That made the goal physically impossible and invalidates V1's dramatic apparent completion losses. Its full source, negative cases, paths and evidence remain immutable in Git history; see ROUTE_INVALIDATION_022_V1_INVALIDATION.md. **Never use V1 as evidence of learned-policy brittleness.**

V2 rejects every mandatory goal and protected divider cell, preserves alternate routing and proves reachability of all goal sites from each prefix after obstruction. Checked 200 eligible prefix situations: 0 target collisions and 0 disconnected mandatory goals. Exogenous closures are synthetic experimental events, not CyberSand ABI features.

## Corrected paired results

50 previously unused, unsealed DEVELOPMENT seed clusters 150100–150149; control versus permanent interior route closure; paired first/second material-erasure interventions, 36/36, 72/64, 128/64 censored, and 128/128 uncensored shadow. Frozen TRAIN-only 021 and 018 students; ex-post hindsight is privileged.

| Policy | Control goals / 600 | Goal-safe closure goals / 600 | Control planner calls | Closure planner calls |
|---|---:|---:|---:|---:|
| guard | 421 | 421 | 2850 | 3390 |
| student021 | 432 | 432 | 2894 | 3434 |
| frozen018 | 432 | 432 | 2894 | 3434 |
| hindsight | 440 | 440 | 2746 | 3216 |

| Horizon/cap | Control guard | Control student | Closure guard | Closure student |
|---|---:|---:|---:|---:|
| 36/36 | 71 | 71 | 71 | 71 |
| 72/64 | 100 | 100 | 100 | 100 |
| 128/64 | 50 | 61 | 50 | 61 |
| 128/128 | 200 | 200 | 200 | 200 |

**Finding:** the goal-safe route obstruction increased planner calls by 540 for the trained 021 student and reduced mean utility from 572.23 to 563.88, but its completion count was unchanged. Its censored 128/64 advantage was 61 versus 50 of 200 paired interventions against the conservative guard. This is a limited synthetic result, not native transfer.

## Execution and qualification

1200 paired conditions, 6000 unique controller rollouts, 340104 deterministic full-state source-replayed receipts, 4 malformed-receipt rejections, 542 closure installations on the guard arm, 2097 changed arm receipt hashes, 0 paired goal changes. 82 negative-case records, original controls, action distributions and trajectories retained in the v2 evidence.

Scientific replay is from fresh instances of the same pinned synthetic host plus separately implemented receipt-field checks, NOT independent world logic or native CyberSand acceptance. The 128/128 cohort is shadow only. Node p95 and peak RSS are unqualified in this V8 run; explicit Node CLI records peak RSS when executed. No deployable SLM checkpoint, no paid resources, no production code, no HELDOUT opening.

Reproduce explicitly: `node experiments/driv3r/training/long_horizon/route_invalidation_022.js /absolute/output_dir`.

## Bounded Node predictor-only measurement

An isolated Node v22.16.0 softmax benchmark on x86-64 Xeon Platinum 8573C, reproducing the 018 probability kernel with the frozen 021 weights and a real public prefix feature vector, produced four independent process p95 measurements of **434–442 ns** over 50,000 timed calls each after 30,000 warmup predictions. Peak process RSS ranged **30,324–30,580 KiB**. The benchmark explicitly validates its reference probability vector and is retained in `bench_route_invalidation_022.js` and `ROUTE_INVALIDATION_022_PREDICTOR_BENCH_2026-11-19.json`. These are *predictor-only* measurements, not full DRIV3R policy, route planning, neural SLM, or native game performance. The original research run's Node RSS remains unmeasured.
