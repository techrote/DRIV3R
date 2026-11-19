# ROUTE-INVALIDATION-022 — frozen policy under an exogenous obstacle

19 November 2026 — DRIV3R long-horizon Draft/HOLD synthetic research only.

## Method

Exact pinned 016/018/020/021 source dependencies; TRAIN-only frozen 021 student and 018 prior compared with conservative physical-inspection guard and a hindsight mode selector. 50 previously unused unsealed DEVELOPMENT seed clusters (150100–150149) with a baseline and a matched deterministic route closure after the prefix, at a still-reachable interior cell on a planned route. Two material-damage interventions were paired, with 36/36, 72/64, 128/64 and separate 128/128 shadow cohorts. The closure is a synthetic intervention, not a claim about CyberSand's host ABI.

## Results

| Controller | No closure goals / 600 | Closure goals / 600 | Baseline calls | Closure calls |
|---|---:|---:|---:|---:|
| guard | 421 | 316 | 2850 | 3474 |
| student021 | 432 | 316 | 2894 | 3474 |
| frozen018 | 432 | 316 | 2894 | 3434 |
| hindsight | 440 | 316 | 2746 | 3132 |

### Horizon/cap goals

| Cohort | Guard | Student 021 | Frozen 018 | Hindsight |
|---|---:|---:|---:|---:|
| control/36/36 | 71 | 71 | 71 | 71 |
| control/72/64 | 100 | 100 | 100 | 100 |
| control/128/64 | 50 | 61 | 61 | 69 |
| control/128/128 | 200 | 200 | 200 | 200 |
| closure/36/36 | 71 | 71 | 71 | 71 |
| closure/72/64 | 100 | 100 | 100 | 100 |
| closure/128/64 | 29 | 29 | 29 | 29 |
| closure/128/128 | 116 | 116 | 116 | 116 |

## Execution and qualification

- 50 world-seed clusters, 1200 paired scenario/intervention records, 6000 unique controller rollouts.
- 325442 action receipts deterministically replayed from initial host state, including interventions; independent receipt-field audit; 4 injected corruptions rejected.
- 542 closure interventions actually installed on the feasible arm, 0 unavailable closure cases, 2097 changed paired receipt hashes and 116 paired cases with a guard/student terminal change.
- 41 original negative case records retained in full. V8 wall time 27714 ms (research harness, not an end-to-end Node game measurement).

## Decision and constraints

Do not promote any policy on synthetic evidence alone. The hindsight controller uses counterfactual outcomes unavailable to the host. 128/128 is strictly shadow research, not native acceptance. Source-faithful same-host replay plus separate receipt audit are not an independent game implementation. Exact-source Node predictor p95, RSS, native CyberSand contract confirmation, and deployable neural checkpoint remain unqualified. No structural HELDOUT opened, no paid services, no game modifications. CLI requires explicit manual execution.
