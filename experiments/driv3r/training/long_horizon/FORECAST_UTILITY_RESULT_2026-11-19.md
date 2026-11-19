# FORECAST-UTILITY-002 — the cost of uncertain gate forecasts

**Status: DEVELOPMENT-only synthetic negative result, exact research source executed in the same V8 context from which the committed function source is serialized.** No real CyberSand host signal, native gameplay, optimizer checkpoint artifact, SLM or structural HELDOUT qualification.

## Research question

The predecessor FUTURE-GATE-ALIAS-001 found an irreducible early/late gate schedule ambiguity. Does a 75%-reliable externally supplied *synthetic* forecast resolve the dilemma, and is classification accuracy an appropriate objective when missed goals cost far more than extra planner calls?

**Instrument:** balanced early/late/irrelevant material events across 36/72/128 decision horizons. TRAIN seeds 1000–1047 and disjoint DEVELOPMENT seeds 3000–3015. Forecast signal is deliberately generated using inaccessible future mode, with exact one-in-four error by seed and 75% correctness on gate cases; it is **not** a currently available host observation and cannot be deployed as-is. Parent-inspired independent JavaScript world, one-cell paint, protected materials, STOP/BACKTRACK, host-like legal action menu, deterministic BFS. Native64 proxy always caps at 64 painted moves; shadow128 evaluated in separately labelled synthetic research cohort.

## Executed campaign

- Two distinct genuine logistic policies trained by deterministic Adam: **360 updates each** (120 updates at each horizon), 432 TRAIN examples; no checkpoint binary committed.
- **1344 development episodes** across 7 policies; **82493 action receipts independently reconstructed and verified** including legal menus, event state, action outcomes; 16 source-executed structural/optimizer/replay/tamper tests passed.
- Gallery: [synthetic route traces](FORECAST_UTILITY_PATHS_2026-11-19.svg) from receipt-verified 72-decision episodes, explicitly not a native framebuffer.
- Full source-executed evidence: `FORECAST_UTILITY_EVIDENCE_2026-11-19.json` including per-mode/horizon/cohort goals, decision counts, action distributions, planner calls, utility and training losses.

## Results

| Policy | Handoff correctness on 144 DEVELOPMENT scenarios | Goals / 192 episodes (native64 + shadow128) | Mean reward-weighted utility |
|---|---:|---:|---:|
| No-cue learned | 96/144 | 128/192 | 525.542 |
| 75%-cue learned classifier | 120/144 | 132/192 | 560.49 |
| Fixed route-overlap | 96/144 | 144/192 | 623.094 |
| Learned utility-aware cue rule | 96/144 | 144/192 | 623.094 |
| Privileged future oracle | 144/144 | 144/192 | 630.094 |

The 75%-accurate cue raises full-cohort decision classification accuracy from 66.7% to **83.3%** because irrelevant edits are correctly rejected, but the resulting policy **loses 12 goals** versus always invalidating a route-overlapping future gate and has substantially lower utility. A cost-sensitive rule calibrated on TRAIN chooses overlap under either cue sign: even a 25% chance of late failure outweighs approximately 11 avoidable early-reopen decisions under reward = 1000 × goal - 2 × decisions - 3 × BFS calls - backtracks. The utility-aware policy therefore recovers the fixed overlap control exactly; it does not improve upon it.

The privileged oracle is a non-deployable upper control. No native advancement or model checkpoint reproduction is claimed. This is a **negative scientific result** about optimizing an uncalibrated handoff classifier against an asymmetric downstream goal-loss objective.

## Exact-source execution

```bash
node experiments/driv3r/training/long_horizon/forecast_utility_lab.js --out /tmp/driv3r-forecast-utility-002
```

The command regenerates the exact evidence JSON and route SVG outside Git, using deterministic pure JavaScript. The published source is the exact JavaScript function that executed for this result, not a reconstruction from a separate local prototype. JSON `wall_ms` is execution-environment-specific; all substantive counts, actions and optimization fingerprints are deterministic. Small weight fingerprints and receipt digests use diagnostic **noncryptographic FNV**, not integrity-grade SHA256. Use Git SHA for source identity.

## Boundaries / next question

This is an independent port of parent-like gate geometry rather than an exact Python source cross-implementation replay. Do not infer official CyberSand ABI fidelity; study must be replicated against committed Python source and genuine native evidence separately. No process RSS or per-inference p50/p95 was available from this execution context; runtime wall clock is recorded. No model binaries or other repository work lanes touched. Structural HELDOUT remained unopened. A practical follow-up is **host-permitted event provenance or predictive confidence with prospective calibration and explicit utility training**, without granting the controller unobservable future schedules.
