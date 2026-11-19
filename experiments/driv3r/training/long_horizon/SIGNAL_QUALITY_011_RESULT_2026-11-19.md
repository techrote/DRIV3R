# SIGNAL-QUALITY-011 — Material confirmation quality and policy transfer

19 November 2026 · **Synthetic opt-in CPU research only. No CyberSand native acceptance.** Draft/HOLD.

## Hypothesis and contract boundary

The previous belief learner reverted to always checking previously painted material, since noisy material-change warnings cost more in lost objectives than they save in travel and replanning. Here we isolate **observation reliability**, not another threshold sweep. Can a TRAIN-fitted ten-feature logistic material-erasure predictor skip unnecessary revisits while preserving final goals when the host supplies an accurate, timely, **hypothetical** material-change receipt?

Both conditions use exactly the same synthetic world geometry, material-erasure event generation, original legal action menu (N/E/S/W/PAINT/ERASE/STOP), target locations, first-target erasure seeds, 36/72/128-decision task definitions, and 64-decision censor. Only warning **observation channel** differs:

- **degraded:** TRAIN erasure prior 12%, DEV prior 28%, warning sensitivity 98% TRAIN / 88% DEV, false warning probability 5% / 10%, delivery loss 3% / 12%, delay one of 0/1/2/4.
- **verified (idealised upper bound):** warning indicates actual material erasure exactly, no drop, no delay. This extra confirmation is **not an established CyberSand host ABI field** and must not be presented as available in native gameplay.
- Non-erasure epochs are modelled separately. No controller except the clearly labelled oracle reads hidden erased state, pending events or future events.

Two independent TRAIN models: 160 seeds (6200–6359) × four horizon/cap configurations = **640 TRAIN rows** and **1,200 explicit CPU optimizer steps per mode**. DEV: 80 disjoint seeds (8200–8279) × four configurations × six controllers = **1,920 rollouts per mode**. Structural HELDOUT not opened. 128/64 is a synthetic native64-censor proxy; 128/128 is separately labelled shadow, not native acceptance.

## Measured DEV results

Each goal denominator below is **320 matched tasks**, not 1,920 independent tasks. Policies are evaluated by paired counterfactual forks from the same prefix.

| Observation / policy | Goals / 320 | Mean utility | Planner calls | Revisits |
|---|---:|---:|---:|---:|
| Either / always revisit | 292 | 773.7094 | 1532 | 320 |
| Degraded / TRAIN-calibrated belief | 278 | 753.9438 | 1326 | 114 |
| Degraded / warning-only | 268 | 727.7094 | 1283 | 72 |
| Verified / TRAIN-calibrated belief | 292 | 801.3000 | 1291 | 80 |
| Verified / warning-only | 292 | 801.3000 | 1291 | 80 |
| Either / privileged oracle | 292 | 803.7938 | 1263 | 52 |

Verified observations allow the learned policy to preserve **all 292 baseline goals** while reducing planner calls by **241 (15.73%)** and revisits by **240 (75.00%)**; mean utility rises **27.5906**. Under degraded feedback the same training strategy loses **14 goals** despite saving planner calls, and utility falls **-19.7656**. Both conditions use their own TRAIN-fitted weights; they are not the same policy with different sensors.

TRAIN Brier: degraded 0.050043, verified 0.000036; both select threshold 0.01 without DEV threshold tuning. Structural HELDOUT remains sealed.

## Native-limit censor, controls and validation

| Cohort | Always | Degraded learned | Verified learned |
|---|---:|---:|---:|
| 36/36 | 72 | 70 | 72 |
| 72/64 | 80 | 74 | 80 |
| 128/64 | 60 | 60 | 60 |
| 128/128 | 80 | 74 | 80 |

The synthetic 128/64 proxy remains censored at **60/80** goals in all policies; separate 128/128 shadows must not be counted as native completions. Never/always/oracle action distributions and utilities matched across observation conditions. Both modes passed same-source deterministic receipt replay, movement/material/menu algebra, legal-action checks, and corrupted-action/position rejection; total receipts replayed **167495**. Full per-seed DEV rows, original paired negative cases, model weights, action histograms, route digests, and representative SVGs are retained in distinct evidence files.

In this V8 run, recorded wall time was 2679ms degraded and 2683ms verified. These are **whole-study timings, not per-inference latency**; V8 here cannot report process RSS. The explicit opt-in Node CLI emits process RSS and source SHA-256 when independently executed; this run does **not** assert those measurements. Replay uses the same synthetic Host plus independent receipt algebra, not a separate Python game implementation or actual CyberSand. No deployable checkpoint binary, paid compute, automatic training on push, game-side change or HELDOUT evaluation.

## Scientific interpretation and next step

The useful gap is **quality of observable material confirmation**, not another classifier cutoff. With a perfectly reliable, timely material-damage indication the trained controller attains substantial efficiency without loss of goals in this cohort. With plausible delay and signal degradation, even strong TRAIN Brier scoring is not sufficient. An official, read-only material-dirty/confirmation receipt could be proposed to CyberSand owners, but **must first be defined and authorised through the official host ABI**; do not implement it in gameplay here. Next experiment: measure precision/latency/coverage breakpoints under intermediate channel quality on independent maps and source-faithful trace replays.

Reproduction (opt-in CPU only, no import-time optimizer): `node experiments/driv3r/training/long_horizon/signal_quality_011.js /absolute/output/dir degraded` and separately `... verified`; manual regression: `node experiments/driv3r/training/long_horizon/verify_signal_quality_011_manual.js`. No routine PR/push training.
