# BELIEF-UTILITY-010 — Risk-calibrated material-loss belief under shifted feedback

19 November 2026. **New synthetic study only; not CyberSand native ABI or production evaluation.** Draft/HOLD.

## Question and experimental design

Can a ten-parameter logistic erasure-risk estimator, fitted to genuinely observed features and a TRAIN-only counterfactual utility threshold, preserve final material goals while reducing expensive revisits? This deliberately stresses the negative OBS-HISTORY-009 result on a distinct standalone synthetic host (13×9, BFS route, wall gaps, protected cells, temporary gate, wrong-material paint repair, remote first-target erasure, delayed/dropped/noisy epoch and warning receipts). DEV has changed material erasure prior, warning confusion, drop probability and occlusion geometry. Policy features exclude hidden erasure state, pending messages and future events; privileged oracle is labelled upper bound only.

TRAIN: 140 seeds × four horizon/cap configurations = 560 episodes; 1,200 full-batch logistic optimizer steps. DEV: 60 disjoint seeds × four configurations × six controllers = 1,440 rollouts. Four cohorts: 36/36, 72/64, 128/64 **synthetic native64-censor proxy**, and separately labelled 128/128 **shadow**. Structural HELDOUT unopened. Shared prefix counterfactual forks and source-replayed actual host receipts; independent movement, material and menu algebra, illegal action and corrupted position rejection.

## Measured V8 results

| Controller | Goals / 240 | Mean utility | Revisits | Planner calls |
|---|---:|---:|---:|---:|
| never | 56 | 132.1 | 0 | 902 |
| always | 178 | 603.0708 | 240 | 1142 |
| warning | 86 | 248.2458 | 56 | 958 |
| learned_50 | 154 | 510.1167 | 194 | 1096 |
| risk_train | 178 | 603.0708 | 240 | 1142 |
| oracle | 178 | 618.725 | 122 | 1024 |

TRAIN threshold frontier chose **0.01** (best mean TRAIN utility 679.0910714285715). Learned TRAIN erasure Brier loss: 0.207626; train erasure rate 0.5428571428571428. Total source-replayed action receipts **67476** across 1,440 DEV rollouts. Complete action distributions, individual per-case outcomes, receipt checks, train scores and paired losses are retained in JSON.

| Horizon/cap | never | always | warning | learned_50 | risk_train | oracle |
|---|---:|---:|---:|---:|---:|---:|
| 36/36 | 14 | 44 | 22 | 39 | 44 | 44 |
| 72/64 | 14 | 60 | 25 | 52 | 60 | 60 |
| 128/64 | 14 | 14 | 14 | 14 | 14 | 14 |
| 128/128 | 14 | 60 | 25 | 49 | 60 | 60 |

Paired against always-revisit, losses/wins/ties: never 122/118/0; always 0/0/240; warning 92/92/56; learned_50 24/22/194; risk_train 0/0/240; oracle 0/118/122. Full original negative cases, exact seeds, horizon, censor, outcome difference and receipt digests are included.

**Scientific conclusion:** TRAIN-selected risk threshold collapsed to conservative always-revisit, achieving the same DEV 178/240 goals and utility 603.0708 with **no reduction in planner calls**. The naive 0.5 cutoff lost 24 DEV goals despite saving 46 planner calls. A privileged future oracle can save visits, but its information is unavailable to the policy. The result is negative for adaptive calibration under this observation shift, not evidence that all belief policies fail.

## Qualification/limitations

Local explicit CPU run in a V8 isolate, total wall time 4277 ms. No process RSS/high-resolution per-inference measurements are available from that environment; opt-in Node CLI captures its *own* runtime and RSS, but this has **not yet been executed independently**. This JS host is not the source-faithful OBS-HISTORY-009 Python simulator, or verified native ABI; its receipts are replayed by a new instance of **the same JS host** plus algebra checks, not independent native acceptance. The evidence uses FNV-32 equality digest, **not SHA-256**; SHA-256 for committed source is recorded by the Node CLI only when actually run. No deployable checkpoint, paid compute, HELDOUT, other workstream, or game-code modifications.

Reproduce explicitly with `node experiments/driv3r/training/long_horizon/belief_utility_010.js /absolute/output/dir`. No training on module import or push. Retain always-revisit as a conservative benchmark while investigating better observability, per-goal verification and risk calibration.
