# OUTCOME-FORK-001 — measured counterfactual supervision repairs proxy-label damage, but does not beat frozen imitation

17 November 2026 · DRIV3R PR #1, Draft/HOLD · **TRAIN-only counterfactual labels; DEVELOPMENT-only outcomes; structural HELDOUT unopened; no native CyberSand evidence.**

## Research question

RETRACE-CHOICE-001 prescribed BACKTRACK for a geometrically blocked one-cell side spur, then discovered that the controller learned the label but lost 128-step painting performance. Here the training labels come instead from **two actual source-faithful synthetic rollouts starting from an identical world state**: force the legal MOVE into the spur versus force BACKTRACK; continue both with the *same restored frozen policy*, same goal and 128-decision limit; compare new painted cells. Tie-break toward MOVE to conserve recovery budget. No DEVELOPMENT outcomes choose labels, hyperparameters or winner states.

This is a *policy-relative* outcome comparison, **not an oracle estimate of optimal action**. Both rollouts can miss better strategies; a one-cell difference does not imply a meaningful causal survival advantage. World/host-legal semantics are synthetic and exceed the native 64-logical-move limitation. The native authoritative observation/action ABI is unchanged.

## Controlled training

Two independently restored 1,033-parameter frozen SEQUENTIAL-PROBE-derived baselines (811, 814), source checkpoint ZIP SHA-256 \`c451c22347197bbab919cd3aaf4e6919bfe840c2c9d9d84d212ab87eb9bff2ea\`. Each seed supplies 32 TRAIN comparisons: four known training geometry families × four orientations × trapped/open. Matched adapters: frozen, always-MOVE, geometry-derived contrastive, and empirically labeled outcome. The three adaptations use exactly 40 Adam updates at each of 36 → 72 → 128 steps, identical natural examples, repeat weights and optimizer settings. Eight actual checkpoints were saved and restored.

**TRAIN counterfactual results:** 64 matched comparisons; MOVE beats or ties BACKTRACK in 53, BACKTRACK wins 11 (all by **exactly one painted cell**). Seed 811 has zero BACKTRACK wins; seed 814 has eleven. Consequently seed 811's outcome and always-MOVE checkpoints are byte-identical. This is not evidence for reliably necessary retracing; it establishes that the earlier geometry rule systematically overstated the value of immediate BACKTRACK.

## 128-decision autonomous DEVELOPMENT results

Sixteen episodes per arm and fork type (two policy seeds × four known DEVELOPMENT world families × two orientations). These are new DEVELOPMENT seeds, not unseen structural families.

| Fork | Controller | First MOVE | New painted cells (mean) | Early STOP | Recoveries (mean) |
|---|---|---:|---:|---:|---:|
| Blocked | Frozen | 16/16 | 123.688 | 0/16 | 4.312 |
| Blocked | Always MOVE | 16/16 | **124.000** | 0/16 | 4.000 |
| Blocked | Geometry contrastive | 0/16 | 98.062 | 7/16 | 9.375 |
| Blocked | **Outcome-supervised** | 9/16 | 123.188 | 1/16 | **3.938** |
| Open | Frozen | 16/16 | **127.562** | 0/16 | **0.438** |
| Open | Always MOVE | 16/16 | 126.250 | 0/16 | 1.750 |
| Open | Geometry contrastive | 16/16 | 125.062 | 0/16 | 2.875 |
| Open | Outcome-supervised | 16/16 | 126.938 | 0/16 | 1.062 |

**Interpretation:** in blocked cases the outcome-trained controller gains +25.125 painted cells/episode over the prior geometry-label controller, with a post-hoc paired bootstrap 95% interval approximately +10.56 to +40.31 (20,000 scenario-paired resamples). But frozen and always-MOVE baselines match or exceed the outcome controller. Outcome training reduced excessive retreat; it did **not** establish a new generalizable recovery capability.

The original four-world TRAIN corpus is small, the scenarios share construction logic across splits, and the two seeds disagree about backtracking. Report the positive comparison to the geometry baseline alongside the negative comparison to frozen/always-MOVE; do not promote this to scientific generalization.

## Qualification and custody

- Source file SHA-256 \`5b840f3a025bec1b37143d431ae291580773cf435f2ebdce74ff436cf313977b\`, Git blob \`e43c63d4d8f1fb042ddae5688f9889a68e8c8e3d\`. Three focused pytest tests passed.
- Eight real checkpoints restored and SHA-256 verified; 128/128 DEVELOPMENT trajectories, **16,049 model decisions**, original host menus, action probabilities, cursor/material edits and final hashes independently replayed. The 138 original evidence files matched their SHA-256 manifest.
- The original CPU training and 122/128 full DEVELOPMENT rollouts ran together but exceeded the caller's wall-time limit before the top-level report was written. The **six remaining** DEVELOPMENT episodes were deterministically reconstructed from the *unchanged saved checkpoints*, then all 128 were replay-verified. This mixed original/resume capture is explicitly recorded; no uninterrupted full-campaign wall-time is claimed. Finalization used 18.369 seconds and had 140,940 KiB observed peak RSS; original training peak RSS is unknown.
- Mean model-choice latency on DEVELOPMENT is about **0.126–0.133 ms**, depending on condition. No paid GPU/API was used. Original checkpoints and 128 full trajectory files are **local verified evidence only**, not yet uploaded as immutable GitHub Actions artifacts; binary weights are NOT in Git history. Source and a compact checksum/metrics ledger are committed in this PR.
- Structural HELDOUT, native host observations, real CyberSand painting and physical GPU/Workbench acceptance **were not tested**. Do not treat the historical d_common fixture as the current canonical CyberSand contract.

## Next decisive test

Construct TRAIN-only scenarios where **BACKTRACK wins substantially** (not merely one ink cell), plus matched scenarios where MOVE clearly wins, while preserving **identical legal menus and equal sensor information**. Require outcomes to remain favorable over multiple suffix policies or lookahead horizons. Then train and test on structurally *different* recoverable failures. This will distinguish a learned conditional recovery policy from a default forward-motion bias.

**Ownership:** This research only changes \`experiments/driv3r/training/long_horizon/**\` on PR #1. DRIV3R main, CyberSand, Neurounner, other worker labs and shared workflows remain untouched. Draft/HOLD, do not merge.
