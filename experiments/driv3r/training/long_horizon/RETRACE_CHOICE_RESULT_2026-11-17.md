# RETRACE-CHOICE-001 — learned local discrimination, worsening 128-decision continuation

17 November 2026. **DEV-only synthetic research; structural HELDOUT sealed.**

## Study

Two previous independent 1,033-parameter sequential-probe policies (seeds 811/814), each adapted with three equal-budget teacher variants through 36/72/128-decision curricula. The frozen policy is the fourth control. Each specialized TRAIN fork offers exactly the same legal action menu in blocked and open spur variants: BACKTRACK, MOVE toward the spur, RELOCATE_START and STOP. Only the second cell beyond the side entrance differs. Each adaptation uses equal natural imitation examples and matched targeted repeat-weighting, 40 Adam updates per stage.

**Training label caveat:** BACKTRACK was prescribed for blocked spurs from geometric dead-end reasoning. It was **not** established by independent suffix counterfactual optimization as globally optimal. Indeed, subsequent rollout results undermine that assumption. The original source-faithful `World` is a synthetic research fixture, not a CyberSand native acknowledgement.

## Autonomous DEVELOPMENT outcomes (two seeds, two known world families, two rotations, 128 decision budget)

| Kind / arm | First-choice label match | Mean painted | Early STOP | Mean recoveries |
|---|---:|---:|---:|---:|
| blocked_spur/always_move | 0/8 | 123.875 | 0/8 | 4.125 |
| blocked_spur/always_retrace | 8/8 | 107.500 | 3/8 | 11.250 |
| blocked_spur/contrastive | 8/8 | 96.750 | 4/8 | 9.750 |
| blocked_spur/frozen | 0/8 | 123.375 | 0/8 | 4.625 |
| open_spur/always_move | 8/8 | 126.625 | 0/8 | 1.375 |
| open_spur/always_retrace | 0/8 | 107.500 | 3/8 | 11.250 |
| open_spur/contrastive | 8/8 | 125.500 | 0/8 | 2.500 |
| open_spur/frozen | 8/8 | 127.500 | 0/8 | 0.500 |

**Interpretation.** Contrastive adaptation learned the prescribed initial-action distinction in all 16 DEVELOPMENT cases (8 blocked + 8 open), but did **not** improve long-horizon painting. On blocked spurs it painted 96.75 versus the frozen 123.375, and prematurely STOPped in 4/8 versus 0/8. The always-retrace teacher also degraded performance. The originally presumed necessary retrace was only necessary to avoid entering a one-cell blind pocket, **not necessary for sustained progress**: painting that pocket and later recovering could be cheaper than immediately consuming recovery budget. This is an important falsification of a proxy teacher definition, not a successful deployed recovery policy.

## Evidence and boundaries

- 64 original episodes and 7882 actual host-legal synthetic decisions; all replay verified from 8 restored checkpoint files.
- Wall 16.97 s, peak RSS 142980 KiB. Per-arm action distributions and latency are in report.json.
- 576 TRAIN label records source-replayed. SHA-256 manifest verified for 74 files (excluding manifest).
- Previous checkpoint ZIP SHA-256: 343e48b10dae6388c0029f5a111f1308f169729bf70169a52c8235539927fcd2
- Source SHA-256 and exact input identities in local report.json.
- No sealed HELDOUT opened, native game mutation, new action, paid compute, or CI training.
- Original trained .npz and trajectories remain in a local evidence archive pending immutable Actions artifact upload; do not commit model binaries to Git.

## Next discriminating experiment

Use **actual suffix-outcome labels** for paired accessible side pockets with exact remaining recovery budgets. Verify whether BACKTRACK improves time-to-goal or terminal survival versus painting the pocket first. Train only on recovered, causally outcome-distinguishing pairs, not geometry-defined proxy labels. Then evaluate 36/72/128+ turn performance across fresh DEV families before sealed HELDOUT.
