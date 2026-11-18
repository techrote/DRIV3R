# DETOUR-RECOVERY-001 — frozen goal policies after identical painted dead-end mistakes

**18 November 2026 · DRIV3R Draft/HOLD PR #1 · synthetic DEVELOPMENT-only falsification.** Starting remote research branch HEAD (inspected before editing): `f9dc08cf18b98e28927b61639d986a2cfa612ac9`. No CyberSand, Neurounner or DRIV3R main writes. The historical `d_common` interface and all existing World/model implementations are unchanged.

## Research question and result

Can the four existing *frozen* tiny goal-navigation checkpoints, previously successful on easier source-derived waypoints, recover from an **identical, externally forced wrong move** into an obstructed painted tunnel and reach a spatial objective across the barrier, within an external 64-logical-move gate?

**Partly.** The goal-conditioned controller arrived in **80/96 (83.33%)**, substantially better than matched goal-blind **10/96 (10.42%)**, but worse than both legal deterministic search controls **96/96 (100%)**. This is a **real negative transfer result for reliable recovery** on the new synthetic challenge, not evidence that a more powerful optimizer is inherently required.

| Controller | Goal arrivals / 96 | Mean post-error decisions | Median per-episode model/planner choice time | Action distribution (post-error) |
|---|---:|---:|---:|---|
| Goal-conditioned frozen neural | **80/96** | 32.104 | 0.1015 ms | {'BACKTRACK': 480, 'MOVE_E': 1226, 'MOVE_N': 186, 'MOVE_S': 596, 'MOVE_W': 594} |
| Goal-blind matched frozen neural | 10/96 | 56.750 | 0.0906 ms | {'BACKTRACK': 482, 'MOVE_E': 1768, 'MOVE_N': 532, 'MOVE_S': 1678, 'MOVE_W': 988} |
| Existing single-position BFS/stack heuristic | **96/96** | **19.500** | 0.0745 ms | {'BACKTRACK': 288, 'MOVE_E': 912, 'MOVE_N': 96, 'MOVE_S': 480, 'MOVE_W': 96} |
| Exhaustive minimum-cost monotone-prefix recovery + empty-cell BFS | **96/96** | **19.500** | 1.2618 ms | {'BACKTRACK': 288, 'MOVE_E': 912, 'MOVE_N': 96, 'MOVE_S': 480, 'MOVE_W': 96} |

Conditioned results by seed: **40/48 seed 711, 40/48 seed 712**. Blind results: **4/48 and 6/48**. No arm edited protected/material cells or invoked synthetic `RELOCATE_START`. No policy STOP was selected; failures exhausted the external move gate.

World-cluster bootstrap (12 DEVELOPMENT world seeds, resampled 10,000 times, deterministic bootstrap seed 23451): conditioned minus blind **+72.92 percentage points**, descriptive 95% interval **[58.333, 87.5]**. Conditioned minus legacy BFS **−16.67 points**, interval **[-29.167, -6.25]**. These are paired *within fixture family*, not a generalization confidence interval to unseen CyberSand maps.

## Design and controls

- Two frozen checkpoint seeds **711/712**, with `GoalRanker` **769 parameters** and **120 prior real Adam updates each**; **zero new training**. Original checkpoints restored from the previously qualified GOAL-NATIVE-GATE evidence archive, SHA-256 recorded in ledger.
- **12 DEVELOPMENT specs** covering `pickets`, `ridges`, `islands`, `zigzag`; **two wrong-move depths (3, 5)**; **two free spatial goals** `(60,48)` and `(55,56)`; **four arms**, **384 episodes**. Historical `trap_corridor` is an existing source-defined challenge, not a new game/runtime modification.
- Every arm starts from the exact same replayed World state for each seed/spec/depth, after exactly 3 or 5 forcibly applied and fully acknowledged `MOVE_E` events into the blocked eastward tunnel. Material, painted history, protected zones, goal, cursor and legal menu are held fixed across controllers.
- All goal targets were checked empty and the exhaustive monotone-prefix planner found a feasible path within the 64-move gate requiring BACKTRACK. The planner enumerates every affordable stack-prefix retreat followed by a shortest path through currently empty cells; it is **complete for this restricted strategy class**, not a complete search over arbitrary future painting/recovery states.
- The existing BFS/stack heuristic also solves **100%** of this fixture, so this particular intervention does not discriminate better planning from a simple avoid-occupied rule. **This is a benchmark limitation** and argues against claiming a new planner breakthrough.

## Evidence and qualifications

Full local campaign: **384 episodes, 13810 action receipts including forced interventions, 8530 neural probability vectors**, SHA-pinned trace manifest; controller proposals and legal-menu action receipts were checked twice (once during construction and again by replay from archived file). All traces replay without protected/material edits.

- **5/5 focused tests passed**, including optional full-archive receipt and probability verification; tampered action receipt is rejected. Source-only run supports 4 tests and skips the archive-dependent fifth.
- CPU execution **27.049 s**, peak process RSS **98,832 KiB**; measured action selection in the table is local CPU time, not native model throughput.
- Six-panel original SVG shows actual replayed cursor paths, not imagined trajectories; same fixed 5-step error is dashed yellow and the goal is a diamond. The complete 384 episode transcripts, checkpoint binaries and per-file hashes are retained outside Git history.

## Critical interpretation and next decision

This challenge is deliberately *out-of-distribution*: trained models saw regular `challenge=normal` maps; all 12 nominal geometry families share the **same purpose-built local corridor shape**. Success differences are still meaningful for controlled recovery within this scenario, **not evidence of cross-geometry generalization**. No structural HELDOUT was opened; no native CyberSand actor, host epoch, game simulation or authoritative receipt was exercised. The 64 move cap was externally imposed in the synthetic harness. Existing `World` menus include a synthetic `RELOCATE_START` not validated as a native action, though none of these four arms used it.

**Next:** diversify start states and obstruction/topology beyond the identical central tunnel before training. For genuinely difficult setups where the original BFS heuristic fails, evaluate budget-aware goal conditioning and planned recovery against a complete state-graph oracle *and* matched frozen controls. Do not use these results to promote a game-side policy. Immutable GitHub Actions checkpoint-artifact publication remains outstanding; no paid APIs or GPU charges.
