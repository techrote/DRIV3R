# UNSEEN-MASK-STRESS-001 — frozen-policy transfer across four new obstruction geometries

**18 November 2026 · CyberSand-style synthetic DEVELOPMENT research only · DRIV3R PR #1 Draft/HOLD**

## Design and custody

The frozen tiny goal-conditioned checkpoints (`711`, `712`) and five earlier intervention policies were tested without training or trigger retuning. Four new deterministic **synthetic** masks — `broken_diamond`, `offset_comb`, `slanted_fences`, `nested_bays` — were authored before examining any outputs. They are different from the previous four layouts, although they share the source World and the four DEVELOPMENT background families, so this is **not** sealed structural HELDOUT. The sealed `archipelago` / `staggered_vault` split was not opened.

The 96×96 source World, immutable historical action fixture, goal schedule (ticks 12 and 30), goals `(67,48)` and `(46,64)`, model parameters, original policy/controller source and 64-logical-move gate were unchanged. Synthetic overlays clear only the central 31×31 test arena before placing obstacles. All original background material remains outside it. The research harness substitutes a distinct overlay builder **only within this local test context**; it never changes `world.py`, the DRIV3R common interface, CyberSand, or another worker's files.

All 16 initial (background × mask) configurations have *both* targets reachable individually under the restricted prefix planner. Unlike earlier studies, no scenarios were excluded after feasibility inspection. Goals are reversed twice at fixed ticks; earlier edits/history remain, so the second goal may become unreachable under host restrictions.

## Controlled results (64 episodes per arm)

| Frozen controller | Reached final goal | Searches | Neural-selected actions |
|---|---:|---:|---:|
| Neural only | **16/64** | 0 | 3,905 |
| Fixed tick-24 handoff | **41/64** | 115 | 1,536 |
| Planner every stage | **40/64** | 176 | 0 |
| Second-switch handoff | **37/64** | 64 | 1,920 |
| Second-switch stall ≥3 | **19/64** | 64 | 2,292 |

**Negative transfer finding:** the previously leading second-reversal handoff reached 37/64, below the fixed-tick handoff's 41/64, reversing the ranking on these new synthetic masks. These are 64 paired scenarios, not 64 independent model seeds. No statistical generalization claim is made.

### By unseen mask, out of 16 matched scenarios

| Mask | Neural | Fixed tick 24 | Planner every stage | Second reversal | Stall ≥3 |
|---|---:|---:|---:|---:|---:|
| Broken diamond | 3 | **16** | **16** | 12 | 4 |
| Offset comb | 5 | **9** | 8 | **9** | 5 |
| Slanted fences | 8 | **16** | **16** | **16** | 10 |
| Nested bays | 0 | 0 | 0 | 0 | 0 |

The fixed-tick policy rescued 25 neural failures with zero paired regressions; second-switch rescued 21 with zero paired regressions. Always-planning rescued 29 but regressed on five neural successes. These are descriptive paired outcomes, not prospectively estimated treatment effects.

## Failure audit: path-dependent continuation infeasibility

`nested_bays` is not merely a bad neural score: all 16 initial situations permit a restricted path to each goal, yet the staged planner's first trajectory can leave the *replacement* target without a legal 64-move path. At tick 12, **16/16** stage-planner traces were already infeasible under the exact restricted prefix search, so that controller issued STOP and never reached tick 30. For the tick-24 controller, 12 of 16 were infeasible at tick 12, and all three cases reaching tick 30 were infeasible there. The second-reversal controller reached tick 30 in all 16 nested-bays scenarios, with **0/16** legal restricted continuations at that switch.

This audit does **not** prove global impossibility: the `prefix_plan` oracle searches backtrack-prefix followed by fresh empty cells and is not a complete search over all host-action histories. It does show that comparing controller success without conditional feasibility accounting can misdiagnose reachability failures as policy incompetence. A genuine complete host-state planner or on-policy recovery learned on independent topology is required.

## Verification and boundaries

- **320** fully executed episodes, **15,132** validated synthetic action receipts, **9,653** neural probability vectors.
- Replay of every trace succeeded *twice*: once during campaign generation and again from retained receipts. Three separate source tests passed, including a forged-menu rejection test.
- Zero illegal protected-cell/material edits, zero native CyberSand receipts, zero newly trained optimizer steps. Every episode respected the synthetic 64-logical-move ceiling.
- Measured campaign 32.24 seconds wall time, peak process RSS 99,092 KiB (local CPU). Action-choice median values and checkpoint SHA-256 hashes are preserved in the machine ledger. These are Python research-runtime timings, not deployment latency estimates.
- Visual gallery is rendered from the actual archived action receipts; no fabricated trajectories.

**Critical limitations:** static overlays only (no reactive material-state changes yet); engineered goal-distance inputs; four predeclared masks sharing backgrounds; potentially incomplete planner; repeated training-family structural patterns; no sealed HELDOUT or native validation; model binaries locally archived, not yet immutable GitHub Actions artifacts. An import plus these CPU tests does not establish scientific reproducibility of the larger model families.

## Next falsification

Freeze the policies and test genuinely state-changing material events with explicit external intervention receipts, plus a complete bounded state-space planner that can distinguish path-history impossibility from poor intervention timing. Do not tune to the sealed HELDOUT. Keep evidence and source under this lab in the existing DRIV3R Draft/HOLD PR.
