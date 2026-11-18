# GOAL-CHURN-001 — two scheduled goal reversals with sparse planner intervention

**18 November 2026 · DRIV3R Draft/HOLD PR #1 · source observed at start `3cc747281f4187`**

## Decision

**Positive for sparse planner rescue, not evidence of generalized autonomous navigation.** In 48 paired DEVELOPMENT synthetic scenarios, frozen goal-conditioned neural control completed the final reversed target in **34/48** cases, whereas all five planner-assisted arms completed **48/48**. The simplest absolute-decision-24 handoff needed **48 planner searches**, versus **96** for a handoff eight ticks after each switch and **144** for planning each stage. It retained **1,122 neural-selected actions**, but the stalled-decision arm retained **1,432 neural actions** at the cost of **75** searches and longer paths. No method was newly trained.

The absolute24-over-neural paired improvement was **14/48 (+29.17 percentage points)** with no regressions. A descriptive four-world-family cluster bootstrap interval was **8.33 to 54.17 percentage points** (10,000 seeded resamples). This is *not* a population-level or independent-world confidence statement; the worlds share one synthetic trap-corridor design.

## Executed protocol

- Frozen source-matched two model seeds **711 and 712**, checkpoint SHA-256 values in the evidence ledger; DEVELOPMENT-only four geometry families × three overlays (`original`, `east_sleeve`, `staggered_piers`) × both goal orders × two seeds × six policies = **288 episodes**, including **48 matched scenarios per policy**. No scenarios were excluded for initial unattainability.
- Initial controlled prelude: five legal eastward paint/movement decisions on a synthetic trap-corridor World. Subsequent action menus are always obtained from the immutable historical World `menu(12)` and executed with `strict_step`, enforcing the external **64 logical-move ceiling**.
- Three target stages: `A → B → A` or `B → A → B`; externally scheduled replacement at decision ticks **8 and 24**. World state, materials and painted history remain unchanged by goal replacements. On early goal arrival, the controller idles *outside* the action interface until the next scheduled replacement; it never fabricates an illegal WAIT action.
- On every retarget event, cached planner paths are discarded even if a previous stage's path was unfinished. Each planner arm is capped at one restricted plan search *per stage*. Planner control is not a complete native optimal-control search: it considers a backtrack-prefix followed by empty-cell BFS.
- Independent replay: all **11,302 continuation action receipts**, **6,292 recorded neural probability vectors**, controller decisions, switching events, SHA-pinned source/checkpoints, and per-episode hashes verified. Seven focused unit tests pass, including action, probability, and switch-state tamper rejection.

## Comparative results

| Policy | Final target arrivals / 48 | Planner calls | Neural-selected actions | Mean executed continuation actions |
|---|---:|---:|---:|---:|
| Goal-conditioned neural only | 34/48 | 0 | 2202 | 45.875 |
| One planner handoff at absolute tick 24 | 48/48 | 48 | 1122 | 38.583 |
| Planner after eight ticks in each stage | 48/48 | 96 | 1152 | 37.000 |
| Planner after four stalled decisions | 48/48 | 75 | 1432 | 41.208 |
| Planner on both goal replacements | 48/48 | 96 | 384 | 36.708 |
| Planner at all three stage starts | 48/48 | 144 | 0 | 36.083 |


**Qualification:** All **288** paired episodes saw both target replacements. No action entered a protected or occupied material cell, and all synthetic logical-move counts stayed at or below 64. All original move history and receipts are retained in the external evidence archive. Campaign CPU wall time **18.534s**, peak process RSS **98068 KiB**, median per-episode controller-selection latency for the absolute24 arm **0.097534 ms**. Local Python measurements only, not native CyberSand latency.

The 6-panel SVG gallery derives **only from archived executed action receipts**, preserves the complete forced prelude and both switch markers, and fixes the same crop for all six arms; it is an illustration of one prespecified comparable failure/rescue scenario, not a replacement for the full comparative ledger.

## Scientific cautions and research consequence

This is a controlled extension of the single-retarget experiment, not evidence of transfer to unfamiliar topology. All geometry families share the same trap-corridor intervention and a frozen material map. External goal replacement and external idling are synthetic capabilities, not proven native CyberSand features. The model sees engineered goal-distance features. The restricted planner has privileged full material geometry and does not solve arbitrary path-planning with painted cells. Neither structural HELDOUT nor native acceptance was opened. The two checkpoints were pre-existing; **optimizer updates: 0**. Full model/replay binaries are retained in a locally verified evidence ZIP rather than an immutable GitHub Actions artifact. CI source-smoke (if run) should not be mistaken for this full verification.

Next bounded trial should use **genuinely non-isomorphic obstacle topologies and changing material state**, with a stronger complete legal-action planner, to determine whether a one-search delayed handoff remains reliable. Preserve weird neural exploration as a separate score (path shape, turns and motif diversity) rather than treating 100% goal arrival as the entire objective.

**Ownership:** Only `experiments/driv3r/training/long_horizon/**` changed in DRIV3R PR #1. Do not merge/promote, change DRIV3R main, mutate CyberSand/Neurounner, or trigger paid compute.
