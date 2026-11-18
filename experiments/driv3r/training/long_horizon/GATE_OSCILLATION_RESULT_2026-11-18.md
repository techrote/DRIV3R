# GATE-OSCILLATION-001 — reversible gate closures, stale cache, and incomplete footprints

**2026-11-18 · DEVELOPMENT synthetic evidence · Draft/HOLD · source-faithful host-menu research, not native CyberSand acceptance.**

## Hypothesis and controlled design

Does immediate route-overlap invalidation actually improve outcomes when material temporarily blocks an already-cached route, and what breaks when the route footprint is truncated?

Use exactly four existing DEVELOPMENT world families with one preselected overlay per family, two opposite spatial goals, and two frozen tiny goal-conditioned rankers (711/712) for neural fallback. Gate cell is chosen from waypoint six of the **initial pre-event** restricted prefix plan, before any controller executes. An unrelated non-overlap material cell is the matched control. Every action is checked against the original synthetic host-legal menu; the 64-logical-move ceiling is external. The native CyberSand ABI is not replaced or modified. The `d_common` fixture remains unchanged.

Event programs:
- **Early reopening:** route gate closes at decision tick 2 and reopens at 4, before the original plan would reach it.
- **Late reopening:** route gate closes at 2 and reopens at 9, after the original plan would attempt the blocked step.
- **Unrelated reopening:** distant material closes at 2 and reopens at 9, with no planned-route intersection.
- **Rapid bounce:** route gate closes/opens/closes/opens at ticks 2/3/4/6.

Policy arms: **blind cached route**, **unconditional invalidation** on every event, **full remaining-route intersection**, **truncated two-waypoint footprint**, **lazy invalidation on next illegal action**, and **frozen neural fallback** after an illegal next action (two seeds). All methods share the same initial prefix-search charge, legal menus and initial world states. Neural fallback is evaluated against the same gates, not trained in this run.

## Executed results

| Arm | Final arrivals | Total initial + follow-up planner searches | Neural-selected actions |
|---|---:|---:|---:|
| Blind stale cache | 16/32 | 32 | 0 |
| Invalidate on every material event | 32/32 | 112 | 0 |
| Full route intersection | **32/32** | 56 | 0 |
| Truncated two-waypoint footprint | 24/32 | 48 | 0 |
| **Lazy next-illegal-action invalidation** | **32/32** | **48** | 0 |
| Frozen neural fallback, two seeds | 56/64 | 64 | 1,068 |

Mode detail for deterministic arms (eight cases per mode):

| Mode | Blind | Unconditional | Full intersection | Partial footprint | Lazy |
|---|---:|---:|---:|---:|---:|
| Early reopen | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Late reopen | **0/8** | 8/8 | 8/8 | **0/8** | 8/8 |
| Unrelated reopen | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Rapid bounce | **0/8** | 8/8 | 8/8 | 8/8 | 8/8 |

**Key falsifications:** detecting a current material overlap is not always worth replanning when the gate reopens before the cached step is due. A two-waypoint cache footprint misses the closing gate and fails all eight late-reopen trials; it is *not* a valid substitute for full-route intersection. Lazy intervention on actual host-menu illegality recovers all 32 deterministic cases with **14.3% fewer searches than full-route invalidation** and **57.1% fewer than unconditional invalidation**. This is engineered bounded evidence, not proof that lazy intervention is universally safe. The unconditionally invalidated policy spends 112 searches vs 48 for lazy on the same 32 matched scenarios.

**Neural fallback:** the two frozen small checkpoints reach 56/64 goals (late gate and rapid bounce each 12/16, early/unrelated 16/16). The neural actions are replayed against saved probability vectors; failures are retained. No checkpoint is trained or changed here.

## Qualification and source provenance

- **224 independently recomputed synthetic episodes**, **5,748 verified host-menu action receipts**, **1,068 recomputed neural action-probability vectors**.
- Seven focused Python unit tests pass, covering material-event schedules, controls, incomplete footprints, heldout rejection, archived replay and forged action/menu/event/probability tampering.
- Local campaign: **11.681 s**, maximum RSS **108,560 KiB**. CPU-only; no paid services.
- Source module dependency Git blob identities were verified at DRIV3R PR #1 HEAD `19397d818af19ecd76d031d92f1bca12c72cde8e` before the experiment; exact source and checkpoint SHA-256 hashes are in the compressed machine-readable evidence.
- Event and receipt streams are complete as compressed immutable files in this same Draft/HOLD PR. Run `python -m experiments.driv3r.training.long_horizon.gate_oscillation_lab verify --out OUTPUT --checkpoints EXTERNAL_CHECKPOINT_DIR` after decompressing evidence JSON and placing the receipts in OUTPUT; the two frozen `.npz` checkpoints are *not in Git*.
- The SVG gallery is derived only from stored executed cursor receipts and event cells, with no simulated/inferred shortcuts.

## Scientific limitations and next step

The scene uses four central overlay masks, a deliberately engineered future-route obstacle, deterministic legal menus and simulated material changes rather than actual reaction dynamics. The planner is a restricted prefix search rather than a complete host-state reachability solver. The neural rankers are existing frozen tiny checkpoints, not d1-2B, LFM2.5-350M or Sprocket-500M fine-tunes. Only DEVELOPMENT was evaluated; structural HELDOUT remains sealed. No native Workbench, game or physics acceptance was performed. Choosing a controller now based on these results risks overfitting.

Next: freeze the three candidate invalidation policies and run a genuinely path-dependent test with ink-history aliasing and surprise occupied-corridor transitions. Explicitly log *when* an action's legality changes relative to replanning, then evaluate on new physical topologies and held-out native read-only receipts. Do not alter CyberSand or its ABI; any proposed game-side observability addition is a separate, non-mutating handoff.
