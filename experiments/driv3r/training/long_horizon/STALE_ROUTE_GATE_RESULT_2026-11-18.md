# DRIV3R / STALE-ROUTE-GATE-001 — material closure along a cached future route

**Date:** 2026-11-18 · **Repository/authority:** `techrote/DRIV3R` Draft/HOLD PR #1 · **Evaluation:** synthetic DEVELOPMENT, not native gameplay or sealed structural HELDOUT.

## Question and predeclared falsification

The prior `MATERIAL-EPOCH-001` reported no arrival benefit from unconditional cache invalidation, but its material changes never closed the next cached action. This experiment explicitly closes **the sixth future waypoint of a cached legal route**, after two actions have been executed (event tick 2), while retaining an independent **off-route material-change control**. The blocker is selected deterministically from the *initial prefix plan*, before observing any policy outcome. It becomes the third future action after the event; the lazy controller first encounters it at tick 5. The distant control blocker is the maximally separated unoccupied central-arena cell, determined from the same original plan.

The original 96×96 synthetic `World`, `strict_step` external 64-logical-move gate, source `prefix_plan`, goal-conditioned checkpoints and four predeclared topology overlays were reused with exact source blob matching. Four DEVELOPMENT world base families were paired with one distinct overlay each, two spatial goals per overlay and both event modes. There are **8 world/goal cases**, **16 material-event cases**, and **96 controller episodes**: four deterministic policies (16 each), plus two independently frozen neural recovery seeds (32 episodes). All initial planner calls were included in the accounting. No unsuccessful scenario was filtered out.

The controllers are `stale_cache` (never invalidate), `epoch_all` (discard on every event), `route_overlap` (discard only if a now-occupied cell intersects the pre-event remaining cached destinations), `lazy_illegal` (replan only when the imminent cached action is no longer legal), and `neural_recovery` (switch to frozen conditioned neural ranking after reaching an illegal cached next action). Every controller uses exactly the same blocker, starting state and initial two actions in its paired case. This is a deliberately intervention-favourable **falsification**, not evidence of naturally occurring material-reaction frequency.

## Actual results

| Controller | Future route obstructed | Unrelated material event | Total arrivals | Total planner searches |
|---|---:|---:|---:|---:|
| Stale cache | **0/8** | 8/8 | 8/16 | 16 |
| Invalidate on every event | **8/8** | 8/8 | **16/16** | 32 |
| Invalidate on cached-route intersection | **8/8** | 8/8 | **16/16** | **24** |
| Replan only when imminent action illegal | **8/8** | 8/8 | **16/16** | **24** |
| Frozen neural fallback, seeds 711 and 712 | 12/16 | 16/16 | 28/32 | 32 |

**A targeted material-event invalidation rescued all 8 paired stale-cache failures.** Equally importantly, it avoided all 8 unnecessary planner searches performed by `epoch_all` after unrelated material edits. In these fixtures, lazy intervention at the actual illegal action also succeeded 8/8, with the same search count but **three decisions later**; future dynamic geometry could change that trade-off.

The frozen neural recovery controller completed 12/16 on-route cases, failing both neural seeds in the enclosed-courtyard/eastern-goal and horizontal-switchback/southern-goal cases. These four failures reached the external decision ceiling without arriving. The neural policies had **534 recorded probability vectors**, not simply heuristic labels. No optimizer updates or new checkpoints were produced. Aggregate results are deterministic on this tiny, designed sample; do not treat 8/8 as an estimated field success rate or claim cross-topology generalization.

## Qualification and reproducibility

- **96 episodes, 2,536 actual host-menu-checked synthetic action receipts, 534 recomputed neural probability vectors.** Source-ledger SHA-256 identifies both frozen `.npz` checkpoints; binaries remain external to Git. Full receipts are stored as deterministic gzip with exact uncompressed and compressed SHA-256 hashes in the evidence ledger.
- Complete local re-execution/replay of all 96 episodes passed. Five focused tests passed, including concrete legal-menu, action, material-event and neural-probability tamper rejection, and a fresh full-campaign rerun/replay. All episodes obeyed the external 64-logical-move ceiling; **zero material/ink/protected-cell collisions**.
- Full single-process campaign ran in **3.924 seconds** at **107,660 KiB peak RSS** (observed in the recorded process). Median per-episode controller-selection calls were approximately **0.005 ms** for cached strategies and **0.067 ms** for neural-fallback episodes; those are local illustrative medians, not native benchmarks.
- The accompanying eight-panel SVG was rendered from the original executed receipt paths and reverified against both frozen checkpoints; it is synthetic research artwork, not a CyberSand native screenshot.

## Interpretation and boundaries

This result **falsifies the assumption that every material change must invalidate a cached route**, while also showing that ignoring a future-route obstruction can be fatal even when the immediate next action is still legal. The relevant distinction is route-state intersection, not merely an incremented epoch. However, the test deliberately derives the obstructing cell from the controller's cached route and all four overlays have centrally cleared regions. These are controlled DEVELOPMENT fixtures; the result does not show how reliably a runtime can obtain all future route footprints or predict reactive physics.

`prefix_plan` is an incomplete restricted backtrack-prefix search, not a full host-state reachability proof. Goal features are engineered Manhattan distances. The neural candidate is a frozen small ranker, not d1-2B, LFM2.5-350M or Sprocket-500M. No true held-out topologies, native CyberSand receipts, GPU training or externally billed inference were used. Import/CI syntax success alone must never be represented as native reproduction.

**Next:** stress selective invalidation when obstacles open and close repeatedly between checks, when the route footprint is uncertain or partially stale, and when multiple painted-history states share the same observable spatial coordinates. Freeze criteria before opening structural HELDOUT. Any canonical host ABI requirement belongs in a read-only interface handoff; no game code changes are authorized here.
