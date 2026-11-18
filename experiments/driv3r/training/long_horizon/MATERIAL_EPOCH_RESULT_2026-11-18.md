# DRIV3R / MATERIAL-EPOCH-001 — frozen sparse policies under scripted material changes

**Date:** 2026-11-18 · **Repository:** `techrote/DRIV3R`, Draft/HOLD PR #1 only · **Split:** DEVELOPMENT synthetic · **Status:** negative for unconditional material-epoch plan invalidation in this bounded challenge.

## Purpose and methods

Following the topology-transfer and goal-churn failures, this experiment directly changes the synthetic material occupancy map at scheduled ticks **20 and 36**, while spatial goals reverse at ticks **12 and 30**. It tests whether automatically invalidating a sparse planner cache on every material-state change improves final-goal arrival. All changes are external scripted research fixtures, **not** CyberSand reaction physics. Neither the game nor frozen `d_common` reference interface was modified.

Four predeclared DEVELOPMENT obstacle overlays, two goal orders, four base world families, two independently frozen conditioned checkpoints and five controller arms give **320 executed episodes** (64 per arm). No feasibility filter excludes failed scenarios. Every menu, action, material mutation, goal change and state hash is archived. Model probabilities are recomputed during replay from exact checkpoint binaries held outside Git.

## Controller outcomes

| Controller | Final goals | Planner searches | Neural-selected actions | Applied material patches |
|---|---:|---:|---:|---:|
| Frozen neural only | **32/64** | 0 | 3806 | 256 |
| Fixed tick-24, cache unchanged | **47/64** | 128 | 1536 | 222 |
| Fixed tick-24, invalidate on material epoch | **47/64** | 175 | 1536 | 222 |
| Planner only on material event | **39/64** | 128 | 1664 | 256 |
| Planner each stage and material event | **48/64** | 304 | 0 | 224 |

**Paired interpretation:** material-epoch invalidation changed controller trajectories and increased planner searches from **128 to 175** (+36.7%) versus the otherwise matched fixed-tick policy, but **did not change final-goal completion (47/64 in both arms)**. The always-planning control arrived in 48/64 scenarios, requiring 304 searches. The neural-only control reached 32/64. On these four material scripts, additional automatic replanning has **no demonstrated goal-completion benefit** and uses more planner work. This does not imply that material events are always harmless: closure of a cached next step was not induced in the qualifying samples, and other reaction geometries may require invalidation.

The event-only policy reached 39/64, with 128 searches. External events were replayed only for episodes surviving to the relevant ticks. All applied mutations were material-only and obeyed ink/cursor/protected exclusion; **no synthetic illegal cell edits** occurred. Where an event could not safely close a cell, the engine records suppression rather than overwriting ink (guard covered by a focused test). All observed patches in this run applied; none required suppression.

## Source and proof

- **16,140** independently verifiable synthetic action receipts and **8,542** neural probability vectors across all 320 episodes; complete original JSONL receipt archive retained as deterministic gzip in this PR.
- 64-logical-move external gate; zero native receipts, zero optimizer updates, no new checkpoints, zero paid GPU/API calls.
- Original eight-panel SVG trajectories rendered from receipts, with distinct neural versus planner segments, material edits and goals. These are **not** native gameplay captures.
- Three focused local tests passed, including event pre/post state hashes, ink-safe closure suppression, and tamper detection for actions/material event/probabilities. Independent full-campaign 4-process replay passed in **12.64 seconds**, parent peak RSS **101,392 KiB**, with separate worker processes. This is a replay benchmark, not uninterrupted end-to-end experiment time.
- Exact source and frozen checkpoint SHA-256 hashes, compressed receipt hashes, scenario-level outcomes and explicit caveats are in `MATERIAL_EPOCH_EVIDENCE_2026-11-18.json`. The committed `material_epoch_verify_publication.py` independently verifies the compressed Git receipt archive against the frozen checkpoints; this verification passed for **320/320 episodes**, **16,140** actions and **8,542** probability vectors in **6.612 s** (4-process replay).

## Limits and recommended next test

This is an exploratory DEVELOPMENT comparison, not a structural HELDOUT experiment. All environments share the historical 96×96 synthetic `World` and centrally overlaid masks, engineered Manhattan goal features and restricted prefix planner. Event scripts were manually specified and are not native dynamic materials; closures were not observed to invalidate a cached next action. The failure to improve completion therefore cannot settle the harder stale-route hazard.

**Next:** construct a deliberately moving barrier that reliably intersects a cached future action, record stale-plan rate and event-to-invalidation latency, and compare targeted *on-demand* replanning with unconditional invalidation. Independently test complete host-state reachability on tiny solvable boards before interpreting `prefix_plan` returns of `None` as actual impossibility. Keep HELDOUT sealed pending a freeze and avoid making game changes.
