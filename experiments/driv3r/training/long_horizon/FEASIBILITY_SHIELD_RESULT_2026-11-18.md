# FEASIBILITY-SHIELD-001 — conservative neural recovery under an explicit 64-move gate

**2026-11-18 · DRIV3R Draft/HOLD PR #1 · synthetic DEVELOPMENT evaluation · no native CyberSand execution**

## Decision: negative for continuous planner supervision; positive bounded rescue

The existing frozen goal-conditioned ranker failed 10 of 32 development trap-corridor recoveries. An always-on restricted planner feasibility shield rescued all 10, but this required **2,032 planner searches**, compared with **338** searches for a simple 24-decision handoff that also completed **32/32**. A planner-only controller completed **32/32** with **624** search calls and fewer logical actions. These are **costs and outcomes of this controlled synthetic benchmark**, not evidence that a production policy or host contract should adopt any of these controllers. Continuous prospective search is not justified by these DEVELOPMENT cases.

The result is falsification of the proposition that continuous feasibility lookahead is needed to rescue frozen navigation in these simple corridors. It does **not** show that timed handoff generalizes to different worlds, nor that the heuristic is a complete native path planner.

## Controlled setup and provenance

- Reused exact source-hash-matched `world.py`, `goal_conditioning_lab.py`, `goal_observability_lab.py` and `detour_recovery_lab.py` from the current research branch. Their Git blob hashes were verified against the remote exact branch before running. No modification to those source fixtures.
- Loaded frozen ranker checkpoints 711 and 712 with 120 training updates each; **no new optimizer updates**. Their SHA-256 values and source hashes are recorded in the evidence ledger and checked during replay.
- The 32 matched case identities comprise 2 frozen seeds × 4 DEVELOPMENT world families (`pickets`, `ridges`, `islands`, `zigzag`) × 1 DEVELOPMENT seed per family × 2 forced-prelude depths (3 and 5) × 2 goals. All controllers see the exact same game state, protected cells, ink history, goal and legal menu per case.
- The fixture uses `trap_corridor`, an artificial, purpose-built cul-de-sac. Action menus, painting, `BACKTRACK`, material exclusion and receipt logging use the existing synthetic `World` implementation. No `RELOCATE_START` was executed.
- The 64-logical-move limit is **externally enforced by `strict_step`**, not the actual native runtime. The structural HELDOUT (`archipelago` and `staggered_vault`) remained sealed.

## Results (32 matched episodes per controller)

| Controller | Goal arrivals | Mean continuation actions | Neural-selected actions | Planner-selected actions | Planner calls | Mean per-episode median selection ms |
|---|---:|---:|---:|---:|---:|---:|
| Frozen neural | 22/32 | 37.312 | 1,194 | 0 | 0 | 0.10141 |
| Always restricted prefix search | **32/32** | **19.500** | 0 | 624 | 624 | 1.24683 |
| Handoff at decision 24 | **32/32** | 34.125 | 754 | 338 | **338** | 0.22375 |
| Lookahead shield, zero slack reserve | **32/32** | 36.375 | 880 | 284 | **2,032** | **3.66633** |
| Lookahead shield, eight-move reserve | **32/32** | 34.375 | 840 | 260 | **1,928** | **3.37055** |

All 32 planner-start states were solvable within 64 logical moves under the **restricted** planner. The feasibility shield simulates a prospective neural action and calls the restricted prefix search again to determine if the goal remains attainable inside the remaining budget. When the proposed action fails this test, it commits to the planner. It is intentionally privileged: it reads the complete synthetic material/ink grid and ordered history, which are not demonstrated to be available to a deployed CyberSand neural action API.

**Cost accounting matters:** selecting 880 neural actions does not mean only 284 planner operations; shield0 made 2,032 full path searches, typically including one or two for each neural action. For the same 32/32 arrivals, a simple fixed handoff was substantially cheaper in this particular corridor. Planner-only was shortest in actions. The neural and hybrid paths remain more meandering, but this is not a measured benefit unless the gameplay objective rewards exploration.

## Audit, latency and failure modes

- 160 complete episodes; **5,814 replay-verified synthetic legal action receipts**, including forced preludes; **3,688 restored-checkpoint probability vectors**; 161 output files with SHA-256 manifest.
- Five focused tests passed, including wrong-action and neural-probability tampering rejection. Independent complete archive replay passed against the final source. Each trace stores source spec, prelude, state and final hashes, full legal menus in the receipts, chosen actions and probabilities.
- CPU campaign **31.885 seconds**; peak RSS **97,940 KiB**. The per-controller selection measurements include planning and Python overhead, are measured on one local CPU, and are not native deployment latency.
- Synthetic legal menus were respected and no protected/material cell was edited. The original gallery is rendered from recorded receipt cursor positions (no invented or shortened path segments).

## Explicit limitations and next experimental gate

1. The same central corridor challenge persists across the four nominal families; this is a strongly correlated DEVELOPMENT benchmark, not structural transfer or statistical proof. The recent separate topology-transfer experiment reported a negative new-model transfer result.
2. The prefix planner is complete only for a monotone BACKTRACK prefix followed by empty-cell traversal; it is not complete for every legal action sequence. It uses privileged full-world information.
3. A 24-decision handoff is tuned to **this observed problem family**. Its perfect score cannot justify use in new maps, dynamic physics or native gameplay. Different horizons or topologies could reverse the conclusion.
4. Frozen rankers, not d1-2B/LFM2.5-350M/Sprocket-500M models, were evaluated. No structural HELDOUT or native action receipts were opened; no model was trained here.
5. Binaries remain in a verified **local evidence archive** rather than immutable GitHub Actions artifacts. Source-smoke CI alone is not a full model replay.

**Next:** falsify timed-handoff success on genuinely different legal topology families and changing material/goal states, with a bounded, sparse (not every-step) planner watchdog and a deployment-realistic observation budget. Leave game-side integration to a read-only interface handoff proposal; do not change CyberSand.
