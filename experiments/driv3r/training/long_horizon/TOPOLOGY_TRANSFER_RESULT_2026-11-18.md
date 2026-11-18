# TOPOLOGY-TRANSFER-001 — route recovery after unseen local obstacle arrangements

**2026-11-18 · DRIV3R Draft/HOLD PR #1 · synthetic DEVELOPMENT experiment · no native CyberSand execution**

## Decision

**Negative transfer:** Oracle-imitation training on two synthetic central-patch motifs did not improve autonomous success on two unseen DEVELOPMENT motifs against the existing frozen goal-conditioned neural policy. The freshly fitted models arrived at **58/64 (90.6%)** goals, compared with **63/64 (98.4%)** for the frozen goal-conditioned controllers. The complete deterministic backtrack-prefix/empty-cell search reached **64/64** goals in fewer actions. A masked-goal ablation reached **0/64** goals. There is no justification to promote the newly trained recovery policy.

This is a falsification of a specific short-horizon imitation-curriculum design, **not** a result about broader 2B/350M/500M neural language models or native DRIV3R actors.

## Controlled intervention

- A source-identified `World` from the existing `world.py` remains the legal action/observation and material fixture. The new laboratory creates a fresh world and deliberately clears its *central 39 × 39 terrain/protection patch*, replacing it with a geometrically explicit obstacle layout. Outside that patch, original base-family terrain is preserved. This is an authored synthetic geometry intervention, **not unmodified native material fidelity**.
- The starting corridor is a cul-de-sac, oriented by a 90° rotation. Preludes force either three or five real, legal synthetic ink moves into the corridor, producing a state that requires history-based backtracking. Goal coordinates are separate from current cursor state. The entire initial world, prior ink, prelude and goal are matched across controllers for each case.
- TRAIN motifs: `ribs`, `shelf`, rotations 0°/180°. DEVELOPMENT motifs: `stagger`, `lattice`, rotations 90°/270°. Four original World base families are represented on DEVELOPMENT, one seed each; the structural HELDOUT (`archipelago`/`staggered_vault`) is **not opened**. The central replacement means the four base-family comparisons are highly correlated. All selected source seeds are even and therefore motif `variation` is zero; this limits seed diversity.
- Only the historical `World.menu()` actions were offered. The external research wrapper enforced at most 64 logical moves before executing any action that would exceed the limit, including the true path-length cost of `RELOCATE_START`. Protected and material painting remained prohibited by the legal menu.
- New tiny `GoalRanker` controllers used the **same existing action/goal feature matrix** as frozen models, without an oracle distance input at inference. Their labels were teacher-provided shortest paths among plans of the form *backtrack zero or more steps along owned ink, then move through previously empty cells*. This is a deliberately restricted deterministic comparator, not exhaustive optimal state-space search or a native planner.

## Real bounded CPU training

Two new random seeds **821** and **822** were each trained with three 64-update Adam phases (192 updates per model), on 36/72/128-labelled curriculum phases. Each phase contained **1,664** legal model-observation training examples; the 26-decision per-trajectory training cap was constant. Thus the stage labels **do not assert actual 72–128 step training episodes**.

| New seed | 36 phase off-policy moves | 72 phase off-policy moves | 128 phase off-policy moves | Final training losses by phase |
| --- | ---: | ---: | ---: | --- |
| 821 | 143 | 148 | 155 | 0.35576 / 0.22081 / 0.26188 |
| 822 | 136 | 139 | 136 | 0.30066 / 0.22110 / 0.23736 |

Each `GoalRanker` contains **769 parameters**. Optimizer updates were genuine NumPy CPU training, not merely labelled synthetic results. The previous models **711** and **712**, each with 120 original optimizer updates, were restored with their original SHA-256 identifiers. Model binaries are kept outside Git in the local evidence archive; immutably hosted Actions artifact publication remains absent.

## DEVELOPMENT autonomous outcomes

**256 complete synthetic episodes**: 64 tasks per controller (2 seeds × 4 base families × 2 DEVELOPMENT motifs × 2 forced detour depths × 2 goals; the test rotation is fixed deterministically by depth and seed). Total independently replayed receipts, including all forced preludes: **11,480**. Original full neural action distributions recomputed and checked: **8,664**.

| Controller | Reached goal | Mean continuation actions | Median model/action selection | Stagger goal arrivals | Lattice goal arrivals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Newly trained goal-conditioned | **58/64** | 38.594 | 0.103884 ms | 26/32 | 32/32 |
| Frozen goal-conditioned | **63/64** | 36.781 | 0.101540 ms | 32/32 | 31/32 |
| Newly trained, goal channels masked | **0/64** | 60.000 | 0.096459 ms | 0/32 | 0/32 |
| Deterministic prefix search | **64/64** | **28.000** | 2.528220 ms | 32/32 | 32/32 |

Paired new/frozen outcomes were **57 both succeeded, 1 newly trained only, 6 frozen only, 0 neither**. Seed 821: new 30/32, frozen 31/32. Seed 822: new 28/32, frozen 32/32. All six new-only failures clustered in `stagger`; the one frozen-only failure was in `lattice`.

The new controller reduced neither failure rate nor action count. Masking explicit goals destroyed completion, confirming that goal features matter in these controlled task pairs; it does not establish route-planning semantics. The apparent base-family sample size overstates independent structural variation, so **no generalization significance claim is warranted**.

## Verification and original trajectory evidence

- Six focused tests passed, including TRAIN/DEV/HELDOUT guards, deterministic controlled starting receipts, distinct motif geometry, restricted oracle feasibility, modified-receipt rejection and full source-exact archived replay.
- Every trace consists of actual synthetic prelude and continuation `World.apply` receipts. For neural controllers, every recorded probability vector, action decision, source-state SHA, legal action menu and final state hash was checked against restored model weights. The replay recomputes the hash manifest and rejected a deliberately tampered action.
- All controllers: **zero protected/material edits**. The original World exposes `RELOCATE_START`; no controller used it in these 256 evaluated episodes. The 64-logical-move cap is external *synthetic enforcement*, not an authenticated native CyberSand host limit.
- Eight evaluation slices consumed a sum of **24.392 s CPU/wall campaign time as separately measured**; their largest observed per-slice process RSS was **97,552 KiB**. Training was executed in independent bounded slices and not included in that 24.392-second value. Median select times are single-policy selection, not host render/physics costs.
- `TOPOLOGY_TRANSFER_TRAJECTORIES_2026-11-18.svg` shows six original case comparisons, each over the corresponding real material/protected-cell pattern. Teal = freshly trained, coral = frozen, dashed slate = deterministic search. Do not interpret the paths as native game footage.

For exact full provenance, action distributions, receipt JSON, checkpoint SHA-256 values and file checksum manifest, see the locally retained evidence ZIP. A concise `TOPOLOGY_TRANSFER_EVIDENCE_2026-11-18.json` ledger is committed alongside source and the report. The source-only GitHub workflow is not equivalent to replaying this full archive.

## Next hypothesis

Avoid another blind increase in model size. Build a new **obstacle-visible, recovery-cost-conditioned** candidate, with explicit local topology observations, legal move budget and a loss that penalizes both futile retreat and missed reachable goals. Predeclare distinct seed-parity perturbations, multiple independent topology mechanisms, and measure actual goal completion and excess steps against both frozen and deterministic planners. Keep structural HELDOUT sealed until the model-selection process is frozen. Only then consider native integration via a read-only interface proposal; never change CyberSand native code from DRIV3R research.

**Governance:** DRIV3R `research/import-consolidated-worm-labs` / PR #1 only, Draft/HOLD, no Ready, no merge, no force push, no external repo mutations, no paid API/GPU. This result does not qualify native inference, ABI equivalence, long-horizon 128-action gameplay or scientific reproducibility from source import alone.
