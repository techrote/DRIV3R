# DRIV3R GOAL-OBSERVABILITY-001 — host-bounded sequential goal changes

**Status:** negative source-matched CPU finding. **Scope:** synthetic DEVELOPMENT only. **Repository:** `techrote/DRIV3R`, existing Draft/HOLD PR #1 only. **Starting source HEAD:** `2972d20ec4d7fa51adb749e81350de48a8c398ec`; the shared `world.py`, `policy.py`, and `train.py` were Git blob-verified against that commit before evaluation. No foreign repository changes.

## Scientific question and negative result

Do two existing 128-stage long-horizon tiny policies make useful host-bounded choices when their external goal changes while the materials, protected cells, painted history, cursor and action menu remain fixed?

**They do not.** Existing `policy.FEATURES` has no target/goal channel and `Ranker.choose(World,horizon)` receives no external objective. The same observed world under two mutually exclusive one-step goals necessarily gives exactly the same model distribution. Across **286 model-state paired interventions** (572 goal conditions), the frozen models chose the correct one-step goal action **261/572 = 45.63%**. The correct action under each goal belongs to the actual host-legal `World.menu`. Any deterministic *goal-blind* policy, regardless of model size, can be correct on **at most 50%** of these equally weighted contradictory pairs. The observed 45.63% includes some non-moving choices. This is an interface/observability failure, not a hypothesis that a different checkpoint alone fixes goal-conditioned behaviour.

The episode-level task used 12 existing DEVELOPMENT worlds (`pickets`, `ridges`, `islands`, `zigzag`; three seeds per family) with original, source-fidelity-generated legal routes, marked every four moves into 4–11 waypoints. All controllers received the **same reference waypoints per world**, switched objectives when the current waypoint was attained, and retained the original materials, protected cells and legal menu. A goal-aware deterministic BFS/stack oracle served as a capability control; it is **not a learned controller** and it is **not a native CyberSand provider**. Both existing tiny checkpoint seeds (`1051` and `1052`) were restored and evaluated without optimizer updates.

## Original autonomous results

| Controller | Ordinary goals reached | Legally perturbed goals reached | Combined |
|---|---:|---:|---:|
| Deterministic goal-aware BFS/stack oracle | 97/122 | 86/122 | **183/244** |
| Persistent-random legal-action comparator | 7/122 | 2/122 | **9/244** |
| Frozen tiny Ranker, optimizer seed 1051 | 1/122 | 0/122 | **1/244** |
| Frozen tiny Ranker, optimizer seed 1052 | 0/122 | 0/122 | **0/244** |

The goal-aware oracle did not achieve 244/244: some constructed future destinations became inaccessible after earlier path painting, or after injected wrong turns. This is a meaningful weakness of the reference task, **not** proof of universal path-planning success. Goal-blind policies may inadvertently encounter a waypoint but cannot condition their choices on its location.

For every episode, the external harness terminates before executing a decision that would exceed **64 synthetic logical moves**. It leaves `World.menu`, `World.apply`, recovery permissions, STOP availability and action receipts unchanged. The 64-move cap is **a synthetic host-side gate** informed by historical native contract evidence, not native execution or native enforcement. A route ending at the cap is recorded as censored rather than a successful 72/128 native run. Existing `RELOCATE_START` is retained in World; its use is separately counted as an **unvalidated synthetic operation**. The reference goal oracle never chooses it. No protected/material edits occurred.

Ordinary+perturbed combined: 96 episodes, **5,330 actual synthetic legal decisions**, 3,072 separately recomputed neural menu-probability vectors, 286 exact-state paired objective interventions. The neural policies each used all 64 logical moves in all 24 of their episodes, without finishing the objective series. Oracle ordinary episodes finished all goals on 7/12 worlds; perturbed episodes on 5/12.

**Action counts across all 96 episodes:** MOVE_E 1,916; MOVE_S 1,326; MOVE_N 1,088; MOVE_W 816; BACKTRACK 168; STOP 12; synthetic RELOCATE_START 4. No editing of solid/material/protected cells. All complete trajectories and per-action neural distributions are retained outside Git in the attached SHA-indexed checkpoint/trace evidence archive.

## Qualification, provenance, resource bounds

- Six source-only tests passed. A separate verifier reconstructed and replayed **96/96** original episodes, **5,330/5,330** original legal action receipts, and **3,072** original neural choice distributions from restored SHA-256-pinned checkpoint files. It also checked all 286 paired interventions, archive hashes, and zero protected/material edits.
- CPU wall time: **7.646 s** for the original evaluation; peak process RSS **114,844 KiB**. Median per-episode neural selection latency: approximately **0.125 ms**. The oracle runs a bounded Python BFS, so its lower reported decision latency is not a fair neural throughput comparison.
- Checkpoints: optimizer seed 1051 (`aeeb475f0db066ac3745f83c84d43fc2ef3720a7900814d8cbb4ed8e1777567f`), seed 1052 (`20ffbd90a173d6c7f98f5e6b0dd45769aa19066e32331c8d7ce4c85d50896471`). Each checkpoint contains 625 real NumPy parameters and 192 cumulative optimizer updates inherited from the prior experiment. **New updates this run: zero.** No paid API/GPU.
- TRAIN models evaluated against DEVELOPMENT worlds only. No fresh optimizer training, structural HELDOUT opening, d1/LFM/Sprocket comparison, immutable Actions checkpoint artifact, or native CyberSand acceptance.
- The trajectory gallery is drawn from original executed cursor receipts, with material/protected masks, original reference goals, and arrived/not-arrived indicators. It is **not** native screenshots and not AI concept art.

## Next bounded experiment

A genuinely **goal-conditioned** small ranker should receive a host-supplied, explicit target channel without changing host-legal menus. Train with counterfactual paired objective labels, curriculum escalation and realistic off-path corrections. Compare its autonomous waypoint arrivals to (a) the frozen goal-blind checkpoints, (b) persistent random, and (c) the deterministic BFS/stack baseline on **new DEVELOPMENT** routes, using the same 64-logical-move synthetic host gate. Preserve a sealed structural HELDOUT, and do not claim native gameplay compatibility without CyberSand-side acceptance.

This zero-update diagnostic has a concrete negative finding; it is not a replacement for the longer-horizon training mission and does not justify additional model size by itself.
