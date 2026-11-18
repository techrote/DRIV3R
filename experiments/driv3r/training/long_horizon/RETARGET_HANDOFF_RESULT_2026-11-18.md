# DRIV3R RETARGET-HANDOFF-001 — scheduled goal replacement, frozen policy, sparse planner fallback

Date: 2026-11-18. Owned laboratory: `experiments/driv3r/training/long_horizon/`. Upstream research base at setup: `techrote/DRIV3R` Draft/HOLD PR #1, remote HEAD `226bbdad46459642f7a0c4de16ead4ad6087a1fe`. All development and measurements here are synthetic research, **not native CyberSand acceptance**.

## Question and intervention

Does a single sparse planner handoff preserve meaningful neural navigation while rescuing failed spatial retargeting? Two frozen, previously trained NumPy goal-conditioned checkpoints (seeds 711/712, 120 optimizer updates each) start at the **same** forced five-step mistake in a deliberately trapped corridor. At continuation decision 8 an external experimenter replaces the spatial goal without changing paint, material cells, cursor, history, host-legal menu, or model weights. Alternate ordering of targets `(60,48)` and `(55,56)` exposes both directions. All controllers receive the same replacement signal, and any cached route is invalidated; no stale plan is executed after the switch.

Four DEVELOPMENT base families (`pickets`, `ridges`, `islands`, `zigzag`) × five material-screen overlays × two target orders yield 40 candidate scenarios. 2 scenarios were excluded **before controller execution** because one of the original goals was already infeasible under restricted prefix search; all exclusion details are preserved. 38 retained scenario instances × two model seeds × six controllers = **456 full episodes**. Structural HELDOUT remained unopened.

## Results (DEVELOPMENT, externally enforced 64 logical moves)

| Controller | Replacement goal arrivals | Planner searches | Neural-selected actions | Mean continuation decisions |
|---|---:|---:|---:|---:|
| `neural` | 58/76 | 0 | 2498 | 32.87 |
| `late24` | 76/76 | 32 | 1778 | 30.08 |
| `stall4` | 74/76 | 98 | 1586 | 27.47 |
| `switch_plan` | 76/76 | 76 | 608 | 23.63 |
| `switch_stall4` | 76/76 | 152 | 380 | 24.08 |
| `plan_per_goal` | 76/76 | 152 | 0 | 24.45 |

The event-blind **fixed decision-24 handoff rescued all 18 neural-only failures** while requiring only 32 searches across 76 paired controller episodes, compared with 76 searches for immediate handoff on goal change. It retained 1,778 model-selected actions, versus 608 for immediate switch planning. The paired arrival improvement is **+23.68 percentage points**; a *descriptive* bootstrap resampling 38 complete scenario clusters (two neural seeds per scenario) gives a 95% interval of **11.84 to 36.84 percentage points**. Clusters share structural families and are not an independent external population.

The stall-4 strategy reached 74/76 with 98 searches. More aggressive switch-plus-stall reached 76/76 but consumed 152 searches (two per episode). Complete planner supervision also reached 76/76 with 152 searches and **zero neural-selected actions**. All 76 per-arm episodes encountered the scheduled switch; none stopped before it in this selected fixture.

The aggregate choose-time measurements include Python neural inference and planner search but exclude World stepping and receipt serialization. Median per-episode choice latency is **not** an end-to-end controller cost measure. Fixed24 mean cumulative selection time is 4.087 ms per episode; immediate-switch mean cumulative is 1.815 ms, with fewer total decisions. Planner-search count and preservation of neural actions are better interpretable here than comparing these wall-time measurements alone.

## Qualification and evidence

The laboratory generated 14,636 replayed original action receipts (including common five-move preludes) and 6,850 recomputed neural probability vectors across 456 episodes. Each archived episode was **independently re-executed from source and matched checkpoints**, comparing exact legal menus, pre/post state digests, actions, planner decision sources, probabilities, externally commanded goal stages and complete trace hashes. Every recorded trajectory preserved zero protected/material edits and a maximum of 64 synthetic logical moves. All controller combinations passed a second full archive replay, plus focused tampering tests for altered action and changed goal. The full archive remains local pending artifact publication; it is not a native receipt set.

CPU-only run: **17.638s**, peak RSS **98,708 KiB**. New optimizer updates: **0**. Paid API/GPU: none. Checkpoint binaries were not placed in Git history. Gallery source renders a six-panel, same-world, same-model, **common-scale** comparison from verified executed receipts; yellow indicates the original goal, cyan the replacement goal and white the switch position.

Model SHA-256: `711-conditioned.npz` `64643cc987d095a94e3dfcd9b5681377f361c2e0df5363f3027bacbe2b8db7df`; `712-conditioned.npz` `111dba8a604c00497d5c5e5726023f67fc06ea62be77a63001122f9bdca5a9a3`. Source digests are in the accompanying machine-readable evidence ledger. Archives are not GitHub Actions immutable artifacts yet; do **not** represent this run as reproducible from GitHub source alone.

## Limitations and recommended next test

The underlying generator still shares one engineered trapped corridor across all four DEVELOPMENT families; overlay variation is not equivalent to independent topology generalization. Initial feasibility selection restricts the inference population. Goal replacement is an **external synthetic instruction**, not an established native CyberSand ABI. The restricted history-prefix/BFS planner sees complete simulated material geometry and does not guarantee arbitrary-state optimality. Models use handcrafted goal-relative Manhattan features. Fixed decision 24 works here but has no universal safety guarantee. Real changing liquids/materials, true unseen structural families, native game acknowledgements and sustained host-compatible 128-step sessions remain untested.

**Recommendation:** retain sparse delayed fallback as the economical DEVELOPMENT baseline, but do not declare it superior to intervention at real uncertainty events before testing genuinely distinct obstacle geometries, repeated goal changes, changing material legality, stronger planner controls and sealed HELDOUT when a formal protocol is frozen. Keep DRIV3R PR #1 Draft/HOLD.
