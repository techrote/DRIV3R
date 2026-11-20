# Corpus Factory v72 — post-fault physical sensor rewitness

**Date:** 2026-11-19. **DATA lane only.** **Host:** synthetic v44 with synthetic v67 erosion and additional hypothetical v72 sensor receipts; **NOT native CyberSand ABI**. **Status:** Draft/HOLD, independent host qualification PASS, no neural training performed.

## Research question

Can a game-policy corpus discriminate *calibration retained from early physical evidence*, *calibration invalidated by a common-bus warning*, *calibration restored for one material by a later physically executed local paint witness*, and *sample source time versus late arrival*? The physical state can stay identical while different observation histories require STOP, inspect/repair A, inspect/repair B, or inspect both.

## Mechanism and experimental controls

- The v44 host accepts every PAINT, ERASE and MOVE. The v67-style physical shock destroys remote A/B paint at a material-specific coupon-derived threshold. Remote shocks never reveal which material was affected.
- Two PRIMARY/SECONDARY raw sensor channels may invert after an isolated-secondary or common-bus warning. Early physical goal paints establish **independent per-material initial calibration**.
- After the warning but before late reports arrive, a scripted arm may physically PAINT_A and/or PAINT_B at the starting hub, observe the PRIMARY channel on an **intact host-observed coupon**, then ERASE it. The resulting postfault witness is valid for that material only. It incurs two extra accepted host actions per material.
- Late measurement (after all disturbances) is delivered before an older measurement. Remote receipts include measurement clocks, while the latest arrival is stale. A common-bus warning invalidates postfault PRIMARY evidence until re-witnessed. Isolated-secondary warnings retain primary calibration.
- Four independent initial calibration states × two fault domains × flip/no-flip × four witness modes = **64 arms** per physical-world group. Instantaneous critical observations match across the arms; witness interventions remain beyond the recent-32-action window. Four physical history groups per world.
- The teacher conservatively revisits unknown or confirmed damaged material, not a privileged actual-painted hidden state. All pre-critical steps (including witness choice) are loss-masked. TRAIN only; DEV, HELDOUT and certificates are evaluator-only.

## Full default corpus

| Split | Unseen worlds | Complete episodes | Host-replayed actions | Matched 64-arm groups | Groups with different correct critical actions |
|---|---:|---:|---:|---:|---:|
| TRAIN | 4 | 1,024 | 157,408 | 16 | 12 |
| DEV | 2 | 512 | 80,480 | 8 | 6 |
| HELDOUT | 2 | 512 | 85,952 | 8 | 8 |
| **Total** | **8** | **2,048/2,048** | **323,840** | **32** | **26** |

**Evaluation-only controls (no neural learning):** instantaneous-observation majority upper bound at matched critical decisions: TRAIN 75.0%, DEV 75.0%, HELDOUT **68.75%**. An aggressive last-source-time PRIMARY policy applying **stale initial calibration** loses at least one actual physical goal in **128 TRAIN + 64 DEV + 64 HELDOUT cases (12.5%)**, inferred from omitted physical repairs. These are *control policies*, not model scores.

### Economic accounting: witness cost is not free

Mean **net** action savings versus unconditional reinspect-both, deducting two PAINT/ERASE actions for each physical witness:

| Split | No witness | A witness | B witness | Both witnesses |
|---|---:|---:|---:|---:|
| TRAIN | 5.750 | 6.5625 | 6.5625 | 7.500 |
| DEV | 6.250 | 6.000 | 8.500 | 8.500 |
| HELDOUT | 9.500 | 14.125 | 11.875 | 15.000 |

**Negative results:** a witness can cost more than it saves. In DEV, A-only witness produces 0.25 fewer saved actions on average than no witness. Individual witness-arm episodes have negative net savings. Choosing when to invest in witness information is therefore a real remaining decision problem, not solved in the scripted prelude. Report witness cost separately rather than presenting inflated gross savings.

## Independent qualification

1. `node physical_rewitness_v72.js NEW_DIRECTORY` fully regenerates host-executed chronological split streams and evaluator-only certificates; destination must not exist. No online API, GPU or paid inference.
2. `node test_physical_rewitness_host_v72.js DIRECTORY` does **not import the generator**. It independently reconstructs v44 world geometry, physical coupons and paint, remote erosion, fault timing, measurement source-time, sensor readings, physically executed rewitness, legal actions, model observations, loss masks, repair route, and terminal success.
3. Two full generations were byte-identical across all seven output files, including all stream and certificate digests. Eight world geometry hashes are distinct across TRAIN/DEV/HELDOUT.
4. A teacher-label tamper with an updated manifest file digest was rejected by the independent host auditor (`teacher labels do not follow physical witnesses`).
5. The evaluator-only scorer `score_physical_rewitness_v72.py` computes no-training shortcut and net action-cost controls directly from certificates; do not provide certificates to a model.

## Limitations and consumer handoff

The host, remote erosion, dual channels, warnings and local-witness telemetry are **synthetic**, not established native CyberSand observations. Only eight unique world geometries were used, and the 112-step detour plus witness actions are scripted rather than learned exploration. A compact symbolic history controller still solves the benchmark. Training a named target SLM is **outside this DATA lane** and has not occurred. Neural closed-loop HELDOUT completion, p50/p95 CPU action latency and RSS, checkpoint hashes, and an immutable GitHub Actions release of full datasets remain outstanding.

**Next consumer:** separate model-trainer lane should train an actual pinned CPU SLM on v72 TRAIN chronological histories, supervise post-critical decisions only, and evaluate sealed DEV/HELDOUT for critical-action accuracy, closed-loop goal retention, net travel cost inclusive of physical witness actions, calibration/event ablation, and explicit SLM latency/RSS. Do not train on certificates or HELDOUT streams. A follow-on data challenge should expose the choice to probe as a controllable action with its actual opportunity cost, and broaden topology/goal-law diversity.
