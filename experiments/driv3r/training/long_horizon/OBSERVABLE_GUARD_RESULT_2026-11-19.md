# OBSERVABLE-GUARD-006 — route-distance guard negative transfer (2026-11-19)

## Objective and controls

Test whether a **source-replayed, host-observable** current-material gate guard can salvage the fixed-patience failures in PATIENCE-TRANSFER-005, without using privileged future-gate states. The policy initially keeps its cached route after an observed closure and rechecks the **current** synthetic material cell each decision; if it is still closed and the blocked cell is within 1, 2, 4, 8 or 12 cached moves, invalidate and replan. Compare with conservative immediate route overlap and prior TRAIN-selected patience. Development topology, reopen ticks and matched seeds are unchanged from transfer predecessor. TRAIN seeds 1000–1047 choose the highest mean utility guard independently per 36/72/128 horizon. The candidate family includes immediate overlap, and the choice is frozen before evaluating DEV 3000–3015.

## Exact experimental outcome

- Source-V8 execution: **4032 synthetic DEVELOPMENT policy episodes**, **245851 independently source-replayed action receipts**, **18 assertions**. V8 elapsed 215696 ms; this is not Node or per-inference latency, and no RSS measurement is available in this run.
- TRAIN choice: **36→overlap, 72→overlap, 128→overlap**.
- DEVELOPMENT immediate-overlap: **140/192 goals**, utility 603.302; selected guard: **140/192**, utility 603.302.
- guard1: **90/192** DEV goals, utility 341.542; paired goals lost **0**, gained **0** versus immediate overlap.
- guard2: **94/192** DEV goals, utility 362.427; paired goals lost **47**, gained **1** versus immediate overlap.
- guard4: **100/192** DEV goals, utility 393.479; paired goals lost **41**, gained **1** versus immediate overlap.
- guard8: **97/192** DEV goals, utility 377.432; paired goals lost **44**, gained **1** versus immediate overlap.
- guard12: **95/192** DEV goals, utility 366.297; paired goals lost **46**, gained **1** versus immediate overlap.

**Decision:** observed gate presence plus proximity, by itself, is inadequate for reliably replacing immediate route-overlap replanning under the tested topological/reopening shift. All TRAIN cohorts chose the conservative overlap fallback, reproducing its DEV goal count. Preserve the guard failures as negative cases rather than claiming learned recovery improvement. Follow-on should test richer observation-history and *actual* recovery after a bad earlier move or material mutation, not another uncalibrated timer.

## Reproduction and caveats

`node experiments/driv3r/training/long_horizon/observable_guard_lab.js /tmp/driv3r-observable-guard`

The runner pins all three upstream Git blobs, rejects source drift, checks synthetic legal menus/state receipts and tampering, and when run under Node records wall time and maxRSS. It is **explicit opt-in CPU work** only; there are no automatic training or GPU steps. Development is not structural HELDOUT, which remains sealed. 128-step `shadow128` outcomes are not native 128-move results: native-proxy paint count stays capped at 64. The patched JS synthetic model is not a Python cross-implementation replay and cannot establish CyberSand native game acceptance. Other original negative cases are retained unchanged.

Evidence: `OBSERVABLE_GUARD_EVIDENCE_2026-11-19.json` (aggregate action distributions, matched failure cases, source replay counts) and `OBSERVABLE_GUARD_PATHS_2026-11-19.svg` (synthetic trajectories).
