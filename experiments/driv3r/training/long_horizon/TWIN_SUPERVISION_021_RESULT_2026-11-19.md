# TWIN-SUPERVISION-021 — bounded paired-fork physical-inspection training

19 November 2026. DRIV3R long-horizon Draft/HOLD; synthetic DEVELOPMENT evidence only.

## Question and method

Can a four-class student trained on observable prefix features improve physically verified recovery of erased material under a 64-logical-move censor? We use the pinned PHYSICAL-INSPECTION-016 host, MULTIPULSE-PHYSICAL-018 material interventions and STOP-RESERVATION-020 controller. Synthetic STOP terminal accounting is **not** established as a native CyberSand ABI rule. There are no hypothetical offscreen damage notifications.

Pinned dependency Git blobs: 016 = 45a426f893203f93fa9a3972d63bd4a110ceaa3d; 018 = 4e7f4b4b20d4db42d601961a7689f3729477aaf6; 020 = 51b43ff4f31299f9ed0b0ff8a9a559b1e9a0a79d.

TRAIN seeds 120100–120169 across 36/36, 72/64, 128/64 and 128/128 gave 280 training situations. Four candidate modes (feasible, late, early, both) were scored across paired hidden material interventions (3,360 TRAIN candidate rollouts), producing 245 feasible and 35 early labels. A four-class softmax was fitted with **1,100 optimizer updates**. Frozen 018 model is an independently trained prior control.

The teacher's label selection is **ex-post counterfactual simulation**, not a host-observable signal. The fitted student sees only public prefix observations plus known static topology and cap. No hidden damage, future outcomes or DEVELOPMENT labels enter its features.

## DEVELOPMENT results

Each batch contains 50 new world-seed clusters, four horizon/cap cohorts and 600 paired conditions. These correlated conditions are **not 600 independent worlds**.

| Controller | Fresh terminal goals / 600 | Shifted terminal goals / 600 |
|---|---:|---:|
| Feasibility + reserved STOP | 420 | 426 |
| Late inspection + STOP | 420 | 426 |
| Early inspection + STOP | 313 | 328 |
| Both inspections + STOP | 413 | 428 |
| Frozen 018 student + STOP | 422 | 440 |
| **New trained 021 student + STOP** | **441** | **440** |
| Ex-post twin-balanced supervisor | 441 | 450 |
| Hidden-intervention-aware supervisor | 441 | 450 |

At **128/64 censored** (200 correlated intervention records per batch), new student: **70/200** versus feasibility **49/200** on fresh worlds; on shifted obstacles **62/200** versus **48/200**. At 128/128 uncensored shadow, all four stronger STOP-reserved controllers reached 200/200 per batch. That shadow success must not be conflated with native 64-move acceptance.

The student used **2,938 planner calls vs 2,854** for feasibility on fresh worlds, and **2,936 vs 2,880** shifted. Better goal completion did not lower planning overhead. Mean synthetic utilities were 585.090 fresh and 584.193 shifted for the student, versus 551.450 and 561.233 feasibility. On shifted worlds, twin-aware hindsight found 10 additional completions beyond the trained student. Privileged intervention knowledge raised utility without further raising aggregate goals.

## Qualification

The run covered **1,200 paired DEVELOPMENT conditions**, **5,238 unique policy rollouts** and **301,050 pinned-host-replayed action receipts**. A separately authored receipt-field validator checked all 301,050 for tick, action-menu, position/movement and local paint/erase semantics; four deliberately malformed receipts were rejected.

A second complete execution of the exact source Git blob reproduced fitted weights, outcome summaries, horizon totals, original negative cases and gallery trajectories, ignoring variable elapsed wall time. The ledger preserves **115 negative-case records** with seeds, intervention choices, modes, utilities and receipt hashes; trajectory SVG pairs are retained. Previous research files and negative cases were not edited.

The first V8 campaign measured 39,538 ms wall time. An isolated equivalent-softmax Node v22.16.0 predictor-only benchmark with 30,000 measured calls gave p50 **383 ns**, p95 **815 ns**, p99 **932 ns**, and peak RSS **30,324 KiB**. The reproducible opt-in Node benchmark is committed alongside the script. This measurement is **not** exact-source end-to-end game-policy latency or RSS; the full Node CLI will measure its own peak RSS on explicit execution.

## Reproduction and decision

Run explicitly from a clone with pinned dependencies, providing an absolute output directory:

    node experiments/driv3r/training/long_horizon/twin_supervision_021.js /absolute/output
    node experiments/driv3r/training/long_horizon/bench_twin_supervision_021.js /absolute/output/TWIN_SUPERVISION_021_EVIDENCE.json

Source-smoke CI is separate from scientific replay and native game acceptance. No automatic CPU optimizer on PR/push, no paid GPU/API use, no deployable SLM checkpoint, no binaries in Git, no structural HELDOUT read, and no game-side writes.

**Decision:** Retain feasibility as conservative safety benchmark. Continue testing the 021 student against harder unsealed 128/64 topologies and route-invalidating events. Do not promote while shifted-world generalization and planner overhead remain unqualified.
