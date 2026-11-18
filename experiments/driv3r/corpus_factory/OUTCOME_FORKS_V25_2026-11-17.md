# DRIV3R Corpus Factory v25 — suffix-outcome-proven branch versus retrace

17 November 2026. **Synthetic research host; not a native CyberSand ABI conformance result or trained SLM.**

## Why v25

The LFM GP350014/015 experiments saw 25/25 or 45 premature RETRACEs after apparently learning locally trapped geometry. The separate long-horizon `RETRACE-CHOICE-001` DEV evidence (17 November 2026) falsified a geometry-proxy teacher: blocked-spur `always_retrace` painted 107.500 versus frozen 123.375, with 3/8 premature stops; contrastive blocked-spur painted 96.750 and stopped 4/8, despite matching 8/8 prescribed first actions. v25 supplies **actual shortest successful suffix outcome costs** instead of treating blocked geometry as a teacher.

## Construction and measurements

- Explicit orthogonal graph host with side spurs, an upper shortcut that can be obstructed, a lower loop, multiple paintable branches, an exit, and an exact two-of-three goal with material A/B recipes. Wrong material can be accepted then erased; the terminal checker rejects residual wrong material.
- Action vocabulary: MOVE_N/E/S/W, BACKTRACK, PAINT_A/B, ERASE, STOP. BACKTRACK is the immediate predecessor edge, not an unconstrained teleport. All steps are host-acknowledged by the synthetic simulator.
- Exhaustive reverse-BFS enumerates `(position, previous_node, base3_paint_status)` over each world. It proves the shortest completed suffix after both candidate first actions and keeps only cases with a **unique best first action** and **regret >=2 steps**; there is no geometry-derived label.
- Four label/obstruction strata per eight-example family: blocked distal spur + BACKTRACK optimal; blocked + MOVE_N optimal; open + BACKTRACK optimal; open + MOVE_N optimal, each with two examples. The distal blocked/open bit is a **distractor** in this version (not a paint target), while closing/opening the upper shortcut changes useful connectivity. Do not claim the distal bit caused the label.
- Each family has identical current local observation, legal action set, goal, budget, and latest 64 position/event entries, despite different optimal first actions. Full remote survey appears only earlier in scripted, accepted host-like prefix movements. Model decision inputs must include that older observation to discriminate worlds.
- TRAIN / DEV / HELDOUT have disjoint generated graph topologies and hashes. DEV/HELDOUT have longer branches; HELDOUT is more difficult by corridor length, but the graph grammar is shared and is **not** native-world generalization.

| Split | Groups | Episodes | Decisions | Unique inputs | Exact duplicates | First RETRACE | First MOVE_N | Correct paint operations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 64 | 904 | 792 | 112 | 32 | 32 | 128 |
| DEV | 4 | 32 | 512 | 512 | 0 | 16 | 16 | 64 |
| HELDOUT | 4 | 32 | 600 | 600 | 0 | 16 | 16 | 64 |

Each split's blocked/open first-action strata are also balanced 50:50. Consequently current-local-only, last-64-only, and blocked/open-only first-action majority controls are at chance (50%) by construction. That is an information-theoretic balance, **not an actual SLM score**.

## Exact-source and reproducibility

- Source SHA-256: `c879024a7c8ac96408a0a507be8f12bc7299b2bc6f8baf6be918e738e5d01a7d`; test SHA-256: `a2aacfba15540435303e5301cda43df3ae9cd85909ff72a5dd83a7722483eb92`.
- Full default manifest SHA-256: `5a55a0987374c422e444a7809a927a086f48968a6b9f63ede2d7a904747ed180`.
- `node test_outcome_forks_v25.js /path/to/generated` passes five standalone checks plus all complete-corpus SHA/reconstruction assertions. Two complete default generations reproduced every shard and manifest byte-for-byte; two quick generations also matched.
- All 128/128 full synthetic oracle episodes reached STOP at the exit, with quota satisfied and no residual contamination. 2,016 exact labelled action receipts were reconstructed and validated; 112 duplicate TRAIN inputs were retained with `training_weight=0`. No conflicting labels or cross-split exact input overlaps.
- Full run wall time observed: 14.78 and 15.76 seconds; peak process RSS: 518,368–519,528 KiB. Local full corpus is roughly 2.9 MB (six JSONL shards); this measures generator workload, **not model latency/RSS**.
- Compact `*_episodes.jsonl` contains the initial history once. `*_decisions.jsonl` stores only `events_added` and `positions_added`, state, legal target, receipt and full reconstructed model-input SHA-256. Never expose episode metadata, targets, suffix costs, regret or counterfactual class as model features.

## Scientific limitations / negative results

This is an intentionally narrow four-way branch/retrace **synthetic** benchmark, not a replacement for v24's changing missions or previous long-horizon painting. Exact shortest-path optimization admits a symbolic planner, including a compact state space. The distal spur blockage alone does not change value; it is a matched irrelevant cue. There are only three paint targets per episode and a quota of two. The prelude and remote scan are scripted. No SLM forward/training, native CyberSand host receipt, external checkpoint, or immutable Actions artifact is supplied. File hashes protect byte identities but do not make the toy host native.

## Consumer experiment

Run pinned actual LFM/Sprocket SLM policy in **closed loop** on new split-disjoint scenes: train on TRAIN, select on DEV, evaluate HELDOUT without label leakage. Compare full survey memory against last-64, blocked-only and symbolic oracle; report first-action stratified correctness, realized suffix regret, quota completion, needless early STOP, wrong paint+erase recovery, time budget, and p50/p95/p99 inference latency and peak model RSS. One-step matched labels alone are insufficient because the previous blocked-RETRACE proxy degraded sustained painting.
