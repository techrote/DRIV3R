# DRIV3R Corpus Factory v73 — endogenous physical witnessing under action cost

**Date:** 19 November 2026. **Lane:** experiments/driv3r/corpus_factory/** only. **Status:** full local synthetic-host audit PASS. **Programme:** existing Draft/HOLD PR #3; no merge, no paid compute.

## Scientific question

V72 varied physical witness interventions but scripted their selection. V73 makes the witness decision depend on remembered cost and calibrated uncertainty. The model is shown the A/B witness cost tariff early in its history, receives local material calibration, remote vibration and common-bus/secondary-fault signals, and then decides whether to witness neither material, A, B, or both. The original cost tariff is absent from the instantaneous decision input. Decisions occur on average 106 host-accepted actions after the start.

Witnessing costs 1 or 6 complete physical PAINT/ERASE coupon cycles per material (2 or 12 legal actions). Costs are actual host-accepted operations, not a free symbolic flag. After witnessing or skipping, the host delivers delayed sensor reports; the teacher conservatively revisits and repairs any remotely damaged or uncertain material, terminating with a host-accepted STOP.

**No privileged future for probe labels:** compute the expected post-probe physical travel/repair/STOP actions over two equally weighted hidden erosion histories and two hidden sensor-fault polarities, using counterfactual executions of all four witness strategies from the pinned v72 synthetic host. Select the minimum expected action cost with fixed NONE/A/B/BOTH tie order, then execute that chosen intervention with its real 1/6-pass procedure. The model input does not contain hidden truth, future outcomes, oracle cost vectors or evaluator certificates. This is an *ex-ante* cost-sensitive deterministic oracle, not a trained policy.

## Qualified corpus

| Split | New, nonoverlapping geometries | Completed episodes | Independently replayed legal actions | Probe decisions NONE / A / B / BOTH |
|---|---:|---:|---:|---:|
| TRAIN | 4 | 512/512 | 74,280 | 376 / 24 / 48 / 64 |
| DEV | 2 | 256/256 | 39,944 | 128 / 48 / 48 / 32 |
| HELDOUT | 2 | 256/256 | 39,552 | 176 / 16 / 16 / 48 |
| TOTAL | 8 | 1,024/1,024 | 153,776 | 680 / 88 / 112 / 144 |

There are 64 matched 16-arm groups. For each world/threshold/calibration/fault-domain combination, four tariffs are crossed with both hidden physical histories and both hidden fault polarities. Every first-choice observation is identical inside its 16-arm group, but 32 of the 64 groups require different optimal witness modes across historical tariffs. All geometries are disjoint across TRAIN/DEV/HELDOUT and locally confirmed disjoint from archived v68–v72 world fingerprints. Duplicate TRAIN follow-up (observation, action) pairs were loss-masked 3,365 times without masking the probe or recovery decisions.

## HELDOUT expected-cost results (symbolic controls, NOT neural scores)

* Cost-aware oracle average remaining cost: **10.50** host-accepted actions, including coupon, travel, repair and STOP but excluding the common pre-choice trajectory.
* Always NONE: mean regret **2.125** actions.
* Always BOTH: mean regret **9.750** actions.
* Fault-domain-only (COMMON_BUS means BOTH, else NONE): mean regret **2.750** actions.
* Best instantaneous-current-observation-only mode for each matched group: mean regret **0.875** actions.
* End-to-end synthetic host completion: **256/256** HELDOUT, **1,024/1,024** across all splits.

The negative result for simplistic probe heuristics is genuine within the fixed synthetic prior, but history plus an engineered expected-cost calculator can still solve this task without an SLM.

## Independent qualification

1. The standalone auditor does not import the v73 generator/teacher or its oracle cost scorer. It reconstructed expected probe costs from shortest-path host geometry and deterministic abrasion counts, independently replayed all 153,776 actions and checked source-time telemetry, legal actions, actual coupon painting/erasing, repairs and terminal objectives.
2. All 64 sixteen-arm groups met counterfactual completeness and equal-current-observation gates. Repeating the entire generation yielded byte-identical output for all six JSONL shards and the manifest.
3. A tampered evaluator cost certificate was rejected even after the modified dataset shard and manifest SHA-256 were recomputed; independent expected economics, not just file checksums, caught the corruption.
4. Local Node syntax and full host qualification PASS. GitHub Actions source-smoke CI on the new exact commit must be checked independently; it cannot substitute for this full audit.

Pinned SHA-256: generator **92cf463e1b09f902823e1d5522a415f3c226aa5d61169bbe93f8e5841c3fb82d**; independent auditor **46fe2d37e733f6cc80195a06fda1cd0a7194c5f00e827ab33f595a44ede18c17**; scorer **332c4482e8a90ed38ee535a80ff983ee6c6b4f82c860e87dbfa17505c0d31b3b**; V73 manifest **63dc5a110db2108ff4eece26f063e6962ad31d69c650ccd0ff8fd3cd1c0a34e5**; parent v72 **3ef2235f86414349ab60b59a687667a0d2966aa109547407271392b1cc4c195a**; underlying host v44 **14d27b999d5eee0a83fe7e5afd9e85ce29670bdd4320024d3465705f7b29f8b6**; parent v70 **aa742b764b8ad05df3a252c6fc1dae9a9b1406959e0c94cdbe0fb312558a64b1**.

## Reproduction and ownership

From repository root run, in order:

    node experiments/driv3r/corpus_factory/endogenous_witness_v73.js /tmp/driv3r-v73
    node experiments/driv3r/corpus_factory/test_endogenous_witness_host_v73.js /tmp/driv3r-v73
    python experiments/driv3r/corpus_factory/score_endogenous_witness_v73.py /tmp/driv3r-v73

Commit only the compact generator, auditor, scorer, manifest, report and lane-local handoff. Full JSONL corpora belong in immutable opt-in Actions artifacts, not Git. **Only TRAIN_endogenous_witness_stream.jsonl is trainable. DEV/HELDOUT and all certificates stay evaluator-only**. The next consumer task is trainer-owned pinned CPU SLM/adapter training with autonomous sealed HELDOUT repair/completion, net action cost, tariff/calibration-memory ablations, p50/p95 inference latency and peak RSS.

**Boundaries:** all observations, material abrasion laws, fault channels, tariff semantics and physical coupons are synthetic experimental host extensions, **not established native CyberSand ABI**. Only eight geometries, a uniform hidden-event prior, two coupon workloads and four probe modes. No trained target SLM, no autonomous neural episodes, no native game qualification, no immutable Actions corpus artifact, and no paid API/GPU spend. The synthetic perfect teacher is not evidence of a deployable learned policy.
