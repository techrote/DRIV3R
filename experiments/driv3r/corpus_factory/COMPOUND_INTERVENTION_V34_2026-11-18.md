# DRIV3R Corpus Factory v34 — accepted compound navigation/painting mistakes

**Date:** 2026-11-18 UTC. **Owner:** `experiments/driv3r/corpus_factory/**`. **Publication:** DRIV3R research PR #1 only, Draft/HOLD. **Host:** synthetic graph simulator, **NOT native CyberSand**.

## Scientific question

Can game-policy learners recover when **two distinct legal but detrimental actions are actually accepted**: an initially bad movement choice and a later wrong-material PAINT, with old remote contamination history outside the last 64 receipts? V33 supervised a bad first navigation action but did not inject any second off-policy painting error after control began. V34 adds that independent action and trains on the resulting *real host successor*, not an imagined label.

## Design and nonleakage

- Reuse v32/v33's five-target, dynamically latched gate and four-way identical-recent-history recovery families, with disjoint TRAIN/DEV/HELDOUT topology groups. Each matched family has four older hidden wrong-target locations and requires one of BACKTRACK/MOVE_E/MOVE_S/MOVE_W.
- For each arm produce three trajectories: clean; accepted suboptimal first navigation with exact post-deviation repair; and **accepted bad first navigation followed by an accepted wrong PAINT_A/PAINT_B at a subsequent oracle-selected painting state**, followed by an exact shortest terminal recovery.
- The later wrong paint is a genuine legal host action with an independently certified minimum **2-action suffix regret**. Each compounded trajectory performs one additional ERASE. The synthetic oracle verifies the actual successor cost after every accepted action, not only at the first step.
- A deliberately incorrect PAINT intervention row has `target` = correct teacher action, `applied_action` = actual wrong action, `training_weight=0`, and explicit teacher-only `intervention` metadata. Consumers must **not** leak receipts, counterfactual costs, hidden scenario identifiers or teacher certificates into the model's pre-action input. Train on the model-visible observation and correct-action labels; use receipts only after action application.
- Initial remote wrong-paint incidents and the initial gate-opening operation remain **scripted host-accepted prelude**, not neural autonomous behaviour. The new second error occurs after learner-start time, but is injected by the experimental harness, not sampled from an actual SLM.

## Default deterministic corpus

| Split | Families | Episodes | Decisions | Accepted bad nav | Accepted wrong PAINT | Erasures | Zero-weight TRAIN inputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 768 | 23,008 | 512 | 256 | 1,792 | 5,568 |
| DEV | 32 | 384 | 19,136 | 256 | 128 | 896 | — |
| HELDOUT | 32 | 384 | 25,856 | 256 | 128 | 896 | — |
| **Total** | **128** | **1,536** | **68,000** | **1,024** | **512** | **3,584** | **5,568** |

Each accepted bad nav adds exactly 2 actions of oracle regret. Each accepted wrong material adds another 2 actions. Four-way, last-64-limited initial decision accuracy has a 25% theoretical ceiling in the matched clean families. The gate counterfactual is strictly consequential in one of four clean arms per family (128 positive, 384 negative controls).

## Reproducibility / local qualification

- Generator: `compound_intervention_v34.js`, SHA-256 `8bc27d89d6950858b5b3205b42acf9f010443176ef7498afb0563a71c516fe51`.
- Immutable-looking *local files only*: six data shards and `manifest.json` with each shard's exact byte count, row count and SHA-256; **not yet an immutable Actions artifact**.
- Two independent default generations produced identical manifests and all six binary-identical JSONL shards. Generator timing: 31.12 seconds, peak RSS 1,113,256 KiB; repeat 30.62 seconds, peak RSS 1,115,200 KiB. These are *corpus generation* costs, not neural model latency/RSS.
- Full default generation asserted the exact shortest-successful-suffix distance after **every actual applied action** across all 68,000 decision steps and all 1,536 successful episodes; accepted action legality and terminal painting/exit correctness were enforced.
- Independent host replayer passed the complete `--quick` corpus: 288 episodes, 12,336 decisions. An independent **sampled** replay of one complete group per split against the full default dataset passed, plus verification of *all six full-shard hashes and byte sizes*. The attempted exhaustive independent default replay exceeded the local execution budget, therefore do **not** label that independent full replay PASS.
- Local tests establish no contradictory complete-observation labels, no cross-split full-input collisions and no cross-split geometry collisions on generated data; the independent sample exercises the same checks. Duplicate TRAIN observations are assigned zero weight, not repeated as independent supervision.

## Scientific limits and required consumer test

This is still a compact synthetic world, not production CyberSand. The goal is two of five painting targets, and an exact graph oracle with global survey history can outperform learning via symbolic planning. Fixed mission grammar, scripted prep history and highly regular four-arm geometry remain potential shortcuts. No actual LFM/Sprocket root weights were updated or run. No native CyberSand evaluation, real SLM p50/p95 latency, peak model RSS, or checkpoint was produced. LFM and Sprocket worker handoffs report absent pinned root-model assets/runtime as execution blockers.

**Consumer:** pin model and tokenizer hashes, TRAIN-only updates, DEV selection, HELDOUT once; compare full durable-history, last-64 and symbolic controls under autonomous closed-loop model actions. Stratify clean, navigation-only and compounded recoveries; report terminal completion, residual contamination, first-action regret, corrective ERASE, illegal raw preference, premature STOP, p50/p95 inference latency and peak resident memory. Preserve negative outcomes, root/adapter hashes and immutable artifact custody.

No CyberSand/Neurounner writes; no other DRIV3R lab ownership changes, expensive runner, merge or main update.
