# DRIV3R Corpus Factory v77 — dynamic obstruction and stale spatial evidence

**Dataset type:** synthetic grid-graph host, NOT native CyberSand or verified compatible with native ABI. TRAIN-only trainable. No neural model trained.

## Mechanism
Two initially open routes lead to distant PAINT_A objective. One remotely changes to blocked in the synthetic host's physical navigation graph. Old and new sensor reports arrive in either chronological or reverse order; 138 or more legal accepted patrol actions separate the last report from the critical junction. Identical current local observations require different MOVE_N / MOVE_S actions depending on the historical source-time reports. The host enforces gate collision, physically accepts the goal PAINT_A, and refuses premature STOP.

Across 24 split-disjoint geometries, TOP/BOTTOM final open states, FRESH_FIRST/STALE_FIRST delivery, and teacher/forced-wrong branches are balanced. Loss-masked legal wrong-corridor detours reach an actual blocked gate, after which recovery is supervised with accepted host movements. All 160 prelude actions have zero training weight; interventions and recoveries are supervised except the forced wrong actions.

## Qualified data

| Split | Worlds | Episodes | Independently replayed legal actions |
|---|---:|---:|---:|
| TRAIN | 12 | 96 | 18,028 |
| DEV | 6 | 48 | 9,564 |
| HELDOUT | 6 | 48 | 9,884 |
| **TOTAL** | **24** | **192** | **37,476** |

All 192 episodes physically reach terminal success. At each world, matched action contrasts share an identical contemporaneous input; balanced current-only first-choice action ceiling is 50%. A symbolic history-aware teacher obtains 100% by construction. In a separate cert-only scorer, the arrival-order shortcut is correct on 75% (144/192), the current-only top preference is correct on 50% (96/192), and a last-128 window also scores 50% because the final sensor report is more than 128 accepted decisions old. Source-time selection is correct on 192/192 by construction. This does NOT measure SLM performance. There are 96 physical, off-policy wrong-corridor recoveries.

A separately written replay reconstructs each host action, gate state, sensor report, observation, painting and STOP. Two full generations byte-identical. After altering a gate report and updating its stream SHA, episode SHA and certificate shard SHA, the independent auditor rejects the mismatch against physical report history. All 24 geometry hashes are unique and split-disjoint.

## Limits and consumer handoff
This is a deliberately narrow two-corridor synthetic game-like benchmark. Corridor widths and gate locations vary, but topology motif remains repeated; reports and physical obstruction are experimental rather than established native CyberSand ABI. Scripted patrol loops add genuine long temporal separation but are not complex autonomous exploration. The whole-history symbolic teacher still solves the task, and no actual SLM model weights, neural held-out rollout, CPU inference latency/RSS or immutable Actions corpus artifact is reported.

Generate: `node dynamic_gate_memory_v77.js /path/to/new-output`. Audit: `node test_dynamic_gate_host_v77.js /path/to/new-output`; ablations: `node score_gate_report_memory_v77.js /path/to/new-output`. Training input: ONLY `TRAIN_gate_memory_stream.jsonl` and positive-weight actions. DEV/HELDOUT streams and all certificates are evaluator-only. Next consumer: pinned CPU SLM, held-out autonomous legal navigation and recovery, last-128-memory ablation, arrival-order shortcut and current-only controls, with p50/p95 latency, RSS, checkpoint/tokenizer SHA-256.
