# Corpus Factory v41 — delayed multi-event obstruction recovery (2026-11-18)

**Status:** qualified synthetic supplement; Draft/HOLD PR #1 only. Not a native CyberSand benchmark or an SLM-training result.

## Scientific purpose

v40 exposed a single old obstruction receipt. v41 forces interpretation of **two or three chronologically ordered remote junction transitions**, interleaved with **one or two unrelated sensor diagnostics**, over 160 actually accepted synthetic-host movement actions. The final relevant transition is at least **72 accepted movements old** when the recovery policy is labelled. A rule that simply follows the oldest actuator event, or the last event irrespective of type, is insufficient. A symbolic interpreter reading the most recent relevant event still solves this fixed graph motif.

For a matched CLEAR/BLOCKED pair, the current local view, legal actions, most recent 64 action receipts and most recent 64 positions agree exactly. Only the older junction event receipts identify the correct route. The sensor events are identical across the pair and **remain in the input when the junction receipts alone are ablated**. Positive pairs need MOVE_E versus MOVE_S, each wrong counterpart incurring at least two extra accepted-action steps. Negative controls preserve the same first action.

## Host execution and provenance

Start with the v32 scripted but host-accepted material-painting prelude. Reconstruct the host before the last 160 actions. At offsets 0, 32, 64, 80 and (for one schedule) 88, insert accepted environmental receipts. Each junction actuation actually replaces the synthetic Host's adjacency graph before further actions execute through the host's legal-action gate. Sensor diagnostics do **not** change state. Check movement acceptance, positions and painted-material receipts against the original 160-action history; recompute an exact shortest-successful suffix using the synthetic v32 oracle and replay to terminal success.

Environmental changes are scripted; they are **not agent-selected actions**. All labels remain synthetic host data, not native CyberSand.

## Full, independently repeated data campaign

| Split | Candidate pairs | Strict positive | Controls | Episodes | Decisions | Topologies |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 512 | 128 | 32 | 320 | 8,912 | 8 |
| DEV | 256 | 64 | 16 | 160 | 8,736 | 4 |
| HELDOUT | 256 | 64 | 16 | 160 | 12,096 | 4 |
| **Total** | **1,024** | **256** | **64** | **640** | **29,744** | **16** |

640/640 oracle terminal successes; 0 observed exact input duplicates, conflicting labels or cross-split topology overlaps. Two complete CPU generations produced byte-identical five JSONL shards and manifests; all 29,744 teacher-labelled records were checked for legal action, schema, historic event presence, SHA-256 integrity and split isolation. The independent spot reinstantiation covered representative positive pairs in all three splits and all four event-order patterns. Full generation durations: 18.28 and 18.01 seconds; peak process RSS: 368,280 and 371,640 KiB. These numbers are **corpus generation**, not model inference.

Generator SHA-256: `25e72a3cbde6fd88893fe9881f5c5277527abed7c028a3f112f82d59058860df`.
Generated manifest SHA-256: `e6b719b2187b74565fa923602e796195d838324d1a0615d2fdbf313ce0ecc9aa`.
Dependency and five shard digests are pinned in `V41_DEFAULT_MANIFEST.json`.

## Consumer contract, limitations and next test

`*_policy.jsonl` rows contain only `input` (model-visible), `action` (teacher label), `weight`. DEV/HELDOUT `*_challenge.jsonl` are evaluator-only: never train on challenge costs, cases or labels. Reproduce with colocated pinned dependencies:

```bash
node delayed_obstruction_v41.js /tmp/v41-a
node delayed_obstruction_v41.js /tmp/v41-b
node test_delayed_obstruction_v41.js /tmp/v41-a /tmp/v41-b
```

V41 still has one fixed junction, repetitive north/root ageing movements, public symbolic survey and deliberate environmental timing. It is not autonomous on-policy SLM experience. No actual LFM/Sprocket weights were trained or evaluated. Model latency/RSS, native CyberSand legality and immutable Actions data artifact custody remain unmeasured. V38 remains the broader longitudinal tool-law/recovery training corpus.

Next: use a pinned actual SLM and TRAIN-only data, evaluate closed-loop HELDOUT matched positives and negative controls with (a) full durable history, (b) junction-only ablation retaining sensor events and (c) recent-64 history. Measure first-action accuracy (junction-ablated positive-pair ideal upper bound 50%), terminal success, realized regret, residual contamination, illegal preferences, p50/p95 latency, model RSS, and exact source/model/tokenizer/adapter/corpus hashes. Native CyberSand requires separate owner acceptance. No paid compute or automatic training job was used.
