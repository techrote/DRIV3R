# DRIV3R Corpus Factory v10 — changing goals, dynamic recovery, motif continuation

17 November 2026. This is a **data** increment in the existing consolidated Draft/HOLD PR #1, not a game patch and not a policy-training run.

## Independent synthetic host challenge

`dynamic_challenge_v10.js` emits three explicitly distinguished, balanced counterfactual tasks, each with two *different legal* labels in otherwise matched local situations:

1. **D5 shutter recovery:** a synthetic remote gate closes while a probe is adjacent to it; the host validates the move/retrace path, takes the actor back to the fork through the same final 12 positions, and the oracle selects the unblocked route. Both siblings have byte-identical local 7×7 material tokens, legal action menu, position, final-12 history, and goal. Only the older receipt/full execution history distinguishes the outcomes. A last-12-only deterministic classifier is information-theoretically capped at 50% on the balanced matched pairs; this is an *aliasing property*, not a measured model accuracy.
2. **D4 goal retarget:** the same junction and legal actions receive distinct new remote beacon positions; the host-graph shortest-path oracle reverses the optimal action.
3. **D3 motif echo:** the same unprotected authoring cursor and `PAINT_SAND` / `PAINT_METAL` menu require the next alternate material based on opposite remote synthetic paint-receipt sequences. This is a cheap continuation task; last-one-paint is sufficient, so it does **not** qualify as a long-memory challenge.

Each row contains `model_input`, `target`, legal-but-wrong and illegal hard negatives, a synthetic-host receipt/prelude, pair/world/topology provenance, explicitly excluded non-feature fields, a short-memory control, and a difficulty label. Deterministic BFS is used for **label verification**, not as a deployed policy.

## Qualified default generation (12 style seeds per topology)

| Split | Disjoint shapes | World groups | Counterfactual pairs | Rows |
|---|---:|---:|---:|---:|
| TRAIN | 18 | 216 | 648 | 1,296 |
| DEV | 6 | 72 | 216 | 432 |
| HELDOUT | 6 | 72 | 216 | 432 |
| Total | 30 | 360 | 1,080 | 2,160 |

Exactly **95,328 synthetic host-applied cardinal movement steps** are checked by the generator; test assertions cover legality, protection, dynamic closure, oracle reachability, paired counterfactual label inversion, full-input-vs-short-input ambiguity, and topology/group separation. No protected entries were accepted. The cross-split near-duplicate guard rejects world groups whose local 7×7 token views are at Hamming distance ≤4 from an accepted world; this run rejected none. The intentionally paired sibling views are exempt from deduplication.

The full JSONL dataset is generated outside Git; the compact `V10_DEFAULT_MANIFEST.json` records exact SHA-256 and byte length. These are NOT immutable hosted Actions artifacts.

| Shard | SHA-256 | Bytes |
|---|---|---:|
| TRAIN | `96893ccb181297e08dc34189d1ee74380d4e420a4e75b15fa793122eca1ad3cd` | 5,616,336 |
| DEV | `56df16574cb63a70bafcfab48478bc1e7d40b6faa135b8b8325a1a7532d5c697` | 1,925,448 |
| HELDOUT | `c8f3e1c481d6f4e0ab4852c717ed24122e017af9a93ed3cca9fd555b74908fe5` | 1,992,912 |

**Exact source Git blobs:** generator `3647f3f9141c0c86312a0e49d9331d823773f238`; tests `a696887d61548ffa5f368d343184bc13965a06c5`. Two complete independent default generations produced byte-identical shard and manifest hashes. `node test_dynamic_challenge_v10.js` passed. Local Node.js 22.16.0 one-shot timings: 1.89 s and 1.81 s, peak process RSS 119,868 and 119,832 KB respectively (single-host measurements, not model inference).

Run from this directory:

```sh
node test_dynamic_challenge_v10.js
node dynamic_challenge_v10.js /NEW/OUTPUT/DIR 12
```

The CLI refuses to overwrite existing output. TRAIN is optimizer-permitted; DEV is for tuning/evaluation and HELDOUT must remain sealed until final assessment. Train only on `model_input`; `target`, `controls`, `hard_negatives`, `evidence`, `provenance` and split IDs are not model features. Do not shuffle sibling rows across partitions.

## Cross-lane consumer opportunity and limits

Read-only consumer handoffs show Sprocket's pinned root still unavailable in the local executor, and GP350-018 has a decoder preflight but **no new full LFM model inference**. LFM's earlier RETRACE-oriented scalar changes produced false-positive/coverage regressions. This corpus is therefore suitable for a **pre-registered root-model experiment** comparing (a) full host-observation/event history, (b) the last 12 positions without older receipts and (c) last-state-only input, on paired HELDOUT shutter situations. A fully model-controlled looped evaluation on *independently generated* environments remains necessary before claiming progress toward native gameplay.

**Limits and negative findings:** This is externally synthetic, not CyberSand's canonical host ABI. Its local action labels are host-synthetic cardinal moves/paint palette, not verified native actions. The oracle makes one correct post-prelude choice; the model did not choose earlier moves, no real SLM weights or adapters were updated, and no closed-loop model performance is claimed. Local motif receipts are generated synthetic exemplars, not native painting confirmations. The symmetric layouts are deliberately constrained; perfect benchmark performance could still arise from simple gate-coordinate comparisons. No expensive Actions training, paid compute, CyberSand/Neurounner write, or DRIV3R merge/Ready action occurred.
