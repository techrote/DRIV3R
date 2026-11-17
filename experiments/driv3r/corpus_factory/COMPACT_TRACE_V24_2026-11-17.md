# DRIV3R Corpus Factory v24 — lossless, consumer-ready sequence corpus

17 November 2026 · DATA ONLY · Synthetic v23 game-policy host, **NOT native CyberSand**.

## Scientific/engineering motivation

V23 exports the entire growing event/position history on *every* decision, making six split datasets ~719 MB uncompressed. That inhibits CPU SLM training even when the model weights and runtime eventually become available. V24 makes no new claim of learned behaviour: it **preserves v23 labels and game states exactly**, while exporting episode-initial history once, then receipt/position deltas and local observations at each decision. An independent decoder reconstructs the *complete original model-visible input* and checks its SHA-256, and the tests compare recovered inputs/labels with the actual v23 oracle replay.

The change specifically avoids truncating crucial old remote programme receipts. A consumer must reconstruct the full causal history; training directly on the current decision row without its episode header would silently turn this into an ambiguous-memory benchmark. `target`, `receipt`, `training_weight`, private scenario metadata and post-action outcomes are **not** input features. Only the reconstructed `model_input` is policy-visible. No file represents a native CyberSand receipt.

## Reproduction

```sh
cd experiments/driv3r/corpus_factory
node test_compact_trace_v24.js
node compact_trace_v24.js /NEW_EMPTY_DIRECTORY   # full
node compact_trace_v24.js /ANOTHER_EMPTY_DIR --quick  # 1 world group/split, 48 episodes
```

Parent v23 Git blob `e24ad8fb5e874452b86533041a0f1a91174039a9`; v22 `b6f744216859eb91bec925a42f053153c85a7d0d`; v21 `19e8b18139263850930666cae119f58f8b993c82`. All three pinned against local source in tests. V24 generator SHA-256 `c7c39280f0cadb29b2fd0dfa3e2f5e516d8e0625c3dcfec3dfe51515bd52c755`; manifest SHA-256 `5c9b13e65969a63f471b5e1f7b1d8e57d203f1996cb7de3bbd451468b60c3a8a`.

| Split | World groups | Episodes | Decisions | Distinct inputs | Exact duplicates | Erases |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 12 | 192 | 12,496 | 12,000 | 496 downweighted to zero | 384 |
| DEV | 6 | 96 | 10,032 | 9,984 | 48 | 352 |
| HELDOUT | 6 | 96 | 15,392 | 15,344 | 48 | 528 |
| **Total** | **24** | **384** | **37,920** | **37,328** | **592** | **1,264** |

Full repeated `model_input` JSON serialization: **709,723,536 bytes** (input-only reference); six compact episode+decision JSONL shards: **134,516,562 bytes**, an **81.05% reduction**, while retaining full replay information. Note these are two specific representations and include different per-row metadata; percentage is not a native runtime or token-count reduction. Compressed ZIP size depends on packaging. The previous v23 six-shard reference is ~719 MB with metadata.

## Qualification

- 48-episode regression replay: all 5,184 decisions reconstruct exact v23 model-input SHA-256 digests and labels; cross-checks independent v23 oracle action replay at three split/variant samples.
- Full default corpus: 384/384 synthetic oracle terminal successes, 37,920 labelled steps, 0 illegal oracle actions. All 24 world groups have disjoint geometry; all 48 programme strings are split-disjoint; four initial mission states equally represented per group.
- No cross-split exact `model_input` digest overlaps and no conflicting complete-input labels. TRAIN's 496 exact duplicates remain with `training_weight=0`, preserving v23 weighting. DEV/HELDOUT duplicates are retained for evaluation but should be reported both by episode and deduplicated state.
- Two independent complete default regenerations: identical manifest bytes and all six shard files. One generation: **11.68 s wall / 102,912 KiB peak RSS** (Node corpus generation, not model inference). Second complete run 12.28 seconds / 101,492 KiB peak RSS.
- Identical information-ablation limitations to v23: last-64/fixed-prefix/fixed-suffix shortcuts fail by construction, but the public eight-operator task still admits a four-state symbolic interpreter and the world has only two workshops.

## Actionable consumer protocol

Read one `*_episodes.jsonl` row per episode (contains `initial_history` and `observation_keys`), then read its ordered rows from `*_decisions.jsonl`. Starting with the initial history, append each `events_added` and `positions_added`; rebuild each observation in the original key order using the row's `state` and restored `history`. Reject a reconstruction whose SHA-256 differs from `model_input_sha256`. Form the SLM prompt from **only** this reconstructed observation; supervise `target` using `training_weight` for TRAIN. Never expose oracle labels/receipts from the current row, `world_id`, `difficulty` or counterfactual-group metadata as input. A real model's chosen actions must be applied to an online host, not simply replayed from these teacher-forced traces.

For next consumer evaluation, pin LFM2.5-350M or Sprocket-500M root+adapter and run fully model-controlled HELDOUT, comparing full causal history against last-64 and symbolic controls. Report goal completion, exact target-material phase, missed corrections and wrong-branch paint, path stretch, rejected actions, p50/p95 model decision latency and peak *model* RSS. **None of those SLM metrics was collected in this run.**

## Limitations and next data increment

V24 is a **storage/consumption improvement, not new scenario diversity**. All training labels remain v23's synthetic oracle; no policy weights, adapters, CPU SLM inference or native CyberSand observations were produced. No immutable GitHub Actions data artifact yet. Other trainers' latest handoffs still report missing pinned root model weights/tokenizers and no dispatchable opt-in CPU experiment. Next Corpus Factory increment should diversify beyond the two-workshop layout and add matched branch/retrace-positive versus retrace-premature episodes with truly distinct 4-way topology, without introducing hidden oracle access or target leakage.

Only `experiments/driv3r/corpus_factory/**` is writable. DRIV3R PR #1 is Draft/HOLD, and CyberSand/Neurounner remain read-only.
