# DRIV3R Corpus Factory v29 — compound remote repair, matched-memory challenge

**Date:** 2026-11-17. **Repository:** `techrote/DRIV3R`, shared Draft/HOLD PR #1 only. **Scientific status:** synthetic-host oracle corpus; **not** native CyberSand, **not** actual model training.

## Question and protocol

V28 distinguished the locations of one remotely damaged paint cell. V29 asks whether an action policy can remember **two accepted wrong-material paintings** after they are both absent from the latest 64 observation events and positions. Each arm contains two remotely contaminated target cells selected from a three-target world. It must visit and erase both, then paint at least two authorized material recipes and finish at the exit. The first accepted paint flips a live gate, so route planning is dynamic.

Generate a matched pair by changing which two cells were damaged while holding fixed the world survey, current node, legal actions, local view, goal, painted-correct count, remaining budget and recent 64 receipts/positions. Require different optimal first actions with **at least two extra actions of shortest-successful-suffix regret in both directions**. The complete earlier history retains both actual painting receipts and their positions, so input labels are well-defined. The first-action ceiling for a last-64-only or survey-plus-current model is 50% on each matched pair; this is **not** an empirical model benchmark.

The prelabel trajectory including both wrong paint actions is scripted, and then **160 accepted moves** eclipse the interventions. All subsequent labels and terminal successes come from the existing finite-state exhaustive reverse-BFS oracle. Metadata and `hidden_interventions` are for analysis, never model input. Training consumers reconstruct only `state`, `initial_history`, `events_added` and `positions_added` in `observation_keys` order. If a training loader exposes episode metadata, it invalidates the experiment.

## Reproduced corpus

| Partition | World families | Accepted strict pairs | Episodes | Decision labels | Expert erasures | Zero-weight exact duplicates |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 59 | 118 | 2296 | 236 | 304 |
| DEV | 4 | 28 | 56 | 1270 | 112 | 0 |
| HELDOUT | 4 | 28 | 56 | 1520 | 112 | 0 |
| **Total** | **16** | **115** | **230** | **5086** | **460** | **304** |

- **230/230** scripted-prelude + oracle-suffix episodes reach valid synthetic terminal; **460** corrective erasures, exactly two per episode.
- **115/115** matched pairs have identical restricted inputs but distinct full inputs and distinct optimal first actions. Minimum wrong first-action regret **2**, minimum remote distance **3 edges**.
- **13 of 128 candidate worlds rejected** because they lacked qualifying strict pairs (5 TRAIN, 4 DEV, 4 HELDOUT). Do not count as positive evidence.
- Exact full-input/target conflicts **0**; cross-split exact input overlap **0**; cross-split graph geometry/world overlap **0**. TRAIN duplicates **304** zero-weighted. DEV and HELDOUT keep weight 1.
- Regression suite verifies complete observation-hash reconstruction for all **5086** decisions, valid/accepted action labels, exact STOP at terminal, two erasures, disjoint splits and manifest SHA-256. Two independent full generations are byte-identical across all six JSONL shards and manifest; two quick generations also byte-identical.
- Generation measured **34.11 s / 540,396 KiB peak RSS** for first full pass and **34.69 s / 486,488 KiB** for second. Local verification **17.22 s / 451,532 KiB**. CPU corpus machinery only; these are *not model inference* metrics.

## Exact implementation identity

- `compound_remote_v29.js`: SHA-256 `8f4aac216b5e2da7dcd2d9600b0b9533190fc31cc8f8c5eb5fa70b1b5616c386`.
- Dependencies: `remote_error_alias_v28.js` `c94a1f563193ca1edc78555a2a2b20eab06732cf28e196b1effe450214b811fb`, `causal_gate_v26.js` `b64925f57b95a20fc1562748a25ba24d7a6cdc01d11787413544aa26451bdd1c`, `outcome_forks_v25.js` `c879024a7c8ac96408a0a507be8f12bc7299b2bc6f8baf6be918e738e5d01a7d`.
- `V29_DEFAULT_MANIFEST.json` SHA-256 `95eb2d10720fbf2bb278d338d55faa0f377ec4c608a67f546813fbd39687f643`; embedded per-shard digests are authoritative.
- Generate with `node compound_remote_v29.js NEW_OUTPUT` and verify with `node test_compound_remote_v29.js NEW_OUTPUT`; `--quick` is smoke-only and must not substitute for published full split.
- Generated JSONL shards are **local/transfer artifacts**, not committed to Git and not yet immutable Actions artifacts. The opt-in CI requirement remains unresolved.

## Cross-lane evidence and next consumer experiment

LFM350's `LATEST_HANDOFF.json` reports that the pinned full-model GP350-006 adapter is conservative (no BRANCH or RETRACE) and later branch/retrace LoRAs suffered premature retraces. Its latest GP350-018 decoder effort has no new root-model forwards or optimizer updates. Sprocket's handoff reports missing pinned 500M root weights/dependencies and no available opt-in model-workflow dispatch. Therefore Corpus Factory v29 **does not claim** to have run either SLM.

A useful consumer trial: train or fine-tune an actual pinned LFM2.5-350M or Sprocket-500M adapter using TRAIN, select on DEV, open HELDOUT once; require full closed-loop model-controlled repair in each paired world, comparing full-history vs last64-only vs survey-only and deterministic symbolic controls. Measure paired first-action accuracy, realized suffix regret, both erased repairs, terminal completion, wrong residual material, illegal/raw preferences, inference p50/p95 latency and peak model RSS. Publish weight hashes and actual root forward/update counts in an immutable Actions artifact.

## Important limitations

- Synthetic graph topology with only three paint targets and finite-state oracle; a symbolic interpreter with full history solves the problem. Causal memory dependence is established **only for restricted first-action contrasts**, not for an SLM or natural gameplay.
- Prelabel painting and drift are scripted; no model-controlled prelude and no native CyberSand observations, actions, physics, painting, or material reactions were tested.
- Local environment has no required pinned root SLM weights; the Corpus Factory lane is DATA, not policy trainer. No model training, paid API, GPU spending, checkpoint claims, or game-repo writes.
- The split separation is at the graph-world and exact-input level, not proof against all combinatorial grammar shortcuts. The dominant matched actions are BACKTRACK versus one of several moves.
