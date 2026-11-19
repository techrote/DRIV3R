# Corpus Factory v62 — surface-relevant temporal invalidation after physical calibration

**19 November 2026.** DRIV3R Corpus Factory programme PR #3 — **DRAFT / HOLD**. **Synthetic graph-host research for CyberSand; NOT native CyberSand execution and NOT neural policy performance.**

## Research problem and experimental design

V61 distinguishes older versus latest safe physical B-probe measurements, but a short symbolic rule that trusts the last receipt always solves it. V62 adds a warning *after* that valid measurement, potentially invalidating it, and a distractor warning for the already-completed A surface. A competent game-policy learner must retain calibrated material state, attribute events to the correct surface, invalidate only B evidence when the new warning targets B, then safely re-probe B before committing protected material. An irrelevant A warning must not trigger B reprobe.

Each matched set crosses independent B **old law x intermediate law** (normal/swapped x normal/swapped) and four post-calibration conditions: **no alert**, **irrelevant surface-A alert**, **repeat surface-B alert with no law change**, **repeat surface-B alert with hidden law flip** (16 arms per world). The B change is *hidden* until physically probed. The host executes two safe material probes in every episode, and a third after a late B warning. A warning may arrive after previous calibration; the agent then experiences a long host-executed legal detour/retrace. All detour steps have training `weight=0`; they are not presented as expert navigation demonstrations. Current observation at the goal is identical across all 16 arms after the forced detour, while the expert critical action is balanced: 8 move/reprobe, 4 PAINT_A, 4 PAINT_B. No hidden law or experiment-arm marker appears in the model input.

## Qualified dataset

| Split | Worlds | Matched 16-arm sets | Successful synthetic episodes | Accepted decisions | Masked detour moves |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8 | 8 | 128 | 12,960 | 8,192 |
| DEV | 4 | 4 | 64 | 10,976 | 8,192 |
| HELDOUT | 4 | 4 | 64 | 20,224 | 16,384 |
| **Total** | **16** | **16** | **256** | **44,160** | **32,768** |

**Independent host replay:** all **44,160** exact pre-action observations matched, all actions legal, all **256** synthetic terminal episodes successful, and all 16 identical-current-state 16-arm sets verified. Four HELDOUT matched sets are a focused diagnostic, not broad generalisation evidence. The old-probe receipt is misleading in 128 episodes. A late B warning occurred in 128 episodes, an irrelevant A warning in 64, and no late warning in 64. Exactly **64** independent host-executed stale-material counterfactual paints failed when a B warning flipped the law; the corresponding stable-law blind paints succeeded. Thus reprobe is necessary for *reliable informed selection*, not because the host prohibits lucky correct paints after a warning.

**Controls:** memoryless current-observation majority and last-probe-only critical-action ceiling **50%** by matched construction (not measured neural accuracy). Surface-aware timestamped symbolic oracle **100%** critical choice (not neural). For matching B stable-versus-flip warnings, observations and prescribed actions remain identical up to the third physical probe; the terminal protected command then differs according to physical material outcome. Irrelevant A warnings produce the same correct critical commands as no-warning histories.

TRAIN has 4,221 additional exact repeated noncritical action records zero-weighted; DETOUR actions in every split are zero-weighted. DEV/HELDOUT and evaluator-only certificates are strictly excluded from training. World geometry/seed grouping matches existing v61 split ownership, so **cross-version combination must maintain these split assignments**. Split data and sidecars contain no model checkpoints.

## Reproduction and provenance

```bash
node experiments/driv3r/corpus_factory/relevance_gate_v62.js /tmp/v62
node experiments/driv3r/corpus_factory/test_relevance_gate_host_v62.js /tmp/v62
node experiments/driv3r/corpus_factory/relevance_gate_v62.js /tmp/v62-again
# Compare all six JSONL shards and manifest byte-for-byte.
```

Generator SHA-256: `33b854e4d60b1e786bfc820fd5a4fdad18140e31e097a68457465c2231bd71ae`

Independent auditor SHA-256: `4b88851c3cbb1c01a37b6e4a06eb60d5ce621167923ee1d25c5147b69077e4b3`

Default generated manifest SHA-256: `43c287c47d47c5a71e98b1a13984cd1c9dfe47694a6f23c749173de6d699a7c5`

Two full generations produced **byte-identical** manifests and all six JSONL shards. Independent host auditor passed; one-byte shard corruption was rejected by its hash gate. The v62 manifest contains parent-module hashes and data-shard SHA-256/byte/row counts. Full raw data are kept out of Git, reproducible from pinned source.

## Scientific limitations and consumer handoff

1. Synthetic v53 host logic and externally injected late surface-A/B warning semantics do **not** establish native CyberSand behaviour or its exact runtime ABI; read native host-legal contracts before game integration.
2. Long detours are legal, host-executed and loss-masked, not expert traversal training. The 4 held-out geometries are small for structural generalisation.
3. Timestamped per-surface receipt parsing remains a sufficient symbolic controller; no real LFM/Sprocket SLM weight update, neural autonomous completion, neural CPU latency/RSS, native validation or immutable GitHub Actions dataset artifact occurred in this lane.
4. A lucky blind paint may succeed after a no-drift B warning. The benchmark evaluates calibrated safe choice under hidden law uncertainty, not an impossible-paint host prohibition.
5. Useful trainer experiment: use chronological TRAIN data only with real pinned SLM adapters, contrast alert-surface relevance and late hidden drift, evaluate sealed 16-arm HELDOUT all-correct, memory reset/truncation and closed-loop synthetic host trajectories; report CPU p50/p95/RSS. Existing long-horizon trainer PR #2 is a separate lane and was not modified.

**Ownership:** Corpus Factory files only on ongoing DRIV3R PR #3, which must remain DRAFT/HOLD. No main, CyberSand, Neurounner or other-lab writes; no paid compute or force-push.
