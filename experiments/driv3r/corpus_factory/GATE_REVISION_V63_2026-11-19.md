# DRIV3R Corpus Factory v63 — streaming remote-gate revision and route-causality challenge

**Date:** 2026-11-19. **Owner:** `experiments/driv3r/corpus_factory/**`. **Repository target:** existing DRIV3R Draft/HOLD programme PR #3 (`research/corpus-factory-postmerge-programme`). **Baseline at implementation:** `689c3ffec19bea1128d052b543d3d10ee2c19adc`; main `85df84666a8bad18f0092b930c08557b8010687f` (read-only).

## Why a new challenge

v60–v62 measured temporal retention of physical material receipts and surface attribution. v63 tests a different game-policy failure: choosing a route through a remote obstruction based on an **obsolete gate bulletin**, after a newer observation arrives *during a live stream* and is then absent from the recent window. The positive case changes the **first optimal physical movement action**; an irrelevant remote control gate changes no route. These are *actual synthetic-host graph transitions*, not detached scalar classification.

## Design / provenance

- Novel split-specific procedural graph families (`unseen_topology_v44.js`), partial surveys and accepted-action prelude (`frontier_memory_v45.js`), chronological event delivery (`offpolicy_repair_v58.js` event stream contract). Source scenario indices begin at 24,000, separate from prior releases. No new action tokens.
- Each world provides an **eight-arm crossed comparison**: pivotal vs irrelevant gate, obsolete gate open/closed, revised gate open/closed. The late `EDGE_GATE` event is injected at the midpoint of a 48/96/160-action TRAIN/DEV/HELDOUT detour; *not repeated* after delivery. Each critical instant has exactly matching model-visible observation, legal menu and local/partial-survey geometry among all eight arms.
- All detour actions are host-accepted and reversible. They explore already surveyed branches, never cross either candidate gate, and are **zero loss weight**. The critical supervised action has focus weight 8. The rest of the host-completing route and two-objective painting sequence are replayed.
- For pivotal gate variants, wrong first choice has at least **two additional graph moves** of shortest-path regret in independently computed host ranks. Irrelevant control changes no choice. Grouping is by full source geometry and split, not individual arm.
- Only TRAIN streams are for fitting; DEV and HELDOUT streams and **all evaluator certificates** are evaluator-only. Explicitly no evaluator arm labels, hidden gate IDs, epoch state or full durable ledger in per-step model-visible input. 942 redundant noncritical TRAIN same-state same-action records loss-masked; conflicting critical historical labels preserved.

## Qualified output

| Split | Worlds / 8-arm groups | Successful synthetic episodes | Legal decisions | Masked intervention actions | Duplicate TRAIN noncritical labels masked |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 8 | 64 | 4,440 | 3,072 | 942 |
| DEV | 4 | 32 | 3,972 | 3,072 | 0 |
| HELDOUT | 4 | 32 | 6,208 | 5,120 | 0 |
| **Total** | **16** | **128/128** | **14,620** | **11,264** | **942** |

Independent `test_gate_revision_host_v63.js` reconstructs each source world, original partial-survey prelude, ordered bulletin and revised gate, 14,620 pre-action observations, legal movement/paint/erase/stop, delayed route decision and terminal success. Does not import v63 generator. Tests: **PASS**; current-state only and stale-first-bulletin controls each score **50%** on 64 pivotal critical decisions; latest-gate symbolic controller **100%**. *These deterministic controls are not neural-model scores*. All 64 irrelevant-control episodes retain identical critical movement. A single world contributes 4 pivotal and 4 irrelevant episodes. Independent structural wrong-turn regret minimum **2 moves**.

- Two full corpus generations: all 7 output files byte-identical.
- Manifest / shard row counts and SHA-256 validated; manually corrupted output rejected.
- Cross-release v62/v63 geometry intersection: **0** (16 vs 16 unique hashes).
- Positive/negative route evidence and exact pre-action host replay audited in a separate implementation.
- Cost: local CPU Node.js, no paid API, no GPU. **No model checkpoint produced**.

## Reproduction

```bash
node experiments/driv3r/corpus_factory/gate_revision_v63.js /tmp/driv3r-v63
node experiments/driv3r/corpus_factory/test_gate_revision_host_v63.js /tmp/driv3r-v63
```

Generation is deterministic; the committed `V63_DEFAULT_MANIFEST.json` pins hashes and counts. The full local corpus is distributed as a release ZIP rather than tracked bulk Git files, pending immutable opt-in Actions artifact publication. Do not automatically train on ordinary pushes.

## Exact SHA-256 provenance

| File | SHA-256 |
| --- | --- |
| generator | `a96e1d03d5f01113beb15d719b52de31216cc1f7e5b39b08f53b86cb59b46f8f` |
| independent auditor | `e39bff1c8af582bb8baaf4f4f768c2ef92d4d53234d8727b93f7dd28598cf767` |
| default manifest | `b039eec6e572aa0ddfd7be83b3a7204b0fdaca5f8af392192b3f8cf07ce8d253` |

## Scientific boundaries and next consumer

This is the **synthetic graph host, not native CyberSand**. Histories before the stream, map surveying and detours are scripted interventions; the subsequent route is observable-survey oracle-labelled. A compact symbolic controller tracking the latest gate event can solve the pivotal cases; there is **no evidence here of neural learning**, LFM/Sprocket closed-loop success, CPU p50/p95, RSS, native game compatibility or immutable Actions dataset artifacts. Only 4 held-out topologies are included: treat these as a focused challenge, not a broad generalization estimate.

**Next consumer experiment (separate trainer lane):** train actual pinned SLM/adapter on chronological v45/v46 gate examples and v63 TRAIN, excluding DEV/HELDOUT and evaluator certificates. Evaluate autonomously in the synthetic host on held-out route revisions, ablate event memory / reset at midpoint / stale-first-gate / route-planner, report accepted-action completion, optimality regret, recovery, CPU p50/p95/RSS, checkpoint SHA-256. Compare against symbolic latest-gate controller before claims about game usefulness.

**Ownership:** corpus_factory only; draft/HOLD PR #3; no merge, Ready, auto-merge, main write, CyberSand/Neurounner write, other-lab edits, paid compute or non-opt-in expensive workflows.
