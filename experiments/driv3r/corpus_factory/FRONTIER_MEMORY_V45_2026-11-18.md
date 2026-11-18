# DRIV3R Corpus Factory v45 — partial-map survey / remote-event memory

**18 November 2026 — SYNTHETIC graph host, not native CyberSand. Data-only lane.**

## New scientific question
Can a game-like policy combine a physically acquired, incomplete map with an old remote obstruction receipt, then discover an initially unseen next painting goal on a new graph topology? V44 gave the agent the entire static graph at every decision. V45 removes that shortcut. Host-accepted movements are required to reveal graph edges. Model inputs carry only visited cells, edges incident to visited cells, current local legality, paint, recent acknowledgement history, old remote events, and public goal coordinates.

A survey prelude executes both candidate route arms through the actual synthetic host, paints the first objective with a wrong material, and physically returns to the decision point. Following two remote-gate announcements and 192 accepted ageing movements, paired open/closed cases present identical local menus, observations and recent-32 movement history. Only the older relevant event distinguishes the correct first route. Unrelated obstruction negative controls retain the same first action. A partial-map-visible frontier labeler (no hidden edges) produces the subsequent online recovery trajectory, including erase, PAINT_A, discover unseen second objective, PAINT_B and STOP.

The survey is **scripted privileged preparation** (it is not evidence an autonomous model explored intelligently). The teacher operates **only on its model-visible observation**, choosing shortest known routes or a deterministic nearest optimistic frontier. Thus labels are not claimed globally shortest when the map is unknown. The full hidden graph is used only to select/validate initially causal counterfactuals and check completed host execution.

## Exact local qualification

- Parent: `unseen_topology_v44.js` original pinned source SHA-256 `14d27b999d5eee0a83fe7e5afd9e85ce29670bdd4320024d3465705f7b29f8b6`.
- Generator: `frontier_memory_v45.js` SHA-256 `8b6e8949f967a58a0e09702166bd8aa9d98b6ff7446079dbe007ee726d3702df`.
- Independent auditor: `test_frontier_memory_v45.js` SHA-256 `ed115e5c24fb0699ddb3c19e59d6559653a4886bc8a0590e2678ac83d0641688`.
- Source world indices use `5000 + index * 211 + trial` to avoid regenerating the v44 prior cohorts. TRAIN/DEV/HELDOUT world sizes are 7–8, 9 and 10–11 respectively; each split has different SHA-256 topology identities.
- TRAIN: **64 worlds, 256 successful terminal episodes, 5,644 decisions**, 2,260 movement labels before visiting target and 1,756 while the goal location is not yet in surveyed graph nodes. Mean fraction visited at initial fork: 25.545%.
- DEV: **24 worlds, 96 successful terminals, 2,652 decisions**, 1,128 before-target-visit movement labels, 982 with unseen goal. Mean initial visited fraction: 21.171%.
- HELDOUT: **24 worlds, 96 successful terminals, 3,436 decisions**, 1,650 before-target-visit movement labels, 1,394 with unseen goal. Mean initial visited fraction: 17.412%.
- **TOTAL: 112 distinct worlds, 448 host-successful closed-loop oracle episodes, 11,732 labelled decisions, 112 causal pairs and 112 negative-control pairs; all positive first movements match the actual shortest path in the true graph, with >=2 steps of counterfactual regret in both worlds.**
- Both full generations byte-identical across 7 files (six shards + manifest). Independent auditor re-ran the accepted synthetic host from each prelude through every one of the 11,732 saved decisions and all 448 STOP successes. Per-step observation replay, a separately coded visible-map-only action planner, per-visit geometry exposure, remote-memory ablation, action legality, split boundaries, duplicate-label checks and shard byte hashes all passed.
- Generation A 3.97s / peak RSS 214,916 KiB; B 3.82s / 215,660 KiB. Independent auditor 16.62s / 970,244 KiB. These are corpus job metrics **NOT neural inference latency/RSS**.

## Consumption / cautions

`*_policy.jsonl` contains only `input`, supervised `action`, and `weight`. The model must see **only `input`**; the grade/checkpoint must not read `action`, internal world seed, teacher or evaluator. `*_challenge_EVALUATOR_ONLY.jsonl` is sealed to trainers. TRAIN only for gradient updates; DEV for selection; HELDOUT once.

This corpus remains one synthetic randomized grid-graph family with publicly named goal coordinates and global reliable remote notices. The two-route survey prelude is scripted, not spontaneous exploration; the learner may also exploit the determinism of the teacher's frontier heuristic. No actual LFM/Sprocket parameters were updated, no fresh model game-like closed-loop trajectories were measured, no native CyberSand host ABI or physics validated. No immutable Actions artifact publication is currently available. This local package requires the pinned v44 source alongside the v45 generator when running outside the branch.

**Next consumer experiment:** compare actual pinned LFM and Sprocket adapter training with v43+v44 vs v43+v44+v45, same legal host menus and unmodified model runtime. Measure HELDOUT fully autonomous goal discovery, legally applied move/erase/paint/STOP, terminal completion, first-action regret, blocked-route revisits, history-ablated vs full-event accuracy, p50/p95 model CPU forward latency and peak RSS. Report real model forwards and updates; no surrogate substitution.

**Ownership:** no cross-lab writes, no CyberSand or Neurounner modifications, no GPU/API spend, no Git main changes, no merge.
