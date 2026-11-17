# Corpus Factory v16 — ordered remote gate receipts, counterfactual replay

17 November 2026. DATA LANE ONLY. Git destination: `techrote/DRIV3R` Draft/HOLD PR #1; no CyberSand/Neurounner writes.

## Science question

V13–v15 model-control problems are largely expressible with short parity formulas, possibly one salient receipt, and one fixed intervention grammar. This successor asks a more specific question: can a model **retain the order** of distant environmental instructions rather than relying on the unordered event inventory, local map, or latest 64 position/events? A `SET_WEST`, `SET_EAST`, or `TOGGLE` instruction changes the synthetic remote shutter; the instruction sequence is noncommutative. The model does not get privileged shutter state and cannot issue instructions. Four-event v15 parity is not reused.

Each world family has 12 deliberately opposed eight-instruction sequences drawn from the same multiset: one `SET_WEST`, one `SET_EAST`, six `TOGGLE`. Every sequence includes 126 physically replayed washout moves following its eight recorded remote station visits (270 prefix moves in all). The final junction observation has the same position, legal actions, objective, goal, local 7×7 material grid, last 64 movement positions/events, and **unordered** event multiset across the family. Six sequences close WEST and six close EAST; the shortest safe first action reverses accordingly. Extraneous BETA-line instructions are included, but do not control the ALPHA shutter.

The decisive departure from earlier corpora is **event-order partitioning**: TRAIN, DEV and HELDOUT use disjoint sets of 12 instruction sequences (36 total of 56 unique permutations). Source world topology groups also remain split-isolated. DEV has two BETA decoys per ALPHA event and HELDOUT three, versus one in TRAIN. Thus a model that memorises the 12 TRAIN command strings cannot directly reuse them on HELDOUT.

## Bounded evidence

| Partition | Unique world/family groups | Episodes | Sparse oracle decision labels | ALPHA sequence split |
|---|---:|---:|---:|---|
| TRAIN | 32 | 384 | 3,456 | 12 TRAIN-only orders |
| DEV | 8 | 96 | 864 | 12 DEV-only orders |
| HELDOUT | 12 | 144 | 1,296 | 12 HELDOUT-only orders |
| **Total** | **52** | **624** | **5,616** | **36 disjoint sequences** |

Across all episodes, 168,480 prelude/washout movement actions and 41,904 expert post-prefix actions were host-replayed. There are 6,240 strictly longer legal hard-negative annotations, zero accepted illegal oracle moves, zero protected entries, and 52 verified counterfactual families. Parent source is **exact Git blob** `b49afca90c307d8126c8f4c1becba667b7ced6f1` (`relay_memory_v13.js`). The local code must use the matching v13 file; runtime `require('./relay_memory_v13.js')` does not modify that immutable parent.

### First-decision information ceilings (balanced 12-case family)

These are **empirically enumerated Bayes-majority upper bounds on the deterministic controlled projection**, not trained-policy accuracy or global impossibility results. They apply within each family's 12 counterfactual first decisions.

| Visible remote-event information | TRAIN | DEV | HELDOUT |
|---|---:|---:|---:|
| Unordered receipt bag / last64 | 6/12 = 50.0% | 6/12 = 50.0% | 6/12 = 50.0% |
| First ALPHA instruction only | 8/12 = 66.7% | 10/12 = 83.3% | 9/12 = 75.0% |
| Latest ALPHA instruction only | 6/12 = 50.0% | 6/12 = 50.0% | 8/12 = 66.7% |
| Latest two ALPHA instructions | 7/12 = 58.3% | 7/12 = 58.3% | 9/12 = 75.0% |

A deterministic full-history interpreter can, by design, reconstruct the final gate state from commands. **No model has been trained or evaluated** in this lane. Any subsequent trained-SLM study must avoid treating rows/episodes from one world as iid independent replicates.

## Qualification and repeatability

`node test_ordered_relay_v16.js` passed **7/7 sections**, including split/order isolation, paired event-bag alias, oracle shortest-path and protected-cell legality, deterministic regeneration, and on-disk shard hash verification with overwrite refusal. Two independent full `node ordered_relay_v16.js NEW_OUTPUT_DIR 2` executions produced **byte-for-byte identical JSONL shards and SHA-256 manifests**. On the bounded Linux Node 22.16 environment, the two full runs required 17.01 and 17.31 s; peak RSS 399,208 and 403,400 KiB. These are **dataset generation**, not SLM inference, measurements. Manifest SHA-256: `403e51b173af2e993652aa193d84f5f6b85e83cc35416039810ad2b22398c81e`.

Source Git blob SHA-1: `5d64ac0558349f5efe7f40ff866204006a2a1fa7`. Test Git blob SHA-1: `26e025ebed249113b17e473baa73b32c65f395cc`. The committed `V16_DEFAULT_MANIFEST.json` records every shard hash; generated ~78 MB JSONL data should not enter Git history. No GitHub Actions artifact is claimed. The local output is only an evaluation-delivery fallback.

## Consumer handoff / evaluation guardrails

`node experiments/driv3r/corpus_factory/ordered_relay_v16.js /fresh/output/directory 2` generates six partition shards. Consumers must feed **only** `row.model_input` into the policy, and never metadata, expert actions, parent host, hard negatives or `controls`. Use TRAIN for adapter/SLM optimisation, DEV for model selection, HELDOUT only once the choice is fixed. Compare full causal event history to last64, unordered bag, first/latest/latest-two ALPHA event controls; report paired 12-way accuracy by world family, autonomous arrival/path stretch/rejected actions, p50/p95 action latency and peak RSS. The replay entrypoint `createHost(ep)`, `observe(host)` and `apply(host, action)` accepts model actions at **every post-prefix step**; expert planner is used only for labels/verification.

## Limits and negative controls

- The synthetic eight-command station visits and BETA decoys are scripted; the actual model does not control the earlier history. Host-defined remote commands are environmental events, not legal worm actions.
- The v13 external host has a narrow U-loop geometry with deterministic gate semantics and no physical sand or paint acceptance; **this is not native CyberSand gameplay nor canonical ABI acceptance**.
- All 12 sequence permutations repeat across world styles *within* a split. World and order partitions are disjoint *across* splits, not independent in every possible dimension.
- The order-selection rule itself intentionally balances labels and weakens cheap suffix controls; this is a designed challenge, not a claim to match natural event frequencies.
- There is no learned-policy result, no immutable Actions artifact, no native Windows NVIDIA acceptance, and no paid computation.

Next: a pinned actual LFM/Sprocket checkpoint should be evaluated on DEV first. The important HELDOUT result is whether the model **generalises to unseen command orders**, not whether it memorises instructions or solves a fixed shortest path after the shutter state is given.
