# DRIV3R Corpus Factory v28 — Remote contamination location counterfactuals

17 November 2026. Synthetic research host, not native CyberSand. Draft/HOLD PR #1 only.

## Research question

V27 reported public painted_correct counters but all wrong-material interventions happened at the first target. This creates a fixed-location repair shortcut. V28 constructs matched pairs in which an accepted wrong-material paint occurs at one of two *different* remote target cells, and its receipt is older than the latest 64 events.

Both arms share the exact same world/remote survey, current position, local view, legal actions, public correctness count, action budget, gate state, last-64 events and last-64 positions. Full older history differs. The reverse-BFS expert selects different first recovery actions in the two arms. Choosing the other arm's expert first action incurs at least two additional successful-suffix steps on either arm.

## Construction

A synthetic host scripts legal travel to one of three targets, paints the wrong material, triggers the first-paint gate transition, returns to a common remote rest position, and performs over 64 accepted oscillating movements. All paired starting observations agree on public recent state and remaining budget. Both erroneous targets are at least three graph edges from the rest point. The agent must travel to the *correct* contaminated location, ERASE, paint legitimate material targets, and finish at the exit.

The entire *post-intervention* suffix is labelled by the v26 exhaustive reverse-BFS successful-terminal oracle, not by a geometric proxy. The prelabel routes are scripted and are NOT evidence of autonomous model decisions. Metadata fields hidden_intervention, difficulty, pair_id, regret and split must NEVER be included in policy observations.

## Qualification

| Partition | Candidate worlds | Strict pairs | Episodes | Labelled decisions | Corrective ERASE |
|---|---:|---:|---:|---:|---:|
| TRAIN | 64 | 59 | 118 | 2,226 | 118 |
| DEV | 32 | 31 | 62 | 1,356 | 62 |
| HELDOUT | 32 | 32 | 64 | 1,630 | 64 |
| TOTAL | 128 | 122 | 244 | 5,212 | 244 |

All 244 oracle recoveries reached valid terminal states with corrected contamination. All 122 selected pairs have identical restricted observations but opposite expert first choices, with >=2 action regret for crossing choices. Thus a deterministic policy restricted to first remote survey plus last 64 receipts has a *theoretical* 50% balanced-pair first-action ceiling; this is NOT measured model accuracy.

Six of 128 candidate worlds (5 TRAIN, 1 DEV) failed the strict-pair screen and are recorded in the manifest instead of being reported as successes.

Independent regression reconstructed all 5,212 complete observations from compact history deltas and checked SHA-256; every oracle label was a host-legal accepted action. There were zero contradictory complete-input labels and zero exact input or geometry hashes shared across partitions. 296 duplicate TRAIN observations were assigned zero weight; DEV and HELDOUT observations retain weight one. Two complete CPU generations produced byte-identical six JSONL shards and manifest. Second full generation: 36.19 s, peak RSS 595,080 KiB; these measure the generator, NOT a model.

Run in a source directory containing remote_error_alias_v28.js, test_remote_error_alias_v28.js, causal_gate_v26.js and outcome_forks_v25.js:
- node test_remote_error_alias_v28.js — two deterministic quick generations and complete local replay
- node remote_error_alias_v28.js NEW_OUTPUT_DIRECTORY — full six-shard corpus, refusing overwrite
- node test_remote_error_alias_v28.js NEW_OUTPUT_DIRECTORY — full reconstruction, partition checks, accepted-action and terminal tests

Full output file SHA-256 checks, exact split counts, source/dependency hashes and negative selection cases are pinned in V28_DEFAULT_MANIFEST.json.

## Next game-policy consumer experiment

Train a pinned *actual* LFM2.5-350M or Sprocket-500M checkpoint on TRAIN, tune on DEV, and run complete autonomous HELDOUT replays. Compare full-history, last-64-only and no-intervention-history controls. Measure paired first-choice accuracy, regret, finish rate, leftover contamination, legal-action rate, p50/p95 real model inference latency and peak RSS.

## Negative and scientific boundaries

These are synthetic two-branch/loop graphs, with three paint targets, static original survey and scripted long distractor walk. A compact symbolic interpreter could solve them; this is not proof of neural memory learning. Exact-hash disjointness does not eliminate structural grammar shortcuts. No named SLM weights were updated, no native CyberSand or real host-ABI acceptance took place, and no immutable GitHub Actions dataset artifact was published. No paid API/GPU compute was used. Scope only experiments/driv3r/corpus_factory/** on existing Draft/HOLD PR #1. Never merge or alter other labs.
