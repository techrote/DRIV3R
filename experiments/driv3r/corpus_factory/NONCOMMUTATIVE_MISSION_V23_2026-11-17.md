# DRIV3R Corpus Factory v23 — order-sensitive public mission programmes

17 November 2026. DATA ONLY. Synthetic game-policy host, **not native CyberSand**.

## What changed

V21–v22 target selection used a fixed three-receipt XOR and two-workshop layout. V23 replaces the **XOR selector** with a public eight-instruction, noncommutative permutation programme. The complete programme, in its accepted receipt order, maps state 0 through eight publicly disclosed four-state transformations. The resulting state has TWO consequences: bit 0 selects NORTH/SOUTH workshop, and bit 1 selects motif material phase. A model must make full sequential use of the old programme; the same current 5x5 local view, legal actions, recent histories, unordered instruction multiset, first two and last two instructions cannot distinguish the four targets.

World geometry, protected cladding, dynamic shutters, accepted painting, corrective erasure, public law pulses, public goal retargeting, STOP checks, and oracle route planning are inherited *byte-for-byte* from v21–v22. Parent source Git blob IDs: `recipe_grounded_v21.js` `19e8b18139263850930666cae119f58f8b993c82`, `curriculum_pulse_v22.js` `b6f744216859eb91bec925a42f053153c85a7d0d`.

Unlike v22, the correct motif phase and workshop can vary **independently**, giving four target states. The five distinct move+paint steps remain host-executable. The source does not expose the private programme result in model inputs. Oracle-consistency assertions recompute the public mission from observed receipts and reject disagreement with private simulator state.

## Reproducible corpus

`node experiments/driv3r/corpus_factory/noncommutative_mission_v23.js NEW_EMPTY_DIRECTORY`

| Split | Disjoint worlds | Disjoint programmes | Episodes | Oracle-labelled actions | Corrective erases | Law/goal pulses | TRAIN downweighted duplicates |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 12 | 16 | 192 | 12,496 | 384 | 192 | 496 |
| DEV | 6 | 16 | 96 | 10,032 | 352 | 112 | — |
| HELDOUT | 6 | 16 | 96 | 15,392 | 528 | 144 | — |
| **Total** | **24** | **48** | **384** | **37,920** | **1,264** | **448** | **496** |

All 384 expert episodes reached valid terminal states, with zero new oracle illegal actions; host protected cells remain impassable. TRAIN/DEV/HELDOUT use disjoint world fingerprints, eight-instruction orderings, and differing recent-history distractor lengths (88/120/160). The static accepted prelude and remote instructions are synthetic and scripted; **not** a closed-loop neural run. There are 496 exact duplicate TRAIN observations with identical labels, explicitly assigned zero weight. No contradictory complete-model-input labels were found in any split.

Six shard JSONLs and a SHA-256 manifest were generated offline. The default shards total 718,690,150 uncompressed bytes and are intentionally **not checked into Git**. Manifest SHA-256: `59848ba2fb4b47a428672a93030321708a38940927b9b5c51f95ca982aa418bd`.

## Discriminative controls (theoretical information ceilings, not model scores)

Each sixteen-member family balances four final programme states at 4/4/4/4, hence 25% optimal four-class accuracy if only the aliased evidence is provided.

| Observation restriction | TRAIN | DEV | HELDOUT |
|---|---:|---:|---:|
| Latest 64 actions/events/positions | 25% | 25% | 25% |
| Unordered inventory + first two programme instructions | 25% | 25% | 25% |
| Unordered inventory + last two programme instructions | 25% | 25% | 25% |
| First three programme instructions | 31.25% | 31.25% | 37.5% |
| Last three programme instructions | 37.5% | 37.5% | 31.25% |

The first/last-three ceilings are computed from the fixed selected banks, not extrapolated. One earlier v23 bank candidate was **rejected**: its first-three prefix alone perfectly separated four final states (100% classification). The corrected bank has deliberately shared two-instruction prefixes/suffixes with limited third-instruction leakage. All 48 exact eight-instruction sequences are split-disjoint. Four-class target prediction is a stronger requirement than initial NORTH/SOUTH routing: the latter remains 50% when deprived of the programme, even though full mission classification is 25%.

## Qualification

- `node test_noncommutative_mission_v23.js` — 9/9 regression sections PASS, including all-world, all-intervention complete oracle replay.
- Two independent full six-shard regenerations — byte-identical SHA-256 manifests and shard digests.
- Separate default generation — 10.61 seconds wall time, peak RSS 84,680 KiB (Node corpus generator; **not model inference**).
- Parent generator Git blob hashes checked against remote source.
- No checkpoint, LoRA, model training, external GPU/API expenditure, or native CyberSand gameplay execution.
- Tests source SHA-256 `1545e628366c92d5508c5a480425a3a6d9a3a5904081f5cfcaed085666fdcbc5`; generator SHA-256 `4ab6dde5bc08fe9ee5860912f3ed8c0cd9e739e309b2418a32c744aa0b918fa7`.

## Limits / consumer experiment

The public programme can be executed by a small 4-state symbolic interpreter. That control must remain reported; v23 does **not** prove that SLM sequence memory or native game generalisation has been learned. The underlying topology still has only two workshops; this increment fixes the XOR shortcut, not topology breadth. Paint/law pulses are synthetic, not chemical reactions. Generator timing/RSS does not imply model throughput. Model inference, new weight updates, and immutable Actions artifact publication are still absent.

LFM Game Policy's current handoff reports frozen LFM2.5-350M best GP006 adapter but no new true full-model forwards because pinned root weights and executable opt-in runner are not available here. Sprocket likewise lacks its pinned 500M root checkpoint/tokenizer and critical Python deps. Neither lab reports a real v23 consumer run. Next: pinned real SLM train on v23 TRAIN, choose on DEV, once-only closed-loop HELDOUT evaluation (all movement/paint/erase/STOP model-controlled), compare full ordered receipt history with first/last-2, latest64, and symbolic-interpreter controls. Measure exact final motif, wrong-branch residuals, illegal/rejected preferences, path stretch, latency p50/p95 and peak model RSS.
