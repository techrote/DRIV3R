# DRIV3R Corpus Factory v54 — true temporal event-stream benchmark

**Status:** local corpus and independent audit PASS; publication into shared DRIV3R Draft/HOLD PR #1 is a separate gate. **Synthetic graph host != native CyberSand.** No model was trained.

## Research question

The v44–v53 `input.durable` field provided the model with *every* historic mission, obstruction, calibration receipt and warning on **every** decision. Consequently, a stateless symbolic controller could repeatedly re-parse the entire history. Large event age did **not** imply that a neural policy had to remember anything. v54 directly closes this particular observation shortcut without inventing new game physics or targets.

## Mechanism

The fully qualified v53 host-accepted trajectories are regrouped into ordered episodes (each ends in STOP). The first observation sends the complete observable scripted prelude. Thereafter, `events_delta` sends only newly received historical events (a multiset difference that handles simultaneous receipts), never the full `durable` ledger. Both `recent` and `recent_positions` are limited to eight actions. All other fields and all legal action tokens remain unchanged. The policy must maintain state across steps to remember older material probes, mission revisions, remote gates and future-B warnings. Source labels, loss weights, split world memberships and episode lengths are preserved without modification.

## Measured outputs

| Split | Worlds | Complete episodes | Decisions | Historical receipts not retransmitted on decision | Conflicting instantaneous-input keys |
|---|---:|---:|---:|---:|---:|
| TRAIN | 32 | 576 | 22,842 | 19,194 | 168 |
| DEV | 12 | 216 | 9,310 | 7,942 | 59 |
| HELDOUT | 12 | 216 | 12,636 | 11,268 | 55 |
| Total | 56 | 1,008 | 44,788 | 38,404 | 282 |

The 55 conflicting HELDOUT instantaneous inputs account for 560 individual decisions. In this specific contradictory subset, the maximum achievable label accuracy of a majority-vote *memoryless input-only lookup* is **65.7143%**. Across all HELDOUT rows the corresponding majority ceiling is **98.4805%**, so aggregate accuracy alone would obscure the memory test. These are deterministic empirical properties of this dataset, not accuracy of a neural model. No conflicting labels exist for identical *full* source observations or identical *observable sequence prefixes* in any split.

**Negative controls:** an input-only controller has no information to determine the correct action when the same v54 observation appears with contradictory labels. A sequence-aware controller can in principle distinguish these from prior event receipts. The actual trainable model must demonstrate that ability; the dataset does not claim success.

## Reproduction

Requires Python 3.11+ standard library, the pinned v53 release archive and its six intact source shards:

```sh
python3 stream_v54.py --source /tmp/v53/corpus --output /tmp/v54-stream
python3 test_stream_v54.py --source /tmp/v53/corpus --export /tmp/v54-stream
```

The generator refuses partial/quick v53 sources, source hash mismatches, altered historical receipts, illegal labels, premature STOP, cross-split topology overlaps, conflicting full-state actions and identical observable prefixes requiring opposite actions. The independent auditor reconstructs and checks every decision against the original v53 source, all materialized output file hashes, split grouping, exact event deltas, eight-action truncation and metadata isolation. Full regeneration was byte-identical; no pretrained SLM was involved.

## Training/evaluation contract

Only `TRAIN_stream.jsonl` is an optimizer input. Each line contains `steps`, an ordered sequence of `input`, supervised `action`, and source `weight`; **reset recurrent state at episode boundaries** and honor source zero weights for duplicate labels without deleting transitions needed for memory state. Train a pinned named SLM or adapter, not a fresh toy surrogate.

`DEV_EVALUATOR_ONLY_stream.jsonl` is for development evaluation, `HELDOUT_EVALUATOR_ONLY_stream.jsonl` is sealed evaluation-only, and all `*_provenance_EVALUATOR_ONLY.jsonl` are quarantined audit sidecars. Never concatenate or show them to a learner. Keep the 56 source worlds and all matching earlier-version topology/seed families split-locked. Do not shuffle decisions within an episode.

Compare actual SLM autonomous closed-loop episodes **with and without reset/per-step event memory**, report protected material accuracy, future-surface warning retention, goal completion, wrong-paint recovery, inference p50/p95 and peak RSS. Give a separate score on the HELDOUT contradictory-input subset. Validate results in native CyberSand host-controlled ABI before claiming game-ready competence.

## Limitations

No new native event API or physics was tested. This is an **observation-scheduling transformation of v53 synthetic-host trajectories**; the inherited prelude remains scripted and receives historical events in bulk at the first model step. The original v53 controller supplies the labels. Complete source episodes already had valid host execution, but this run did not execute a target model against the host. The dataset proves *insufficiency of current-step observation* for some cases; it does not prove learnability by LFM350, Sprocket, or any other SLM, nor measured latency/RSS. Immutable GitHub Actions dataset publication is outstanding.
