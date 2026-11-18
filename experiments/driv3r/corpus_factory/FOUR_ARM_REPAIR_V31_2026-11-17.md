# DRIV3R Corpus Factory v31 — four-way remote-contamination memory challenge
17 November 2026 · DATA ONLY · Synthetic host, **not** native CyberSand · Existing DRIV3R Draft/HOLD PR #1

## Why v31
V30 made the most recent PAINT receipt identical between two remote-error histories, reducing a one-receipt shortcut but retaining a 2-choice east/west graph and a 50% restricted-history ceiling. V31 makes **four** equally likely older wrong-material locations (west/east/south/north) require **four different legal optimal first moves**, while preserving the same last PAINT receipt, latest 64 events, current observation fields and no-memory/last-paint-only public projections.

## Host and ground-truth construction
Five paint targets on orthogonal graph corridors: four remote variable-error targets plus one common remote final-error target. Every prefix uses accepted native-*synthetic-host* actions to travel, deliberately paint wrong material at its variable target, return to the rendezvous, travel to the same common target, paint wrong again, return, then make 160 accepted legal distractor moves. No teleportation or hidden intervention bypasses the synthetic Host action receipt. The first accepted paint toggles a side-loop gate. The gate can create an alternative route but is not guaranteed to be consequential on an optimal recovery; do not claim this is a strict dynamic-obstruction causality benchmark.

A fresh exact reverse BFS solves shortest successful recovery suffixes over 5 ternary target states × graph node. Terminal requires two correctly painted targets, no wrong paint anywhere, a triggered gate and successful STOP at the common exit. A replay must ERASE both wrong paints, repaint a sufficient set of targets, and finish. The predecessor node affects movement action spelling (BACKTRACK vs cardinal MOVE) but not unit-cost shortest distance. For every strict family, each of the other three first actions incurs at least **2 additional actions** according to the exact suffix oracle.

The four-case family has identical public **no-memory**, **latest-PAINT-only**, and **last-64** projections; only older durable PAINT receipts identify the branch to repair. The restricted first-action Bayes ceiling is therefore **25%** for a uniformly weighted family. This is not a claim about full closed-loop episode outcomes or actual model accuracy. The bounded public view is checked against an 8,192-character JSON limit (not an actual tokenizer token cap).

## Full deterministic run
| Split | Disjoint graph groups | Four-way families | Episodes | Decisions | Corrective erasures |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8 | 64 | 256 | 7,328 | 512 |
| DEV | 4 | 32 | 128 | 6,208 | 256 |
| HELDOUT | 4 | 32 | 128 | 8,448 | 256 |
| **Total** | **16** | **128** | **512** | **21,984** | **1,024** |

All 128 candidate families qualify. No excluded worlds. Each first label is balanced exactly (BACKTRACK / MOVE_E / MOVE_S / MOVE_W: 128 each over all splits). The minimum wrong-first regret is 2 actions. The 21,984 full-model-input SHA-256 hashes are distinct, with no conflicting full-input labels, no exact cross-split input overlap, and no cross-split geometry overlap. A conservative near-duplicate fingerprint preserving the ordered material-receipt memory, material law, legal actions, position and local view finds no duplicates across TRAIN. This does **not** mean every path is semantically unique: the graph grammar remains simple and correlated.

Independent test reconstructs every observation from episode history and delta records, compares the public-only bounded view, **replays every action on the synthetic Host and recomputes every target using a fresh exact planner**, verifies action receipts and terminal completion, and proves the four-way restricted aliases. Two full generations produced byte-identical six JSONL shards and manifest. Second run: 12.96 s generation, 410,164 KiB peak RSS; independent full corpus replay: 15.38 s, 521,404 KiB peak RSS (these are **corpus tooling** figures, not SLM inference metrics). SHA-256 identities for generator, dependencies, shards are in `V31_DEFAULT_MANIFEST.json`.

```bash
node experiments/driv3r/corpus_factory/four_arm_repair_v31.js /tmp/driv3r-v31
node experiments/driv3r/corpus_factory/test_four_arm_repair_v31.js /tmp/driv3r-v31
```

The destination directory must not exist. Datasets are regenerated rather than committed to Git. TRAIN/DEV/HELDOUT are **disjoint in graph corridor lengths** (4–11 / 14–17 / 21–24); groups and recipe/gate variants are fully pinned. Holdout selection is not used for model hyperparameter tuning. The episode header contains `hidden_interventions` **for audit only**: consumers must pass ONLY reconstructed `model_input` or the `bounded_model_input` to the model, never episode IDs, labels, oracle regret, split or hidden metadata.

## Actual-model consumer experiment
For pinned and *executably loaded* LFM350/Sprocket root or adapters: train weights on TRAIN, tune on DEV, open HELDOUT once. Compare durable full-history bounded view against latest-PAINT-only and no-memory/recent64 ablations. Run **all** subsequent actions in a closed loop on the synthetic Host; report first-action accuracy, four-way family consistency, terminal success, residual wrong paint, excess suffix cost, invalid raw preferences, p50/p95 CPU forward latency, peak process RSS, checkpoint/source/seed SHA-256. Do not call the symbolic shortest-path oracle a trained policy. No target SLM weights were trained here.

## Honest limitations
- The new 5-target four-corridor synthetic graph remains much simpler than CyberSand's native action/observation ABI. Current results cannot establish game-legal native behaviour, general visual geometry skills or transfer.
- Four-way memory is necessary for the selected first-action labels, but a symbolic reader of old PAINT receipts can solve the task. No measured learned memory yet.
- All prefix painting and 160 post-paint travel moves are scripted (but every operation is accepted by the synthetic Host). Only the labelled suffix is expert-controlled.
- The side gate toggles but is not proven to alter the optimal action in these cases. Do not interpret this as proof of dynamic gate adaptation or of physics/material-law reasoning.
- Materials are two abstract recipe labels A/B rather than native reaction laws. No opt-in Actions artifact for generated shard binaries was dispatched; no paid API/GPU compute.

Ownership: all writes confined to `experiments/driv3r/corpus_factory/**` on the existing Draft/HOLD PR #1 branch; no other DRIV3R lab, main, CyberSand or Neurounner edits.
