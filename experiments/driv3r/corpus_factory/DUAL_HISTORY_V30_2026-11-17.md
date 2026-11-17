# DRIV3R Corpus Factory v30 — dual remote-error memory with last-PAINT aliasing
17 November 2026 · Synthetic host only · Draft/HOLD PR #1 · DATA lane

## Research question
Previous v29 paired recovery cases concealed remote contamination from the most recent 64 events, but the last older PAINT receipt could still identify the correct arm. A truncated one-receipt reader therefore remained a shortcut. V30 asks whether distinct remote error histories can require opposite optimal actions **after the most recent PAINT receipt is also held identical**.

## Intervention and legal trajectory
Two deliberately wrong-material paints occur at separate remote targets in a three-target grid graph. The latest wrong PAINT is always at the shared northern target; the *older* wrong PAINT is either at the west or east target. After both, 160 accepted host-legal distractor movements return to an identical rendezvous. The dynamic gate changes state after the first accepted paint, and all movement, painting, erasing and acknowledgements use the existing synthetic host.

The first optimal recovery action is BACKTRACK for one member and MOVE_E for the other. The inferior opposing action incurs at least two additional successful-suffix steps, as established by the v26 reverse-BFS oracle (not hand-labelled proxy geometry). Both episodes must erase both wrong materials, repaint enough correct material, and finish at the prescribed exit.

A compact public-only observation encoder records terrain topology, legal actions, painting recipe, local view, recent events, positions and older event receipts. Its serialized length is capped at 2,048 **characters** per decision; no claim about tokenizer tokens is made. Hidden interventions, oracle regret and pair IDs remain outside policy input.

## Corpus and qualification
| Split | Groups | Paired worlds | Episodes | Decisions | Erasures |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8 | 64 | 128 | 2,368 | 256 |
| DEV | 4 | 32 | 64 | 2,016 | 128 |
| HELDOUT | 4 | 32 | 64 | 2,720 | 128 |
| Total | 16 | 128 | 256 | 7,104 | 512 |

Every world qualifies; 0/128 are excluded. Balanced optimal first choices: 128 BACKTRACK and 128 MOVE_E. Each selected pair has identical local/public state, recent 64 events, no-memory encoding and latest-PAINT-only encoding, but distinct full histories and opposite optimal first actions. Accordingly a classifier limited to either projection has a 50% balanced first-action ceiling. HELDOUT has minimum three-edge remoteness strengthened to eight, and disjoint graph geometry from TRAIN/DEV.

Independent regression reconstituted all 7,104 full observation objects from initial history and decision deltas, then verified SHA-256 of both full and bounded model inputs; all action receipts were accepted, terminal states succeeded, and each episode required exactly two corrective erasures. Across partitions: zero conflicting complete-input labels, cross-split identical inputs or cross-split identical topology, and zero exact duplicate input rows. Six JSONL shard SHA-256 values and dependency hashes are pinned in V30_DEFAULT_MANIFEST.json. Two independent complete generations yielded byte-identical shards and manifest. One generation took 4.20 s wall time and 181,132 KiB peak RSS (generator figures, **not** SLM inference).

Run (requires Node.js and committed v25/v26/v28 dependencies):
```bash
node experiments/driv3r/corpus_factory/dual_history_v30.js /tmp/driv3r-v30
node experiments/driv3r/corpus_factory/test_dual_history_v30.js /tmp/driv3r-v30
```
The output directory must not exist. Generated JSONL shards are intentionally not committed to Git; consumers can regenerate them and check the manifest hashes. An immutable Actions artifact is still outstanding because no opt-in workflow dispatch is available through this executor.

## Controls and limitations
Evaluate real pinned LFM350/Sprocket policies on HELDOUT twice: (A) complete history / all durable receipts, (B) latest-PAINT-only, and (C) recent-history/no-memory. The model must control **all recovery actions**, not merely classify the initial decision. Report matched-pair terminal success, residual wrong paint, realized suffix regret, legal preference rate, p50/p95 CPU forward latency and peak RSS. Train on TRAIN, select on DEV, open HELDOUT once.

This is a synthetic three-target graph grammar, with scripted *legal* prelabel interventions and an oracle controlling labelled suffixes; it is not native CyberSand, a full autonomous-prelude benchmark, or learned memory competence. A simple symbolic full-history interpreter can solve it. LFM350 and Sprocket root weights were not loaded or updated in this lane; current cross-lane handoffs report model/dependency and opt-in execution blockers. No paid API/GPU, native CyberSand, Neurounner or other-lab writes.
