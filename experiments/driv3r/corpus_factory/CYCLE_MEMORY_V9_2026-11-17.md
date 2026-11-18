# Corpus Factory v9 — loopy history-aliasing challenge (17 November 2026)

This is **new, deterministic external-synthetic corpus generation**, not a game patch or a new policy trainer. It directly addresses v8's negative result: a previous-cell-only policy equalled the full-history policy on every v8 held-out corridor. Here, two large graph cycles share a return stem. A host-applied visit of either remote branch returns to the same junction via the **identical last 12 positions**, with equal local material grid, action menu and immediate heading. The next correct branch reverses between counterfactual siblings. The full route/probe history distinguishes them; a previous-cell/12-step-only input cannot.

The challenge also includes a distinct **remote material-law** pair: the same surveyed west-sand/east-metal corridors have reversed optimal branch costs under dry and wet laws. The law is an explicit input; oracle labels follow the cheaper surveyed branch. These two task types are not conflated.

## Generated corpus (default 16 style seeds per topology)

| Partition | Unique topology templates | World groups | Paired counterfactuals | Labeled decisions |
|---|---:|---:|---:|---:|
| TRAIN | 36 | 576 | 1152 | 2304 |
| DEV | 12 | 192 | 384 | 768 |
| HELDOUT | 12 | 192 | 384 | 768 |

**Total scripted host-applied prelude steps:** 270336. **Pairs:** 1920. All candidate moves at the fork are checked against the independent synthetic cardinal action menu; the host trace is checked cell by cell against the open/protected sets. Counterfactual siblings share a group and partition. Geometry templates (span,height) are disjoint across splits; HELDOUT uses larger spans. Deliberate sibling duplicates in restricted observations are intrinsic to the experiment and must not be removed as accidental duplicates.

Run `node experiments/driv3r/corpus_factory/test_cycle_memory_v9.js`; generate with `node experiments/driv3r/corpus_factory/cycle_memory_v9.js /NEW/OUTPUT/DIR 16`. The generator refuses to overwrite prior outputs, writes a SHA-256 and byte-count manifest for each shard, and explicitly marks only TRAIN optimizer-eligible.

**Critical limits:** this is a *one-action counterfactual decision corpus*, not a successfully learned long-memory navigation policy. The previous exploration route is **host-scripted and verified**, not selected by a trained model. The graph is deliberately constructed so local and short history are aliased; this is a discriminative benchmark by design, not evidence of general-world robustness. Synthetic material tokens and cardinal actions do **not** claim native CyberSand ABI parity. No official held-out model result or checkpoint is claimed. Large generated JSONL files are produced on demand and are not stored in Git; immutable Actions upload remains an open follow-up. No automatic CI dispatch or paid compute.
