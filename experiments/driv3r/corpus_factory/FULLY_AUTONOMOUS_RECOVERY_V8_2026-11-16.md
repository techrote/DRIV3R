# Fully autonomous synthetic recovery v8 — 16 November 2026

**Scope:** DRIV3R research only, `experiments/driv3r/corpus_factory/**`. External host, synthetic materials and obstacles. Draft/HOLD PR #1.

## Research question

Does executed-route memory improve **complete model-controlled recovery trajectories**, rather than only choosing the first arm at a scripted fork? Each synthetic host episode starts independently at (1,13). The fitted policy chooses every subsequent move, including entering an initially ambiguous fork, recognizing dead ends and retracing. The host checks cardinal legality, protects designated cells and applies a seeded paint motif to each newly entered traversable cell. Paint type is **not** chosen by the model.

Each pair has identical local 5x5 materials and menu at the hub, but opposite goal-connected branches outside sensor range. The TRAIN and DEV worlds use straight corridors; HELDOUT also includes an independently parameterized switchback corridor grammar. There is no native CyberSand execution.

## Controlled experiment

Two 14-feature pairwise logistic policies were fitted independently for 24 deterministic CPU epochs on the **same** 6768 TRAIN teacher decisions, 96 complete source episodes, and 6768 chosen-vs-legal-alternative comparisons. The history-aware arm uses per-candidate visit counts, immediate reversals and recency. The sham has an identical input dimension and training rows, with those five slots masked. Inference-time memory masking is a third negative control. Neither optimizer sees DEV or HELDOUT.

| Evaluation | History-aware solved | Sham solved | Memory masked at inference |
| --- | ---: | ---: | ---: |
| DEV (24 episodes) | 24/24 | 0/24 | 0/24 |
| HELDOUT straight (64 episodes) | 64/64 | 0/64 | 0/64 |
| HELDOUT **unseen switchback** (64 episodes) | 64/64 | 0/64 | 0/64 |

Unseen-switchback mean distinct positions: memory 64.25, sham 6; mean two-step reversals: memory 0.5, sham 171. Each 64-episode set is **32 paired seed groups**, not 64 independent worlds. All model actions are validated against host legal menus; no protected cells are entered.

## Exact-source reconstruction and caveats

Run `node fully_autonomous_recovery_v8.js /NEW/output/directory`. This writes a TRAIN-only JSONL corpus, exact JSON report, and original matched-path SVG; it refuses to overwrite an output directory. The generator pins explicit train/dev/held-out seeds, world SHA-256 hashes, counterfactual pair identities and host-applied decision histories. The reproducible compact report is also checked into the research path.

- TRAIN JSONL SHA-256: `6ae2de0fde163616ebf2887c2b92f350e96f848b858af16ed335e64df34286c3`
- Memory-weight JSON SHA-256: `9f08f64759d8ba5501a4d5b5005de9b0b4ebb89e972db2b533cdc377f1596257`
- Sham-weight JSON SHA-256: `a6532ce86406b9a45e55fc962e329898497dd1df25837b413e50b79fd58e52c9`
- Report SHA-256: `c7b8d030d0a4ab94722350963b356984da3aa6c40583377920c0aac3c8711ec4`

**Scientific limits:** This is a constructed tree-corridor environment; the supervisor is a frontier-first synthetic heuristic. It is not a full CyberSand action menu, material simulation, runtime receipt, native model checkpoint, or autonomous paint-choice demonstration. Shared scenario grammar creates substantial correlation. The switchback holdout is distinct at the corridor-construction level, not proof of unrestricted topology transfer. Do not promote this model or imply native/production acceptance.

There are no Actions artifacts in this run: only the source and small source-matched report/gallery are checked into Git, while full generated TRAIN corpora remain reproducible locally rather than in immutable hosted storage. No game-side changes, paid compute or merge.

## Crucial causal qualification: one-step memory alone is sufficient

A fourth dimension-matched 24-epoch logistic learner receives only the *immediate previous-cell* history signal, with all longer visitation and recency signals masked during training and inference. It uses identical TRAIN scenario decisions, hard negatives and optimizer settings. Results:

| World-disjoint split | Full route history | Previous-cell only | No memory |
| --- | ---: | ---: | ---: |
| DEV (24 episodes) | 24/24 | 24/24 | 0/24 |
| HELDOUT straight (64 episodes) | 64/64 | 64/64 | 0/64 |
| HELDOUT unseen switchback (64 episodes) | 64/64 | 64/64 | 0/64 |

**Correction to interpretation:** The success gap against zero-memory sham does NOT demonstrate that long-horizon memory is useful. Immediate anti-reversal alone gets the same complete-episode success and path statistics as full memory. The sham gets stuck near the entrance in short two-step oscillations, so that baseline is not a controlled test of the later ambiguous fork. The earlier scripted-prelude v7 test is a separate, narrower finding. Neither experiment establishes fully autonomous long-memory-dependent recovery.

Future study must construct genuinely loopy, model-generated histories in which the last step cannot reveal the explored bad arm. Compare full history against short-memory controls while holding the action policy, training volume and external generator constant. Do not promote the v8 result as evidence of long-memory necessity.

- Short-memory weight JSON SHA-256: `81e6f6e15fe043a14856df4f243d398d5ba919cdf69cb0c68f2a1863236e28b1`.
- Updated report SHA-256: `c7b8d030d0a4ab94722350963b356984da3aa6c40583377920c0aac3c8711ec4`.
