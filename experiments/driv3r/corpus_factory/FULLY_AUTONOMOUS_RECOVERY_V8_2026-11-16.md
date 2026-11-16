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
- Report SHA-256: `055bf7a9b990a91cb0f027c401f8074bd9a7615cc64889eb85c6c1ba57294378`

**Scientific limits:** This is a constructed tree-corridor environment; the supervisor is a frontier-first synthetic heuristic. It is not a full CyberSand action menu, material simulation, runtime receipt, native model checkpoint, or autonomous paint-choice demonstration. Shared scenario grammar creates substantial correlation. The switchback holdout is distinct at the corridor-construction level, not proof of unrestricted topology transfer. Do not promote this model or imply native/production acceptance.

There are no Actions artifacts in this run: only the source and small source-matched report/gallery are checked into Git, while full generated TRAIN corpora remain reproducible locally rather than in immutable hosted storage. No game-side changes, paid compute or merge.
