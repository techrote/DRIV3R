# Corpus Factory v11 — delayed causal memory with sealed grammar shift

17 November 2026 · research-only data challenge for CyberSand-style navigation and authoring.

The v8 fully autonomous corridor control found that immediate previous-cell memory matched full memory (64/64 on each tested geometry). The v9/v10 datasets constructed counterfactual short-history aliases, but v10 used a 12-position control, and its motif task remained solvable from the most recent paint receipt. This increment raises the discriminative horizon to **32** and introduces an actual periodic motif where 32 most recent paint receipts are insufficient.

Three paired challenges use the **same present local 7×7 material view, legal action menu, present goal/cursor, and last 32 host-applied positions or paint receipts**. Opposite correct actions require older causal evidence:

- `delayed_shutter`: an off-screen gate probe was denied, then the actor returned to the junction and underwent 48 identical stem moves. The safely reachable route is verified by shortest-path search with the remembered gate closed.
- `delayed_material_law`: an older DRY/WET material-cost revision changes the minimum-cost remote route, followed by identical approach and >32 common moves. A weighted graph oracle verifies the label.
- `periodic_motif`: a 37-cell repeated painting pattern has 36 common cells and one discriminative cell. After another 36 common cells, the next symbol matches the older discriminative one. The last 32 accepted paint receipts are identical across opposing labels.

Train, DEV and HELDOUT use **disjoint topology dimensions and grammar categories** (plain loops, one-spur loops, two-spur loops). The common twin-loop backbone remains the same, so this is a bounded shift rather than independent CyberSand realism. Paired siblings stay in the same partition. A Hamming-distance-4 cross-partition near-duplicate hub-view guard rejects non-sibling lookalikes. Only `model_input` may be consumed as features; all top-level labels/controls/evidence/scenario IDs must be excluded.

Synthetic host trajectories are executed and checked for cardinal, open, non-protected movement; oracle labels are checked against the legal menu. This is **not** proof of learned policy competence, a native ABI qualification, or a real model checkpoint. Deterministic/oracle planners label examples only. HELDOUT remains sealed from optimization or hyperparameter selection.

Generate locally without overwriting an existing directory:

```sh
node experiments/driv3r/corpus_factory/test_stateful_challenge_v11.js
node experiments/driv3r/corpus_factory/stateful_challenge_v11.js /NEW/OUTPUT/DIR 16
```

Each output contains SHA-256-manifested TRAIN, DEV, and HELDOUT JSONL shards. Keep large generated corpora outside Git; use opt-in immutable Actions artifact publication when approved. The next consumer should compare full event/paint history against matched last32 and zero-history inputs on TRAIN/DEV, freeze model/hyperparameters, and then measure HELDOUT paired action accuracy and closed-loop transfer to independently generated synthetic worlds. No optimizer is run in this data lane.

## Default full-generation qualification

Two independent V8 runs of the committed source's pure dataset generator produced byte-identical records on all three splits, with exact paired aliases and legal oracle choices validated. The generation was executed in the research tool's JavaScript runtime, **not** Node CLI; the committed Node CLI test itself has not been executed. A separately checked pure-JS SHA-256 implementation (empty and `abc` known vectors) hashed the exact JSONL payloads. The following digests can be verified by the Node CLI in an independent executor:

- TRAIN: 1536 rows; 11687034 bytes; SHA-256 `d04cd14c109e0c8fa389c2c58791c03ec4c7306625d0c085edbef8dfcc9b5dc6`
- DEV: 576 rows; 4506714 bytes; SHA-256 `28118499ee8e768f60b0a2d25f568b22aa176d24b3b39675896174124e414f91`
- HELDOUT: 576 rows; 4612320 bytes; SHA-256 `76444f56d1852d5daab6aeb79141f09ac1918b76bc76ea8d984ed51ac3c6ade1`

All 2,688 rows cover 1,344 balanced counterfactual pairs and 448 worlds. The host prelude validator counted 222,976 cardinal movement steps, 65,408 accepted synthetic paint receipts and zero protected-cell entries. The corpus and SHA metadata remain locally generated and are **not yet immutable GitHub Actions artifacts**.

## Source-matched material-consistency remediation

The initial v11 generator emitted randomly mixed material tokens on traversable left/right corridors while the material-law oracle treated the entire western branch as sand and eastern branch as metal. That was physically inconsistent. Corrected the synthetic host's rendered corridor tokens to west=sand / east=metal, left the surrounding material noise and protected cells unchanged, and added explicit per-material costs to the old law-change receipt. Revalidated the full default corpus and regenerated the SHA-256 shard manifest. This correction supersedes previous v11 shard hashes; no earlier default corpus was published as an Actions artifact.

- TRAIN: 1536 rows, 11707002 bytes; SHA-256 `98e39823bdee55a3576704f7394183e51bf79a61c93fa39f0e83c6020988b50c`
- DEV: 576 rows, 4514202 bytes; SHA-256 `ccec8834b0d7962801c34bbd3ae3fd65a45bea88cc893fb536cb60073a8900d7`
- HELDOUT: 576 rows, 4619808 bytes; SHA-256 `af7395afebcb60fc82bb2cab4b9079f62ffc7cd13a1f60521188c94a034b3132`
