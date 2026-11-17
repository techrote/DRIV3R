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
