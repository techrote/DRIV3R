# DRIV3R Corpus Factory v21 — recipe-grounded game-policy corpus

17 November 2026. DATA ONLY, synthetic host; **not native CyberSand**. Existing Draft/HOLD PR #1, lane `experiments/driv3r/corpus_factory/**`.

## Why this repairs a scientific defect

V20's synthetic oracle painted from `world.palette` and the ordered canvas strip, yet `observe(host)` exposed neither the world's permitted palette nor a reliable canvas-slot identifier. A learner restricted to `model_input` could not infer the correct paint action in unseen worlds, even though the privileged oracle finished every episode. V20 remains preserved; v21 makes the *law publicly observable*.

V21 adds `public_recipe` containing the two material names, a five-cell motif vector, the historical phase rule `OCHRE_XOR_VIOLET`, and zero-based indexing. The local 5×5 observation of a canvas tile now includes its public slot (0–4) and workshop branch. It does **not** expose the hidden active workshop, private receipt-derived phase, full world geometry, split or variant. The oracle uses the public motif with the existing receipt phase.

TRAIN has two material-law patterns, DEV one distinct pattern, and HELDOUT two other distinct patterns. Thus a model must use the visible recipe to generalise painting to previously unseen patterns. This is not evidence that any actual SLM has learned the rule.

## Measured corpus

| Split | World groups | Episodes | Oracle decisions | Exact repeated model inputs | Unique model inputs |
|---|---:|---:|---:|---:|---:|
| TRAIN | 20 | 160 | 6,880 | 240 | 6,640 |
| DEV | 8 | 64 | 3,920 | 12 | 3,908 |
| HELDOUT | 8 | 64 | 4,848 | 12 | 4,836 |
| **Total** | **36** | **288** | **15,648** | **264** | **15,384** |

In TRAIN, 240 exactly repeated, same-label full model inputs are retained with `training_weight:0` rather than deleted from replay histories. All other TRAIN rows have weight 1. DEV/HELDOUT retain all rows at weight 1. The generator **throws on conflicting labels for identical full observations**. An independent corpus audit found zero such conflicts.

All 288 expert episodes complete in the synthetic host. There were 288 shutter events, 104 deliberately refused prelude protected moves, zero accepted protected entries and zero subsequent illegal expert actions. The historical three-receipt counterfactual families still show balanced NORTH/SOUTH first moves under last-64 or any one-/two-receipt restriction (50% theoretical ceiling); this is not measured neural-model accuracy. Additional tests show same-workshop siblings needing opposite paints at the same local tile when the old receipt-derived phase differs.

Qualification: **11/11 standalone Node.js regression sections PASS**, including two independent byte-identical full six-shard generations, oracle trajectory replay, public recipe audit, unseen-law split checks, legal-actions and duplicate weights. One independent full generator run took 2.94 seconds and peaked at 73,776 KiB RSS (Node.js 22.16.0, Linux). These are *generator*, not SLM inference, metrics.

Source Git blob: `19e8b18139263850930666cae119f58f8b993c82`.
Test Git blob: `14a4e8b69341e110af85cb211f88bc111962c63e`.
Generated `manifest.json` SHA-256: `0b98ffd189c4b8eea30a384852627b42055013fa4b15293f7ed81d25e4f2bdb5`.
`V21_DEFAULT_MANIFEST.json` pins the six JSONL shard hashes, byte sizes and row counts. Large generated shards are deliberately **not committed** to Git.


**Direct identifiability falsification:** At a matching first canvas cell across all 288 episode variants, v20 produced 56 identical complete model-visible observation repeats, **44 of which demanded conflicting paint actions** (e.g. identical input required PAINT_SAND in one world and PAINT_METAL in another). The v21 corrected projection produced 8 repeated inputs and **zero conflicting labels** among the same 288 candidate states. Regression #10 enforces that property for all world groups. This demonstrates that the v20 hidden palette was a real data defect rather than a speculative convenience issue.

## Reproduction and consumer contract

```sh
node experiments/driv3r/corpus_factory/test_recipe_grounded_v21.js
node experiments/driv3r/corpus_factory/recipe_grounded_v21.js NEW_EMPTY_OUTPUT_DIRECTORY 1
```

Fit real named SLM/adapter on TRAIN with the sample weights, choose hyperparameters on DEV, evaluate HELDOUT only after selection. In closed loop, call `makeRecord` → `createHost` → `observe` → `apply` with actual model-selected actions every step. Never pass `private_target`, entire `world`, the oracle or evaluation labels as inference features. Compare full causal history versus last64/two-receipt/zero-history, report terminal paint correctness, wrong-workshop contamination, protected refusals, path stretch, latency p50/p95 and model RSS.

## Honest limitations

V21 still has a two-workshop synthetic topology, long *scripted* prelude and a small XOR instruction rule; it has not demonstrated real learned memory or broad level transfer. The new paint law is well-specified, not proven mastered. Native CyberSand action/observation acceptance has not been established; no actual LFM or Sprocket SLM weight updates, inference rollouts or immutable Actions artifacts were produced.

Independent trainer-lane handoffs report pinned root-model/tokenizer/dependency caches are missing and the connector cannot opt-in dispatch their CPU model workflow. **Do not substitute another toy/surrogate trainer for these experiments.** The next useful experiment is full, model-controlled held-out painting with a pinned real LFM/Sprocket adapter and transparent runtime/resource measurements.

No paid API/GPU, CyberSand/Neurounner write, DRIV3R main modification, or other lab ownership violation. PR #1 remains Draft/HOLD.
