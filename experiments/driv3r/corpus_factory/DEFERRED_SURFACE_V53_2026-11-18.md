# DRIV3R Corpus Factory v53 — deferred cross-surface physical warnings (18 November 2026)

**DATA ONLY · synthetic host != native CyberSand.** Published only on DRIV3R's shared Draft/HOLD PR #1 research branch. No native game changes or SLM weights updated.

## Question

Can the consumer recover a non-current surface warning after completing another mission, including when that B warning is interleaved with a second A-surface warning? An agent that keys only on the latest global warning, ignores future-surface warnings, or trusts stale physical calibration should make observably wrong decisions.

## Intervention

**Eighteen** arms per split-isolated world. First objective is A, second B. Early B warnings occur either at the first policy decision or simultaneously with A's second recalibration warning. B's hidden contextual law independently remains the same or flips, and the warning itself does not reveal which. The physically accepted safe PAINT_A probe, material observation, ERASE, and later correct B paint resolve the ambiguity. A remains the active objective until completed. No new legal action tokens or privileged law bits enter model input.

Each world provides seven matched B-no-drift/B-drift physical-outcome pairs, four negative controls showing irrelevant B warnings do not alter the protected A paint, and two controls in which simultaneous A/B warnings do not divert A's immediate action. Unlike v52, the warning is about a *future* objective; the oldest deferred warning is up to 88 host-accepted actions before arrival at its B goal.

## Exact corpus

| Split | Worlds | Successful synthetic episodes | Policy decisions | Safe host probes | Matched B physical pairs | TRAIN zero-weight exact repetitions |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 32 | 576 | 22842 | 1184 | 224 | 11414 |
| DEV | 12 | 216 | 9310 | 444 | 84 | 0 |
| HELDOUT | 12 | 216 | 12636 | 444 | 84 | 0 |

All **44,788** policy records are generated from host-accepted movement/painting/erasing/stopping, including 1,008 successful synthetic episodes and 2,072 physical safe paint probes; no residual probe paint. The independently reconstructed oracle and host replay check every input, legal action, accepted terminal state, safe probe, physical B material, the seven counterfactual pairs and relevant negative controls. Two full generations are byte-identical, including all six shards and manifest.

### Statistical controls and leakage

- TRAIN/DEV/HELDOUT worlds: 32/12/12; 56 unique v53 world geometries, no same observation across splits and no contradictory action labels.
- External comparison with locally supplied v44–v52 release archives: no v53 geometry overlaps earlier versions, and no prior cross-split geometry conflicts. v43 has no comparable geometry hash. This comparison is local evidence, not a universal historical guarantee.
- TRAIN repeated identical model inputs are retained for exact replay but `weight=0` on later occurrences; DEV/HELDOUT remain unsampled. The evaluator-only challenge JSONL must **never** be exposed to models at training or inference.
- Training split's duplicated rows may overstate naive corpus size: use nonzero weights, sample by world/scenario family, and report per-family metrics.
- A stale B calibration that ignores the future warning would select the wrong protected B material in all seven B-flip arms of every world. This is an **analytic deterministic counterfactual**, not an autonomous neural rollout.

### Consumer instructions

```
node experiments/driv3r/corpus_factory/deferred_surface_v53.js NEW_OUTPUT_DIR
node experiments/driv3r/corpus_factory/test_deferred_surface_v53.js NEW_OUTPUT_DIR TRAIN
node experiments/driv3r/corpus_factory/test_deferred_surface_v53.js NEW_OUTPUT_DIR DEV
node experiments/driv3r/corpus_factory/test_deferred_surface_v53.js NEW_OUTPUT_DIR HELDOUT
```

Do not run full three-split independent replay in one unbounded Node process: memory consumption grows with the retained input-dedup map. Isolated per-split replay passed. Model inputs are only `policy.input`. `policy.action` and `weight` are labels, never features. Group by source split, geometry hash and world seed, including cross-version group dedup.

## Limitations and next scientific decision

This is a *synthetic graph host* exercising an experimental warning mechanism; the observational and protected-goal semantics are not validated against native CyberSand. A simple symbolic per-surface event/receipt parser still solves these episodes. Prelude and survey are scripted host actions; model-led discovery was not evaluated. No LFM350/Sprocket checkpoint trained or run, no neural CPU latency/RSS and no native gameplay metrics. Shards are local immutable-by-hash release files, **not yet GitHub Actions artifacts**. Source-smoke CI does not constitute dataset scientific reproduction.

A consumer should train actual pinned small-language-model/adapter weights on v50–v52 versus v50–v53 TRAIN, keeping family-balanced batches, then use HELDOUT autonomous closed-loop policies. Compare full history versus dropped future-B alert, all-B-alert, material-result and short-window ablations; report protected-goal damage, terminal success, independent A/B mission adherence and CPU p50/p95 inference+RSS. Require model/optimizer/tokenizer/checkpoint hashes and no HELDOUT supervision.