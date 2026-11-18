# Corpus Factory v52 — temporal invalidation of host-executed physical probes

**Status:** locally qualified on 2026-11-18; synthetic host, **NOT native CyberSand**. Writable scope only `experiments/driv3r/corpus_factory/**` on existing Draft/HOLD DRIV3R PR #1. No model training, no paid compute or branch change outside PR #1.

## Research question

V51's one-warning/one-calibration task is solvable with a one-shot `warning => probe` rule. What if the calibration becomes outdated *after* the agent departs the safe site, while it is returning to paint a protected objective? V52 injects a second observable, non-diagnostic surface warning during the host-executed return route. A physically valid earlier PAINT_A receipt must be invalidated; the agent must retrace to its safe site, perform another actual PAINT_A, observe the new physical result, ERASE it and return to the original objective. New warning alone does not expose the hidden law and must never be confused with a positive physical measurement.

## Reproducibility and execution

From `experiments/driv3r/corpus_factory` run:

```bash
node realert_v52.js /tmp/driv3r-v52-a
node test_realert_v52.js /tmp/driv3r-v52-a TRAIN
node test_realert_v52.js /tmp/driv3r-v52-a DEV
node test_realert_v52.js /tmp/driv3r-v52-a HELDOUT
node realert_v52.js /tmp/driv3r-v52-b
# sha256sum /tmp/driv3r-v52-{a,b}/* and compare corresponding files
```

All source, parent and shard SHA-256 digests and split counts are fixed in `V52_DEFAULT_MANIFEST.json`. The generator writes six policy/evaluator-only JSONL shards and `manifest.json`; do not commit large generated shards, upload them to immutable opt-in Actions artifacts when a lane-authorized workflow is available. The CLI requires a **new nonexistent** output directory, preventing accidental overwrite. Full generation ran twice and the seven files were byte-identical.

| Split | Different geometries | Completed synthetic episodes | Labelled legal decisions | Safe physical probes | Repeated warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 32 | 384 | 14,420 | 640 | 288 |
| DEV | 12 | 144 | 5,488 | 240 | 108 |
| HELDOUT | 12 | 144 | 6,948 | 240 | 108 |
| **All** | **56** | **672** | **26,856** | **1,120** | **504** |

Each scenario has 12 arms, 4 strict matched physical-result contrasts after a second warning, no-warning and single-warning negative controls, repeated-warning/no-drift controls, and opposite prior-calibration contexts. Initial gate/mission history and partial-map surveys remain. Training records duplicated exactly are kept for replay but assigned weight zero (6,519 records); evaluator-only arms and latent law flags must never be model-visible. Tested geometry signatures: no overlaps with provided v44–v51 releases.

## Independent evidence

The audit reconstructs observation assembly, teacher actions, host action acceptance, material-result receipts, warning injection and terminal protected-goal outcomes separately from the main generator. Independently replayed **all 26,856 legal labelled decisions** to **672 successful synthetic terminal episodes**. Per-shard SHA-256 and parent-source checks passed. No cross-split observation or geometry identity collision and no conflicting labels were observed. Four causal second-probe contrasts were asserted per geometry: identical observations and action prior to the second physically executed PAINT_A, differing material outcomes, then differing correct protected-goal commands; deleting only the post-second-warning receipt equalizes each relevant goal-decision input.

**Deterministic negative control:** a stale-warning symbolic controller ignores the second notice, incorrectly trusts the first physical calibration and continues toward the goal. It completed 392/672 episodes but irreversibly damaged protected targets in the other 280/672, including 60/144 HELDOUT failures. The current timestamp-aware oracle succeeded 672/672. These are **symbolic controls, not measured neural accuracy**. A tiny per-surface parser can still solve all cases; V52 therefore increases the memory/intervention burden but cannot establish SLM generalization without actual weight-training and autonomous closed-loop evaluation.

## Remaining blockers / consumer gate

1. The synthetic `SURFACE_ALERT`, protected-goal failure and context-specific tool-law mechanics have **not** been mapped to canonical CyberSand native observation/action ABI. Native semantic acceptance requires a separate READ-ONLY investigation, not a game modification.
2. No root LFM350/Sprocket fine-tuning or inference was carried out in the Corpus Factory lane. No neural latency/RSS results, native terminal trajectories or immutable Actions dataset artifacts exist.
3. Training may overfit symbolic event codes or repetitive geometry. Run held-out sampled-history, post-alert receipt-ablation, event-timestamp permutation, delayed-invalidation, surface-specific and full-trajectory controls on the actual pinned root model. Compare v50+v51 to v50+v51+v52, and report autonomous terminal success, legal-action rate, protected-goal damage, action regret, PAINT/ERASE/STOP correctness, CPU p50/p95 inference and peak RSS. Keep heldout labels and challenge metadata evaluator-only.

**Safety:** no named SLM trained, no API/GPU expenditure, no writes to CyberSand or Neurounner, no automatic push-triggered training or changes to other DRIV3R laboratory directories. PR #1 remains Draft/HOLD, no merge or Ready.
