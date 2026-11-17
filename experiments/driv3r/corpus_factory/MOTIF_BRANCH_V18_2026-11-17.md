# Corpus Factory v18 — ordered relay → protected-canvas motif authoring

17 November 2026 · Data lane · synthetic host only · DRIV3R PR #1 Draft/HOLD.

## What changed

This is **not another route-only permutation corpus**. V18 introduces a new, fully replayable **authoring** task. A synthetic actor starts at a corridor junction after a scripted long prelude of ALPHA relay instructions and irrelevant BETA events. The ordered ALPHA commands determine whether the NORTH or SOUTH work area is authoritative. The actor must navigate to that area, step around an impassable protected cell, paint **five individual cells** with the correct alternating material motif, and return to the eastern exit. The inactive work area must remain unpainted.

The executable host offers `createHost`, `observe`, and `apply`; movement, paint overwrite, ERASE, accepted receipts and illegal-action rejection are implemented. All navigation and painting actions **after the prelude** can be selected by a consumer SLM. No visual oracle marker or privileged motif target is exposed in the model observation. Incorrect painting can be corrected via ERASE; early arrival at the exit is not success. The local view is a 7x7 character grid, not a native CyberSand image or material snapshot.

The model-visible rules explain instruction semantics and both possible painting laws, but do not disclose which branch is currently correct. ALPHA history, not the latest local view, determines that choice. These are fictional synthetic commands and brush semantics, **not native CyberSand host-legal ABI fixtures**.

## Dataset and science controls

| Split | Worlds/families | Episodes | Oracle-labelled decisions | Instruction gap |
|---|---:|---:|---:|---|
| TRAIN | 4 | 48 | 1,800 | adjacent SETs |
| DEV | 2 | 24 | 948 | one intervening TOGGLE |
| HELDOUT | 2 | 24 | 996 | 2–6 intervening TOGGLEs |
| **Total** | **8** | **96** | **3,744** | split-disjoint order families |

The generator verifies **3,744 replayed expert actions, 480 accepted paint receipts, zero accepted illegal oracle actions, zero protected-cell entries and 96/96 completed expert episodes**. Every episode contains five protected-aware paint operations; the remaining actions traverse the corridors and bypass. No actual SLM policy was executed.

Every balanced 12-case world family shares exactly the same initial 7x7 grid, objective, action menu, last-64 history and unordered event inventory, but has six NORTH and six SOUTH first-actions. The family-conditional information ceiling for those restricted views is **50%**. First-two and last-two ordered ALPHA tokens are also identical across siblings. First-three/last-three shortcuts can remain **7/12 (~58.3%)**; full ordered histories permit deterministic composition. A simple finite-state SET/TOGGLE interpreter can solve the hidden branch decision, so this is **not evidence of sophisticated episodic reasoning**. Paint sequencing and protected-cell traversal are the added behavioral requirements.

TRAIN/DEV/HELDOUT use different world identifiers, rendered styles, topological arm lengths and source instruction sets, but each split still contains only two arm lengths. Source order partitions are deliberately inherited from v17: do not combine v17 and v18 as independent examples or assert out-of-family generalization from local corpus qualification alone. In particular the small eight-world evaluation has limited statistical power; report family-level outcomes.

## Exact qualification

- `node test_motif_branch_v18.js` — **7/7 regression sections PASS**: split integrity, alias invariants, end-to-end painting success, protected-cell rejection, premature-exit rejection, isolated model observation, deterministic regeneration, on-disk SHA validation and overwrite refusal.
- `node motif_branch_v18.js NEW_DIR 2` executed twice, with all **six JSONL shards and manifest byte-identical** between fresh output directories.
- Generator measured **0.76 / 0.51 s** elapsed, peak process RSS **137,284 / 137,096 KiB** on Node 22.16.0 Linux. These figures describe data generation, not neural inference.
- Generator Git blob: `e120babfa7826a2cfb7deb3866f6b7871ed74a45`; regression Git blob: `f7dbb9043ad49145835e1f1bbc42d86131f85d70`. Order source v17 blob: `014a25750f368f68b3397d0e413f7faca5ea4c67`; transitive v13 host blob: `b49afca90c307d8126c8f4c1becba667b7ced6f1`.
- Manifest SHA-256: `a11358b0e6b2ef4676eeb316655de577b0ba95576a887fdd62b64f4108eda097`; canonical disk output `V18_DEFAULT_MANIFEST.json` captures exact row counts, bytes and SHA-256 for all six shards. Full shards are generated on demand, not included in Git.

## Consumer protocol

Run from `experiments/driv3r/corpus_factory/`:

```sh
node test_motif_branch_v18.js
node motif_branch_v18.js /new/output/directory 2
```

The consumer receives **only** `row.model_input` or `observe(host)`; never pass `privileged_host`, `scripted_environment_events`, `target`, `phase`, `group_id`, `hard_negatives`, or metadata as model input. Fit on TRAIN, use DEV only to select, and seal HELDOUT until the evaluation configuration is fixed. Evaluate paired 12-way first action and *model-controlled* edit completion, number of accepted but wrong paints, necessary erases, protected-cell action rejection, path stretch, and inference latency p50/p95 plus model RSS. Compare full event order to last64, event-bag, first-two and last-two ablations. Do not calculate confidence intervals treating the 3,744 within-world decision rows as independent.

This host is independent of production CyberSand. It establishes no native Godot brush semantics, protected material semantics, material-paint receipts, ABI compatibility, GPU performance or checkpoint fitness. There is no SLM/adapter training here; trainer lanes own that experiment. No paid compute, external-repo writes or model binary additions.
