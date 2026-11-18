# DRIV3R — CAP-CONDITIONING-001 (2026-11-18)

**Outcome: mixed / no qualified autonomous improvement.** A small neural ranker trained on matched counterfactual recovery allowances substantially reduced **futile BACKTRACK** actions but also incorrectly refused many reachable historical goals. The first-action accuracy advantage over equal-update fixed-cap training has a motif-clustered descriptive confidence interval that includes zero. These results should **not** be described as solved long-horizon recovery.

## Exact-source boundary

- Original DRIV3R consolidated Draft/HOLD PR #1 branch `research/import-consolidated-worm-labs`; pre-experiment source commit `9ed4d813565e24cab41fcc41d5f90c1c91c76b43`.
- Synthetic `World`, `policy`, `history_alias_lab`, and `budget_calibration_lab` match Git blobs at that source commit. All source edits are inside the owned `experiments/driv3r/training/long_horizon/` directory. No changes to `d_common` or native game sources.
- Frozen synthetic `World.menu(max_recoveries)` is used as a *research counterfactual*. Changing a recovery cap after the same traversal is **not** established as a native CyberSand API capability.
- The task-specific teacher gives BACKTRACK iff the target is on the history stack at depth <= remaining allowed recoveries; otherwise it says STOP. The perfect deterministic stack-reachability rule is an analytic reference, not an imported neural policy.

## Experiment

- Two newly initialized 24-hidden-unit neural ranker seeds: **833, 834**. Three arms: `paired` (cap 2/4/6 examples on **identical states**), `static` (same optimizer/example budget, but cap 3 repeatedly), `cap_blind` (same paired training with all four intervened cap-derived feature coordinates zeroed).
- Each arm/seed trained with actual NumPy Adam optimization: 96 updates per nominal stage × 3 stages (36, 72, 128), **288 updates per checkpoint**, 1008 examples per stage; six checkpoint files. Stages indicate an optimizer schedule, **not** 36/72/128 executed gameplay decisions in this experiment.
- TRAIN: 14 history motifs with length-12 paths, source TRAIN terrain seeds; DEVELOPMENT: 20 untouched length-14 motifs, separate synthetic seeds. The DEVELOPMENT cap values **1, 3, 5** are absent from paired training. Structural HELDOUT remains sealed.
- Exact same prehistory, painted ink, protected cells, materials, cursor, heading, goal, and action menu for all three cap values in each DEVELOPMENT triad. Only four normalized cap-derived observation features were intervened, consistent with the earlier `BUDGET-CAP-INTERVENTION-001` audit. All oracle labels are computed from exact history depth.

## Complete first-decision DEVELOPMENT census

| Controller | Correct / 2880 | Futile BACKTRACK | Missed reachable BACKTRACK | BACKTRACK choice inversions / 960 triads | Probability nonmonotone / 960 |
|---|---:|---:|---:|---:|---:|
| `paired` | **2759 (95.80%)** | **9** | 112 | **0** | 284 |
| `static` | 2671 (92.74%) | 201 | **8** | 1 | 952 |
| `cap_blind` | 2669 (92.67%) | 149 | 62 | 0* | 0* |

\* An invariance control necessarily has identical cap inputs for the same state and cannot demonstrate successful cap reasoning by having no inversions.

The paired model eliminated 192 of 201 baseline futile backtracks (~95.5% reduction), **but missed 104 additional reachable backtracks**. Aggregate accuracy increased by **3.06 percentage points** (88 of 2880 decisions); the motif-clustered bootstrap descriptive 95% interval was **−0.174 to +6.563 percentage points**, including zero. This is not sufficient evidence of a robust positive generalization effect.

There is meaningful seed instability: the paired checkpoint for seed 833 achieved **1411/1440** correct (29 incorrect), while seed 834 achieved **1348/1440** (92 incorrect). Both seeds reduced futile retreat; seed 834 was particularly conservative on reachable goals. Deterministic stack-reachability remains **100%** on this analytically constructed task.

## Autonomous continuation and negative controls

- **189** sampled seven-step-ceiling synthetic continuation episodes were separately executed with real host-legal action receipts: **245** actual continuation actions, all source-replayed. The `paired`, `static` and `cap_blind` arms all reached the historical goal in **49/63** sampled continuations each. **No autonomous target-arrival advantage** was established.
- Counterfactual continuation terminates when the historical feature extractor's fixed-12-credit menu differs from the altered synthetic-cap menu. This prevents inventing an unsupported action feature mapping; some episodes therefore have zero or very few continuation actions, and these are **not** full 7-step achievements.
- At the frozen first decision, **8640** complete neural probability vectors were computed and independently replayed. **960** state/seed/cap triads were tested per arm. All six checkpoint byte hashes were verified.
- Six focused unit tests and independent full evidence replay passed. CPU-only training/evaluation wall time was **19.087 seconds**; peak process RSS approximately **188 MiB**, model-only median inference **0.017035 ms**, p95 **0.032038 ms**. The full evidence `report.json` retains precise measured figures and SHA-256 hashes.

## Evidence and limitations

The adjacent `CAP_CONDITIONING_EVIDENCE_2026-11-18.json` is a compact source-matched ledger of checkpoint hashes, outcome statistics, per-seed accuracy, archived decision-vector and trajectory-file hashes, and caveats. The original executable source/tests/gallery and a six-panel SVG showing complete host-legal prehistories and subsequent sampled policy actions are included in this PR. The full probability vectors, all 189 sampled action receipts, and six `.npz` checkpoints are retained in the external evidence archive, **not in Git history**; immutable GitHub Actions artifact upload remains outstanding.

Do not describe this as native CyberSand acceptance, a model-level proof of causal monotonicity, generalized navigation, or successful 72–128-step autonomous behavior. There was no paid API/GPU spend and no HELDOUT access.

## Next decision

Use the source-matched negative/positive pairs to test a **cost-sensitive loss** that penalizes false retreat and missed recoverable targets separately, then compare **actual autonomous goal hits** rather than optimizing first-action accuracy alone. Only proceed to 72–128 genuinely executed decision trajectories when those controllers beat the stack oracle on an uncertain-observation task where the deterministic oracle is no longer trivially perfect.
