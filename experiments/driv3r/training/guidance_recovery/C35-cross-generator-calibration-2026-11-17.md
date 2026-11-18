# C35 — Cross-generator executed-return calibration and guarded recovery

**17 November 2026 · DRIV3R Guidance & Recovery · REJECT for promotion**

This is **external, synthetic, host-legal CyberSand-like research**, not native CyberSand validation or a production ABI change. All changes are confined to the Guidance & Recovery laboratory of DRIV3R's single Draft/HOLD PR #1. The historical `d_common` interface is immutable and is not claimed to supersede CyberSand's native contract.

## Question and novelty

C34's neural head modified thousands of actions but failed to beat a goal-rearming controller. C35 tests whether independently trained, host-observed **executed eight-step return differences** generalize from two independent maze generation algorithms to a **withheld loop-erased random-walk graph generator**, and whether calibrated estimates improve *actual* recovery under signed host guidance.

C35 introduces:

- **DFS** and **randomized Prim** 7×7 maze spanning trees for training; **Wilson loop-erased random-walk trees** and **Wilson-braid** trees with 14 added links withheld from training. Each grid is a 15×15 material field with 97 traversable cells (111 for braid).
- An actual 24-move prefix executed through `World.apply()` and then an off-history goal change. A second host-issued goal change may occur at decision 50. Goal candidates and shortest paths are used **only to build fixtures**, never to rank actions or construct inference features.
- 15 host-visible features; three independently seeded, 32→16 hidden-unit Tanh return heads; **668 training samples, 210 separate tree/Prim validation samples, 224 CPU AdamW updates per head (672 total)**. Targets are clipped differences between actual host-executed eight-step BACKTRACK versus forward continuations.
- Four paired policy arms: frozen C18-A0; C34-RESET non-neural control; C35-SIGNED (bounded signed mean return bias on BACKTRACK); C35-GUARD (only positive, three-head-consensus, low-uncertainty bias). Two further **zero-training random initialization** and **sign-inversion** controls test whether head initialization and change direction alone account for results.
- Matched no-cue, weak/strong BACKTRACK, and weak/strong CONTINUE conditions, both personality signs and guidance weights; two independent test-seed cohorts, 70-decision horizon; host-legal menu refreshed on every action; candidate selection remains stochastic.

## Return calibration on separate generator families

| Evaluation family | Counterfactual states | Mean absolute return error | Zero-prediction MAE | Sign accuracy (excluding near-ties) |
|---|---:|---:|---:|---:|
| DFS (fresh seeds) | 288 | 0.668 | 0.724 | 66.0% |
| Prim (fresh seeds) | 184 | 0.607 | 1.065 | 84.1% |
| **Wilson (held-out generator)** | **278** | **0.591** | **0.828** | **80.8%** |
| **Wilson-braid (held-out)** | **232** | **0.528** | **0.537** | **70.1%** |

The withheld Wilson result demonstrates a measurable predictive relationship between observable state and eight-step counterfactual return. On the braided generator, the small MAE reduction over predicting zero is not persuasive. Neither proves useful *long-horizon* neural policy control by itself. The return proxy remains a heuristic teacher, not a shortest-path or native-game oracle.

## Matched, executed gameplay outcomes

*Guided completed goal events; each cohort has 256 guided episodes per policy. Episodes with a second goal can earn two goal events. These counts are not independent Bernoulli trials.*

| Policy | Cohort A | Fresh-seed B | Combined |
|---|---:|---:|---:|
| C18-A0 frozen baseline | 80 | 70 | 150 |
| C34-RESET non-neural rearm | 84 | 74 | 158 |
| C35-RANDOM zero-optimizer head | **90** | 81 | 171 |
| **C35-SIGNED trained return head** | 89 | **89** | **178** |
| C35-GUARD three-head uncertainty gate | 86 | 86 | 172 |
| C35-INVERT trained heads with output signs inverted | 82 | 75 | 157 |

C35-SIGNED gained **20 goal events** against the matched C34-RESET controller. A paired cluster bootstrap, resampling **128 distinct graph/condition clusters** with 6,000 fixed-seed draws, gives a descriptive 95% interval **[+5, +36] goal events**. But the untrained random head also gained 13 events. The trained head's **incremental +7 over random** has bootstrap interval **[−7, +20]**. This means learned policy value remains **unproven**, even though the direction-aware estimate appears causally relevant: reversing its sign loses 21 events versus the trained signed head.

**Strict nonregression failed:** C35-SIGNED regressed against C34-RESET in 5 condition groups in A and 2 in B; GUARD regressed in 6 and 2. With just four guided episodes per family/case/switch/advice-strength group, these results cannot establish robust qualification. Braided and Wilson results remain sensitive to random graph choice, cue quality and action sampling. There was no tuning after observing cohort B.

## Action safety, no-cue invariance and personality

All five intervention variants (including ablations) matched C18-A0's **action and ink hashes in 64/64 unguided episodes per cohort**, for **128/128 paired no-cue checks per model**. No synthetic host-rejected action or protected-cell modification occurred in the 3,840 total matched executions. A small number of raw unmasked model top choices were illegal (for example C35-SIGNED: 3 in A, 0 in B); they were excluded by the host's legal menu. Do **not** extrapolate this synthetic result into native-game safety.

The evaluation also retained guided action-type counts by signed personality and reported painting/INK counts separately from completion. Aggregate INK action counts changed under guidance. No turn-direction/personality acceptance gate or independent native fidelity test has been passed. Therefore **personality preservation under active guidance remains unqualified**.

## Reproduction, source and custody

- Three saved CPU heads reproduced byte-for-byte over **two full optimizer invocations plus a separate source-matched training invocation**. Their SHA-256 values are recorded in `C35-summary.json`; binaries are retained only in the downloadable immutable-hash **local research archive**, not Git history.
- Source dependencies `c34_independent_maze.py`, `c32_offstack_recovery.py` and `c28_temporal_option.py` were verified as byte-identical Git blobs at the inspected DRIV3R branch head `4c8448d045e61e5888844b04a0b298dcfe5879c9` before preparing C35. The final report and source evidence must be checked against the newer exact branch HEAD when publishing.
- **8/8 targeted tests** passed locally. The 24 selected gallery trajectories were independently replayed using the same generated geometry and actual legal actions, verifying prefix, graph, full action and ink hashes. Their full cohort traces and per-family evidence hashes are preserved in the archive.
- Independent bounded evaluation may be executed with `python c35_eval_partition.py <archive-root> <out-dir> <family> --seed 351701 --checkpoint-dir <checkpoint-dir>` for each of the four families; repeat with seed `551704`, then the corresponding `c35_ablation.py` runs. These 16 partitions reproduce the exactly reported policy control inputs. This **is not a claim of native engine scientific reproduction**, and the original checkpoint dependencies must be provided.
- No paid GPU/API resources, routine-push training or new workflow was used. **Immutable GitHub Actions artifact custody and native CyberSand validation are still outstanding.** The retained local ZIP is interim evidence custody only.

**Decision: REJECT C35-SIGNED and C35-GUARD for promotion.** Retain checkpoints and negative results for further scientific comparison. The next meaningful gate is a larger, strictly withheld return-calibration evaluation with a genuine untrained/label-shuffled control that is beaten condition-wise, not merely in aggregated goal counts, and an explicit guidance-preserved personality audit. Do not alter CyberSand, Neurounner or DRIV3R main; retain Draft/HOLD PR #1.
