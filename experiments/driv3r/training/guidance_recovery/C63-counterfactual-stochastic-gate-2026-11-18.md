# DRIV3R Guidance & Recovery — C63 stochastic counterfactual recovery gate

**18 November 2026 · Draft/HOLD · NO PROMOTION.** Owned location: `experiments/driv3r/training/guidance_recovery/**` in **techrote/DRIV3R PR #1 only**. No edits or write actions against CyberSand, Neurounner, DRIV3R main, native/Workbench or the immutable historical `d_common` interface.

## Purpose and advance declaration

C62 corrected C61's unequal optimizer budgets but still found no reliable out-of-family remote-goal improvement against a shuffled delayed-reward control. C63 tests a different failure mode: **might an episode-level learned guidance gate decide when to activate the already effective non-neural observed-FRONTIER recovery option instead of RESET, without suppressing useful exploration or imposing deterministic control?**

The gate is a 12-feature PyTorch linear sigmoid head over ONLY the announced remote goal, host-observed action history/frontiers, present host menu-derived facts, personality sign and current issued cue strength. Hidden world connectivity, protected map, shortest-path oracle, future rewards and actual counterfactual outcomes are forbidden at inference. The selected option is **soft FRONTIER or RESET** for one recovery suffix; subsequent host-legal action selection is stochastic and retains unchanged host-authorized action menus. Gate probability is bounded to `[0.1, 0.9]`, using one source-deterministic but outcome-independent uniform draw shared across all learned and coin-control policies. No online policy rewrites or native changes.

Training labels are derived from pairs of actual matched TRAINING synthetic-host rollouts: soft advantage = 12×goal-completion difference + 0.045×novelty difference − 0.012×length difference + 0.02×remaining-distance improvement. Optimizer: **128 matched CPU AdamW steps per head**, 64 examples per batch, both heads with exactly the same initial weights, training prefixes, observable features and minibatch contract. SHUFFLED permutes counterfactual advantages within topology/cue/damage strata to break action/credit assignment, retaining outcome marginals. COIN chooses FRONTIER 50% of the time with the same draw. No held-out labels influence training or threshold selection. Independent deterministic refits reproduced identical model weights.

C63 seed `630118`, 106-step overall horizon with prefix tick 9, fixed weak 0.35 / strong 1.0 / none cue arms, 0% and 50% memory erasure, five familiar/Wilson training cohorts vs never-trained Kruskal and fresh Wilson/familiar test cohorts. Past C60 historical, C61 and **all 76 reconstructed C62 graph identities** excluded before generating tests; reconstructed C62 first/last hashes checked against committed C62 evidence. Generated 68 training graph identities, 96 test identities. The fixed prior remote-goal prefix filter accepted **43 training** and **66 test** graphs, of which **45 are genuinely out-of-distribution Wilson/Kruskal**. Excluded graph receipts and skipped-prefix reasons are retained; do not treat all generated graphs as scored trials.

## Main held-out outcome

Completed remote goals, pooled **weak + strong** cues in the **intact-memory** condition. The 50% memory-damage condition reuses the same graph/goal identity and must not be counted as independent confirmation.

| Controller | Kruskal 44 | Wilson 46 | Familiar 42 | Total 132 |
|---|---:|---:|---:|---:|
| C18-A0 with issued cue | 8 | 4 | 8 | 20 |
| C34-RESET | 10 | 10 | 14 | 34 |
| C60-FRONTIER, non-neural | 12 | 13 | **21** | **46** |
| Frozen C60-ALIGNED | **13** | **14** | 16 | 43 |
| **C63-ALIGNED** | **10** | **12** | **20** | **42** |
| C63-SHUFFLED | 10 | 12 | 20 | 42 |
| C63-COIN | 10 | 11 | 20 | 41 |

**Negative result:** learned ALIGNED and wrong-credit SHUFFLED achieved *identical goal counts in every topology family*. C63-ALIGNED tied RESET on never-trained Kruskal, underperformed frozen C60-ALIGNED and the non-neural FRONTIER policy there, and failed the strict nonregression gate: **2 Kruskal losses**, **1 Wilson loss** and **1 familiar loss** against matched RESET conditions. On 22 Kruskal graph clusters its paired goal delta over RESET was exactly 0, with a descriptive graph-cluster bootstrap 95% span approximately **−5 to +4 goals**. On fresh Wilson it gained two goals (interval approximately −2 to +7), again unseparated from SHUFFLED.

Across 132 intact-memory guided conditions, the aligned gate selected FRONTIER 99 times, SHUFFLED 98 times and COIN 65 times. The two learned policies chose different options on only **5/132 conditions**, despite differing checkpoint weights and genuine optimization. Mean predicted activation probabilities were approximately 0.720 (aligned) and 0.712 (shuffled). This is largely learning **FRONTIER's overall advantage** rather than learning conditional causal eligibility. It does not establish differential hint understanding, personality recovery or robust calibration. It suggests that the current 12 observable prefix features are insufficient for reliable counterfactual gate discrimination, or the available training labels/graph mix offer too little informative separation.

The frozen C60 neural explorer slightly outperformed the non-neural frontier policy in Kruskal and Wilson but not familiar topologies on this cohort. This does not imply that C60 has robust transfer or is eligible for promotion; prior negative results and this small selected sample remain applicable.

## Host execution and actual trajectories

**396 matched condition rows, 1,584 distinct host-executed suffixes** (four independent concrete controllers per condition), and **2,772 policy outcome assignments** including deterministic selection/reuse of complete counterfactual executed suffixes for the three gating controllers. These 2,772 assignments must NOT be described as independently simulated rollouts: no synthetic trajectories were interpolated or fabricated. The two legal counterfactual suffixes were actually executed once per matched condition and the learned gate selected one. **132/132 unguided condition rows** had identical executed action and paint hashes across all policies. Zero host rejections, protected-cell changes or illegal painting across all unique executions.

The source-fixed gallery selection uses up to two first eligible graphs per family under strong guidance with intact memory; **48 selected paths were cold re-executed** from frozen host prefixes/checkpoints and matched action/ink hashes. The 48-panel SVG shows only actual start/goal and host-executed coordinate traces, deliberately no invented obstacle geometry.

Full 20-action movement proxies versus RESET for ALIGNED (n=97 matched segments): direction-entropy +0.0025, turn-rate −0.0498, reverse-rate −0.0682, unique-position ratio +0.0506, painting fraction +0.0500. These are merely summary statistics over **selected non-terminated 20-action segments**, not full-trajectory Meanderer/Curler/Brancher personality qualification. Different durations and path termination cause survivorship bias.

## Reproduction / limitations

Unpack the complete local owner-evaluation ZIP so `lab/`, `frozen/`, `publication/`, `results/` and `results63/` are siblings. Dependencies: PyTorch CPU, NumPy, safetensors, pytest. Execute:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=frozen/research-source python lab/c63_counterfactual_gate.py fresh-results
PYTHONPATH=frozen/research-source pytest -q lab/test_c63_counterfactual_gate.py
python lab/c63_gallery.py results63/C63-traces.json results63/C63-gallery.svg
```

The first command writes fully fresh synthetic-host training and evaluation evidence to a new directory. The pytest evidence tests intentionally target the unchanged `results63/` owner archive; do not mistake them for a fresh-run test unless the regenerated files are independently compared or relocated. The source is sha256-bound in `C63-summary.json`; lossless full results are in `C63-executed-ledger.zlib.b64` with its extractor. Models are tiny `safetensors` files, retained in the owner ZIP **not** in Git history. Repo source-smoke CI alone cannot prove scientific reproduction without the frozen host and model artifacts. Native CyberSand ABI/material fidelity, whole-trajectory personality, and immutable GitHub Actions artifact custody remain unqualified. This is exploratory, not preregistered confirmatory evidence, and no paid APIs/GPUs were used.

## Disposition

**HOLD / reject C63-ALIGNED.** This experiment quantitatively falsifies the useful independent-selection claim on the current family-split sample: identical held-out goal performance to the shuffled control, multiple paired regressions, less success than frozen FRONTIER. A better next experiment would use verified host-issued intervention opportunities and delay-conditioned feedback with a separately specified, high-contrast training cohort, or train improvements to the frontier policy itself rather than a redundant high-probability wrapper. Do not promote any learned model until it outperforms the simple FRONTIER control and passes nonregression, legal-action, personality and native-acceptance gates.
