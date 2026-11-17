# C47 — Differential full-goal guidance, strict held-out transfer

**Repository:** `techrote/DRIV3R`, consolidated Draft/HOLD PR #1 only. **Scope:** `experiments/driv3r/training/guidance_recovery/**`. **Decision:** **HOLD — reject neural promotion.** This is a synthetic game-like host study, not native CyberSand acceptance.

## Research question and outcome

C46's goal-supervised three-way policy arbitration collapsed to the unguided C18-A0 controller, partly because 91/128 training labels favored A0. C47 directly asks whether **full-episode goal-event differences between C40-PAIR and C34-RESET**, excluding A0 from arbitration targets, support a selective neural guidance mechanism. It trains on full paired *host-executed* outcomes rather than local distance or ten-step oracle regret. A confidence threshold selects C40 when predicted goal-event improvement exceeds 0.15 and otherwise retains RESET; there is no retrospective selection during live evaluation.

**Negative result:** ALIGNED completed 34 guided goal events on unseen Wilson/Kruskal maps versus 31 for RESET, 35 for unconditional C40-PAIR, and **34 for graph-shuffled neural controls**. ALIGNED improved no outcome over SHUFFLED (0-goal difference; paired graph-cluster descriptive 95% bootstrap interval **−4 to +4**), and regressed in two conditions against RESET. This is not a successful learned intervention. The 3-goal gain over RESET has a wide interval (**−3 to +9**). C40's small aggregate lead also carries three condition-level regressions. There is no model promotion.

## Protocol

- **Training:** 64 independent DFS/DFS-braid/Prim/Prim-braid graph clusters. All three guidance arms (weak BACKTRACK 0.35, strong BACKTRACK 1.0, strong CONTINUE 1.0), both personality signs, detour/forward tasks, and single/dual goals. For each of **192 graph-cue records**, execute independent matched legal-host trajectories with C18-A0, C34-RESET and frozen C40-PAIR, identical initial worlds and RNG seeds. C40 minus RESET completed-goal events form the target: **10 positive, 8 negative and 174 exact ties**. Baseline and neural have equal opportunity to succeed; no tie is silently labeled a neural win.
- **Neural heads:** Two equal 19→12→1 Tanh MLPs from matched random initial parameters, with 32 CPU AdamW epochs, **256 updates per trained head**. Training weights differential observations at 4.0 and ties at 0.35 (about 11.4-fold relative weight). The shuffled neural head receives *its own shuffled labels and its own derived loss weights*, avoiding the subtle label-leakage error where aligned weights would disclose original nonzero targets. Entire three-arm label vectors are rotated between graph clusters within family/case/personality/switch strata, preserving marginals and within-graph arm dependence while forcibly preventing identity permutation. A frozen random head and a fixed non-neural ranker-dispersion rule are controls.
- **Input boundary:** Only the current host-observed nine-feature local state, two current cue fields, the personality sign, known goal-switch condition, and five **frozen** C40 ranker statistics derived from the host-legal action menu. No maze graph, future success, true route, counterfactual label or protected-cell bypass enters deployment. The train-set normalization mean/scale is frozen and applied to blind/online data. Neural threshold **+0.15** and rule threshold **0.4** are fixed in source without tuning to held-out results.
- **Blind:** 16 disjoint Wilson/Kruskal graphs, **48 cue records**. Results: 47 C40-vs-RESET ties, one C40 win, zero losses. ALIGNED selected C40 on 17 records but missed the only differential win; incremental goal benefit **zero**. SHUFFLED likewise selected 17 and gained zero. This blind cohort contains too few discriminative examples for a meaningful calibration claim.
- **Online:** 48 further disjoint Wilson/Kruskal graphs in three independent seed cohorts, four arms (including no guidance), **1,344 policy-labeled episodes** over seven policies; only three base controllers physically execute each condition and selectors replay the exact corresponding full base trajectory. Each guided policy therefore has **144 guided episodes**; all graph identities disjoint between train, blind and online. No guidance 48/48 action and paint hashes match the original C18-A0 baseline for every policy.

### Outcomes on held-out online graph clusters

| Model | Guided goals | Condition regressions vs RESET | Guided C40 selections |
|---|---:|---:|---:|
| C18-A0 | 34 | 2 | 0 |
| C34-RESET | 31 | 0 | 0 |
| C40-PAIR | 35 | 3 | 144 |
| **C47-ALIGNED** | **34** | **2** | **61** |
| C47-SHUFFLED | 34 | 3 | 75 |
| C47-RANDOM | 31 | 0 | 25 |
| C47-RAW-RULE | 33 | 3 | 83 |

Paired graph-level differences: ALIGNED minus RESET **+3** (95% descriptive bootstrap **−3 to +9**; 4 positive/2 negative graphs); ALIGNED minus C40 **−1** (−5 to +3); ALIGNED minus SHUFFLED **0** (−4 to +4; 2 positive/2 negative graphs). The 95% intervals are descriptive bootstrap quantiles, not formal proof of any direction.

## Executed-behaviour qualification

All five targeted local regression tests pass, including actual graph-host observation construction, legal-action feature shape, thresholded abstention behavior, graph-group label shuffling and train-with-updates, and no-guidance action/painting parity. Independently replayed selected original controller paths match their recorded action and painting hashes. All full online records report **zero host rejections** and unchanged protected-mask digests. The gallery is made from the *executed synthetic traces*, not from oracle or imagined paths.

ALIGNED and SHUFFLED checkpoints are bit-identical across independently repeated deterministic optimizer runs (in-memory tensor comparison). Their SHA-256 hashes and full training loss curves are in `C47-summary.json`. Trained binary weights remain **outside Git history** in the local, SHA-256-manifested archive. Immutable GitHub Actions checkpoint custody has **not** been achieved; this is not repository-only reproducibility.

## Interpretation and limitations

Only **9.4%** of training records, and **2.1%** of blind records, distinguish C40 from RESET by goal outcome. The resulting class scarcity, family-specific execution dynamics, and selection at time zero give little evidence that an observation-conditioned neural intervention can predict otherwise rare success differences. The positive aggregate ALIGNED return is matched by SHUFFLED and therefore cannot support learned recovery superiority. Classification accuracy on all examples would be misleadingly high because ties dominate; C47 instead reports strict win precision and incremental completed goals.

The host is a synthetic challenge approximation; the historical d_common fixture is not CyberSand's authoritative native action/observation contract. This model cannot be claimed accepted for CyberSand gameplay. The classifier never receives mid-trajectory correction, does not quantify wall-clock recovery latency, and whole-trajectory personality behavior is only descriptively counted (visitation confounds turn ratios). None of these limitations is removed by exact local replay hashes.

**Next research priority:** Increase the *differential-event rate* with a preregistered set of valid, deliberately forked goal/recovery situations, then run graph-cluster-separated paired tests with meaningful numbers of both C40 wins and RESET wins. Avoid additional small MLP runs on 90–98% tied labels. Prioritize genuine post-mistake, mid-trajectory policy adaptation while preserving the legal-host and personality controls; do not modify the game.
