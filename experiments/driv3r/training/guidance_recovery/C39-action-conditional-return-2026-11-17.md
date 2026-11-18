# C39 — Action-Conditional Return Prediction

**Scope:** `experiments/driv3r/training/guidance_recovery/**` only. External synthetic CyberSand-like host. DRIV3R Draft/HOLD PR #1 only. No native game changes, no model promotion, no paid GPU/API and no automated training on push.

## Scientific question

C38 learned the **sign** of BACKTRACK's relative return better than shuffled controls, but did not improve complete synthetic trajectories. C39 asks whether predicting the return of **each host-legal candidate action** (rather than one global return/backtrack sign) helps executed, cue-conditioned recovery without changing unguided behavior or protected-cell permissions.

The training-label oracle independently executes up to ten legally verified host actions after each candidate. Those suffixes are **offline targets only**; the deployed policy uses the current nine-dimensional host observation, the 25-dimensional features of each of up to nine candidate actions, current legal-menu mask, executed history and host-issued guidance. The oracle's maze connectivity, shortest routes and future actions are not inputs at inference. STOP is controlled by the existing frozen policy, not retrained.

The model is a small 34→32→16→1 MLP, trained by real CPU AdamW on 856 candidate/return pairs from 168 states across 56 independent **tree** graph identities; 140 optimizer updates per trained model. SHUFFLED uses the identical initial parameters, update schedule and within-menu permuted labels; RANDOM has identical initial parameters but no updates. Both trained checkpoints reproduced byte-identically in independent optimizer executions.

C39 attaches a bounded (±0.8) candidate-specific logit adjustment to the frozen C34 re-arming recovery controller **only during active BACKTRACK-directed recovery phases**. A per-state directional-preference trust region limits the adjustment; legality still belongs entirely to the host. There is no intervention without host guidance.

## Held-out results

The offline label test contains **288 independently executed state sets from 96 disjoint tree/braided graphs**, with no graph hash shared with the training set or online evaluation set. Mean oracle regret (lower is better): **ALIGNED 0.321875; SHUFFLED 0.371875; RANDOM 0.329514**. Oracle-top-choice agreement across 288 states: **161 ALIGNED; 102 SHUFFLED; 154 RANDOM**. Despite this ranking advantage, ALIGNED's mean regret advantage is just 0.0500 over SHUFFLED (graph-cluster descriptive 95% bootstrap interval **−0.1014 to +0.2076**) and 0.0076 over RANDOM (−0.0007 to +0.0167). There is no statistically conclusive improvement in oracle regret. The by-family differences are large and opposite in direction: the held-out braid and tree distributions do not yet generalize uniformly.

The actual online study used **six fresh-seed strong-guidance cohorts**, each with matched no-guidance / strong BACKTRACK / strong CONTINUE arms, across tree/braided graphs, detour/forward geometries, personality ±1 and optional second goal. Across **96 evaluation graph identities and 1,440 policy episodes** (960 guided), guided goal events were:

| Policy | Goal events (192 guided episodes each) | Condition-level regressions vs RESET across cohorts |
|---|---:|---:|
| C18-A0 frozen baseline | 57 | 17 |
| C34-RESET non-neural | 69 | 0 |
| C39-ALIGNED | **74** | **2** |
| C39-SHUFFLED | 71 | 0 |
| C39-RANDOM | 69 | 1 |

Paired graph-cluster descriptive 95% bootstrap intervals for ALIGNED's total goal-event advantage: **+5 vs RESET [−1,+11]**, **+3 vs SHUFFLED [−3,+9]**, and **+5 vs RANDOM [0,+10]**. These descriptive intervals and small absolute event counts do not justify superiority claims. ALIGNED fails the **zero condition-level regression** gate despite modest aggregate gains.

An additional **320 matched weak-guidance policy episodes** on 32 graph identities reused the same geometry/seeds as strong cohorts A/B; the weak cue strengths were 0.35 rather than 1.0. Combined guided goals: baseline 19, RESET 20, ALIGNED 21, SHUFFLED 20, RANDOM 20. This narrower weak-guidance pilot cannot establish generalization across all six cohorts.

## Safety, personality, and custody

- Across the combined 1,760 online policy episodes: **zero executed host rejections**, **zero modifications of protected cells**, and **six raw illegal top choices prior to the original legality mask**. Preserve the distinction between raw proposals and executed actions.
- All **96 paired no-guidance trajectories per candidate** retained identical action and ink hashes to frozen C18-A0. Online policy candidates changed thousands of guided per-state scores, not merely an inactive model head.
- For ALIGNED, the proportion of personality-preferred over personality-opposed turning movements in strong guided episodes was 226/(226+179)=55.8% for personality −1 and 246/(246+194)=55.9% for +1. RESET was 55.5% and 55.9%, respectively. This is descriptive only; it **does not prove whole-trajectory personality preservation**, particularly because sampled trajectories, ink output and visited states differ.
- The independent audit reconstructed and re-executed **240 displayed synthetic trajectories / 11,354 actual host actions**, reproducing their action and ink hashes. The 12-panel HTML gallery uses these executed paths, not imagined oracle routes. All three graph splits are hash-disjoint: training, offline label holdout, and online policy evaluation.
- Source and compact metrics are committed only in this owned research directory. The checkpoint files and full trajectories are kept out of Git; this run retains a SHA-256-verifiable local archive. **Immutable Actions checkpoint custody remains outstanding** until an approved bounded manual artifact workflow supports checkpoint upload. Do not treat the local archive as Actions custody.

## Qualification and disposition

Five targeted local regression tests pass. Source syntax and dependency imports pass locally against the C38 bundled source snapshot and prerequisite model files; exact original source identity is documented by `C39-archive-receipt.json`. The historical `d_common` interface is never changed or claimed canonical. This is *synthetic-host* work; native CyberSand legality/physics and physical Workbench evaluation remain untested.

**Decision: DO NOT PROMOTE C39.** Preserve ALIGNED as a weak positive research candidate and SHUFFLED/RANDOM/RESET as essential controls. C39 improves aggregate completion modestly but has two condition regressions, wide paired-cluster intervals, and no demonstrated general policy or personality acceptance. Next research should target more physically distinct unseen generators and explicit learning of long-horizon value, while preserving no-guidance fidelity and eliminating every condition-level regression. All new work stays in DRIV3R PR #1, Draft/HOLD.

### Reproduction notes

From the archive, add `dependency/` and `source/` to `PYTHONPATH`; restore `prerequisite/prerequisites` into `root/prerequisites`, `prerequisite/results` to `root/results`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/c39_action_value.py root out
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/c39_action_value.py root out --weak-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/c39_audit.py out out
python -m pytest -q source/test_c39_action_value.py
```

The source resumes previously recorded cohort files to avoid re-running expensive episodes; **for a complete independent reproduction, use an empty output directory**. The archive contains all original evidence files, prerequisite checkpoints, full executed trajectories and SHA-256 hashes. This repository's source import alone does not establish reproduction.
