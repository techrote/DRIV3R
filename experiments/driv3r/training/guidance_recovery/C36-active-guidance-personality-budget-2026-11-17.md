# C36 — State-matched personality budget under recovery guidance

**17 November 2026 · DRIV3R Guidance & Recovery · REJECT for promotion · synthetic host only**

This study is confined to `experiments/driv3r/training/guidance_recovery/**` in the sole Draft/HOLD DRIV3R PR #1. It does not edit CyberSand, use a native CyberSand runtime, or claim canonical ABI equivalence. C35 was already on the remote branch when C36 began; this experiment deliberately tests a distinct active-guidance *personality* hypothesis rather than relabeling C35's cross-generator return-calibration study.

## Question and experimental intervention

Can the host-legal C34 rearming policy preserve a signed directional personality preference *at the same observation and action menu* while retaining long-horizon recovery? If a learned return head is added, does it improve completed goals beyond a zero-optimizer random initialization?

For every guided decision the intervention computes both the frozen baseline's logits and the recovery controller's logits **on the identical legal menu**. A bounded projection reduces deviations in the conditional left-versus-right movement preference to at most 0.045 probability units, without replacing the controller's overall movement versus BACKTRACK/STOP allocation. The projection is applied only to moving-action logits and never changes host legality or issues actions. Policy sampling remains stochastic.

The five matched arms are C18-A0, C34-RESET, C36-STYLE (projection without head), C36-LEARNED (projection plus newly trained return head) and C36-RANDOM (projection plus an untrained same-architecture return head). All except the baseline preserve C34's rearm behavior. Training uses the C34 independently generated tree family and 318 short counterfactual return comparisons executed through the synthetic host; the new head receives **80 CPU AdamW updates**, with source-fixed seed 360013. It is evaluated without retuning on two fresh seed cohorts including the withheld braided family.

Each cohort uses 32 independently generated maze graphs, 160 episodes per policy (128 guided and 32 no-cue), tree/braid topology, detour/forward goals, optional second host goal change, both personality signs, two hint strengths, and no/weak/strong BACKTRACK or CONTINUE advice. The decision horizon is 60. **No claims of model or native policy generalization to C35's Wilson graphs follow from these smaller fixtures.**

## Held-out recovery

*Guided goal events, not independent Bernoulli trials; switched episodes may complete two goals.*

| Policy | Cohort A | Independent B | A + B | Condition regressions vs C34-RESET (A/B) |
|---|---:|---:|---:|---:|
| C18-A0 | 30 | 39 | 69 | 7 / 6 |
| C34-RESET | 35 | 43 | 78 | 0 / 0 |
| C36-STYLE | 34 | 43 | 77 | 1 / 0 |
| **C36-LEARNED** | **36** | **49** | **85** | **2 / 1** |
| C36-RANDOM | 36 | 44 | 80 | 0 / 0 |

Learned recovery gained seven goal events against C34-RESET and five against the untrained random control across these small cohorts, **but failed strict condition-wise nonregression (2 groups in A, 1 in B)** and provided no replicated superiority in A, where LEARNED and RANDOM tied. Neither the learned head nor personality projection qualifies for policy promotion. C36-STYLE itself was marginally worse than C34-RESET in A.

## Active-guidance personality finding

| Policy | Mean absolute within-state signed-turn shift, A | B | Intervened guided states A/B |
|---|---:|---:|---:|
| C36-STYLE before → after projection | 0.0031 → 0.0008 | 0.0025 → 0.0006 | 57 / 35 |
| C36-LEARNED before → after | 0.0050 → 0.0012 | 0.0035 → 0.0007 | 91 / 45 |
| C36-RANDOM before → after | 0.0026 → 0.0007 | 0.0025 → 0.0006 | 50 / 37 |

These are **within-observation counterfactual distribution measurements**, with 3,827–4,584 guided states per intervention per cohort, not observed long-run trait retention. Existing controllers already perturb the signed-turn probability minimally on most states, so the projection rarely binds. On-policy trajectory turn rates and INK/LIFT action counts still differ materially among arms because the agents visit different states and choose different BACKTRACK actions. **Active-guidance personality acceptance is NOT established.**

## Verification, negative results and scope

- Four dedicated regression tests passed: per-state legal movement mass preservation and turn budget, matched no-cue invariance, host legality, and grouped cohort schema.
- All candidate policies matched C18-A0's **32/32 uncued action and ink hashes per cohort**, without synthetic host rejections or protected-material changes.
- Some unmasked raw model top choices were illegal (C34-RESET: 1 in A, 6 in B; C36-LEARNED: 1 in A, 0 in B); host legal-menu masking prevented execution. This synthetic protection must not be mistaken for a native-game safety result.
- The 988-byte learned checkpoint reproduced **byte-identically** on a separate optimizer run (SHA-256 `a76d43e12b67c389d8fc5a3e898797a945cd9eb2084df7e5ad9bf2e32693be59`). The frozen C28, C12 and C34 prerequisites are externally provided and provenance-linked, not recreated in this study.
- Twelve representative host-executed trajectories were independently rebuilt from the same maze seeds and legal moves. Their complete action and ink hashes matched.
- Full matched cohort trajectories, dependency snapshots, the checkpoint replicas and SHA-256 manifest are in the associated local evidence archive. **Immutable GitHub Actions artifact custody remains unsatisfied**; no checkpoint binary is added to Git.

**Decision: REJECT C36-STYLE and C36-LEARNED for promotion.** Retain C34-RESET, C35 trained/random comparisons and C36 negatives as research controls. Prioritize a genuine turn/personality gate on held-out policy trajectories, an increased number of graph-cluster replicates and a learned candidate that beats *both* random and sign/label-shuffled controls condition-wise. Do not merge/promote DRIV3R PR #1.
