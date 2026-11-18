# DRIV3R Guidance & Recovery — C48: braid-enriched hard-case counterfactual supervision

**Disposition: REJECT / HOLD.** Synthetic-host training improved the number of informative comparisons, but C48-HARD did not achieve a reliable neural improvement over the non-neural controller, shuffled-label control or the untrained random gate. This does **not** validate any native CyberSand gameplay contract.

## Research question and exact scope

Could a classifier trained on **full 78-step, host-executed** pairwise goal-outcome differences, with an oversampled *braided training graph family*, more reliably decide whether to use the previously trained C40-PAIR recovery head instead of the non-neural C34-RESET controller? Previous C47 learned from 192 paired guidance-arm samples with just 18 nonzero differences.

C48 freezes the upstream C47 source (Git blob `a89d852a858de828415686098b74efe205e9567c`) and C40 action head, trains only a small episode-start classification head, and does not change any host legality, observation, game-side code, action menu, physical solver, materials or production defaults. The old `d_common` fixture remains unchanged. Training never sees graph IDs, future outcomes, protected cells or oracle paths as deployed inputs. The 19 input features originate in the legal host observation, guidance cue and bounded scores of the frozen C40 head. The two candidate policies are executed in matched seeded world copies for the *training labels only*.

## Preregistered-by-code design and actual runs

- **Train:** 128 disjoint DFS-braid/Prim-braid graphs, with both detour/forward cases, both personality signs, switched/non-switched goals, and two BACKTRACK advice strengths. All 256 matched comparisons are retained rather than only favorable outcomes. Label counts: **44 C40 wins, 14 C40 losses, 198 ties**; nonzero rate **22.66%**, versus 9.38% in C47. This is a constructed training-distribution change; it must not be interpreted as deployment prevalence.
- **Optimizer:** identical 19→12→1 Tanh heads, initialized from the same seed; matched 36 epochs of CPU AdamW for the aligned-label and graph-stratum shuffled-label arms, 288 updates per head. Wins and losses receive weight 12, ties 0.20, independently from each model's OWN labels. The graph-level shuffle preserves arm and morphology stratum, with a nonidentity permutation. Two independent optimizer runs reproduced byte-identical state dictionaries and losses.
- **Decision:** episode-start controller switch is allowed *only* for active BACKTRACK cues, at logit >1.0 (sigmoid >0.731) for both learned/shuffled arms. CONTINUE/no-guidance retain C34-RESET's guidance selection. A 20% deterministic random intervention arm serves as a non-neural comparator. Existing C47-ALIGNED is evaluated from its original checkpoint, with normalization reconstructed from its immutable source training records.
- **Blind:** 16 previously unseen Wilson/Kruskal graphs, 48 guided comparisons per model, were reserved before online rollout; there were **zero nonzero controller differences** in these episodes, so no model could demonstrate discriminative improvement on that blind subset.
- **Online:** 96 disjoint new graphs (64 DFS/Prim tree/braid; 32 Wilson/Kruskal out-of-generator) and four cue arms per graph. **2,688 fully executed episode records** = 96 graphs × 4 arms × 7 policies, including **288 guided episodes per policy**. The 40 selected replay trajectories were independently re-executed to match movement and painting hashes.

## Held-out online result

| Policy | Guided goal events | Condition regressions vs RESET | Neural PAIR selected among 288 guided |
|---|---:|---:|---:|
| C18-A0 (unguided baseline) | 88 | 15 | 0 |
| C34-RESET | 100 | 0 | 0 |
| C40-PAIR (always-on) | **109** | 7 | 288 |
| C47-ALIGNED | 104 | 6 | 90 |
| **C48-HARD** | **103** | **3** | **94** |
| C48-SHUFFLED | 101 | 4 | 98 |
| C48-RANDOM (20%) | 104 | 1 | 39 |

The C48-HARD advantage over RESET was **+3 goals** from four favorable and four unfavorable graph clusters; graph-cluster bootstrap **95% descriptive interval −5 to +12**. C48-HARD versus C47-ALIGNED was **−1** (interval **−8 to +6**); versus C48-SHUFFLED was **+2** (interval **0 to +5**) from only two favorable graph clusters. The random gate outperformed C48-HARD on aggregate goal events and suffered fewer condition-group regressions. These outcomes neither establish a meaningful learned advantage nor satisfy strict nonregression.

The ALWAYS-ON C40 candidate achieved nine more goals than RESET but still regressed in seven condition groups. Thus the mixture's poorer aggregate performance is not explained by C40 universally failing; **reliably identifying the right intervention opportunities remains unsolved**. In particular, the new fixed threshold rejected several successful C40 recoveries on withheld Wilson/Kruskal cases. No threshold was tuned on these test cases.

## Safety, personality and replay

All 96 matched unguided trajectories per policy reproduced C18-A0's movement and ink hashes exactly, and every online record had zero executed synthetic-host rejections. No policy-side mutation of protected cells was permitted by the synthetic host. Raw illegal action proposals, where present, were excluded by the host legality mask. The mean whole-guided-path fraction of preferred-versus-opposed painting was **0.5906 for C48-HARD** and **0.5902 for RESET**, a descriptive similarity only: it cannot establish trajectory-level personality equivalence or prove absence of local stylistic loss. There is no native CyberSand replay.

All five targeted tests passed. The render-only SVG gallery uses 20 replay-verified panels (five matched graphs × four policies), displaying actual executed positions and protected-cell masks, not imagined oracle trajectories.

## Falsification, limitations and decision

The more discriminative synthetic *training* data did not produce a convincing held-out neural advantage. The out-of-generator blind cohort had zero differentiating outcomes, the trained head remains unaware of distant maze topology, the high-tie problem persists in wider distributions, and the system can only switch at episode start. Short-horizon and even full-episode offline utility labels may not approximate which particular **mid-trajectory** guidance action is beneficial.

**Do not promote C48-HARD.** Hold DRIV3R PR #1 as Draft/HOLD. Next bounded hypothesis: collect *mid-trajectory* intervention opportunities only where independently executed continuation outcomes diverge; predeclare a held-out independent generator split, a host-legal fallback and a conservative personality nonregression gate. Evaluate against RESET, C40 and nonidentical shuffled-label controls, not solely against unguided A0. Do not alter CyberSand or Neurounner.

**Evidence custody:** Research source, tests, this report, compact results and executed-path gallery belong in the sole authorised `experiments/driv3r/training/guidance_recovery/**` branch path. Full trajectories, optimizer traces, checkpoint binaries and frozen dependencies are in the SHA-256-manifested local evidence archive. Binary checkpoints have **not** been placed in Git. Immutable GitHub Actions artifact custody and exact-HEAD CI remain pending until published/verified; the archive alone is not an immutable Actions artifact. No scientific reproducibility is claimed from the DRIV3R import alone.
