# C41 — Independent Prim-Topology Transfer and Personality Audit (17 November 2026)

**Research isolation:** DRIV3R `experiments/driv3r/training/guidance_recovery/**` only, within existing Draft/HOLD PR #1. The C34/C39/C40 host is synthetic; no native CyberSand ABI qualification, production-model promotion or external-repository mutation.

## Experiment

C40's pairwise ranker appeared four goals better than C34-RESET on four favorable graph clusters, but its offline uncertainty spanned zero. C41 tested that claim on independent topology generation and retrained a **balanced tree/braid** pairwise ranker under matched shuffled-label and random-initialization controls.

The new **Prim** generator selects randomized frontier edges rather than C34's depth-first spanning tree; **Prim-braid** opens ten additional corridors. All four families still occupy the same 15×15 logical grid with 24 legally executed prefix moves, host-legal action menus, protected cells, painting actions, host cues and optional second goal. These are not new CyberSand physical environments; the holdout is an independent graph-construction algorithm, not an independent host implementation.

Training: 80 distinct graphs (40 DFS-tree, 40 DFS-braid), 240 states, 1,258 legal candidate examples. The unchanged C40 within-menu soft-target cross-entropy/AdamW trainer executed **1,680 optimizer updates for each of two trained heads**. A second independent optimizer execution reproduced all three serialized checkpoint hashes byte-for-byte. C41-MIX uses executed counterfactual suffix returns as **training targets only**. C41-SHUFFLED permutes labels within legal menus using an identical optimizer schedule; C41-RANDOM is untrained. The frozen C40-PAIR checkpoint is the historical comparison.

Holdouts: 144 disjoint graphs and 432 independently executed offline ranking states; another 96 graphs in three cohort seeds generated 2,304 host-executed online episodes, covering six policies, four guidance arms (none, weak BACKTRACK, strong BACKTRACK, strong CONTINUE), both personalities, detour/forward goals, first/second goals and four generator families. All training/offline/online topology hash sets were pairwise disjoint.

Cohort C used seed 773111 after the original 743111 seed produced a **challenge-generation rejection** in a Prim forward/switch fixture (no qualifying desired path within 40 deterministic retries). The replacement was chosen for fixture construction before observing C outcomes, not for performance selection. This is a design limitation, and the rejected seed is retained here.

## Matched held-out online goal events

| Policy | Guided goal events / 288 episodes | DFS tree | DFS braid | Prim tree | Prim braid | Regressed conditions vs RESET |
|---|---:|---:|---:|---:|---:|---:|
| C18-A0 | 107 | 19 | 33 | 32 | 23 | 20 |
| **C34-RESET** | **119** | **26** | **36** | **29** | **28** | **0** |
| C40-PAIR | 117 | 22 | 40 | 28 | 27 | 11 |
| **C41-MIX** | **124** | **25** | **40** | **29** | **30** | **7** |
| C41-SHUFFLED | 122 | 26 | 37 | 29 | 30 | 0 |
| C41-RANDOM | 119 | 26 | 37 | 29 | 27 | 2 |

The original C40 improvement **did not replicate** in these new graph clusters: C40-PAIR completed two fewer goal events than C34-RESET, and regressed in eleven cohort/condition cells. C41-MIX completed five more than RESET, but also suffered seven condition regressions and beat SHUFFLED by only two goals.

A paired graph-cluster bootstrap (4,000 descriptive resamples, not confirmatory inference) gave C41-MIX minus RESET **+5 events [−3,+14]**, minus SHUFFLED **+2 [−8,+12]**, minus RANDOM **+5 [−5,+15.0]**, and minus historical C40 **+7 [0,+15]**. The last is not evidence of broad robustness: many models and fixtures have been iteratively tried, and all other intervals include zero.

On the 432 blind offline ranking states, mean oracle regrets (lower better) were MIX **0.259722**, SHUFFLED **0.270602**, RANDOM **0.266204**, and historical C40 **0.291204**. All four families showed varying performance, and the measured MIX/SHUFFLED margin is small.

## Safety and personality caveats

All six policies preserved **96/96 matched no-guidance movement and painting hashes**. There were **zero synthetic host-rejected executions** and no protected-set mutation. Across 2,304 episodes there were 13 raw illegal top-choice proposals subsequently filtered by the host-legal mask; a raw preference for an illegal action is not equivalent to legal action execution.

C41-MIX produced 710 personality-preferred and 570 opposite-direction ink decisions over 12,283 guided executed actions; RESET produced 747 and 554 over 11,872. These are different whole trajectories and indicate **no demonstrated whole-trajectory personality equivalence**. The inherited C39 per-decision signed-preference projection is insufficient for that claim, and a physical native flight/painting acceptance test remains necessary.

The probe provides a **negative transfer result** for historical C40 and an **inconclusive positive aggregate** for C41-MIX. Neither model meets the strict zero-condition-regression criterion or a convincing learned-over-shuffled advantage.

## Reproduction and custody

The runnable source and targeted regression tests are committed here. The archive manifest identifies the complete prerequisite snapshot, exact C40/C41 checkpoints and full synthetic trajectories. A source-matched local C41 run passed seven combined C40/C41 tests; independently rerun AdamW checkpoints matched SHA-256 exactly. The compact JSON evidence records source/checkpoint/trajectory digests, cohort metrics and the negative outcomes.

**Important limitation:** Full checkpoint/prerequisite binaries and full episode logs are preserved in a local SHA-256-indexed research archive, **not** in Git history or immutable GitHub Actions artifacts. Reproduction from the repository alone remains unqualified, and artifact custody still requires an approved bounded, opt-in DRIV3R-only Action. No routine push/PR-triggered training, external paid compute or new research workflow was created.

**Disposition: HOLD.** Do not merge, mark Ready, promote a model, or modify DRIV3R main. Next prioritize cluster-independent neural versus shuffled replication and active-guidance whole-trajectory personality bounds before investing in further model complexity.
