# C40 — Pairwise Action Ranking for Host-Legal Recovery

**Repository:** techrote/DRIV3R, consolidated Draft/HOLD PR #1 only.
**Scope:** experiments/driv3r/training/guidance_recovery/**.
**Status:** synthetic research control; NO PROMOTION, no native CyberSand changes.

## Experiment

C39's action-conditional scalar return regression offered modest, uncertain recovery improvements and two condition-level regressions. C40 retains C39's 34→32→16→1 MLP and unchanged host-legality enforcement, stochastic controller, and per-state personality projection, but trains against a within-menu distribution over independently executed candidate returns rather than raw scalar return. Specifically, the target is softmax(executed candidate return / 0.35). Future return labels are calculated offline by the synthetic host and are never visible during inference.

Matched controls: C34-RESET non-neural policy; frozen C18-A0; C40-PAIR trained on aligned labels; C40-SHUFFLED trained on within-menu permutations with the identical initial weights and optimizer schedule; C40-RANDOM untrained but initialized identically. C39-ALIGNED is additionally scored on the same new offline states.

## Source and optimizer

Training: 56 distinct tree graphs, 168 host-executed decision states, 850 legal candidate examples. Real CPU AdamW ran 7 epochs, **1,176 parameter updates per trained head**. A second complete same-seed optimizer execution produced byte-identical model parameters. Exact source/checkpoint SHA-256 values appear in C40-compact-evidence.json. Three new targeted local tests passed.

Independent splits: 112 new offline candidate-ranking graph hashes (336 states, tree and braided graphs), plus 64 fresh online evaluation graphs, with **zero graph-hash overlap** among training/offline/online. Online test: four matched fresh-seed cohorts totaling 960 synthetic policy episodes, including 640 guided episodes; two graph families, changed goals, host-issued BACKTRACK/CONTINUE cues, movement and painting, protected cells, and two personality signs.

All four online cohorts were evaluated with the same frozen intervention parameters and fixed candidate checkpoints. This is a continuation of the existing C34 world generator, not a physically distinct topology generator or native game runtime.

## Executed held-out results

| Model | Guided goal events (128 guided episodes) | Condition-level regressions vs C34-RESET |
|---|---:|---:|
| C18-A0 | 48 | 7 |
| C34-RESET | 50 | 0 |
| **C40-PAIR** | **54** | **0** |
| C40-SHUFFLED | 49 | 2 |
| C40-RANDOM | 51 | 0 |

C40-PAIR differs from RESET in 53/128 guided trajectories; SHUFFLED in 39/128 and RANDOM in 27/128. All five candidates preserved **64/64 unguided matched action+ink hashes**. Across all 960 episodes: **zero executed host rejections**, **nine raw illegal top choices** filtered by the unchanged host legal menu. This does not prove native CyberSand safety.

Graph-cluster descriptive bootstrap (4,000 resamples, 64 graph identities): C40-PAIR versus RESET **+4 goals [1,8]**; versus SHUFFLED **+5 [1,9]**; versus RANDOM **+3 [0,7]**. The apparent gain comes from only **four favorable clusters against RESET**, with no losses in these particular cohorts. Given sparse events, repeated research iteration and correlated geometry, descriptive intervals must not be interpreted as robust statistical superiority.

Offline mean oracle regret, lower is better, over 336 separately executed ranking states: PAIR **0.244940**, SHUFFLED **0.310714**, RANDOM **0.347619**, previous C39-ALIGNED **0.293452**. However, graph-cluster 95% descriptive intervals for PAIR's regret advantage include zero: **[-0.022321,0.158929]** against SHUFFLED and **[-0.002679,0.209844]** against RANDOM. PAIR differs substantially between tree (0.155952 regret) and braided graphs (0.333929).

## Replay, limitations and decision

Independent audit replayed **160 selected synthetic trajectories, totaling 7,076 actually executed legal actions**. All replayed action and ink hashes matched source records. The complete local SHA-256-manifested research archive contains the dependency/prerequisite snapshot, source, checkpoints, optimizer evidence, full online/offline trajectories and an executed-trajectory HTML gallery. The repository contains source, tests and compact checkpoint-hash-linked metrics, **not binary checkpoints or full trajectory files**.

Active-guidance personality retention across whole trajectories has NOT been qualified; differing executed paths change turn and painting counts even when local per-state preference projection is bounded. No native CyberSand runtime/Workbench validation, immutable GitHub Actions checkpoint custody, or generalization to fundamentally different graph generators has occurred. Historical d_common remains a fixture, not an ABI replacement.

**Disposition: retain C40-PAIR as a promising synthetic research candidate, do not promote.** Larger genuinely independent maze-generator replication, paired whole-trajectory personality thresholds, and native owner-gated validation are required. PR #1 stays Draft/HOLD; no branches, PRs, workflows, or external repository writes were added.

## Reproduction

The standalone local research archive restores prerequisite model files under a root/prerequisites directory, the C28 checkpoint under root/results, and the three frozen C39 reference heads under root/reference. Set PYTHONPATH to the archived dependency, source, and C40 source directories. On a clean, empty output directory, run the C40 trainer, then C40 audit, and pytest for test_c40_pairwise_rank.py. **Do not reuse prior C40 cohort JSON files when claiming an independent reproduction**; the trainer supports resuming existing results. The repository import alone does not establish scientific reproduction.
