# RECOVERY-FORK-001 — Synthetic near-exhausted recovery forks

17 November 2026 · DRIV3R single Draft/HOLD research branch · **Mixed: targeted rescue success, open-route overavoidance**

## Question and method

The preceding FAILURE-MINING-001 run found only four strict survival improvements over imitation in 288 sampled TRAIN states. Can *legally generated, genuinely recoverable failures* supply a useful learning signal without tuning more reward coefficients?

This experiment synthesizes paired near-exhausted states using the **existing unmodified `World.apply` / `World.menu`** fixture: eleven legal sacrificial MOVE/BACKTRACK cycles along distinct unpainted cells, one legal `RELOCATE_START`, and one legal eastward MOVE into an obstructed corridor. The resulting state has twelve used recoveries, menu `MOVE_E, MOVE_W, STOP`, and a realistic goal instruction toward the east. Choosing east seals the return route and forces STOP after five further painted cells; choosing west preserves a legal open-space continuation of at least twenty moves. A deterministic, source-replayed 24-step legal-MOVE continuation certifies each training and DEVELOPMENT pair; it is an **existence/viability witness**, **not the trained worm's policy**.

Only 96×96 synthetic one-cell painting worlds are used. This fixture does not certify CyberSand's production host ABI, native physics, authorization, or 64-logical-move ceiling; `RELOCATE_START` is synthetic and should not be interpreted as a real native teleport.

Two independently initialized 961-parameter rankers (seeds **681, 682**) were compared. Each began with 54 actual Adam baseline updates (18 at each 36 → 72 → 128 stage). The frozen ranker remained unchanged; the two adapted policies each received **96 additional Adam updates** (32 per stage) on identical inputs. Each stage included 64 naturally sampled host-legal TRAIN examples plus **16 TRAIN recovery forks**, 80 samples per arm. The matched *imitation* arm retained the goal-chasing teacher's `MOVE_E`; the *rescue* arm assigned the certified `MOVE_W` at those same forks. No DEVELOPMENT or held-out outcomes informed these labels. The 16 forks per seed are replayed at three curriculum stages and should **not** be represented as 48 independent geometries.

## Synthetic DEVELOPMENT results

Eight known-family DEVELOPMENT specifications (four structural families × two seeds), with two prefix variants each, evaluated per seed. Each arm rolled autonomously up to 128 **post-fork** decisions, following the same 35-decision legally replayed prefix. They were not teacher-forced after the fork.

| Metric (combined two seeds) | Frozen | Matched imitation | Matched rescue |
|---|---:|---:|---:|
| Risky east first choice, 32 trapped forks | 32 | 32 | **0** |
| Safe west first choice, 32 trapped forks | 0 | 0 | **32** |
| Premature STOP in trapped post-fork episode | 32/32 | 32/32 | **12/32** |
| Mean post-fork decisions | 6.0 | 6.0 | **101.94** |
| Mean new post-fork painted cells | 5.0 | 5.0 | **101.56** |

The positive result is **real in the deliberately constructed recovery trap**: both seeds learned the safe fork, and the checkpoint-restored model's subsequent legal movement sustained far more painting. It is not proof of general long-horizon survival, because construction and evaluation share a canonical eastward cul-de-sac pattern.

### Essential negative control: east is now viable

The same exhaustion-prefix procedure on **normal, open DEVELOPMENT geometry** produces a verified 24-step eastward continuation. All arms receive the same eastward goal instruction. Selecting west is not illegal, but it sacrifices progress along a viable target-facing route.

| First action on 32 open controls | Frozen | Matched imitation | Matched rescue |
|---|---:|---:|---:|
| East (viable and goal-directed) | 32 | 32 | **6** |
| West (unnecessary retreat) | 0 | 0 | **26** |

This is a substantial **false-recovery bias**. In seed 681, rescue chose west on 10/16 open controls; in seed 682, 16/16. Masking the rescue model's eight recovery features at the trapped fork did **not** reverse any of its 32 safe-west choices. Consequently, this experiment **cannot attribute its improvement to correct interpretation of exhaustion state**. It might be learning a brittle local directional rule tied to similar location, goal and recovery values.

### Ordinary surprise-retargeting DEVELOPMENT controls

The original, unmodified normal DEVELOPMENT rollouts (128 and 160 decisions, four known training-family geometries per seed) were also replayed. Results combined over 16 episodes per arm:

| Metric | Frozen | Matched imitation | Matched rescue |
|---|---:|---:|---:|
| Instructed target contacts | **39** | 33 | 37 |
| Mean unique painted cells | **143.00** | 134.38 | 138.00 |
| Early STOP | **0/16** | 4/16 | 3/16 |

Targeted rescue **does not surpass frozen imitation-trained policy** on this broader DEVELOPMENT measure. Do not promote it as a generally improved worm.

## Training, performance and verification

| Source-matched campaign | Seed 681 | Seed 682 |
|---|---:|---:|
| Frozen checkpoint SHA-256 | `d6753df075276ead0ba9ddbadee20858b7d8da3a94d7f1d71d313c3428dc1b78` | `399bcf95420d75906a8e1cef098f59850f01e169fb8e7b05cacd37725389361f` |
| Imitation checkpoint SHA-256 | `aabe2781e72dddf06a99e9188fc138a91b590039eadb3059086fe5d05217d10b` | `d523535e768a3b26d2c58b7e78caec817a0d6ebe5cfae7576746f8d9a78a6756` |
| Rescue checkpoint SHA-256 | `92462cfcd097fff96a5e4ff5f8776110770219afea9c6189c20315c75ae9ec87` | `812cbdb35878aa03ec37993d4c68bd2363ac6e643fbd6a9817f2354377f13d3d` |
| Targeted / standard / open episodes | 48 / 24 / 16 | 48 / 24 / 16 |
| Autonomous actions (targeted + standard) | 5,336 | 5,023 |
| Process time (seconds) | 22.61 | 23.01 |
| Peak Linux RSS (KiB) | 123,160 | 123,672 |
| Rescue mean trapped action latency (ms) | 0.118 | 0.118 |

The `run` + `verify` process restored all six NumPy checkpoint files, re-executed **96 targeted + 48 standard** original trajectories, replayed **96 TRAIN fork stage-examples**, verified 32 open controls and their complete first-action distributions, recomputed all source and evidence SHA-256 manifests (**102 files per seed**), and rejected deliberate evidence tampering. All **five focused tests passed**. Full artifacts include material-rendered PNG traces, full legal menus, complete model probabilities, actual state hashes, training metadata, exact source hashes and checkpoint hashes. Noninteracting SOURCE files were compared by Git blob SHA against the previously fetched DRIV3R branch; evidence generated from matching local copies.

Checkpoints are **local immutable-run outputs**, not yet uploaded as immutable GitHub Actions artifacts. They were intentionally excluded from Git history. No paid inference or GPU training was used.

## Scientific assessment and next objective

The previous sampling bottleneck is resolved *for a specific synthetic failure type*: legal prefixes can be constructed so the two action choices differ sharply in empirical survivability, and small models can learn that local contrast. But the unconstrained policy overgeneralizes retreat to open, viable corridors, making the global utility ambiguous.

Next test: **contrastive risk/open-state supervision** with exactly matched optimizer budgets, spatially counterbalanced corridor orientation and separate normal-route controls. Also compare recovery-feature ablations and a goal-conditional context feature, rather than adding another hand-tuned scalar reward. Continue using TRAIN-only labels and disjoint DEVELOPMENT observations. Do not open the structurally sealed heldout families until the evaluation protocol is frozen.

**Boundaries:** DRIV3R research only. Source and report in `experiments/driv3r/training/long_horizon/` on the existing single Draft/HOLD PR #1. No native CyberSand changes, Neurounner changes, extra PRs, workflows, merge, Ready promotion or default-provider changes. All strong results above are explicitly *synthetic DEVELOPMENT*, not native acceptance or held-out structural generalization.
