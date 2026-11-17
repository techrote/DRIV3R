# RETARGET-DAGGER-001 — on-policy imitation under surprise goals

17 November 2026 | DRIV3R Draft/HOLD PR #1 | negative synthetic DEVELOPMENT result

## Question and isolation

Does teacher-labeling the worm's *own visited states*, while changing its spatial objectives during training, improve autonomous 36→72→128+ decision behaviour?

Two independently initialized 961-parameter ranker campaigns (seeds 619 and 620) start from the same recovery-conditioned model: 180 initial Adam updates (60 at each of 36, 72, 128). Each adapted arm then receives an identical **60 additional Adam updates** (20 per stage), preserving the pretrained model and optimizer state. The frozen reference receives no additional updates. Both adaptations resample exactly **768 TRAIN student-visited states per stage** from on-policy trajectories for training; both visit 576, 1152 and 2048 states per stage and seed before resampling.

- **Frozen**: pretrained controller; no on-policy adaptation.
- **Static adaptation**: student-visited imitation under a fixed spatial goal.
- **Retarget adaptation**: same additional optimizer/sample budget, but goals change at decisions 32, 72, 112.

Teacher labels train the model but **do not control evaluation actions**. Synthetic legal-menu deviations are imposed at specified actions. The published, unchanged GOAL-RETARGET-001 fixture evaluates 128- and 160-decision autonomous runs on four familiar DEVELOPMENT families (seeds 3001/3101/3201/3301), in normal and trap variants. It is **not a sealed structural holdout**. No native CyberSand execution, 64-move extension, or real Workbench acceptance occurred. The historical synthetic World fixture is unchanged.

## Combined surprise-retarget DEVELOPMENT results

32 matched evaluations per arm across the two seeds:

| Controller | Targets painted during active instruction | Mean unique painted cells | Premature STOP | Mean 8-step post-switch distance gain |
|---|---:|---:|---:|---:|
| Frozen (180 updates) | 67 | **140.906** | **1/32** | **3.135** |
| Static-adapted (240 updates) | 57 | 132.625 | 4/32 | 2.581 |
| Retarget-adapted (240 updates) | **67** | 133.531 | 3/32 | 3.055 |

**Negative result:** Retarget adaptation increases instructed targets from 57 to 67 relative to *equally optimized* static adaptation, but does not beat frozen pretraining on instructed targets. Retarget adaptation paints 7.375 fewer cells per episode and stops earlier more often than frozen pretraining. Retarget versus static adaptation: mean ink difference +0.906 cells, 10 wins/13 ties/9 losses over 32 matched fixtures, paired-bootstrap approximate 95% interval **[-12.97, +14.22]**. Retarget versus frozen: **[-18.28, +1.38]**. No reliable improvement in long-horizon competence.

## Static-instruction cross-check

32 matched evaluations per arm:

| Controller | Mean painted | Premature STOP | Targets painted |
|---|---:|---:|---:|
| Frozen | **137.406** | 3/32 | 32 |
| Static-adapted | 135.438 | 4/32 | 32 |
| Retarget-adapted | 130.906 | 4/32 | 32 |

Seed 619 retarget arm paints 137.938 cells on the retarget fixture versus 139.313 frozen; seed 620 paints 129.125 versus 142.500 frozen. Frozen has fewer updates, so causal *matched-budget* comparison is the static versus retarget adapted contrast.

### Retarget action counts, both seeds

| Action | Frozen | Static-adapted | Retarget-adapted |
|---|---:|---:|---:|
| MOVE_E | 1568 | 1326 | 1432 |
| MOVE_N | 998 | 1036 | 936 |
| MOVE_S | 888 | 1052 | 944 |
| MOVE_W | 1055 | 830 | 961 |
| BACKTRACK | 26 | 13 | 44 |
| RELOCATE_START (synthetic) | 33 | 43 | 28 |
| STOP | 1 | 4 | 3 |

The distribution changes do not demonstrate improved planning or motif preservation.

## Checkpoints, replay, and resource audit

Six real saved checkpoints and 192/192 original episodes replayed from restored models, with **26,337 total sequential decisions**. Verification checks exact model action probability dictionaries, legal menus, painted materials, state/trajectory identities, and aggregate outcomes. There are 103 SHA-256-manifested evidence files per seed; ZIP CRC passes. **14/14 new and predecessor tests pass locally.**

CPU campaign times **21.048 s** and **20.095 s**; peak process RSS **136,996 KiB** and **135,368 KiB**. Measured neural action latency is about **0.12–0.13 ms** in this CPU research environment, not native Workbench latency.

Checkpoint SHA-256:

| Seed | Model | SHA-256 |
|---|---|---|
| 619 | frozen | 0d8df57d129ea73c309914928df6652fe7be150b5866578d76b1be77c0e57ee5 |
| 619 | static-adapted | fb801a013f0aa11c7ad1612dcb68d34bc90668168626d5d49fa6caa52f975e0f |
| 619 | retarget-adapted | 283054afcb3beb5834a8849d8e6ab15d1ca38150f5d842bcc58c341c9ecf7b2d |
| 620 | frozen | c6fc4690fa505f134343f6ad9837c12c19a03fb3a1aa0a3078ad1806f28ceaeb |
| 620 | static-adapted | 439f66a60d3951afe6db4389d0b95930a5d08d24a457a87266d3b478becc2c01 |
| 620 | retarget-adapted | d893f72653f19215b93e6ab4302a7d7b70dc6d39c4ab51ad41852f1de4c5699f |

All checkpoints remain **outside Git history**. Their complete local source-matched trace archive is verified, but **immutable GitHub Actions checkpoint-artifact publication remains outstanding**. No paid API/GPU usage, external repo writes, sealed holdout access, or native receipts.

## Next experiment

The negative result suggests (without establishing causality) that existing imitation labels overvalue immediate goal progress and insufficiently account for post-switch painting opportunity, survival and recovery debt. Test cost-sensitive mixed objectives and distinct post-switch painting metrics against matched optimizer controls before any structural-heldout or native claims. Preserve Draft/HOLD; no merge or Ready promotion.
