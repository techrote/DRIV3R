# P15 — Protected-position topology transfer and frozen P14 preference ranking

**16 November 2026 · DRIV3R Draft/HOLD PR #1 · locally executed, not Actions-qualified.** Research-only synthetic W0. No CyberSand/Neurounner writes, human ratings, native painting, root-language-model inference, or new optimiser updates.

## Controls and scientific question

Do W16's external protected-position veto and a modest option-preserving movement heuristic retain structure across new mask geometries? Does P14's genuinely CPU-trained retrospective preference ranker generalise beyond its original rectangle masks?

Three predeclared geometry-only masks (alternating teeth, a two-gate cage, and staggered islands) were frozen before running trajectories. Across nine P14-held-out seeds × Architecture/Greebles/Pipework × three masks × three policies, P15 executed **243 real frozen-W16-head synthetic-W0 trajectories** and **30,405 forward passes**. The original W0 remained the authoritative action-selection host, and every model action was host-legal. Policies: original late-gated W16, destination-position veto, and the same veto with small model-logit bonuses for locally available future exits and unvisited destinations. The bonus is a transparent heuristic, not a trained reward or owner preference.

## Actual outcome measurements

| Policy | Position-safe | 48-new-position goal | Protected-position contacts | Mean new positions | Mean branches |
|---|---:|---:|---:|---:|---:|
| Original late STOP | 14/81 | 81/81 | 400 | 107.52 | 18.56 |
| Destination veto | 81/81 | 77/81 | 0 | 97.15 | 16.84 |
| Veto + exit-aware bias | 81/81 | 76/81 | 0 | 98.81 | 18.07 |

Across 81 *correlated* family/seed/layout comparisons, exit-aware novelty exceeded plain veto in 35, declined in 24, and tied in 22; mean difference +1.67. It cost one additional 48-position completion. Resampling all families and layouts together by the **nine source seed blocks** (10,000 preseeded bootstrap replicates) yielded nominal 95% percentile intervals of [+0.012,+3.111] novel positions and [-0.0494,+0.0247] goal-completion fraction. This is not strong evidence of policy generalisation.

Concrete failure cases: Greebles 2789, alternating-teeth destination veto reached 43 positions; Greebles 2819 reached 27 (teeth) and 46 (cage); Pipework 2741 cage reached 29. Exit-aware policy introduced different Greebles cage failures: 2753 reached 39 and 2801 reached 47. No restricted policy required a forced STOP during this sample; failure arose through changed movements.

## Frozen P14 learned-ranker transfer — negative result

Reloaded P14's actual 500-step-Adam seven-feature ranking head, SHA-256 fece7f68288df0c1fc5777b10c8c68694206842f48d49fae0533e84bb1db91f9. **No retraining on P15 data**. Of 211 non-tied algorithmically labelled comparisons, the transferred learned ranker agreed in **189/211 (89.57%)** versus **205/211 (97.16%)** for P14's transparent simple safety-first comparator. The exact source priority achieves 211/211 tautologically. Thus the learned model transferred worse than the transparent baseline.

Labels are deterministic protection/completion/construction proxies, **not owner/human aesthetic preferences**. The score is retrospective, not an online controller. New topology does not imply entirely untouched worlds: P15 deliberately reuses P14's nine held-out seed identities, but neither model was trained on those held-out seeds.

## Source, receipts, and qualification

Pinned input P11 archive SHA-256: 9a9f61038bb1a4f68395816a8296fb8957a8a8f166610921c2a9db6502308d9c.
W16 frozen auxiliary-head SHA-256: f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9.
P15 source Git blob: ab12d550022835630dd8f952c267b1d9669009be.
P15 tests Git blob: 965e25e6b89c0a52fe0521cf7fe978e349510113.
Results JSON SHA-256: 3b5a0b7921637add986fd1b44acec252a50e6da1d7d725787db57df32dcae129.
Matched-pair JSON SHA-256: 9082622a0d57bf605f560a382f7ccd06acb9645b477f154a84e41eec0b44d5cb.

**243 W0 model trajectory/PNG pairs, 162 unique trajectory byte identities, 81 original P11 PNGs byte-exactly reconstructed, 30,405 genuine W16-head forwards, all action receipts legal; eight local adversarial/unit tests passed.** The original binary source, model and trained ranker remain external immutable evidence; binary checkpoints are not Git-tracked. The source-integrity integration test skips when pinned P11 input is absent and passed when present locally. This is NOT a new hosted DRIV3R Actions reproducibility claim.

Reproduction from the preference_eval directory:
  python -m unittest -v test_p15_topology_transfer
  python p15_topology_transfer.py --p11-artifact /path/to/immutable-P11.zip --p14-ranker /path/to/P14-synthetic-pairwise-ranker.pt --out /tmp/p15 --images

The locally preserved data include P15-results.json, P15-matched.json, P15-proxy-pairs.json, P15-index.json, the full hash manifest and a 243-specimen source PNG bestiary. The binary artifact is not committed to Git.

## Decision

Do not promote any of these external filters or rankers into CyberSand production. A one-cell safe destination is NOT proof of protected-material paint safety. Even an action-preserving safety filter can reduce goal reachability, and algorithmic preference models can underperform transparent rules on unfamiliar geometry. Next gate: owner-approved canonical per-command paint write-set/protection receipts, and genuinely blind owner-rated creativity comparisons. Keep PR #1 Draft/HOLD and unmerged.
