# P18 — novelty versus scenario-adversary population tournament (local CPU evidence)

**16 November 2026 · techrote/DRIV3R · existing Draft/HOLD PR #1 exclusively.** This is a deliberately simpler **48-decision** population-selection control built on the imported P0 synthetic host. It complements, but is **not directly comparable numerically to**, the existing P17 three-goal/96-turn mission study. No native CyberSand changes or claims.

## Question and original experiment

Do novelty-selected parents or TRAIN-only adversarial scenario selection improve broad survival/recovery/painting policy populations relative to reward-based PBT and random ancestry? The experiment keeps six separate niche-conditioned, genuinely Adam-trained 20-feature/24-hidden-unit MLP populations. Every arm has identical initial seeds and training/selection/holdout world identities; only parent selection or TRAIN scenario sampling differs. Later-generation parent changes alter experienced states, so this is not a strict single-variable causal experiment.

Two independent cohorts (**676, 677**) train **288 candidates** across six niches × four strategies × (2 initial + 2 + 2 offspring) with **16,537 actual Adam updates**. Challenge PBT spends **1,728 extra training-only probe rollouts** (current policy and synthetic oracle on all four allowed scenario families). Selection uses two disjoint seed streams × four TRAIN scenario families. Final 48-decision evaluation is untouched: three other seeds × two withheld geometry families. The scenario mix includes obstacles, protected/material-like cells, dead-end detours, painting, parent-tree RETRACE, STOP, and a mid-horizon goal switch. This synthetic World is not validated for current native material or action ABI equivalence.

## Held-out results

| Strategy | Mean synthetic reward | Mean painted cells | Retraces | Early STOP | Cross-niche descriptor diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reward-only PBT | **31.3030** | **30.40** | **8.60** | **33.33%** | 0.4845 |
| TRAIN-only hardness PBT | 29.6962 | 29.13 | 6.44 | 44.44% | 0.4970 |
| Novelty pressure (normalized selection descriptors) | **17.3966** | 18.03 | 2.50 | **73.61%** | **0.5293** |
| Random-parent PBT | 30.0103 | 29.21 | 8.08 | 37.50% | 0.4668 |

The novelty arm reaches the highest descriptor distance while catastrophically losing continuation, recovery and useful paint coverage. Its goal switch is reached in only **31.94%** of held-out episodes (reward PBT: **76.39%**). For these cohorts, challenging the current policy with hardest TRAIN scenarios **underperforms** reward-only training. This negative result is retained, not tuned away. Two cohorts and synthetic heuristic reward do not support claims of robust superiority or learned aesthetics. A mechanically masked host guarantees legality and disallows painting over obstacles/protected cells by construction, not by safety learning.

## Qualification, reproducibility and scientific limits

Seven local source guardrail tests pass. The separately implemented `p18_replay_audit.py` reloads **all 288 checkpoint JSON files**, replays **96 selected checkpoint×rank evaluations** covering **576 original held-out episodes / 20,484 host-legal actions**, verifies **48 original trajectory galleries**, all checkpoint lineage SHA-256 and **387/387 overall manifest entries** with zero mismatches. The replay shares the P0 synthetic physics/legality code; it is not an independent native game replication.

Local complete evidence includes every original optimizer-trained weight set, selection lineage, seed hashes, split manifest, full selected held-out summaries, 48 original traced SVG+JSON gallery pairs, and manifest; Git keeps the source, test, audit program and a condensed but complete metadata ledger with all **288 initial/final model SHA-256 hashes**. No large weights are committed to Git. No GitHub Actions run, paid API/GPU or immutable Actions artifact was created. The locally downloadable archive is not guaranteed durable outside the conversation.

Reproduce with Python 3.11+ / NumPy 2.3.x and no PyTorch requirement:

```bash
cd experiments/driv3r/training/population
OPENBLAS_NUM_THREADS=1 python -m unittest test_population_p18_tournament -v
OPENBLAS_NUM_THREADS=1 python population_p18_tournament.py --output /tmp/p18-676-677 --cohorts 676 677 --initial 2 --generations 2 --offspring 2 --training-worlds 3 --selection-worlds 2 --heldout-worlds 3 --horizon 48
OPENBLAS_NUM_THREADS=1 python p18_replay_audit.py /tmp/p18-676-677 --output /tmp/p18-replay.json
```

Every output directory must be new. No scripts execute automatically on PR/push. Main is untouched. Existing DRIV3R PR #1 remains Draft/HOLD.

## Next discriminator

Rather than maximizing an unconstrained handcrafted novelty descriptor, use a **minimum viable continuation constraint** (avoid premature STOP; demonstrate post-switch recovery) then investigate within-niche portfolio diversity. P17's richer three-stage mission remains the stronger test of complete goal sequencing, and synthetic ABI fidelity must be checked against read-only CyberSand sources before proposing a game-side handoff.
