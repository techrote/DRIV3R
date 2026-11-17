# C33 — Irregular multibranch recovery: out-of-family negative neural control

**Date:** 17 November 2026. **Repository:** techrote/DRIV3R, existing Draft/HOLD PR #1 only. **Owner:** `experiments/driv3r/training/guidance_recovery/**`. **Scientific disposition: NOT PROMOTED.** No changes to CyberSand, Neurounner, DRIV3R main, d_common or production settings.

## Question

Does C32-WATCH's improvement generalize when the source corridor has additional irregular side branches and cyclic connections, and can a tiny head trained on actual short-horizon executed returns outperform a fixed local probe rule?

## Host-executed test

Start from C32's 27 legally executed movement prefix and its off-history changed goal. Before issuing advice, add new open cells to the protected map by one of two deterministic generators. `twigs` grafts stochastic 2–8-cell walks from multiple previously traversed anchors. `braid` creates longer Manhattan connections between separated anchors, introducing additional loops, junctions, misleading exits and genuine shortcuts. Each world is reconstructed from a fixed seed. `braid` is **withheld completely from optimizer training**. This is out-of-family graph connectivity relative to the training grafts, but not full unseen maze topologies: both generators retain C32's underlying U corridor and synthetic 15×15 grid.

Evaluation spans four original goal/obstacle situations (`off-early`, `off-mid`, `shortcut`, `lure4`), two personality polarities, two guidance intensities (0.35/1.0), no cue, and weak/strong BACKTRACK and CONTINUE advice. It includes two independent withheld-seed cohorts, two graph seeds per scenario/arm/personality, a **125-action horizon**, 128 different topology fingerprints, and 2,560 actual legally executed policy episodes across four policies. There are 512 unguided and 2,048 guided policy episodes. Graphs added 5–36 open cells; host-protected cells remain immutable.

Policies: frozen C18-A0 baseline, original C32-WATCH, and two C33 variants. `C33-FIXED` gives a bounded additional probe bonus to the baseline's preferred fresh goalward action; `C33-HEAD` instead uses a small neural return predictor to choose among such actions. Both remain stochastic and apply only following explicit positive host advice. Neither receives topology identifiers or forbidden-cell bypasses.

## Training and controls

C33's 8→12→1 CPU head trained for 120 real AdamW updates on 462 action-menu samples from `twigs` environments. Labels were **executed ten-action local continuation returns**, not oracle BFS outputs or symbolic scenario labels. Training included 246 positive-return samples; the training MAE was 0.54485. Two independent training executions reproduced an identical SHA-256 checkpoint `f407881b60515aba159f585cb0480052d633c950260a46d879119e44519a63ac`. This is supervised fitting of an auxiliary head, not demonstrated end-to-end game-policy learning.

The independent C18-A0 frozen baseline, original C32-WATCH and extra fixed-bonus C33-FIXED are all source-matched and evaluated on identical worlds, arm/cue settings and stochastic seeds. Reproducibility controls check unguided action and ink hashes, source hashes, trained tensors, protected materials and host rejections.

## Results

| Policy | Cohort A guided | Cohort B guided | Combined /512 | Condition-level regressions vs C18 |
| --- | ---: | ---: | ---: | --- |
| C18-A0 | 124/256 | 127/256 | 251 | Reference |
| **C32-WATCH** | **138/256** | 135/256 | **273** | **2 groups in A, 2 in B** |
| C33-FIXED | 136/256 | **136/256** | 272 | 2 groups in A, 1 in B |
| C33-HEAD | 136/256 | **136/256** | 272 | 2 groups in A, 1 in B |

C32-WATCH retains an aggregate benefit of 22 completions across the 512 guided trials, but fails the previously required zero-regression condition gate. Its weaknesses include strong/weak BACKTRACK advice at deceptive `lure4` locations. The fixed-bonus variant removes one regression in B but introduces no reliable total improvement.

**The neural head changed no executed decisions whatsoever.** `C33-HEAD` and `C33-FIXED` produced identical action hashes in **640/640 paired episodes** (320 in each cohort), and every head-vs-baseline local probe ranking remained unchanged. A lower optimizer loss is thus *not* evidence of a learned gameplay benefit.

For each of the three recovery policies, **128/128** matched unguided action and ink trajectories remained byte-identical to C18-A0. No host action rejections, raw illegal top choices or protected-material modifications were observed in the complete evaluation. Eleven selected local C32+C33 regression tests passed; synthetic assertions independently recomputed host legality. Gallery images reconstruct the exact seed-specific map and plot actual sampled movement rather than a predicted route.

## Limitations and decision

This result is a **structural stress-test / negative model finding**, not a qualification or a deployment recommendation. The 15×15 environments still inherit the C32 U-prefix; locally observable obstruction features may overinform the agent. Longer native gameplay, genuine material reactions, new goal-switch sequences after advice, multi-source cue reliability, physical CyberSand action/observation equivalence and active-guidance personality behaviour have not been established. There are only two seed cohorts, and no statistical significance claim is made from modest absolute differences.

**Reject C33-HEAD and C33-FIXED as promoted policies. Keep C32-WATCH only as a synthetic research control.** The trained checkpoint remains in the separately hashed research archive, not Git history. An immutable GitHub Actions checkpoint artifact is still missing; the local archive is interim custody only. No CI training is configured to run automatically. Full repository-based scientific reproduction and native CyberSand acceptance remain outstanding.

The next experiment should replace U-derived grafts with topology-independent graph generators and test goal switches with explicit *new* host cues, then use held-out actual returns as a hard neural-ranking improvement gate; do not spend more cycles optimizing a head whose decisions saturate to an unchanged fixed policy.

## Reproduce locally (requires retained prerequisite checkpoints)

```bash
PYTHONPATH=experiments/driv3r/training/guidance_recovery python -m pytest -q experiments/driv3r/training/guidance_recovery/test_c32_offstack_recovery.py experiments/driv3r/training/guidance_recovery/test_c33_irregular_recovery.py
PYTHONPATH=experiments/driv3r/training/guidance_recovery python experiments/driv3r/training/guidance_recovery/c33_irregular_recovery.py /path/to/extracted/prerequisites-root /tmp/c33-run --n 2 --horizon 125
```

The retained archive includes the exact prerequisite checkpoint layout, full executed trajectories, evaluation topology fingerprints, gallery, source snapshot, model tensor hashes and training-menu receipt. `C33-summary.json` in this directory is the compact source-hash-linked ledger.
