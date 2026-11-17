# C25 — Online, executed-feedback cue trust (negative replication)

**Date:** 2026-11-17 · **Ownership:** `techrote/DRIV3R` Draft/HOLD PR #1 only  
**Status:** **Reject C25-ONLINE and C25-NOMEM; retain frozen C12 and untrained C18-A0.**  
**Scope:** `experiments/driv3r/training/guidance_recovery/**`; no game code, gameplay assets, production providers, or other workers changed.

## Question

Can a worm learn to arbitrate host advice using the consequences of **its own recent executed moves**, rather than C24's retrospective reputation signal? This is intentionally a small, bounded optimizer and synthetic-game research test, not native CyberSand qualification.

## Implementation and information boundary

- Frozen C12 is the trained predecessor; C18-A0 is the retained untrained host-attested trail-to-goal control. The trained residual sees a `(10,36)` menu consisting of C18's `(10,32)` menu plus four **past-only** features: recent signed Manhattan progress, repeated visits within the short current-cue window, fraction of choices following the present advice, and progress on those followed moves.
- Features are recomputed after host-applied actions, never from an oracle future or an invented replay history. Cue expiry or changed goals invalidates historical signals. Target outcome labels and map topology families never enter the policy's inference features. The offline training label *does* use matched host-executed counterfactual continuations; this remains a limitation for deployment.
- C25-ONLINE learns an eight-input, sixteen-hidden-unit BACKTRACK residual (scale 0.85) on top of C18-A0; C25-NOMEM uses the same model but zeros only the four online inputs. No-cue logits are **exact frozen C12**, all other action logits are unchanged, and host menu legality remains authoritative.
- 1,267 matched training menus (499 with nonzero online memory); 2,534 forced-action counterfactual suffixes; 795 near-tied, 239 negative and 233 positive return advantages. Group-balanced AdamW, 12 epochs, 240 updates per head. C25-ONLINE surrogate objective decreased from 0.0001421 to −0.0041109, but this is not evidence of improved gameplay.

## Held-out outcomes

All episodes were fresh-seed stochastic actions in source-pinned synthetic host geometry: reverse/ahead/shortcut/bridge, withheld hook/zigzag layouts and shifted shortcut/bridge geometries, both drawing personalities, no guidance and weak/strong positive/misleading advice. The original C24 source and prerequisite model digests are retained in the evidence bundle.

| Policy | Cohort A (6 independent scenario sources) | Cohort B (8 fresh sources, longer horizon) |
|---|---:|---:|
| Frozen C12 | 445 / 576 | 585 / 768 |
| C18-A0 control | **457 / 576** | **620 / 768** |
| C25-ONLINE | 461 / 576 | 612 / 768 |
| C25-NOMEM | 457 / 576 | 619 / 768 |

*Guided success counts; the no-guidance control is excluded from this table.* The two cohorts contain 2,880 and 3,840 evaluated trajectories, respectively, across all four policies. C25-ONLINE gained four goals in A but lost eight in B relative to A0. The *strict condition-level gate* fails for both trained candidates in both cohorts.

- Cohort A C25-ONLINE: **+4 net** guided successes, but regression on **shortcut / strong CONTINUE**, 32 → 31. C25-NOMEM: three regression groups.
- Independent B C25-ONLINE: **−8 net** guided successes; four regression groups, including legitimate strong BACKTRACK 31 → 29, misleading strong CONTINUE 44 → 41, and both shortcut BACKTRACK strengths. C25-NOMEM: three regression groups.
- Unguided matched trajectories remained byte-identical to C12 for **144 episodes per model in A and 192 in B**, including exact action and ink hashes. Across all 6,720 evaluated trajectories, no synthetic host rejections, illegal raw top-choice commands or protected-material modifications were observed.
- Personality-favoured paint opportunities were evaluated after the executed trajectories diverged, so the counts are *not* direct causal comparisons. For guided trajectories in A: A0 384/1,576; ONLINE 381/1,563. In B: A0 488/2,082; ONLINE 483/2,051. These do not demonstrate personality improvement.
- A first positive goal-distance reduction after the induced mistake was recorded for each episode, along with time to completion and repeated visits. These are *observable synthetic recovery proxies*, not native CyberSand navigation timing or a ground-truth judgement of spatial understanding.

Both optimizer repetitions produced byte-identical model checkpoints and source-matched training hashes. This is **local deterministic repeatability**, not an assertion of independent scientific reproducibility or native physical acceptance.

## Reproduction / evidence

Run from the included `source/` snapshot with CPU PyTorch, NumPy and safetensors available:

```bash
python c25_online_recovery.py .. ../final_a --train-n 8 --eval-n 6 --epochs 12 --horizon 42 --eval-seed 250999
python c25_online_recovery.py .. ../final_b --train-n 8 --eval-n 8 --epochs 12 --horizon 48 --eval-seed 251999
python -m unittest discover -q -p 'test_*.py'
python c25_gallery.py .. ../final_a ../gallery
```

Full CSV-equivalent JSON receipts, complete stochastic outcomes, checkpoints, exact source snapshot and 20 source-reconstructed SVG trajectory panels are in the SHA-256-manifested external research bundle. Only inference/trainer/test/gallery source and this report plus the evidence receipt are committed to Git; no checkpoint weights or large trajectory binaries enter Git history. Immutable Actions artifact custody is **not** achieved; the archive is provisional. Full latest-HEAD CI and real CyberSand gameplay qualification remain open.

## Decision and next research target

**Reject both C25 models.** Online signals are genuine executed-host observations and fit the synthetic contract, but the optimizer overfits the training counterfactuals. C25 shows stronger learning loss improvement than C24 but no robust policy lift. Narrow the next study to *intervention-triggered* reward from sustained stuckness/revisits, and pre-register the same strong-backtrack and misleading-advice nonregression tests. Keep the host action/observation contract authoritative and unchanged.
