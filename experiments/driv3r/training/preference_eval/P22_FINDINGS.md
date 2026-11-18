# P22 — actual LFM350 trained adapter tensors and first-divergence action margins

17 November 2026 · DRIV3R Preference Lab · existing [Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1). **This is a real historical model-weight and original-model-output audit, not a new forward pass or training run.**

## Experiment and source custody

Re-open the three exact original pretrained LiquidAI/LFM2.5-350M revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3` GP350-012 Actions archives independently reviewed in P21: stock #14248333879, GP006 #14244806761, GP010 #14245172930. All original ZIP hashes, W0 host-applied 36 × 128 action menus/observations/positions and PNG hashes are checked by existing `p21_real_lfm_bestiary.load()`. Then check **original safetensors model adapter weights**, including the byte-identical nested GP006 step96 checkpoint in the GP010 source ZIP. All 24 tensors (172,032 trainable parameters; rank-8, alpha-16 q/v LoRA in 12 source projections) are finite, shape-aligned and original-hashed. The step24, 48 and 72 snapshots each change every effective low-rank target matrix relative to GP006 step96. Summed per-target Frobenius norm of the effective-matrix change: 1.27794 / 1.63063 / 1.85288 respectively. **These are parameter-space distances, not predictions of behavioural strength**; step48/72 do not have matching real closed-loop GP350-012 outputs.

## Causally restricted same-observation comparison

For each matched world, only compare actual model six-way logits **until and including the first differing host-applied action**. Once one worm takes a different action, its later observation is no longer a counterfactual for the other model; such later logits are excluded. Verify greedy choice using the *recorded legal action menu*. Discard branch-margin probes when BRANCH was not legal.

- 12 matched original worlds; two source seed blocks (9803, 9821), 2 sizes × 3 material families, exactly 128 applied actions per arm.
- GP006 and GP010 diverge in 4/12 complete original paths; **8/12 remain identical**.
- 1,061 legally comparable same-observation BRANCH logit margin pairs; GP010 shifts the average per-world BRANCH-vs-best-other margin by +6.41 architecture, +5.03 Greebles and +6.81 pipework logit units. This is **broad branch pressure even in families that did not visibly branch**.
- Parent GP006 selected BRANCH on 0/1,061 shared states; child GP010 selected BRANCH on 3/1,061. The child ultimately makes 9 BRANCH actions in three Greebles worlds, including after paths diverged. These are **not nine matched causal action flips**.
- Divergence begins at ticks 1, 1 and 84 in three Greebles worlds; the fourth is a pipework RIGHT→LEFT difference at tick 55, with no BRANCH.
- Most changes produce neither a new niche nor a verified aesthetic improvement. One 64-cell Greebles case changes coverage from 129 to **89** distinct positions after BRANCH at tick 84. Both paths remain legal, and neither proves native materials/paint safety.

**Interpretation:** a genuinely updated pretrained-LFM LoRA can substantially raise BRANCH relative logits while usually preserving the original greedy path. Rare early action-margin crossings can radically alter the eventual painting. The GP010 learner was explicitly initialized from GP006; **these are not independently initialized artistic personalities**, and two original seed blocks cannot establish generality.

## Reproducibility and next experiment

The committed `P22_pinned_adapter_audit.py` reproduces the per-world decision/weight audit, generated original A4 SVG pages and source hashes from original ZIP inputs; 7/7 local tests pass, including forged observation, forged decoder choice, nested adapter mutation and byte-identical double-run checks. The evidence ledger is `P22_EVIDENCE.json`; checkpoint/model binaries are not in Git.

No new LFM root forwards, optimizer updates or CPU latency measurements were possible: pinned base model files and matching transformers runtime are not locally present, and the available GitHub connection has no explicit opt-in workflow dispatch. A source-only smoke run is not model inference. No human/owner preference labels were fabricated, and **no new reward ranker is warranted**. The next meaningful comparison is genuinely changing-observation GP350-018 sampled versus greedy full-model decoding on world-disjoint seeds, plus source-original Tiny/Population/Franken specimens for four distinct creative-niche ratings. Send the broad BRANCH-margin effect and the single major coverage loss as hard negatives to Population/Distillation by this handoff, without writing their directories.

**Repository boundary:** only Preference Lab files on DRIV3R PR #1; Draft/HOLD; no merging or external repo writes. Synthetic W0 ≠ native CyberSand.
