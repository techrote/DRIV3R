# C23 — Joint return-constrained, opportunity-conditioned recovery

**17 November 2026 · DRIV3R guidance/recovery · research-only negative promotion**

## Question and method

C22 demonstrated legal painting opportunities but learned almost no episode-return gain. C23 now trains two bounded 32→12→1 INK sidecars with **actual host-executed counterfactual episode suffix returns**, plus an opportunity-conditioned personality objective, policy-distribution trust anchors, and balanced outcome/cue/personality groups. Train families: hairpin/staircase; fully disjoint held-out topologies: rotated/slitted hook/zigzag.

The host still controls action menus, material protections, actual cursor movement and execution. Frozen C12, untrained C18-A0 and C22-STRONG form independent controls. Advice arms are none, weak (0.35) and strong (1.0), crossed with soft hint strengths of 0, 0.35 and 1.0, both reversal and forward goals and two personalities. No new native ABI or CyberSand runtime assertions are made.

C23 collects **336 on-policy legal two-sided INK menus**, comparing the highest-scoring personality-favoured legal INK action with a legal opposition action. At each state, matched stochastic 24-decision continuations are actually executed in cloned host worlds under C18-A0. This yields **1,344 forced-first-action suffix trajectories** with common-random-number pairs. The 336 mean counterfactual returns comprise 47 positive, 55 negative and 234 near-tied pairs (threshold ±0.05). These targets are noisy and dominated by ties; they do not prove an effective signal exists in every training menu.

A differentiable action-probability objective uses the signed return advantage to favour or discourage the preferred INK decision, with opportunity-conditioned preference retention and KL/drift anchors. C23-GUARD has stronger trust anchors; C23-RETURN deliberately relaxes them. Each gets 175 real CPU AdamW updates (350 total). All changes remain restricted to the **host-legal preferred INK logits while host advice is active and both INK opportunities exist**; BACKTRACK, other logits and all no-cue actions are identical to C18-A0 or C12 as applicable. Training is **offline counterfactual policy optimization**, not end-to-end RL on updated C23 rollouts.

## Two independent held-out stochastic cohorts

Each cohort has 3,000 trajectories (5 policies × 600 matched episodes): 480 advice-present plus 120 no-cue trajectories per policy. The second uses a fresh independent evaluation seed. The optimizer itself was rerun for both cohorts, yielding byte-identical candidate checkpoint files.

### Cohort A — seed 341117

| Policy | Guided goal completions | Favourite legal INK probability on 1,227 identical held-out menus |
|---|---:|---:|
| C18-A0 | 425/480 | 25.688% |
| C22-STRONG | 426/480 | 26.066% |
| C23-GUARD | 425/480 | 25.985% |
| C23-RETURN | 424/480 | 25.934% |

### Cohort B — seed 887741

| Policy | Guided goal completions | Favourite INK probability on identical held-out menus |
|---|---:|---:|
| C18-A0 | 428/480 | 25.688% |
| C22-STRONG | 425/480 | 26.066% |
| C23-GUARD | 424/480 | 25.985% |
| C23-RETURN | 425/480 | 25.934% |

The C18-A0 personality-favoured INK mass is 25.6883%, compared with 25.9851% for C23-GUARD and 25.9342% for C23-RETURN. Both retain a substantial gap to C12's 26.4852%; improvements in choice mass do not establish stable long-horizon personality.

**Strict nonregression gate failed for both candidates in both cohorts.** Cohort A GUARD lost a valid weak-BACKTRACK completion (48→47); RETURN lost two (48→46) and one forward/misleading-BACKTRACK completion (59→58). Cohort B GUARD and RETURN lost two weak valid-BACKTRACK completions (42→40), and both lost two misleading-CONTINUE reversals (55→53). Positive effects elsewhere were small and not stable. Do not promote C23.

No-cue matched sequences and final ink hashes remained **120/120 exact per candidate per cohort**, relative to C12, including C18-A0 and C22-STRONG controls. Both held-out cohorts recorded zero host-rejected actions, zero illegal raw top-one selections and protected-cell invariants after every host action. Source/trajectory receipts and execution logs are in the local research archive, *not* Git checkpoint custody.

## Training and artifact identity

- Source SHA-256: `f862f9224ec8d5d9d5adc4cbb8c6f4b761164d9a0c004e3702848a118f9adc98`
- C23-GUARD safetensors SHA-256: `5bb24ca4ed7678d97aba22a4ca74bb9629599dee1c10bebb9157ce3a96a874cd`
- C23-RETURN safetensors SHA-256: `c82b9e964ad6b4ad6a5cec6c487b5a7096e874e6f326a11db320905e93bd5378`
- Data tensor/advantage/group SHA-256: `6adf85b690d300af43e2d32425ce620cc8f08e2db6e4bea1d532480885d76c67`
- Source-matched CPU local tests: **156/156 pass**, with a PyTorch test warning about casting an autograd tensor to float (not a failing assertion).
- Independently repeated optimizer runs produced byte-identical candidate checkpoint files; held-out evaluation seeds differed and were **not** byte-identical by design.
- All seven C12 prerequisite model checkpoint SHA-256 values are present in each `metrics.json`; the source Git commit is separate from model custody. Checkpoints are excluded from Git history. Existing historical source import alone does not prove independent scientific reproducibility.

## Limitations and disposition

C23 was measured against a synthetic CyberSand-like 15×15 host, **not the canonical CyberSand native actor/observation ABI**. Counterfactual values depend on a frozen C18-A0 continuation policy and a 24-decision suffix, while held-out trajectories run for 50 decisions. Source state sampling is on-policy under C18-A0 only, so the optimizer can suffer covariate shift. The candidate effects are small; no confidence interval or statistical significance claim is warranted. Both gate failures are reported rather than tuned away by repeated benchmark selection.

**Disposition: reject C23-GUARD and C23-RETURN. Keep C12 as the preferred trained checkpoint and C18-A0 as the strongest untrained reference; preserve C22-STRONG as a personality comparator.** Next research should use a fresh, genuinely informative correction signal or history-conditioned cue reliability, not additional scalar favourite-INK mass training on mostly tied forks.

Only `techrote/DRIV3R` Draft/HOLD PR #1 is writable. All source paths under `experiments/driv3r/training/guidance_recovery/**`. Do not merge, change the game, or spend paid compute.
