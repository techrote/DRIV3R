# C60 — Host-legal remote-goal exploration with actual on-policy optimizer updates

**Disposition: RESEARCH HOLD — synthetic-host only; 18 November 2026.**

Repository: `techrote/DRIV3R` Draft/HOLD PR #1, `experiments/driv3r/training/guidance_recovery/**` exclusively.

## Purpose and experimental boundary

C59 proved an explicit observed-route eligibility gate avoids regressions by abstaining when remote navigation is unknown, but could not discover routes. C60 asks whether reward-driven **stochastic exploration from currently authorized actions** can increase genuinely remote-goal completion without hidden maze access or loss of self-directed movement. The original C59 and predecessor research modules remain unchanged. This is external synthetic CyberSand-like evaluation, not native acceptance or a replacement game ABI.

## Actual bounded optimizer and controls

The learned controller has a twelve-feature linear action head, appended as a **bounded ±1.8 logit** correction to the frozen host-legal RESET action distribution. The features use only contemporaneously offered actions, previously observed directed edges, visited positions/action history, published goal coordinate and personality sign. Crucially they do **not** consult hidden protected cells, a complete maze graph, a shortest-path oracle, or a counterfactual outcome during inference. Each executed action is revalidated by the synthetic host before application.

Actual CPU AdamW optimizes policy-gradient log probabilities over discounted rewards from **executed on-policy** trajectories, with entropy support and a small KL-to-RESET penalty. Training rewards favor first visits (+0.65), local goal progress, successful goal arrival (+8), and penalize premature stopping; no oracle route labels are trained. A wrong-credit SHUFFLED control receives randomly reassigned delayed returns while retaining actual state/action observations and the same model/optimizer configuration. The untrained RANDOM parameter vector is retained in the metadata but was not evaluated as a policy.

Training: 16 distinct fresh familiar-family graphs, 11 eligible host-executed prefixes, 88 on-policy episodes and **8 CPU AdamW minibatch updates per head**. Control has 8 matching updates. Independent single-thread deterministic refits reproduced both trained weight tensors exactly. No paid compute. These are small research heads, not SLM/LoRA models.

Policies: unguided C18-A0, C34-RESET, non-neural local NOVELTY, non-neural observed-frontier route option, trained ALIGNED and train-time reward-SHUFFLED control. Host cue strengths: none, weak BACKTRACK 0.35, strong BACKTRACK 1.0. Historical menu/action corruption: 0% and 50%.

## Held-out results — completed remote goals

32 independent graph identities (16 previously unseen Wilson/Kruskal, 16 new familiar-family), 25 eligible remote-goal host-executed prefixes, 150 matched conditions, **900 independently executed policy suffixes**. Each row has 15 baseline goals from 50 guided episodes, as below.

| Policy | 0% memory erasure (goals / 50) | 50% erasure (goals / 50) | New sites (0%, 50%) |
|---|---:|---:|---:|
| C18-A0 | 12 | 12 | 294, 294 |
| C34-RESET | 15 | 15 | 402, 402 |
| C60-NOVELTY | 16 | 16 | 428, 428 |
| C60-FRONTIER | 18 | 18 | 505, 518 |
| C60-ALIGNED | 17 | 17 | 539, 559 |
| C60-SHUFFLED | 14 | 14 | 458, 462 |

ALIGNED gained **+2 goals** against RESET and **+3** against SHUFFLED per memory condition, with zero paired losses against RESET in this small selected cohort; FRONTIER gained +3 against RESET, demonstrating that the learnable explorer does not beat a simple explicitly observed-route heuristic. ALIGNED increased newly visited sites substantially, but novelty is only a proxy for useful exploration.

The key **negative transfer** result is that ALIGNED tied RESET on unseen Wilson/Kruskal remote goals (4 versus 4 from the eligible unseen-family remote-goal suffixes). Both extra learned goals occurred in fresh familiar-family braids, not new topology families. Non-neural NOVELTY and FRONTIER each scored 6 unseen-family successes, versus RESET 4; these are small exploratory differences, not robust generalization. **No learned OOD navigation improvement is established.**

Graph-cluster bootstrap descriptive 95% for ALIGNED relative to RESET is 0 to +5 goals at both corruption levels; these intervals are adaptive, uncorrected and are not a promotion criterion. All other observed contrast/negative cases remain in the full ledger. Only two neural training gradients batches per epoch were used; the small optimizer budget is a limitation.

## Personality, safety and independent replay

All 50 matched no-guidance conditions preserved identical movement and painting hashes across every policy. All 900 synthetic-host suffixes executed only freshly legal actions with **zero** rejected attempts and unchanged protected-cell hash relative to the source snapshot. The full ledger contains every policy outcome, seed, graph SHA, correction strength, hash and memory intervention.

For first-20-action matched episodes with both policies surviving 20 actions, ALIGNED versus RESET had +0.035 unique-position ratio, −0.017 turn rate and +0.018 normalized directional entropy at 0% corruption (n=38 paired samples); at 50% corruption the corresponding differences were +0.043, −0.025, +0.030 (n=38). These demonstrate measurable movement changes but **do not** prove preservation of Meanderer/Curler/Brancher traits or artistic personality; short/early-ending runs are censored.

A separate gallery verifier reconstructed source-matched world/prefix state and re-executed **16 representative actual trajectories** across unseen-frontier improvement, familiar learned improvement, shared failure, and shared success. Their action and painting SHA-256 evidence matches the full online ledger. The SVG shows only executed movement, never invented walls or routes.

## Custody and reproducibility

The repository receives text-based executable source, tests, compact summary, compressed full executed ledger, extractor, replay-verified SVG and this report. Checkpoint binaries are **not committed to Git**: the owner archive includes both small `.safetensors` checkpoints and frozen historical prerequisites. Immutable Actions artifact custody and a native CyberSand execution are still missing. This is reproducible *external synthetic host* behavior under pinned dependencies, not qualified production science.

Independent refit log: `aligned independent refit exact parameter equality: True updates: 8`; `shuffled independent refit exact parameter equality: True updates: 8`. Six targeted unit tests pass.

## Promotion decision

**HOLD.** The online signal is localized to fresh familiar-family cases; the aligned head does not outperform RESET in Wilson/Kruskal, cannot beat the simple frontier heuristic, and whole-trajectory personality is unqualified. Retain the negative transfer result and avoid model promotion. A better next experiment would raise the number of *discriminating unseen-topology* successes, predeclare heldout graph seeds, and vary reward mix/horizon without choosing weights after viewing holdout outcomes.
