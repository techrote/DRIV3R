# DELAYED-TARGET-001 — terminal target-outcome imitation versus short-horizon cost

**17 November 2026 · DRIV3R Draft/HOLD PR #1 · negative DEVELOPMENT result.**

## Question and preregistered comparison

The preceding COST-RETARGET-001 result suggested that its five-action score overvalued immediate progress. Does a **12-decision** teacher lookahead, with a strong *actual target contact* reward, improve autonomous retargeting? Tested four controllers with identical 961-parameter architecture, initial weights per seed, TRAIN-state reservoir, optimizer minibatch budget, target instructions and DEVELOPMENT perturbations:

- `frozen`: recovered base policy (180 Adam updates, 60 per 36/72/128 stage).
- `imitation`: base plus 20 ordinary imitation updates per stage.
- `short_cost`: base plus 20 pure five-action cost-label updates per stage (not the earlier study's 70/30 mixture).
- `delayed_target`: base plus 20 twelve-action outcome-label updates per stage, with a +22 bonus **only for newly painting the current target**, +0.25 final goal-distance gain, +0.25 new paint, -1.5 recovery expenditure and -12 for stopping. Unlike the earlier five-action score, this score emphasizes terminal contact. Every counterfactual action and teacher continuation is checked against `World.menu(12)`.

Two independent seeds (`631`, `632`) each used 64 student-visited TRAIN snapshots per stage. Only the first candidate action was counterfactual; the subsequent eleven actions came from the fixed recovery teacher. No future goal instruction was revealed. This is a **teacher-outcome imitation** experiment, not reinforcement learning or self-directed 12-step planning.

## Source-matched DEVELOPMENT results

Evaluation comprised **16 matched episodes per arm**: training seeds 631/632 × two previously established DEVELOPMENT world seeds (3001/3101) × normal and trap-corridor × 128/160 decisions, with goal switches at 32/72/112. The two world families are pickets and ridges, **not a structural held-out generalization test**. There were 64 episodes and 7,909 original model-selected/replayed decisions across arms.

| Metric | Frozen | Imitation | Five-step cost | Twelve-step outcome |
|---|---:|---:|---:|---:|
| Instructed target epochs reached | 37 | **39** | 31 | **3** |
| Mean unique painted cells | **141.06** | 134.69 | 124.38 | **81.75** |
| Premature STOP episodes | **0/16** | 4/16 | 5/16 | **12/16** |
| Eight-decision post-switch painted cells (sum) | **382** | 367 | 337 | **210** |
| Recovery actions, total | 47 | 51 | 41 | 39 |
| Mean policy decision latency, ms | 0.126 | 0.125 | 0.126 | 0.123 |

The outcome-trained model reached **34 fewer instructed targets** and painted **59.31 fewer cells per episode** than the frozen control. Matched-pair bootstrap (20,000 resamples; seeded; descriptive, not a world-family-generalization interval) puts its painted-cell delta against frozen at **-80.00 to -38.56**, and its goal-epoch delta at **-2.50 to -1.75**. It reached fewer targets in all 16 matched episodes. Against matched imitation its mean painted-cell delta was **-52.94** (bootstrap 95% -72.50 to -34.00).

**All 12 STOP actions in the outcome arm were forced by menus offering only STOP**; this was not an explicit stop preference at the decision of termination. A more plausible explanation is that earlier learned movement exhausted legal options. This is not formally isolated as a cause.

## Why this matters

The longer teacher horizon did produce new supervision: **240/384** delayed labels differed from imitation and **193/384** differed from five-action labels. Yet just **119/384** sampled candidate-action sets contained any new-target contact within twelve steps. Outcome labels were often decided by proxy paint/distance/recovery scores, not actual target hits. Delayed labels also favoured synthetic `RELOCATE_START` in 48/384 samples (12.5%) rather than the near-zero imitation frequency. These are candidate mechanisms, not causal diagnoses.

This is a strong **negative result for the tested delayed-score construction**, not evidence that delayed credit or longer lookahead is generally harmful. It does not justify promotion of cost-sensitive targeting. Next: isolate terminal-contact target scarcity and vary horizon with matched, density-balanced target examples; separately test recovery action penalties. The frozen and imitation controls should remain preserved, without tuning against this same DEVELOPMENT set indefinitely.

## Provenance, verification and boundary

- **8 real NumPy checkpoints** (four per seed): all restored and verified for parameter identity; each model has 961 parameters, with base 180 Adam updates and adapted 60 additional updates where applicable. Checkpoint identities are in each `seed_*/report.json` and the source-bound `seed_*/SHA256SUMS.json`.
- Original traces include full action probability distributions, legal menus, painted-cell edits, state hashes, fixed goal schedules, original 96×96 PNGs and complete TRAIN counterfactual labels.
- `--verify` re-executed **64/64 original traces and 7,909 decisions** from restored model weights and checked **84/84** payload SHA-256 entries. Focused and predecessor tests: **10/10 PASS**, including probability-tampering rejection.
- CPU campaign time: **18.502 s / 19.112 s**; peak process RSS: **123,720 / 124,044 KiB** on the experiment host. These are not large-SLM inference timings.
- Verified local dependency Git blobs match DRIV3R branch at inspected HEAD `f0088f64706ce22b48e77fd7e5ee73b3c5b1b87c`, including `cost_retarget_lab.py`, `goal_recovery_lab.py`, `goal_retarget_lab.py`, `retarget_onpolicy_lab.py`, `goal_conditioning_lab.py`, `world.py`, `policy.py` and `train.py`.
- Checkpoint binaries and original JSON traces are **not committed to Git**; retained in locally verified evidence ZIP pending a permitted opt-in Actions artifact route. No claims of immutable Actions qualification.
- All work is scoped to `experiments/driv3r/training/long_horizon/**`, source-faithful *synthetic* worlds only. No actual native receipts, no sealed structural heldout use, no physical CyberSand gameplay, no paid compute and no external repository writes.

**Disposition:** retain negative result, do not merge or promote the Draft/HOLD PR.

## Exact checkpoint SHA-256 ledger

| Training seed | Model | SHA-256 of original NPZ | Adam updates |
|---|---|---|---:|
| 631 | frozen | 304c5ded6e8122ebec368ed77b3509106989b4bfaf2347c06bad039d1bbdc516 | 180 |
| 631 | imitation | aa5fa0c4188bfb060e0f2f9a5b5ecb266416e5677a050eba3777fddf7d598ce2 | 240 |
| 631 | short_cost | 63269c817dfe92d76d07d70e217acc14510c8846d2a73a255ff7b5f4f03acaa0 | 240 |
| 631 | delayed_target | c7204890d49cba15c01ed171d787c088eaa24ff105c5a9cae5211b08588c6ede | 240 |
| 632 | frozen | 4b8a13581c5b597028112f37b900c09cc94a8bc28e8df931c1783afb4a271565 | 180 |
| 632 | imitation | dde1a02438a199aece530a937ae096f93e63951c81937dfee6e9eabaf9620a1c | 240 |
| 632 | short_cost | b7d7d74c875f46c253e87eddb8b0c579bcbac7a1a03febdaa77cbc6d9954e1fc | 240 |
| 632 | delayed_target | 5c2ae40f5a9b74a41f123550353695077f7e8c187ed30f388bd5be68acbc729f | 240 |

Exact executed trainer SHA-256: e36bd22fc41528203eb75528e62303b5889f685b08943a597be68378c3dcc3e3. Original locally verified evidence ZIP SHA-256: bb860ca38d78ce3525808ce78d5e64dcb35676806d1b145175722810d5561222 (88 archive entries; not an Actions artifact).
