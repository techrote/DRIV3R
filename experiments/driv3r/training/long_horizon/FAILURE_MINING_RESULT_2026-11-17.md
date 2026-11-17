# FAILURE-MINING-001 — Risk-stratified near-failure student suffixes

17 November 2026 · DRIV3R Draft/HOLD research PR #1 · **negative DEVELOPMENT result**

## Question and design

The predecessor `FAILURE-OUTCOME-001` observed only one 24-decision survival counterfactual in 144 uniformly sampled TRAIN states. This run intentionally mines observable near-failure states using **TRAIN-only** trajectories and four predetermined risk strata, while retaining low-risk controls. It asks whether empirical 16-decision frozen-student continuations provide better labels than straightforward imitation.

The synthetic 96×96 host fixture and its 12-recovery budget are unchanged. At each of 36 → 72 → 128 curriculum stages, each seed creates two adapted 961-parameter policies from its same 90-update frozen recovery model. Adaptations receive the same 48 TRAIN states and 20 Adam updates per stage. The matched control uses the existing imitation labels; the treatment selects a different legal non-STOP first action only when the 16-step student continuation has noninferior survival and goal contacts, plus a strict improvement in survival, goals or painting. STOP is excluded from nontrivial counterfactual candidate pairs, but remains available in the host menu. The target objective and risk strata were fixed before DEVELOPMENT evaluation. No gradient or model weights from DEVELOPMENT are used for training.

**Important boundary:** all results are on a synthetic CyberSand-like host, not its authoritative native ABI. The source fixture permits synthetic `RELOCATE_START` and 128–160 decisions despite the original M3/M4 native 64-logical-move limitation. This is exploratory model research only, not native acceptance.

## TRAIN risk mining

Each seed sampled 48 risk states per stage (144 total) from 16 TRAIN-only world/variant visits and replayed legal candidate continuations. Predeclared tiers include dead ends with both recovery actions, single exit or tight recovery budget, partially restricted choices, and unconstrained controls. The archive contains every selected state's original legal-action prefix, world hash, current menu, frozen-student suffix action series, painted-cell/recovery outcomes and label choices.

| Training measurement | Seed 661 | Seed 662 |
|---|---:|---:|
| Candidate states selected | 144 | 144 |
| States with a survival-distinguishing pair among non-STOP actions | 14 | 14 |
| TRAIN labels changed from imitation | 22 | 24 |
| Strict survival gain over imitation chosen by guard | 0 | 4 |

State enrichment found non-STOP survival distinctions, but almost none could safely improve upon the original imitation label. Not every mined survival-distinguishing pair implies a useful learning signal; some alternatives mainly induce a synthetic relocation and an early policy STOP. This is a negative result regarding the available failure supervision, not proof that future failure mining is futile.

## Matched DEVELOPMENT results

Each seed evaluates 8 known-topology DEVELOPMENT worlds (four families, normal and trap-corridor) at 128 and 160 decisions, giving **16 episodes per arm per seed**, 96 original episodes and 12,789 replayed decisions total. These are unseen DEVELOPMENT seeds 3201, 3301, 3401, 3501, not sealed structural heldout.

| 32 episodes per controller | Frozen | Risk-matched imitation | Risk-mined suffix |
|---|---:|---:|---:|
| Reached instructed goal phases | **73** | 72 | 60 |
| Mean unique painted cells | 131.44 | **136.13** | 122.16 |
| Premature STOP | 6 | **4** | 9 |
| Post-switch painted steps | 715 | **734** | 672 |
| Recovery actions | 88 | 99 | 112 |

Paired 32-episode bootstrap (5,000 fixed-seed resamples) for suffix minus imitation painting: **−13.97 cells**, 95% percentile interval **[−26.91, −2.91]**. Against frozen: −9.28 cells, interval [−20.88, +1.38]. The paired tests are exploratory and not structurally independent across related terrain worlds.

Seed 661 is mixed (suffix 137.31 painted cells vs 139.25 matched imitation, 33 vs 33 goals). Seed 662 is sharply adverse (107 vs 133 cells, 27 vs 39 goals, 8 vs 3 STOP). Treatment controls were source-matched and training budgets identical, but only two seeds were run.

**Interpretation:** targeted sampling substantially increases counterfactual diversity relative to a uniform source sampler, yet the particular guarded teacher fails to outperform matched imitation. The improvement of supervised labels on individual frozen suffixes does not translate into long-horizon autonomous rollouts. The treatment should not be promoted or interpreted as native gameplay competence. A likely contributor is training a model on relatively few preferential forks with only a one-step perturbation and a frozen-student suffix; that mechanism is not independently isolated.

## Verification and provenance

- 6 real `.npz` checkpoints were saved and independently reloaded; they are **not in Git history**. Three 961-parameter arms per seed, 90 frozen base updates and 60 matched adaptation updates per adapted arm.
- Each saved episode stores complete probabilities, requested action, host-legal menu, material edits, pre/post state digests and goal phase. Verification replayed all 96 episodes and recomputed aggregate metrics.
- All 288 TRAIN-state paired records replayed from their original action prefixes, checked current source-world state and legal menu, and re-executed every recorded counterfactual suffix from restored frozen checkpoints.
- Seed 661: 48 episodes, 6,794 decisions, 53 SHA-256-verified evidence files, 29.719 s CPU campaign, 253,828 KiB peak RSS. Seed 662: 48 episodes, 5,995 decisions, 53 evidence files, 30.709 s, 243,372 KiB peak RSS.
- Mean neural decision latency per model arm: 0.20–0.29 ms in this local CPU environment (includes Python host overhead).
- Four focused tests passed, including source-split, non-mutation, legal-continuation and tamper-rejection tests. The source hash is in both original evidence `report.json` files; checkpoints have individual SHA-256 identities.
- No native receipts, no sealed structural heldout opened, no paid inference/GPU, no production game edits.

The original source-matched experiment archive and trajectory comparison PNG were created locally. **Checkpoint binaries are not yet uploaded as immutable Actions artifacts**; they should be retained outside Git until artifact publication is explicitly available. Commit only this script, focused tests and ledger to the existing Draft/HOLD PR #1. Do not merge or mark Ready.

## Next test

A follow-up should create diverse TRAIN-only *actual* recovery-budget exhaustion events with verified surviving alternatives, rather than selecting on generic hazard proxies. This requires a larger variety of viable counterfactual branch fates, followed by a frozen-label comparison and disjoint DEVELOPMENT evaluation. A new adapter or larger SLM is not justified until the short-horizon supervision reliably improves a held-out behavioural metric.
