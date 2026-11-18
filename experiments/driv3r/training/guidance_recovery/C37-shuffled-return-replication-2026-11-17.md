# C37 — Neural return-head replication against label-shuffled controls

**17 November 2026 · DRIV3R Guidance & Recovery · REJECT neural promotion · synthetic-host research only**

## Question and controls

Does the C36 neural return head improve *executed* recovery over both a fixed goal-rearming controller and a same-architecture neural head trained on randomly permuted labels? Three independent fresh-seed evaluation cohorts contain 96 distinct maze graphs with no graph-hash overlap; the braid family is completely absent from training. Each cohort includes tree/braid topology, detour/forward goal cases, optional second goal change, two worm personality signs, weak/strong guidance and no-guidance, with host legality verified on every action. The decision horizon is 60; goal counts are **goal events**, not independent Bernoulli trials.

C37 trains two nine-input return heads on the same 320 examples derived from host-executed counterfactual continuations. LEARNED uses correctly matched labels; SHUFFLED receives the identical observations and a permuted target vector. Both start from **identical random weights**, receive **100 CPU AdamW updates** with identical mini-batch orders, optimizer settings and training budgets. RANDOM retains those untrained weights. C34-RESET (fixed controller), C37-STYLE (fixed controller with within-state movement style projection) and frozen C18-A0 complete the six matched arms. Every action is sampled stochastically from an identical legal menu; no topology oracle is used at inference.

## Training and replicated checkpoint measurements

| Head | In-sample MAE against actual targets | Independent checkpoint replicas |
|---|---:|---|
| LEARNED | 0.6104 | byte-identical |
| SHUFFLED | 0.6633 | byte-identical |
| RANDOM | 0.8497 | byte-identical untrained control |

Training-target fit is not evidence of better navigation. All training input fields are host-observed; labels derive solely from legal offline counterfactual suffix execution. Checkpoint SHA-256 digests and training dataset/receipt hashes are retained in `C37-compact-results.json` and `C37-evidence-receipt.json`; the checkpoint binaries themselves are **not** committed to Git.

## Held-out goal events

| Policy | A | B | C | Total |
|---|---:|---:|---:|---:|
| C18-A0 | 40 | 46 | 34 | 120 |
| C34-RESET | 46 | 49 | 41 | 136 |
| C37-STYLE | 46 | 49 | 40 | 135 |
| C37-LEARNED | 49 | 50 | 40 | 139 |
| C37-SHUFFLED | 49 | 51 | 41 | 141 |
| C37-RANDOM | 46 | 48 | 40 | 134 |

There are **2,880 source-matched evaluated episodes** (2,304 guided and 576 uncued), comprising 384 guided episodes per policy. LEARNED gains +3 completed goal events over C34-RESET and +5 over RANDOM, but completes **two fewer** than the label-shuffled control. On the three independent cohorts, LEARNED minus C34-RESET is +3, +1, −1, while LEARNED minus SHUFFLED is 0, −1, −1. The learned head changed actual guided action hashes in 147/384 episodes relative to C34-RESET and 127/384 relative to SHUFFLED, so its failure is **not** simply an inactive model.

Condition-level nonregression against C34-RESET fails in **2/3/4 groups** across cohorts A/B/C for LEARNED (9 regressions in total). The label-shuffled arm also produces 1/3/2 condition-group regressions. This strongly argues against treating the limited raw aggregate improvement as learned transferable recovery skill.

### Paired held-out graph-cluster bootstrap

Resampling 96 graph hashes, retaining paired policy results within each graph (10,000 deterministic draws), produces descriptive 95% event-difference intervals:

- LEARNED minus C34-RESET: +3 events; [-7, +13], graph-level wins/ties/losses 13/74/9.
- LEARNED minus C37-SHUFFLED: -2 events; [-8, +4], graph-level wins/ties/losses 4/86/6.
- LEARNED minus C37-RANDOM: +5 events; [-8, +18], graph-level wins/ties/losses 15/71/10.

None excludes zero. These are paired cluster-resampling descriptions and do not establish statistical significance or cover all dependence between deliberately matched interventions.

## Personality and safety

All **32/32 no-guidance action and painting hashes per policy per cohort** match frozen C18-A0. None of 2,880 episodes resulted in a host-rejected executed action or a protected-material modification. Some unmasked policy top choices were illegal before the synthetic host legal-menu mask; they were not executed, and this is not a native-game safety proof.

The active-guidance style projector bounded within-state directional preference divergence. However on-policy trajectories vary substantially by head because different states are visited, so no whole-trajectory personality preservation claim is justified. Both training labels and tests are CyberSand-*like* synthetic fixtures, not an acceptance of CyberSand's real host-legal ABI.

## Replay, source identity, research disposition

Four targeted regression tests passed. Sixteen representative paths were independently replayed through legal synthetic-host menus, reproducing all action/ink hashes. The gallery includes identical starting geometry with C34-RESET, LEARNED, SHUFFLED and RANDOM. Full cohort traces, optimizer replicas, dependency snapshots and an SHA-256 manifest are retained in the external local evidence archive. Its model binaries are deliberately excluded from Git history. **Immutable GitHub Actions checkpoint custody remains outstanding.**

The C34/C36 and foundational dependency Git blob identities were checked against the exact DRIV3R research branch before training. No game files, external repository or production defaults were modified. Only this lab's new C37 files should be added to existing Draft/HOLD PR #1.

**Decision: REJECT C37-LEARNED for promotion.** A lower training loss and 3 more aggregate goal events are insufficient when the label-shuffled control does better and strict condition-level nonregression fails. A next experiment needs more independent graph-generator clusters, calibrated uncertainty and a whole-trajectory personality gate rather than another in-sample training-loss target. No merge/Ready promotion.
