# CONTACT-DENSITY-001 — delayed reward factorial: negative result (2026-11-17)

**Repository:** techrote/DRIV3R · **PR:** #1 Draft/HOLD · **branch:** research/import-consolidated-worm-labs. **Inspected remote HEAD before editing:** aef486030ddc1c7959b06e62a3b4668768fd682a. All nine pre-existing imported source files used by this trainer exactly matched their Git blob SHA-1 at that head; the new trainer's SHA-256 was 206d1ae83d29bed206ad98687c23d1d386d09890f9641142035d4cbd0857b31d. No external repository changes.

## Hypothesis and controlled design

The preceding DELAYED-TARGET-001 collapsed under a twelve-step outcome teacher. Separate sparse target-contact supervision from recovery penalties using a 2×2 factor design: terminal new-goal contact reward 0 or +22, recovery-spend penalty 0 or 1.5. All four configurations reuse **identical** deterministic twelve-step TRAIN counterfactual rollouts, frozen-student-visited states, input features, 961-parameter rankers, initialization and optimizer update counts.

Arms: `proxy_only` = contact 0/recovery 0; `recovery_only` = 0/1.5; `contact_only` = 22/0; `full` = 22/1.5; `balanced_full` = full reward with 50:50 resampling of states in which any candidate can paint the current target. Comparator arms: `frozen` (no adaptation), `imitation` (ordinary label). The balanced arm maintains 64 samples per stage but **reduces unique training states** via replacement, a substantive confound. Sparse contact is tested only through this particular resampling intervention.

Two independent train seeds 641/642; original baseline 180 Adam steps, six adapted arms +60 Adam steps each (20 per stage of 36→72→128). Novel DEVELOPMENT seeds 3059/3159/3259/3359 across four *previously seen structural families*, with normal/trap corridors, fixed goal-instruction seed 681, 128/160-decision autonomous trajectories and switches at 32/72/112. These are **not sealed structural heldout worlds**.

## Autonomous DEVELOPMENT results

Combined 32 episodes per arm (8 development worlds × 2 horizons × 2 training seeds). No teacher continuation controls actions during evaluation. Instructed targets count unique instruction epochs reached during an active instruction, not mere proximity.

| Controller | Targets reached | Mean unique painted cells | Early STOP | Recovery actions | Post-switch painted windows |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 64 | 138.38 | 3/32 | 111 | 737 |
| Imitation | 62 | **139.81** | 3/32 | 80 | 753 |
| Proxy only | 0 | 64.22 | 28/32 | 99 | 287 |
| Recovery only | 13 | 115.69 | 15/32 | 47 | 621 |
| Contact only | 1 | 92.91 | 24/32 | 137 | 469 |
| Full reward | 15 | 109.59 | 18/32 | 66 | 577 |
| Contact-balanced full | 8 | 98.41 | 18/32 | 75 | 482 |

Paired painted-cell comparisons across 32 matched seed/world/horizon tuples, W/T/L:

- Full − imitation: −30.22; 5/2/25; episode-bootstrap 95% interval [−43.25, −18.06].
- Balanced full − full: −11.19; 10/2/20; interval [−25.91, +3.22].
- Contact only − recovery only: −22.78; 6/3/23; interval [−38.66, −6.34].
- Recovery only − proxy only: +51.47; 25/3/4; interval [+35.31, +67.25].

These exploratory bootstrap intervals resample matched episodes (20,000 seeded replicates), not independent families; models share data and only two random initializations were tested. No multiple-comparison correction.

### Training contact prevalence and action distribution

At stage 128, 10/64 TRAIN snapshots for seed 641 and 21/64 for seed 642 had any twelve-step candidate that contacted the new target. Resampling produced exactly 50% contact-eligible examples, but only 35 and 37 distinct states per stage, respectively. Simply duplicating target-contact candidates did **not** restore performance.

Aggregated action counts over the 32 DEVELOPMENT episodes per arm (N/E/S/W, BACKTRACK, RELOCATE_START, STOP):

| Arm | N | E | S | W | BACKTRACK | RELOCATE_START | STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 1150 | 1155 | 1128 | 995 | 94 | 17 | 4 |
| Imitation | 1195 | 966 | 1272 | 1041 | 53 | 27 | 3 |
| Proxy only | 295 | 1106 | 569 | 85 | 19 | 80 | 28 |
| Recovery only | 902 | 1502 | 691 | 607 | 2 | 45 | 15 |
| Contact only | 390 | 1709 | 742 | 132 | 38 | 99 | 25 |
| Full | 889 | 1361 | 662 | 595 | 3 | 63 | 18 |
| Balanced full | 650 | 1536 | 686 | 277 | 2 | 73 | 18 |

Several failed controllers show strong eastward bias, fewer recovery traversals and early exhausted legal options. This is *consistent with*, but does not independently identify, a shortcut policy.

## Verdict

**Negative.** Recovery penalties help relative to an unpenalized proxy, and also improve the contact-reward arm, but neither component recovers imitation's long-horizon performance. Duplicating contact-eligible states at fixed sample/update budget fails. There is no evidence that contact sparsity alone explains delayed-reward collapse. The objective's autonomous distribution shift and the twelve-step teacher's mismatch with durable policy behaviour remain plausible, unisolated causes. These results do **not** warrant promotion to production.

## Exact checkpoints (SHA-256; each 961 weights)

| Arm | Seed 641 | Seed 642 |
|---|---|---|
| Frozen | a4c92566b104df20aca06ded6bfc341acd67340081c7f986f9911f6880f5ecea | a6af5059a72a7d5acc005104e62ea78dc777dd754337210b8487924f04687eb7 |
| Imitation | 06e78f7cba6f8b55f44630abaedcf87d57a7ef68cae1def87e1e6c1242136773 | fe717e77d215bbd5f869cb7533eda8f77ca2cac9d2cad0a225944abbd6b5bbb7 |
| Proxy only | e10d76ac8777b449ab61b2103937995b9642f4c2d8ca08b30b0b8e55cff4fd5d | f73a7e3822764830b12520d3e4540d99bf13b09b980b8ab26707b12f3e688aa7 |
| Recovery only | f034cec5999c5f2cc1c31a5b87cbf107dbc7f5b5c83d9fe6e9cbfef1c299d7e0 | 186f900716c013aee368465e2984b784ea99585d58f259a6fd8d3b3b75d59fa6 |
| Contact only | 40b37e4555fc5bf7817526c35a5c8f17b952e1cd84816e635ffc7a78a5c8bb35 | 563db440d80a89ff7843c4793f226f7623cc06eba575ef66f460efbb4b04340b |
| Full | 06a2c8f6e0d0eda1f0f65559a1e082009e0ed9e89ea4ae3b96f0dc8fe65b8587 | 5016c501537ed6e0acbc998e65038993b364b4311e4328f1137045e463561a61 |
| Balanced full | c9578017ab2ddacf822e6abec7b712d8438dcab8784b75e6275e76d0cd859735 | f30991d941da48bdb6659a85e4980aa96ef3ccd7efb61ec14a9a4946b112ae0e |

## Qualification and limitations

- 14 actual checkpoints generated and restored from NPZ; **224/224** ORIGINAL DEVELOPMENT traces re-executed, **25,014** decisions verified from requested action probabilities, host legality, material transitions and world hashes.
- **149/149 evidence SHA-256 entries per training seed** verified. 5/5 focused source/host/replay/tamper tests passed.
- CPU campaign elapsed 27.428 and 29.137 s. Peak process RSS 128,520 and 131,388 KiB; mean neural decision latency about 0.124–0.129 ms depending on arm, CPU-specific.
- Full local evidence ZIP contains checkpoints, original JSON trajectories/probabilities, PNG gallery originals, counterfactuals, evidence manifests and exact source SHA-256 records. **Immutable GitHub Actions artifact publication is outstanding**; binaries intentionally excluded from Git history.
- The synthetic host is an approximation; CyberSand native action menu, epoch, ACK and its 64-logical-move limit are not validated. Structural heldout untouched, no real game trace/natively played acceptance, no trained d1-2B/LFM/Sprocket adapters, no paid spend.
- Earlier research import and source-smoke CI alone do not constitute model reproducibility.

**Next:** use TRAIN-only student-failure/recovery trajectories to learn longer-term outcome value with frozen DEVELOPMENT fixtures, rather than further tune scalar rewards against these observed environments. Preserve imitation as a non-negotiable control.
