# DRIV3R Corpus Factory v69 — Source-time receipts, stale confirmation and safe reinspection

**19 November 2026.** Corpus Factory DATA lane; synthetic v44 physical-host extension; Draft/HOLD programme PR #3. **Not native CyberSand and not a trained neural-model result.**

## Research question

V68 repaired the privileged physical-state shortcut but depended on idealized verified remote status received after abrasion. Trainer SIGNAL-FRONTIER-012 (19 Nov) showed that hypothetical imperfect feedback can lose completed goals relative to conservative revisiting, motivating explicit scrutiny of observation provenance. V69 tests the different failure mode where a **truthful remote material measurement becomes obsolete before delivery**. A later-sampled receipt may arrive first; naïvely trusting message arrival order can cause mission failure.

## Experimental contract

The host physically paints two distant materials and performs a sacrificial durability coupon, then delivers 112 legal, zero-weight detour movements. There are unverified disturbances at detour steps 8, 21 and 62, physical material snapshots at steps 38 and 72, and possible signed synthetic receipts at steps 80 and 90. For each surface separately, the receipt mode is NONE (no certification), EARLY_DELAYED (old measurement delivered late), LATE (new measurement), or REORDERED (new measurement delivered before a subsequently arriving old one). Six physical shock sequences, two durability thresholds and 16 crossed A/B receipt modes yield **192 causal arms per geometry**.

All arms of a world share the same immediate input at the supervised decision. The prior temporal history and reliability of the evidence determine whether STOP, travel to A, travel to B or reinspection of both is appropriate. **Every scripted probe, paint setup and detour action is excluded from supervised loss.** Later repair actions use only local, legal host observations. Evaluator-only certificates include the real physical state and must never be fed into a model.

Under this deliberately monotone erosion rule, a historically observed ERODED result remains trustworthy until paint repair. A historical INTACT report predating the later unverified vibration is unsafe and triggers reinspection. The teacher's decision depends on report measurements and timestamps, not privileged knowledge of the actual remote paint condition. The entire observation channel and receipt semantics are invented research abstractions, not asserted native CyberSand features.

## Full generation and independent audit

| Split | Disjoint worlds | Successful synthetic teacher episodes | Independently replayed host-legal actions | Arrival-only control lost goals |
|---|---:|---:|---:|---:|
| TRAIN | 4 | 768/768 | 119,816 | 96 |
| DEV | 2 | 384/384 | 58,328 | 48 |
| HELDOUT | 2 | 384/384 | 63,688 | 48 |
| **Total** | **8** | **1,536/1,536** | **241,832** | **192** |

The independent validator imports the underlying v44 physical host but not the v69 generator or its teacher. It reconstructs accepted legal actions, local observations, shocks, measurement times, out-of-order delivery, loss masks and repairs. It audits 8 complete 192-arm groups, 96 physically matched provenance cohorts and 16 indistinguishable all-unverified history groups. Two complete generations were byte-identical. A deliberate certificate change with a correspondingly updated manifest hash was still rejected by the validator.

## Deterministic controls (not neural scores)

- The source-time, conservative teacher succeeded in all 1,536 synthetic cases and saved **4.140625 actions per episode on average** versus always reinspecting both objectives.
- An arrival-order-only interpretation failed to preserve remote goals in **192/1,536 cases (12.5%)**, including **48/384 HELDOUT**; 120 of these failures included a REORDERED arm.
- The physically omniscient upper bound requires **3.994792 fewer actions per episode** than the conservative teacher, on average. That difference represents unavoidable cost where observation certainty is insufficient.
- The STOP-only action is correct in only **16.67%** of these cases. This is a synthetic-host control, not a learned-policy performance estimate.

## Limitations and next consumer experiment

There are only eight distinct world geometries, although the 1,536 causal arms are disjoint across TRAIN/DEV/HELDOUT by world. The 112-movement detour is teacher-forced. A symbolic timestamp ledger remains sufficient; this benchmark does not prove actual SLM temporal reasoning or closed-loop autonomous CyberSand navigation. No target SLM weights, CPU latency/RSS measurements or immutable GitHub Actions corpus artifact were produced in this lane.

Use the existing separately owned trainer lane to train a pinned target SLM on **chronological TRAIN-only v68/v69 streams**, holding DEV and HELDOUT sealed. Compare autonomous source-time memory against arrival-order, history-reset, conservative always-reinspect and privileged evaluator-only upper bounds. Report goal preservation first, then travel/inspection savings conditional on completion, CPU p50/p95 inference, peak RSS and model/checkpoint hashes. Synthetic telemetry must not be presented as native host legality.

**Qualified source SHA-256:** generator `355b7e49727baa99f8af3da4a2d405286195781da41b2cbd32a26e8cb14354e9`; independent auditor `a47dcfa808f15449037632505fee6ab1ac183dc532199863025754e89602d15b`; dataset manifest `a3a33f5c3da33ed4d5e5c65e58f42bd2c3ae2463f9ad7a6a358ba403c40c2fdc`.
