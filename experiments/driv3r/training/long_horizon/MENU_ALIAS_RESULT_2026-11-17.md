# MENU-ALIAS-001 — Equal-menu risk discrimination and observation aliasing

17 November 2026 · DRIV3R long-horizon laboratory · **Negative transfer / verified information bottleneck**

## Research question and finding

The predecessor CONTRASTIVE-FORK-001 separated 32/32 trapped from 32/32 open first decisions, but the cases could advertise different offered-action menus. This experiment holds both menu composition and cardinality exactly fixed, comparing a *visible near blockage* and a *distant hidden blockage* with an otherwise viable forward route.

**Result:** The visible-contrastive policy chose the correct escape on **3/16 visible near traps**, **0/16 distant traps**, but correctly advanced on **16/16 open controls**. Universal-retreat controls escaped **16/16** in both trap types but unnecessarily retreated on **16/16** open controls. The predecessor's perfect first-choice discrimination does not transfer to menu-matched conditions.

The far trap and open route have **bit-for-bit identical entire neural feature matrices**, identical first legal menus, goal and history features, but different distant static material and opposite viable decisions. Across 16 DEVELOPMENT pairs, the first action and full probability distribution matched for all four controllers (**64/64 policy comparisons**). No deterministic memoryless policy receiving only these observations can make both conflicting first decisions correctly. This proves a *representation-level information bottleneck* under the tested observation interface; not a limit on all possible policies.

## Exact source, ownership, and evidence boundary

- The sole writable repository is techrote/DRIV3R, branch research/import-consolidated-worm-labs, existing Draft/HOLD PR #1. Starting exact remote HEAD inspected: ee5a6eccb606e2908d11a9e5cd88fe24fbedb20d. That commit changed another worker's corpus_factory laboratory and is untouched.
- Three new files, all in experiments/driv3r/training/long_horizon/: menu_alias_lab.py, test_menu_alias_lab.py and this report.
- Source SHA-256 of the *actually executed* menu_alias_lab.py: ab812b1f21d3de6aa3a51562b82cb2f76f8a67dc15c70d74401dc4a90edbf739. The eight unmodified dependency sources were individually Git-blob matched to the starting remote HEAD.
- The synthetic one-cell World and World.apply enforce offered actions. Terrain edits happen to **initial static materials/protected cells before** replaying the same legal recovery prefix; no runtime cursor, ink, path, recovery counter or offered menu is assigned. This is **not** a native CyberSand ABI, real Workbench, or physical-acceptance test. No writes to CyberSand, Neurounner, d_common or any other laboratory.
- No structural HELDOUT data opened. DEVELOPMENT fixture family seeds and corridor grammar were used in earlier research; this result is not blind structural generalization.

## Controlled design

Training seeds **703** and **705** were selected using only TRAIN-world viability qualification, before reviewing model outcomes. Attempted seed 701 failed a TRAIN corridor-witness check and was excluded before evaluation. Each model has **961 parameters**. Frozen baseline receives 54 Adam updates (18 per stage); three adapted arms each receive another 96 updates (32 per stage), total 150. All adapted arms see **exactly the same samples and legal masks** and differ only in supervision labels.

At each 36 → 72 → 128 curriculum stage, sample 64 natural TRAIN states and 32 paired near-trap/open TRAIN states from four families and four cardinal orientations. The 16 family/rotation pairs recur across the stages; 96 stage-labelled cases per seed is **not** 96 unique structural situations. Training never shows distant-trap examples. DEVELOPMENT evaluation uses four established families and two predeclared orthogonal orientations (0°, 90°), 3 scene types × 4 policies × 8 configurations × 2 model seeds = **192 source-replayed 128-budget episodes**.

The fork starts after **12 actual synthetic recovery operations**. All three scene types offer exactly the same first three legal actions (goal-forward MOVE, reverse MOVE and STOP). The near trap adds SOLID at (53,48), the distant trap keeps original terminal obstacle at (55,48), and the open control removes only that terminal material/protection. A source-replayed 24-step non-neural viability witness confirms near/far forward forced STOP after 3/5 moves, retreat survival ≥20 moves and open forward survival ≥20 moves. The witness is a geometric certificate, **not autonomous model performance**.

## DEVELOPMENT outcomes — 16 episodes per controller per scene

| Controller | Near trap: correct first | Far trap: correct first | Open: correct first | Mean new paint near / far / open |
|---|---:|---:|---:|---|
| Frozen | 0/16 | 0/16 | 16/16 | 3.00 / 5.00 / 97.75 |
| Goal-forward imitation | 0/16 | 0/16 | 16/16 | 3.00 / 5.00 / 94.19 |
| Universal retreat | 16/16 | 16/16 | 0/16 | 97.13 / 97.13 / 98.63 |
| **Visible contrastive** | **3/16** | **0/16** | **16/16** | **26.44 / 5.00 / 94.06** |

Visible-contrastive improved near-trap decisions only for seed 703 (3/8); seed 705 remained at 0/8. Near versus open current-feature vectors differ and so are, in principle, separable, but the small optimizer did not learn reliable escape under this exact equal-menu curriculum. Far versus open vectors are exactly the same, so neither this model nor a larger deterministic function of the same first-step observation can distinguish them. The open label means goal-forward, whereas retreat could also survive; first-choice discrimination is not proof of long-term objective attainment.

All **192** original autonomous episodes were independently replayed from the restored checkpoints, totaling **10,159 model decisions**. All 138 selected STOP actions happened only when STOP was the sole legal menu choice; early termination was downstream from previous moves, not voluntary STOP preference.

## Training, resource use, verification

| Seed | CPU wall for campaign | Peak RSS | Verified episodes | Replayed decisions | SHA-256 manifest entries |
|---|---:|---:|---:|---:|---:|
| 703 | 23.704 s | 124,224 KiB | 96/96 | 5,407 | 127 |
| 705 | 22.270 s | 123,540 KiB | 96/96 | 4,752 | 127 |

Local neural decision latency is approximately 0.10–0.14 ms, hardware-specific and not comparable to larger adapters. **4/4 focused tests** passed, including tampered-evidence rejection. All eight genuine checkpoints were restored; original neural probabilities, full legal menus, post-action material and state hashes, training-case witnesses and both SHA-256 manifests were verified. Combined local evidence archive: **258 entries**, ZIP CRC pass, SHA-256 **673bc4e1026b91139f63471258a2208e4a9bcc58aeacedbca00a25ab18322ef2**.

### Original checkpoint SHA-256 identities

| Seed | Policy | SHA-256 |
|---|---|---|
| 703 | frozen | fab8d2c2e8e0152ca4ed967e47ac401eca213aa4db47a7f045711437a8e1985c |
| 703 | goal_forward | 7d5c2d973443b0510592c07878a1fca4a82dca5ffc9b933c1881e01dcbbf8360 |
| 703 | universal_retreat | 0bbb5cfb661c958526ecd5f26fffba36ac486588cad59df6791f0588b7e8dc46 |
| 703 | visible_contrastive | f642610e060bfc5a2055a08edce38f55bd95c175eed42c1f091da4a207ed0a14 |
| 705 | frozen | b4ab018ef60451cbb0a3de3a12b76ee1f702d8a9deecaf538d3a9890af8be87a |
| 705 | goal_forward | 97953159fec8a7fbd3f78e0c5f9b611230b7a7d6725aceec8bb8ce4c65989297 |
| 705 | universal_retreat | e467bc848bb3c839f291b5ead9e4bf72b7cebf40f08ec89f918be00483c9fd55 |
| 705 | visible_contrastive | 6ccefee812915a9d9da220ed90a4de23c51fe84aada634f82217759809f24576 |

## Limitations and next decisive control

The experiment *intentionally* constructs a strict observational alias: this is an impossibility demonstration for deterministic **first-decision** policies under the current feature map, not evidence that additional perception, memory, active probing or native host observations cannot distinguish the cases. These are three material variations of a constructed corridor, not three independent topological families, and the geometric continuation witness is not native physical ground truth. The recovery count and horizon are beyond the old native reference semantics.

Next, test a **research-only observation-sufficiency** candidate: genuine bounded farther-forward occupancy sensor or a feasible sequential probe/reassessment policy, compared against the unchanged feature map with legal menus and equal compute. Evaluate false retreat, distal hazard decisions and post-fork continuation on different DEVELOPMENT layouts before opening any sealed HELDOUT. Native-side contract changes would require a separate read-only owner proposal; never write to CyberSand.

**Remaining:** original checkpoints and traces are local ZIP evidence, **not immutable GitHub Actions artifacts**; no native receipts, sealed HELDOUT or larger-model adapters. No paid compute, experiment-specific workflow, other branch, PR, merge, Ready promotion or foreign-repo mutation.
