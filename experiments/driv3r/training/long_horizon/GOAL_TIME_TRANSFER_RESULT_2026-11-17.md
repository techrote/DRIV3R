# GOAL-TIME-TRANSFER-001 — outcome labels preserve target contact but do not improve arrival latency

17 November 2026 · `techrote/DRIV3R` · [sole Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1) · **single-seed synthetic DEVELOPMENT negative result**.

## Research claim and model-side experiment

The previous GOAL-UTILITY-001 counterfactual found consistent time-to-target advantages across goal-swapped branches. This run tested **actual supervised adaptation** of a 1,033-parameter NumPy neural policy, rather than another diagnostic. The test compares three checkpoint-restored policies using the exact same host-legal synthetic observation/menu structure:

- `frozen`: the existing contrastive checkpoint, no extra optimization;
- `arrival`: 45 additional genuine Adam updates on TRAIN-only, suffix-outcome time-to-target labels;
- `paint`: 45 additional genuine Adam updates on matched TRAIN states, but using painted-cell advantage labels.

The time labels were accepted only when **both distinct frozen and contrastive suffix controllers** preferred the same first action by at least three actual arrival decisions. Unreached goals received the 97-step capped time in the 96-decision TRAIN continuation. The painting labels likewise required at least three cells' advantage under both suffixes. Both label sources use counterfactual host-legal `BACKTRACK` and side `MOVE` from **identical** initial states; no teacher-privileged futures were supplied to the evaluator.

Only `specs('train',2)` generated TRAIN fixtures; four rotations and two TRAIN-only exit-length pairs `(2,2),(4,3)` supplied 128 distinct goal situations. DEVELOPMENT used seeds `3000,3100,3200,3300`, 4 rotations and two previously unused exit-length pairs `(3,4),(5,2)`. These are **new corridor lengths, not a distinct structural map family**: no claims of architecture-wide generalization. The synthetic 96×96 world and its `RELOCATE_START` do not validate the native 64-logical-move ceiling, native host ABI, or Workbench. All material/protected geometry is authored **before** the 28-step, 11-backtrack source-replayed legal prehistory. At the decision fork both actions remain in the host menu.

## Bounded completed run and exact outcome

The intended two-seed campaign was interrupted on its second seed: only seed **811** reached a finished checkpoint set and complete DEVELOPMENT evaluation. This report does not report a fictitious second result, uninterrupted CPU duration or full multi-seed acceptance. Original seed-811 checkpoints and all 192 original episode transcripts are preserved in the locally verified archive.

| DEVELOPMENT result (64 episodes/arm, 128-decision ceiling) | Frozen | Arrival supervised | Paint supervised |
|---|---:|---:|---:|
| Goals reached | **64/64** | **64/64** | **32/64** |
| Mean capped first-visit decision | **4.00** | **4.00** | 67.00 |
| Mean additional painted cells | 15.375 | 22.156 | **83.484** |
| Episodes ending with STOP | 64 | 64 | 36 |
| Actual sequential decisions | 1,112 | 1,546 | 5,443 |

In all 32 north-goal DEVELOPMENT cases, `frozen` and `arrival` both chose BACKTRACK first and reached in five decisions. In all 32 south-goal cases both chose the rotated side move first and reached in three decisions. **Arrival-labelled training did not improve these already-saturated decisions at all**, although it changed subsequent painting behaviour. It may have reduced premature recovery waste elsewhere, but the experiment does not isolate that.

The painting-control labels were overwhelmingly one-sided: 120/120 robust paint labels chose BACKTRACK, including circumstances where the target was south. The trained painting arm consequently **missed all 32 south targets**, despite painting more cells. In contrast, time supervision was balanced: 64 BACKTRACK and 64 side labels across 128 accepted cases. This independently corroborates that optimizing painting alone can destroy explicit goal responsiveness, but not that the new time adaptation outperforms its frozen teacher.

Importantly, most recorded early STOP events occur **after** target arrival. STOP is offered by the synthetic host; do not interpret post-arrival termination as a premature failure. In these traces the frozen/arrival STOP actions were all on STOP-only menus. The paint arm had seven STOP decisions with another action still legal; two were in failed south-goal episodes. No automatic post-goal STOP was programmed. This task therefore **does not demonstrate sustained 72–128 decision survival**; 128 is a maximum evaluation budget, not the achieved duration.

## Reproducibility, verification and limits

- **Three real NPZ checkpoints** restored successfully: 1,033 parameters each, frozen at 255 cumulative updates; adapted treatments at 300 updates. Checkpoint SHA-256 for frozen `d5bb12e68b876c02a64342e7ca504db17ce63f5880b788adf41f84422dc575b3`, arrival `9b52754d089ea47ec5fa6227a8e4874df714400feb5d551ffba3d15005abaff2`, paint `e432191f21593ded1d1972183db235fb3ebe337a6752325c51a59a190f2eb4f0`.
- **192/192** independent complete host-state/model-probability replays PASS; **8,101 decisions**, original legal menus, material edits, actions, state hashes and complete original probability vectors checked. **196/196** manifest entries pass SHA-256 inspection. Five focused tests PASS.
- A prior 200-second campaign timeout interrupted the attempt before seed 814 completed; no trustworthy continuous campaign walltime or peak process RSS was retained from that process. A subsequent evidence reconstruction and source-checkpoint check took about 14 seconds and 126,220 KiB maximum RSS; a full independent replay took about seven seconds and 98,960 KiB maximum RSS. These resource numbers are specifically **verification/reconstruction**, not original training timings. Mean model probability latency on source-replayed DEVELOPMENT episodes is approximately 0.024–0.029 ms per decision (excludes world transition/serialization).
- Source dependency Git blobs were checked against the exact inspected DRIV3R research head prior to development; all local sources matched. Original model/checkpoint file hashes and TRAIN-only label evidence are in the compact JSON ledger. No new Actions workflow, routine-push training or paid model/API/GPU charge. Checkpoint binaries remain in an original locally SHA-256 verified ZIP, not committed to Git or published as immutable Actions artifacts.
- **Structural HELDOUT untouched**. No native CyberSand / physical Windows acceptance. No DRIV3R main, external repo or canonical `d_common` mutation.

**Conclusion: bounded negative result.** Robust goal-time labels avoid the destructive one-sided painting supervision but do not beat the existing saturated frozen model. A genuinely discriminating next experiment needs new failure-to-exit layouts where the initial goal-conditioned choice is *not* already correct for frozen/contrastive controllers, and at least two independently trained seeds. Freeze these seed-811 DEVELOPMENT worlds instead of tuning them repeatedly.

Only `experiments/driv3r/training/long_horizon/**` may be committed to the existing Draft/HOLD PR #1. No merge or Ready promotion.
