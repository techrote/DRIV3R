# GOAL-TIME-REPLICATION-001 — independent second-seed completion and causal goal-cue control

17 November 2026 · `techrote/DRIV3R` · consolidated [Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1) · **two-seed DEVELOPMENT negative result**.

## Question and experimental controls

The original GOAL-TIME-TRANSFER-001 completed only neural initialization seed 811; the second run was interrupted before its checkpoint and evaluation were complete. This follow-up executes **the unchanged, source-verified GOAL-TIME-TRANSFER-001 trainer** for the missing independent seed **814**, then checks whether its outcome confirms the earlier result. It also tests the causal influence of explicit goal-coordinate features by pairing north and south targets on *exactly the same synthetic state and host-legal menu*, and zeroing only the six goal features in a diagnostic (non-executed) forward pass.

All three restored policies have 1,033 parameters: a frozen prior contrastive ranker; a 45-update TRAIN-only arrival-time adaptation; and an equal-budget 45-update painting-based negative control. The TRAIN arrival labels require agreement between two different suffix policies with a three-step arrival advantage; frozen source weights are unchanged. The 36→72→128 earlier curriculum belongs to the pretrained checkpoint, not a newly rerun curriculum. No native ABI, simulation or game implementation was modified.

Source identity was compared to **exact Git blobs** at `9372e1ccfcd91926f9eafa0aaab79a829cdc25b4`: `goal_time_transfer_lab.py` `122ecdef41793e7a9bb26d9f8abddb60698e60e3`, `goal_time_finish_lab.py` `63c4ad71e6516c5cfca7265ee111e7ac45acb4ac`, `goal_conditioning_lab.py` `243294176699abc56386eaf12b95580843499b55`, `recovery_debt_audit.py` `8d75bfb4ba2dbbf4e00bac9d2da7a4749621be68`, `sequential_probe_lab.py` `90911809af0cf2ae332fbc74441f598410d0dc9d`, `world.py` `c7e33b7801c3a9d6c1f00f11f466732b312ddc92`, and `policy.py` `c61573bdd4dc603ef38594b5b764fec1ff2142ab`. All checked equal to the executed local copies. The source OUTCOME-FORK ZIP SHA-256 is `bb0be90bb1cceab1b42496a5cac66f39d7b2a881de25228fba1a87a7a642ace1`.

## Complete DEVELOPMENT outcome

64 matched episodes per controller **per independent model seed**: four terrain families × four rotations × two previously untrained corridor-length pairs × north/south goal. Each episode has a 128-decision ceiling. The intentionally unchanged DEVELOPMENT configurations are shared across seeds; **this is independent optimizer-seed replication, not structural generalization**.

| Model | Seed 811 target visits | Seed 814 target visits | Pooled visits | Pooled capped mean arrival | Pooled mean newly painted cells |
|---|---:|---:|---:|---:|---:|
| Frozen | 64/64 | 64/64 | **128/128** | **4.00** | 32.98 |
| Arrival-trained | 64/64 | 64/64 | **128/128** | **4.00** | 41.63 |
| Paint-trained | 32/64 | 32/64 | **64/128** | 67.00 | **105.02** |

All 64 failed paint-trained goal visits were **south targets**. The painting controller overwhelmingly learned BACKTRACK (all 120 of its robust TRAIN labels favoured it), then missed the incompatible southern exit while painting substantially more elsewhere. Arrival labels were exactly balanced at 64 BACKTRACK and 64 side movement in the 128 TRAIN examples. The frozen model was already at the ceiling for target visits and first-contact timing, so **the arrival-trained policy did not outperform the baseline**. Additional painting after goal contact is not evidence of superior goal navigation.

The seed-814 training and evaluation campaign completed in **32.839 seconds of CPU wall time**, with **131,720 KiB peak RSS**. Seed 811's original uninterrupted campaign wall time remains unknown and is not retroactively invented. Seed-814 per-decision model probability inference was approximately **0.030–0.034 ms**, excluding world transitions and serialization. All model actions were checked against the same synthetic host-owned legal menus.

## Causal observation ablation

For each of 32 DEVELOPMENT fork states and both model seeds, the north and south goals were swapped without changing terrain, materials, protected cells, ink history, recovery budget, world-state SHA, or legal-action menu. The only changed input channels were the **six explicit host-observable goal-coordinate features**. Their zeroed versions must produce exactly identical probability vectors for both targets; this invariant was checked for every pair.

| Policy | Correct first actions (128 target cases) | First-action switches on goal swap (64 state pairs) | Switches with goal cues masked |
|---|---:|---:|---:|
| Frozen | **128/128** | **64/64** | 0/64 |
| Arrival-trained | **128/128** | **64/64** | 0/64 |
| Paint-trained | 64/128 | **0/64** | 0/64 |

The frozen and arrival-trained policies **genuinely respond to the explicit target cue at the first decision**; painting-only adaptation abolishes that response. The masked comparison is a *prediction-level diagnostic*, **not** a host rollout or evidence that the masked controller necessarily fails an entire episode. The original unmasked first-action predictions were matched to the first actions in the 384 actual source-replayed trajectories.

## Qualification, evidence, and limitations

- Original seed-811 evidence: **192/192 episodes, 8,101 decisions**, its original **196-file SHA-256 manifest** and checkpoint identities independently verified.
- Newly executed seed-814 evidence: **192/192 episodes, 15,534 decisions**, its independent **196-file manifest**, all neural probability vectors, host-legal action receipts, material receipts, final states and three newly trained/restored checkpoint identities verified.
- Combined: **384/384 episodes, 23,635 actual decisions, six restored checkpoint identities**. Nine local focused source-contract tests passed across original and new test files. No sealed structural HELDOUT opened.
- Six-checkpoint SHA-256 identities, manifest hashes, exact source identity and a compact causal-control ledger are in `GOAL_TIME_REPLICATION_EVIDENCE_2026-11-17.json`. The full new causal paired-probability transcript has SHA-256 `6b1228f63d8b07817702e6b42dbd449e7752a49e3e45cfce8a3084097a97375b` in the retained local archive.
- `GOAL_TIME_REPLICATION_TRAJECTORIES_2026-11-17.svg` plots six **original** movement/painting paths, with targets and final cursor positions; panels use different scales, so lengths must not be compared by eye.

**Not qualified:** model performance on structurally different exit/failure maps; early recovery choice when the frozen model is wrong; autonomous goal planning across 128 useful sequential decisions; native CyberSand execution or its physical limits. Actual goals were normally reached in only three or five decisions after the scripted legal prefix. The simulated `RELOCATE_START` and 128-decision bound are not proof of native actor acceptance. No pretrained d1, LFM2.5 or Sprocket adaptation was attempted.

**Artifact custody limitation:** Original complete seed-811 and seed-814 checkpoints and per-decision trajectories were retained as locally hashed archives. The current connected GitHub actions do not provide an artifact-upload endpoint; they have **not** been published as immutable Actions artifacts. Checkpoint binaries are intentionally excluded from Git history. Compact source/verification and negative-result evidence belong only in this existing Draft/HOLD PR; there must be no push/PR creation in external game or neural repositories.

**Research conclusion:** This is a repeated, causally supported **negative result for incremental goal-time adaptation** on already saturated synthetic fork fixtures, plus a repeated **negative control for painting-only supervision**. The next model-side experiment should sample new TRAIN-only recoverable failure-to-exit topologies where the frozen controller fails, compare at least two seeds on unchanged DEVELOPMENT topologies, and measure goal completion and latency independently from painting volume. Structural HELDOUT remains sealed until a candidate is frozen.
