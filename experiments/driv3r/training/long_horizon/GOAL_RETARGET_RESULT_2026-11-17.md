# GOAL-RETARGET-001 — surprise-goal retargeting after recovery-aware training

**17 November 2026 · techrote/DRIV3R PR #1 · synthetic DEVELOPMENT only · NOT a native gameplay result**

## Falsifiable question

Does the GOAL-RECOVERY-001 conditioning improvement persist when an existing tiny
neural worm is given a new spatial goal **without retraining**, after 32, 72 and 112
sequential decisions, on different DEVELOPMENT seeds? Is the improvement actually
associated with using the live recovery features?

## Protocol and isolation

- Trained two matched **961-parameter** nonlinear rankers (`goal_only`,
  `goal_recovery`) per seed (617, 618), from identical initial random weights,
  identical TRAIN-only examples/labels and 60 genuine Adam steps at each
  36→72→128 curriculum stage (**180 optimizer updates/checkpoint**).
- `goal_only` receives eight recovery features masked to zero. Both models see
  explicit goals and identical host-legal action menus. A third *evaluation-only*
  `recovery_masked` control uses the trained recovery model's exact weights but
  zeros those eight features; it is NOT a third trained checkpoint.
- Evaluated **unseen DEVELOPMENT seeds 3001/3101/3201/3301**, four familiar
  synthetic world families (`pickets`, `ridges`, `islands`, `zigzag`), with
  normal and `trap_corridor` fixtures, at 128 and 160 decisions. Worlds are
  native-derived synthetic only; the sealed structural families
  (`archipelago`, `staggered_vault`) remain untouched.
- Identical precomputed target instructions for all arms. `static` uses one
  target throughout; `retarget` changes the active target precisely at steps
  32/72/112. Target schedules are generated from the pre-rollout world and
  never from the arm's subsequent trajectory; no instruction leakage.
- Four repeatable host-legal forced deviations at steps 19/57/91/131 when
  alternatives exist. Every unforced step is model-chosen, not teacher-driven.
  No additional free MOVE or illegal recovery action is injected.
- Original source-replayed `World` state transitions include material/protected
  exclusion and non-native synthetic `RELOCATE_START`, which is **not**
  evidence that CyberSand permits native teleportation or >64 logical moves.
- The first eight actions after each retarget are evaluated by net Manhattan
  progress toward the new instruction; exact targets painted during an active
  epoch are measured separately. This is a directional responsiveness measure,
  not a guarantee of eventual path completion.

## Results — 192 episodes, 27,391 real sequential decisions

| Condition | Arm | Episodes | Mean painted cells | Early STOP | Targets painted while instructed | Mean first-8-step distance gain after switch |
|---|---|---:|---:|---:|---:|---:|
| Static | Goal-only | 32 | 140.406 | 1 | 32 | — |
| Static | Goal + recovery | 32 | **142.313** | **0** | 32 | — |
| Static | Trained recovery, features masked | 32 | 136.063 | 3 | 32 | — |
| Retarget | Goal-only | 32 | 140.406 | 1 | 81 | 3.396 |
| Retarget | Goal + recovery | 32 | **141.000** | **0** | **82** | **3.646** |
| Retarget | Trained recovery, features masked | 32 | 139.031 | 2 | 76 | 3.458 |

Matched pairs (goal + recovery minus goal-only): static painted-cell difference
+1.906/episode (4 wins, 21 ties, 7 losses; percentile paired bootstrap 95% CI
approximately [-0.844, +6.688]); retarget difference +0.594/episode
(8 wins, 12 ties, 12 losses; paired bootstrap 95% CI approximately
[-0.844, +2.188]). Across 32 retarget pairs, the recovery model paints
**one more active instructed target**, with switch-window directional gain
higher by 0.250 cells per eight-step window on average.

**Interpretation: small, inconsistent effect, not evidence for transfer.**
The masked recovery model degrades, confirming the trained model sometimes
uses recovery inputs, but the apparent advantage over goal-only is close to
zero across seed/world conditions and the effect is not robust under paired
bootstrap resampling. Static and retarget conditions use different target
regimes and should not be compared as though they had equal intrinsic difficulty.

All observed STOP actions are host-legal. No model can certify the real native
CyberSand 64-move contract from this synthetic engine. The high `branch_cells`
measurement counts graph degree >=3 on connected ink, not aesthetic novelty or
motif quality, which remains to be assessed independently. No paid API/GPU,
no real-world game writes, no sealed heldout, and no larger-model adapter.

## Independent replay, resource cost, and artifact provenance

| Campaign | Episodes/decisions replayed | Checkpoints | Training+evaluation CPU walltime | Peak RSS |
|---|---:|---:|---:|---:|
| seed 617 | 96 / 13,712 | 2 | 18.603 s | 193156 KiB |
| seed 618 | 96 / 13,679 | 2 | 18.323 s | 196752 KiB |
| **Total** | **192 / 27,391** | **4** | **36.926 s** | per-process measured |

Each archived run has **123 SHA-256-manifested files**, and fresh restored
checkpoints reproduce the original per-action probability dictionaries and all
27,391 host-legal transitions. The experiment has five focused tests,
including original replay / tamper rejection / sealed-family exclusion.
Model-action latency is recorded separately per arm and run in `report.json`;
roughly 0.11–0.14 ms in this CPU environment, not a native frame cost.

SHA-256 of exact trained checkpoints:

| Seed | Arm | SHA-256 |
|---|---|---|
| 617 | goal_only | `0a2405d70089495256073368f946be02c88d93ac8400536e388e234b8b40a4a8` |
| 617 | goal_recovery | `839ef6a448e09d5aa353b05efa83b0b4cd85c4e9d1739df8667ac4208c9f74c8` |
| 618 | goal_only | `6e7b698557a487b0510df65c1408b3702d188b5cb6c55b8c5234973a02788f89` |
| 618 | goal_recovery | `5a3a32a13a89fc9e6dbf9cc0630f1a75fc09dde408080485e9951bfebcc94de8` |

Exact source SHA-256:
```
{
  "goal_conditioning_lab.py": "6e00bfdfbaecc8195a86cb626bbbcbb65184a449adc5b7272d5642553cf02659",
  "goal_recovery_lab.py": "7a8bf710a644b2a42c7e3fb032036f7aa246665e995683891ce9db5fc3caa290",
  "goal_retarget_lab.py": "88ba7013f3bc8eb1ec8f728c5d704aeeb6aafd4ebc6cf964f534dc14e7f8b14b",
  "policy.py": "f3e6469b592e68c0d3410dca88e107b215f1bdb1c16f7ebd73232ea9470cbb82",
  "train.py": "9ed2c9ac49fa397c12f0b8256e4d4439bd3a9294b002b54276b702a1e3ddcacb",
  "world.py": "6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a"
}
```

The local checkpoint/trajectory bundles are **not yet immutable GitHub Actions
artifacts**. They must not be represented as CI-qualified or native-accepted.
Only text source, tests and this negative/mixed-result ledger belong in Git.

## Next discriminating experiment

Make objective-switch response the optimized behaviour rather than merely an
out-of-distribution probe: introduce TRAIN-only randomized mid-rollout goals,
train with a matched on-policy goal-retarget curriculum, and keep this exact
DEVELOPMENT fixture unseen. Independently score post-switch regret, motif
continuity, and survival; do not modify CyberSand's authoritative ABI.
