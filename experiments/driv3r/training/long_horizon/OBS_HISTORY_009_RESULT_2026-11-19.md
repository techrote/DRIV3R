# OBS-HISTORY-009 — sparse/delayed material feedback and counterfactual revisit policy

**19 November 2026 | Draft/HOLD research, opt-in CPU only.** Synthetic CyberSand-*style* spatial/material host, **not the official CyberSand ABI or native game acceptance**. This continues RECEIPT-RECOVERY-008 rather than modifying it. All files belong to `experiments/driv3r/training/long_horizon/**`; sealed structural HELDOUT remains unopened.

## Question and design

When an already-painted objective can be erased remotely but its material event receipt is noisy, stale, delayed, or dropped, can learned history-conditioned policy choose whether to *travel back and verify/repair* more effectively than a simple rule? Does better teacher-label classification transfer into more achieved final goals?

Deterministic 13x9 synthetic worlds include two wall/gap constraints, additional disjoint DEV blockers, protected cells, transient path obstructions, one wrong-material PAINT requiring local repair, plus one possible remote erasure of the first objective. The event occurs after the first material objective, with a 60% erasure prior and probabilistic warning/epoch message channel: 70% warning on erasure, 30% false-warning on no erasure; delay is 0, 3, 7, 10 or dropped. The controller may observe current cell ink, its own action/paint history, legal moves, material messages actually delivered, decision budget and reachable distance; **it may not read latent erasure, seed, the future or pending messages**. The privileged teacher/evaluation oracle is explicitly separated.

Every controller executes a shared, host-replayed paint-and-route prefix. At the final decision it can STOP or revisit the first target, inspect local ink, repair when necessary and then STOP. Utility is `1000 * all_material_goals_completed - 2 * actions - 3 * planner_calls - backwards_moves`. TRAIN-only teacher labels are generated from the counterfactual utility of both *executed* choices on the same world prefix. Two separate 11-weight logistic policies train for 1,600 full-batch CPU steps each: one with prior message/action history and one with only current decision information (history fields masked). No DEV labels enter optimization.

TRAIN: 120 seeds (1100–1219), 480 seed/horizon/censor configurations. DEV: 40 disjoint seeds (3100–3139), 160 configurations. Six paired controllers, **960 evaluated episodes and 47,042 independently re-executed same-source action receipts**. Four conditions: 36/36, 72/64, 128/64 (hard synthetic native64 proxy), and separately labelled 128/128 **shadow-only**. Structural HELDOUT was not sampled, opened, evaluated or trained on.

## DEV results

| Controller | Material goals / 160 | Mean utility | Revisit decisions / 160 |
|---|---:|---:|---:|
| Never revisit | 65 | 294.6250 | 0 |
| Always revisit | **117** | 583.7375 | 160 |
| Revisit on observed warning | 81 | 384.9438 | 43 |
| Trained with history | 104 | 521.4438 | 58 |
| Trained current-only | 105 | 524.3312 | 70 |
| Privileged outcome oracle (upper bound) | **117** | **604.0375** | 52 |

History learner fit TRAIN's counterfactual labels with **79.79%** agreement, versus **74.17%** for the current-only learner. Yet on independent DEV it lost **13 goals and 62.2937 mean utility** relative to always revisiting; the current-only learner lost 12 goals and 59.4063 utility. This is a **negative transfer result**: extra history-label fit did not translate into greater robustness. The warning-only controller lost 36 goals versus always revisiting, demonstrating the cost of dropped and noisy event receipts.

The oracle matches always-revisit goal completion but uses only 52 visits and improves mean utility by 20.3. It proves there is theoretical room to save work if the system learns the *cost-sensitive probability of remote damage*, but the observed features and simple linear fit have not recovered that benefit.

### Horizon/censor separation: achieved goals per 40 matched DEV worlds

| Horizon / action cap | Never | Always | Warning | History | Current | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| 36 / 36 | 17 | 23 | 17 | 18 | 17 | 23 |
| 72 / 64 | 17 | 40 | 25 | 38 | 39 | 40 |
| 128 / 64 proxy | 14 | 14 | 14 | 14 | 14 | 14 |
| 128 / 128 shadow | 17 | 40 | 25 | 34 | 35 | 40 |

The 128-decision synthetic **64-action censor** makes all policies equal at 14/40. The 128-action shadow raises always-revisit to 40/40; these are **not CyberSand native 128-move results**. Success counts are final material configurations, not stage counter estimates.

## Qualification and exact execution

- Source run locally with Python bytecode compilation, explicit bounded CPU optimizer and deterministic DEV campaign. Same-source fresh-host receipt replay, deterministic re-fork equality, independent per-receipt action/position/material algebra checks and tampered illegal-action rejection all passed. No independent native host replay was performed.
- Measured peak Python process RSS approximately **111 MiB** (111,000 KiB range). Exact process RSS, wall time and controller p50/p95 timings are captured in `OBS_HISTORY_009_EVIDENCE_2026-11-19.json`. The learned-decision timing includes Python route-feature computation; it is not an external SLM/adaptor inference benchmark.
- Checkpoint is eleven scalar coefficients per model stored as text with the evidence, not a deployable model binary or immutable Actions checkpoint. No paid GPUs, commercial APIs, model downloads, synthetic HELDOUT inspection or CyberSand runtime mutations.
- The generated evidence carries the *local exact source SHA-256* and complete paired loss metadata with seed/horizon/cap/receipt SHA; the representative SVG traces are synthetic and should not be confused with native PNG sessions.
- Run explicitly: `python experiments/driv3r/training/long_horizon/obs_history_009.py /absolute/output/directory`. No training on import, push or PR.

## Decision

**Retain always-revisit as the stronger conservative synthetic baseline where achievement matters.** The apparent ability to classify training futures is insufficient when false negatives cost an entire painted mission. Next, consider a calibrated belief model with risk-sensitive expected-utility action selection and richer *verified host-observable* local checks; test transfer to independent topology and corruption rates without opening structural HELDOUT. Do not publish a CyberSand-facing provider or rely on the synthetic observation fields without an independently verified host-contract handoff.
