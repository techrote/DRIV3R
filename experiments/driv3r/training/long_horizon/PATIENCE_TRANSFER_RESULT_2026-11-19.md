# PATIENCE-TRANSFER-005 — unstable material reopening and shifted obstacle topology

**19 November 2026 · DEVELOPMENT-only, source-replayed synthetic game-policy stress. Draft/HOLD.**

## Scientific question

OBSERVED-GATE-DEFER-004 selected immediate route-overlap invalidation at 36 decisions and three-decision patience at 72/128; its DEVELOPMENT gain was only +1.583 utility over overlap under fixed gate reopening and topology. This new run freezes the TRAIN set and policy-selection procedure, but shifts the **DEVELOPMENT** lattice and observed material-event timing: barriers at y=6/10 rather than 7/9, alternate gap row cycling [2,5,11,14] rather than [3,4,12,13], and **early** reopening at tick 6, 8, 11 or 15 rather than always 8. Late gate reopening stays at the terminal boundary. The controller gets only current observed events, not future schedules; no structural HELDOUT is opened. This is an explicit synthetic out-of-distribution DEVELOPMENT stress test, not sealed HELDOUT or native acceptance.

The program transforms two pinned Git blob sources with unique-match checked replacements. The original TRAIN seeds 1000–1047, Adam/hand-off training (2×360 optimizer updates), original policies, utility and world menu, independent action/state receipt replay and 64-paint censor are preserved. Shifted DEVELOPMENT seeds 3000–3015, horizons 36/72/128 and the separately labelled shadow128 cohort are unchanged in quantity. No new SLM/LoRA checkpoint is trained or claimed.

## Result — previous patience selection fails to transfer

| Controller | Previous goals /192 | Shifted goals /192 | Change | Previous utility | Shifted utility | Shifted BFS calls |
|---|---:|---:|---:|---:|---:|---:|
| lazy | 96 | 90 | -6 | 358.833 | 341.542 | 264 |
| overlap | 144 | 140 | -4 | 623.094 | 603.302 | 320 |
| epoch | 144 | 140 | -4 | 622.094 | 602.302 | 384 |
| defer1 | 140 | 129 | -11 | 599.302 | 546.469 | 304 |
| defer3 | 128 | 115 | -13 | 542.750 | 473.042 | 288 |
| train_selected | 144 | 128 | -16 | 624.677 | 540.479 | 296 |

**Key negative result:** the previously TRAIN-selected policy falls from **144/192 to 128/192 goals**, whereas immediate route-overlap moves from **144/192 to 140/192**. Shifted mean utility favours overlap by **62.823** rather than the earlier +1.583 selection advantage. There are **12 cases where route-overlap arrives and selected patience does not**, and **0 converse cases**. Full original negative evidence remains in the predecessor; new case-specific action distributions and sample receipt trajectories are retained below/in JSON.

### By horizon, mode and censor

| Horizon | Mode | Censor | Overlap goals | Selected goals |
|---|---|---|---:|---:|
| 36 | early | native64_proxy | 14/16 | 14/16 |
| 72 | early | native64_proxy | 16/16 | 12/16 |
| 128 | early | native64_proxy | 0/16 | 0/16 |
| 128 | early | shadow128 | 16/16 | 16/16 |
| 36 | late | native64_proxy | 14/16 | 14/16 |
| 72 | late | native64_proxy | 16/16 | 8/16 |
| 128 | late | native64_proxy | 0/16 | 0/16 |
| 128 | late | shadow128 | 16/16 | 16/16 |
| 36 | irrelevant | native64_proxy | 16/16 | 16/16 |
| 72 | irrelevant | native64_proxy | 16/16 | 16/16 |
| 128 | irrelevant | native64_proxy | 0/16 | 0/16 |
| 128 | irrelevant | shadow128 | 16/16 | 16/16 |

Every 128-decision native64-proxy outcome remains explicitly censored at 64 paint moves; shadow128 is synthetic extrapolation. Do not count its completions as native support.

## Qualification and reproducibility

- **2880** DEVELOPMENT rollouts, **175014** actions independently reapplied through the local synthetic state/host menu, **16 internal assertions PASS** including seeded optimizer determinism, legality, event replay and receipt-tamper rejection.
- Original unchanged upstream JavaScript blob: `4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873`; parent transformation blob: `b24321ece46e839ec29458922a0031476c0700fe`; prior aggregate evidence blob: `dd1abb1d8b7c2507c0daec823e81981933caf5e5`.
- In-tool V8 full scientific run elapsed **107128 ms**; this is whole-study wall clock, **not** per-decision inference latency. V8 did not expose process RSS or native game timing. The published Node CLI emits wall clock, RSS and maxRSS when explicitly executed on a machine with Node.
- Run explicitly: `node experiments/driv3r/training/long_horizon/patience_transfer_lab.js /tmp/driv3r-patience-transfer-005`. No new workflows or automatic training hooks were created.
- Synthetic JavaScript parent semantics are not an official CyberSand host ABI, not independent Python cross-implementation replay, and not real native CyberSand gameplay. Native receipts: **0**. Structural HELDOUT opened: **no**. Paid resources: **none**.

## Decision

**Reject three-decision patience as a reliable default** on current evidence. The small earlier gain is highly topology/reopening sensitive. A conservative route-intersection policy has better goal preservation on shifted material dynamics, even when it uses additional planner calls. The next substantive research step is online adaptation using observable reopening/legality history, and at least one actual retained trained checkpoint with quantified CPU inference latency/RSS; do not add privileged knowledge of the future to the action policy.
