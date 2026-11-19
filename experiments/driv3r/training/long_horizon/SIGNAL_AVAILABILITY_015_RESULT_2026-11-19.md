# SIGNAL-AVAILABILITY-015 — causal observation availability and stress calibration

**Date:** 19 November 2026. **Exact experimental source:** `5ba2c2b170af661861829e3bea7ecf5a407ebb68` (Git blob) on `04626dac746aec2a1ad4b0e48552e6b66729345a`. Research Draft/HOLD, not CyberSand native acceptance.

## Controlled intervention

Derived from 014's same synthetic game-policy host (legal movement/paint/erase menus, moving gates, material damage, route replanning and chronological receipts). Preserves 36/36, 72/64, 128/64 and **separate** 128/128 shadow horizons. Fits the same 14-feature logistic controller (850 bounded updates) on 70 TRAIN world seeds. A separate 40-seed TRAIN-only calibration set covers independent, 100%-correlated, missing and delayed channel environments. DEV uses 60 previously unseen world seeds, each evaluated at four horizons (240 paired records; NOT 240 independent worlds).

Interventions are a secondary channel permanently absent, secondary messages delayed beyond the decision instant, primary channel outage, and common-mode failure shifts. The baseline dual-channel model's fitted weights and cutoff remain identical for all imperfect DEV modes. These material notifications are HYPOTHETICAL, not official CyberSand ABI observations.

`availability_guard`: if both distinct channel messages have not causally arrived, use always-revisit; otherwise use the original trained cutoff. `robust`: same weights but TRAIN-only risk-first cutoff across all four calibration environments.

## Paired DEVELOPMENT outcomes

| Observation condition | Always goals | Learned goals | Stress cutoff goals | Availability-guard goals | Learned planner calls saved | Guard planner calls saved |
|---|---:|---:|---:|---:|---:|---:|
| independent | 197 | 197 | 191 | 197 | 155 | 119 |
| shift25 | 197 | 195 | 185 | 195 | 160 | 128 |
| shift50 | 197 | 193 | 185 | 193 | 165 | 132 |
| shift100 | 197 | 193 | 185 | 193 | 161 | 132 |
| no_second | 197 | 193 | 185 | 197 | 161 | 0 |
| late_second | 197 | 193 | 185 | 197 | 161 | 0 |
| primary_outage | 197 | 197 | 192 | 197 | 127 | 0 |
| perfect | 197 | 197 | 197 | 197 | 175 | 0 |

The availability guard prevents the missing/late-channel regressions by reverting to unconditional inspection, giving up planner-call savings. It does **not** prevent correlated false negatives when both channels report a benign epoch. The independently trained model also loses goals if notifications are missing.

**Negative result:** the risk-first stress calibration selected threshold `0.7` with `0` avoidable goal losses among 640 TRAIN calibration task records. Despite this, it loses DEV goals in unseen worlds, illustrating insufficient coverage and conditional distribution shift rather than a reliable safety guarantee. Always-revisit remains the conservative baseline.

## Qualification and limitations

- 15360 explicit synthetic DEV policy rollouts, 575020 legal action receipts replayed against fresh same-host instances, and action/menu/movement/material/chronology/corruption checks passed. No independent native or Python host replay.
- Deterministic independent-mode rerun matched all fit parameters, paired negatives, action distributions and outcomes. Previously committed 014 independent control reproduced 197 goals and 1,030 trained planner calls. The late-channel and absent-channel conditions generated identical complete DEV summaries; future pending messages did not leak.
- All evidence remains divided into two append-only JSON ledgers, including per-policy action counts, per-seed/horizon outcomes and action receipt hashes, plus a matched SVG trajectory gallery. FNV32 receipt hashes are equality aids, **not cryptographic provenance**.
- Node v22.16.0 **isolated 14-feature predictor-kernel** latency: p50 114 ns, p95 204 ns, p99 266 ns over 50,000 calls; process RSS 31,477,760 bytes at end and peak 30,740 KiB. **Not an end-to-end Node 015 or native CyberSand latency/RSS measurement**; those remain outstanding.
- No real SLM/LoRA checkpoint, paid GPU/API costs, HELDOUT access, default provider changes or game-side modifications.

## Explicit bounded reproduction

```bash
node experiments/driv3r/training/long_horizon/signal_availability_015.js /absolute/output/directory
```

No optimizer is triggered by import, PR or routine push. Source-smoke CI must not be confused with this scientific replay or native acceptance. Next: test physical/material observables already authorised by the host rather than adding speculative notification channels.
