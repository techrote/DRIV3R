# SIGNAL-CORRELATION-014 — recovered notification provenance and common-mode transfer

**Date:** 19 November 2026  
**Parent programme HEAD:** `1677482f451d106c1cb6598eb2a08da5f5cf3354`  
**Source git blob:** `3bc548bcd43cd52b9d24f7f4ad0c59a78e48509b` (derived opt-in experiment).  
**Qualification:** CPU V8 synthetic execution, same-host replay against fresh instances and separate receipt-algebra checks. Not native game acceptance.

## Why this experiment was necessary

In `dual_channel_013.js`, the synthetic host queues `{channel:"primary"|"secondary"}` but delivered messages were constructed without the `channel` property. Consequently, secondary-specific learner features remained zero and `warning_both` could never identify distinct channels. This is a **scientific instrumentation error**, not a CyberSand defect. The original 013 source/evidence remain untouched for historical provenance. This new 014 source explicitly carries `channel` through delivery and rejects noncausal, untagged replayed messages. The repaired `warning_both` completes **182/240** (independent) versus its invalid historical **140/240** in 013; the corrected independently trained policy still completes 197/240 on its originally matched DEVELOPMENT worlds.

## Transfer test: train on independent channels, shift DEV error correlation

TRAIN always uses independent primary and delayed-secondary alert/loss/delay draws for `independent` and every `shiftXX` arm. The **same fitted coefficients and cutoff** are retained for all shifted DEV arms, with no DEV tuning. For each DEV seed, the second signal switches to common-mode alert/loss/delay failure with probability 0, .25, .50, .75 or 1. This is a *seed-level* shift; the four horizon/cap evaluations for each seed share the same choice, so 240 task records do **not** represent 240 independent world seeds. A separately TRAIN-fitted `shared` arm and a hypothetical perfect channel supply comparison controls.

| Channel condition | Trained goals | Mean utility | Planner calls | Saved vs always | Lost baseline goals |
|---|---:|---:|---:|---:|---:|
| single | 193/240 | 703.6833 | 1024 | 161 | 4 |
| independent | 197/240 | 720.2708 | 1030 | 155 | 0 |
| shift25 | 195/240 | 712.3167 | 1025 | 160 | 2 |
| shift50 | 193/240 | 704.3625 | 1020 | 165 | 4 |
| shift75 | 193/240 | 704.3625 | 1020 | 165 | 4 |
| shift100 | 193/240 | 703.6833 | 1024 | 161 | 4 |
| shared | 191/240 | 695.8250 | 1021 | 164 | 6 |
| perfect | 197/240 | 722.6000 | 1010 | 175 | 0 |

The conservative always-revisit baseline completes **197/240** with mean utility **704.2708** and **1185** planner calls in all matched worlds. The fixed TRAIN-independent controller saves **155** planner calls at zero missing goals in the unshifted synthetic cohort, but loses **2** goal records at 25% shift and **4** at 50–100% shift (paired losses at seeds **15130** and **15147**, each duplicated across the 128/64 censor and 128/128 shadow cohorts). The separately TRAIN-fitted shared-channel controller loses **6** goals. Hence the observation-redundancy advantage is fragile to common-mode failures.

## Scientific integrity and interpretation

- **2240** TRAIN counterfactual rows, **6800** optimizer updates, **11520** DEVELOPMENT policy rollouts, and **428,109** synthetic action receipts reproduced against fresh same-source host instances.
- All eight modes passed legal-action, movement/material receipt-algebra, chronology, channel-provenance and two corrupted-action/coordinate rejection tests per mode. The exact V8 source used for this execution is the 014 Git blob above. Independent Python/native replay was **not** performed.
- Full per-episode receipts hashes, action distributions, fitted weights, TRAIN cutoffs, 128/64 censor versus separate 128/128 shadow, matched negative cases, and causal delivery counts are retained in two append-only evidence ledgers. The FNV32 receipt hash is a comparison aid, not cryptographic evidence.
- This study used 60 DEV world seeds repeated across four horizon/cap conditions, not 240 independent worlds. Channel observations are an engineered synthetic facility, **not** an established CyberSand host ABI.
- No SLM/LoRA checkpoint was created; this trains logistic recovery controllers only. V8 elapsed whole-study time was recorded, but exact-source Node peak RSS, per-inference p50/p95, an independently implemented host, and native CyberSand acceptance remain **unqualified**.
- Structural HELDOUT stayed sealed; no real-game, engine, gameplay or other DRIV3R laboratory files were changed.

## Reproduce explicitly

```bash
node experiments/driv3r/training/long_horizon/signal_correlation_014.js /absolute/output/directory
```

No optimization occurs on import or routine PR/push. The opt-in CLI writes eight full ledgers plus a trajectory gallery; the source validates identical TRAIN parameters for shifted modes. Source-smoke CI is distinct from execution of this scientific experiment.

**Decision:** continue to use always-revisit as the conservative control. Investigate genuine host-observable confirmation and risk-sensitive adaptation under correlated failure before promoting any redundant-notification strategy. The original 013 claims must be read with the missing channel-provenance limitation.
