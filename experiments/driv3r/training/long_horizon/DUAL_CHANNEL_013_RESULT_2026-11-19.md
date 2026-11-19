# DRIV3R DUAL-CHANNEL-013 — independent vs shared material-feedback failure
**Date:** 2026-11-19  
**Source parent:** `b1c1683161df9b60f35e9326a49969766eed9019` (`signal_frontier_012.js`, Git blob `63b0f42a1419caf1d17395aa36a85af953cbd7bb`).  
**Qualification:** explicitly executed synthetic V8 CPU run; NOT native CyberSand, nor external checkpoint training.

## Scientific question and outcome
Does adding a second, *delayed* material-damage notification improve learned recovery, and does this benefit survive common-mode failure? Train separate 14-feature logistic controllers with 850 deterministic optimizer steps per observation mode, with TRAIN-only safety calibration (zero avoidable missed TRAIN goals before utility). Evaluation uses paired independent DEVELOPMENT seeds; sealed structural HELDOUT remains unopened.

**Result:** the independent secondary signal preserved **197/197** always-revisit completed goals while saving **155/1,185 planner calls** on this DEVELOPMENT cohort. The shared-failure double-notification learner completed only **191/197** goals despite the same nominal 98% sensitivity per channel; single-channel completed **193/197**. Perfect hypothetical signal completed 197/197 and saved 175 planner calls. The independent case is a *synthetic positive*, not a proof of safe game-side deployment. Small sample and hand-designed notification independence matter.

| Hypothetical observation mode | Always-revisit goals | Trained goals | Trained mean utility | Trained planner calls | Warning-any goals | TRAIN cutoff |
|---|---:|---:|---:|---:|---:|---:|
| single | 197/240 | 193/240 | 703.6833 | 1024 | 185/240 | 0.005 |
| independent | 197/240 | 197/240 | 720.2708 | 1030 | 195/240 | 0.01 |
| shared | 197/240 | 191/240 | 695.825 | 1021 | 185/240 | 0.1 |
| perfect | 197/240 | 197/240 | 722.6 | 1010 | 197/240 | 0.005 |
Baseline always-revisit: **1,185** planner calls; **197 goals/240** and **704.2708** mean utility in every matched condition. Privileged counterfactual oracle: **197/240** goals, **981** planner calls (not accessible to the controller). Each mode includes 240 DEV tasks over **36/36, 72/64, 128/64 (censored), 128/128 (shadow)**. Do not conflate synthetic 64-move caps with native validation.

## Design
Each synthetic world has host-legal movement, obstacles/gates, local material painting/erasing, a hidden erase event, optional delivered messages, path planning and a STOP action. All controller features derive from messages already delivered, current position/route length, horizons and previous action history, **not hidden damage**. The oracle uses future counterfactual outcome strictly as an evaluation ceiling. Damage prior shifts from 16% TRAIN to 30% DEV; DEV topology has an additional obstacle. Seed ranges: TRAIN 12100–12169, DEV 15100–15159, four horizons each.

All imperfect primary channels have 98% sensitivity, 1% false-positive rate, 1% packet loss and 0–1 decision arrival lag. Secondary arrives three decisions later: independent mode draws new alert/loss/delay noise; shared mode reuses exact primary alert/loss/delay draws (perfectly correlated failure). Single mode has no secondary; perfect mode is an idealised zero-error primary. **The second channel is a hypothetical observation mechanism, not an official CyberSand ABI** and no native production change is proposed.

The action-control policies are never revisit, always revisit, warning-any, warning-both, TRAIN-calibrated learned probability and privileged post-hoc oracle. No structural HELDOUT, paid compute, production modifications or checkpoint binaries. Four models were fitted only to synthetic TRAIN.

## Reproduction and integrity
Explicit CPU only:
```bash
node experiments/driv3r/training/long_horizon/dual_channel_013.js /absolute/output/directory
```
The script has no import-time optimizer or automatic PR/push trigger. It writes four complete JSON campaigns and an SVG trajectory gallery locally. This checked-in evidence contains all four source-replayed campaigns, exact TRAIN weights, policy action distributions, model thresholds, per-task hashes, negative cases and paths.

Each campaign: **280 TRAIN counterfactual rows; 850 optimizer steps; 1,440 DEV controller rollouts; 2 corrupt receipt rejects**. Same-host fresh-instance replay plus distinct receipt algebra is NOT an independent native game implementation. Matched always/never action trajectories were hash-identical across four campaigns. No exact committed-source Node process RSS or per-inference latency has been claimed; the evidence records V8 whole-study elapsed wall time only. No external model/checkpoint artifact existed for this study.

## Decision
**Keep always-revisit as the deployment-conservative synthetic baseline.** Independent redundant information can enable a limited planner-call reduction, but duplication that shares failure mechanisms is not protection. Next: independently qualify causal observation availability and evaluate feedback correlation under more severe domain shift, ideally against read-only native host receipts. Do not alter CyberSand host ABI on this evidence.
