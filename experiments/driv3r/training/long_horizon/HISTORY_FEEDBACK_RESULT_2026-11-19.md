# HISTORY-FEEDBACK-007 — observed material recovery policy (DEVELOPMENT)

2026-11-19. DRIV3R research-only, source-pinned synthetic simulator. No production CyberSand runtime or host ABI changes.

## Purpose and design
Train a four-feature logistic handoff to choose immediate route-overlap replanning or three-step deferral after observed material closure, using an earlier material close/open pulse as a history cue. Pulse is generated in TRAIN on 75% early / 25% late cases, at ticks 2/3 before the principal closure at tick 5. The online policy receives only observed material history, horizon and gate coordinate, never future mode or reopening time. DEVELOPMENT changes wall geometry and reopening schedule, following PATIENCE-TRANSFER-005.

Each learned arm uses 432 TRAIN paired labels, 900 bounded full-batch cost-weighted optimizer steps. Ablation zeros the history feature. The main benchmark retains paired immediate-replan, fixed-defer3 and TRAIN-selected controls. 36, 72, 128 decisions; native64-style logical paint cap and separately distinguished shadow128.

## Measured DEVELOPMENT results
| Arm | Goals / 192 | Mean utility | Planner calls |
|---|---:|---:|---:|
| overlap | 140 | 603.302 | 320 |
| defer3 | 115 | 473.042 | 288 |
| train_selected | 128 | 540.479 | 296 |
| learned_history | 140 | 603.01 | 312 |
| learned_nohistory | 140 | 603.302 | 320 |

Observed-history learner saves 8 planner calls versus overlap without losing any goals on this DEVELOPMENT set, but its utility decreases by 0.292 per episode. The no-history learner matches immediate replanning. **This is not a demonstrated game-policy improvement.**

Paired comparisons against immediate replanning:
- defer3: goals lost 26, gained 1; utility wins 32, losses 69, ties 91; mean utility delta -130.260417
- train_selected: goals lost 12, gained 0; utility wins 24, losses 48, ties 120; mean utility delta -62.822917
- learned_history: goals lost 0, gained 0; utility wins 8, losses 12, ties 172; mean utility delta -0.291667
- learned_nohistory: goals lost 0, gained 0; utility wins 0, losses 0, ties 192; mean utility delta 0

## Qualification and limitations
3264 synthetic DEVELOPMENT episodes, 3264 same-source independent replays and 198260 action receipts. 19 assertions pass, including tamper rejection and TRAIN-only sample and learned cohort checks. Full action distributions, negative cases and SVG examples are retained in the evidence ledger.

The prior event pulse is an engineered proxy, not a confirmed native CyberSand observation. Replay is same-source JS, not cross-language or native game acceptance. V8 aggregate wall time 157474 ms. Per-inference latency and process RSS **were not measured** by the V8 runner; an explicit Node CLI records wall time/RSS on execution. Source-smoke should not be confused with scientific replay. No checkpoints in Git and no sealed structural HELDOUT.

## Decision
Retain immediate route-overlap replanning as the robust baseline. Next test genuinely observable history under independently generated event schedules and larger recovery/goal-change distributions; only promote a learned rule after it improves held-back DEVELOPMENT goals or utility without concealed future labels.
