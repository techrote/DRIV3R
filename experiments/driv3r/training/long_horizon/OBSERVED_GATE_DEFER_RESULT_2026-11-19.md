# OBSERVED-GATE-DEFER-004 — when to invalidate a route after a witnessed material closure

**19 November 2026 · DEVELOPMENT-only synthetic experiment · Draft/HOLD.** No structural HELDOUT, native CyberSand receipts, production host change, paid API, model binary, neural checkpoint or automatically triggered CPU training.

## Hypothesis and method

An observed material closure intersects the current cached route, but the future reopening schedule is unknown. Compare eager full-route invalidation with legality-only lazy repair, one-to-four-decision deferred invalidation, and two-to-four-action proximity triggers. Can a TRAIN-selected bounded patience rule preserve goals while reducing unnecessary searches, without a privileged future cue?

Strictly transform the already committed \`forecast_utility_lab.js\` at Git blob \`4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873\`; each replacement fails on source drift. Its deterministic geometry, immutable-ink constraints, host-like menu, BFS, material event receipts, independent state/action replay and 64-logical-paint cap are retained. Event time and changed material cell are observed; future mode is not passed to the new defer/selection policies. The old privileged and exogenous-forecast arms remain comparison controls, not deployable features.

For each 36/72/128 synthetic decision horizon, select one of eight candidate invalidation rules solely by mean reward-weighted utility on TRAIN seeds 1000–1047, balanced across early/late/irrelevant modes. The 128-horizon selection uses synthetic shadow128 training; apply that policy to both clearly separated 128 test cohorts. Evaluate DEVELOPMENT seeds 3000–3015, 64-move-censored proxy and explicitly labelled synthetic shadow128.

## Executed source-replay results

The deterministic V8 scientific source run executed **2,880 DEVELOPMENT episodes** across 15 arms and **178,289 independently replayed legal action receipts**, with **16 checks PASS** including optimizer determinism, menu legality, 64-vs-128 censor, material-event replay and tampered receipt rejection. The parent 7 arms were reproduced with identical baseline goals and utility.

| Controller | Goals / 192 | Mean utility | Planner searches | Backtracks |
|---|---:|---:|---:|---:|
| Lazy legality-only | 96 | 358.833 | 256 | 1512 |
| Full-route overlap | **144** | 623.094 | 320 | 128 |
| Unconditional epoch | **144** | 622.094 | 384 | 128 |
| Deferred 1 decision | 140 | 599.302 | 320 | 256 |
| Deferred 2 decisions | 122 | 502.865 | 320 | 384 |
| Deferred 3 decisions | 128 | 542.750 | 256 | 256 |
| Deferred 4 decisions | 128 | 541.542 | 260 | 316 |
| Proximity threshold 2 | 94 | 348.219 | 260 | 1460 |
| Proximity threshold 3 | 98 | 368.594 | 266 | 1410 |
| Proximity threshold 4 | 98 | 368.208 | 272 | 1362 |
| TRAIN-selected controller | **144** | **624.677** | **272** | 224 |

**TRAIN choice:** 36 → full-route overlap; 72 and 128 → deferred 3 decisions. On the disjoint DEVELOPMENT scenarios, selection matches eager-overlap's 144 goals and improves mean utility only **+1.583**; it saves **48 planner searches** but incurs **96 additional backtracks** across 192 paired episodes. This is a narrow, reward-sensitive pilot benefit, not robust native superiority.

The generic defer3 controller loses **16 goals** versus overlap, entirely due to late-closure 36-decision failures. Proximity-triggered controllers fail more frequently. Early reopen and late closure remain observationally identical at event time. The reward function heavily penalises lost goals; controller selection is therefore horizon-conditioned rather than a universal patience timer.

**Censor:** All 128-horizon native64-proxy goals are paint-censored; 128-move shadow completions must not be interpreted as native gameplay support. The 36/72 positive result remains synthetic only.

## Reproduction and provenance

\`\`\`sh
node experiments/driv3r/training/long_horizon/observable_gate_defer_lab.js /tmp/driv3r-observed-defer-004
\`\`\`

This command is **explicitly invoked**, bounded CPU work; no PR/push handler triggers it. The source checks the exact upstream Git blob, then regenerates complete per-mode aggregates, representative receipt-based path gallery, optimizer/source checks, wall-clock and Node process maxRSS. The V8 execution above also publishes its full generated evidence and SVG at this exact PR HEAD. The in-tool V8 has no process RSS or per-inference latency counters; do not infer those from the cumulative study wall time. A separately instrumented Node run is still needed for source-exact RSS and latency profiling. Action receipt hashes are noncryptographic FNV diagnostics; Git blob IDs identify committed source.

## Limitations and next decision

This is a deterministic **JavaScript synthetic port**, not a source-identical independent Python replay and not official CyberSand ABI acceptance. TRAIN choice is discrete policy selection, not new neural optimizer updates or an immutable checkpoint. Original upstream logistic models retrain as controls; these new rules are deterministic. Balanced early/late reopening is deliberately simple and must be challenged with variable observed material-recovery times and different topologies before considering deployment. Retain failed late-closure cases, separate all controls and the 64-move censor, and keep structural HELDOUT sealed.
