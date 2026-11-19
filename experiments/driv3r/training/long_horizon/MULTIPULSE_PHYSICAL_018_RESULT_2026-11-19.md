# MULTIPULSE-PHYSICAL-018 — causal inspection timing and terminal censor

**19 November 2026.** DRIV3R Draft/HOLD long-horizon programme. Parent commit `51b6235a8e3ded7a6c5ea48b3d2b5994356c2581`; pinned and unchanged `physical_inspection_016.js` Git blob `45a426f893203f93fa9a3972d63bd4a110ceaa3d`. Explicit opt-in CPU experiment, not routine CI.

## Question and causal experiment

Test whether a long-horizon controller should physically recheck an earlier painted material goal **immediately after an earlier risk**, **after all later risk opportunities**, or not at all when the remaining logical move budget makes a return impossible. Use the original synthetic host, its local material observation and menu, but no hypothetical offscreen notification.

Two controlled, explicitly offscreen material interventions: first just after the source-faithful prefix, second only after the host-executed final 128-horizon PAINT. The first is 0/1 damage; the later is 0/1 where the 128 case reaches the trigger. A second zero means *preserve* the existing A material, not restore previously damaged material. We reset the first intervention's reference state in both pair arms and test identical public observations/features; hidden damage is never an input. All interventions are reconstructed in replay.

**TRAIN:** 70 world clusters, 280 public decision prefixes and 3360 policy rollouts. A real 1100-update four-class softmax optimizer learns among never/early/feasible/both from counterfactual TRAIN-only teacher choices, goal-first under the stricter STOP+materials criterion. Teacher label frequencies: never 86; early 124; feasible 70; both 0.

**DEVELOPMENT:** 50 disjoint world clusters, 600 event/clock paired conditions and **5400 sequential policy rollouts**, yielding **302652 action receipts**, all replayed by fresh instances of the exact pinned host with interventions rebuilt. 2/2 deliberately corrupted receipts rejected. Two complete V8 reruns reproduced every scientific row, learned coefficient, model choice, receipt hash and paired exception. Source-smoke is not scientific replay.

## Measured DEVELOPMENT outcomes

`material` counts painted objectives regardless of STOP. `terminal` is the **additional experimental criterion** requiring a STOP receipt *after* material completion, to prevent counting runs that exhaust the cap without terminal action. This stricter criterion is **not asserted as the native host ABI**; it is a conservative synthetic diagnostic.

| Controller | Terminal / 600 | Material / 600 | Nonterminal material-only | Mean utility | Planner calls |
|---|---:|---:|---:|---:|---:|
| Never recheck | 199 | 199 | 0 | 209.413 | 2500 |
| Early only | 302 | 333 | 31 | 407.82 | 3148 |
| Late only (unreserved) | 352 | 425 | 73 | 551.44 | 3104 |
| Both (unreserved) | 352 | 433 | 81 | 558.493 | 3448 |
| Conservative feasible guard | 425 | 425 | 0 | 559.42 | 2860 |
| Tight guard (+1) | 425 | 425 | 0 | 559.42 | 2860 |
| Cap-aware timing rule | **426** | 433 | 7 | **568.033** | 3080 |
| Trained softmax student | **427** | 427 | 0 | 562.7 | 2868 |
| Post-task physical-state oracle (not deployable) | 425 | 425 | 0 | 568.69 | 2730 |

*Oracle means privileged knowledge of A **at the late checkpoint**, not future events or the best conceivable policy; it can underperform early scheduling in censored tasks.*

| Horizon / logical cap | Feasible guard terminal | Cap-aware rule terminal | Student terminal | Early terminal | Late terminal |
|---|---:|---:|---:|---:|---:|
| 36/36 | 76 | 76 | 76 | 52 | 52 |
| 72/64 | 100 | 100 | 100 | 100 | 100 |
| 128/64 **censored** | 49 | 50 | 51 | 50 | 0 |
| 128/128 **shadow** | 200 | 200 | 200 | 100 | 200 |

**Result:** the trained student gained **2/600 strictly terminal goals** relative to the conservative guard, and the simple cap-aware rule gained **1/600**. This is a small, exploratory DEV result, not credible statistical evidence for promotion or native performance. The cap-aware rule increased mean synthetic utility by 8.613; the student by 3.280. Paired losses/negative cases (171) remain in the evidence ledger.

**Important accounting correction:** unreserved late/both checks often exhaust the move budget without issuing STOP, despite having all materials correct. They yield 73 / 81 material-complete but nonterminal cases respectively. The older synthetic material-only metric remains reported, but terminal completion must not be conflated with it. Even the cap-aware rule has 7 such cases. The 128/64 cap is never combined with the separate 128/128 shadow.

## Reproducibility and limitations

Run explicitly on a checkout at the pinned long-horizon source: `node experiments/driv3r/training/long_horizon/multipulse_physical_018.js /absolute/output/dir`. Runner refuses altered host blob. It produces JSON action distributions, all paired rows, negative cases, model coefficients and SVG trajectory gallery. It never automatically trains on import, ordinary PR or push.

Two completed **V8 scientific** runs were deterministic; 100,000 softmax predictions took 293 ms (2.930 microseconds average) in the ephemeral V8 experiment process, not an independently qualified Node per-inference p95, native latency, or game end-to-end timing. Node peak RSS is **unmeasured in this run**; explicit Node runner records it. No neural SLM checkpoint or immutable Actions artifact was produced.

This is a **same-source synthetic host replay with explicit experimental interventions**, not an independent CyberSand host replay, source/native acceptance, an observed real game trace, or verified notification ABI. Material observation is local at the physically reached cell. The structural HELDOUT remains unopened. No model binaries, paid CPU/GPU/API jobs or native/game changes.

**Decision:** retain conservative feasibility guard as baseline pending multi-seed independent host validation, real receipts and terminal ABI confirmation. Cap-aware scheduling is a promising narrow research candidate; two extra trained successes do not establish robustness. Do not promote or merge. Preserve all historical negative results.
