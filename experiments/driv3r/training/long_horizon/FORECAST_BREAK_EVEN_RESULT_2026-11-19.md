# FORECAST-BREAK-EVEN-003 — how accurate must a gate forecast be?

**Result type:** Audited, source-pinned decision-theory sensitivity of existing verified DEVELOPMENT rollouts, **not** additional model training, native gameplay, or newly executed episodes at hypothetical cue reliabilities. Structural HELDOUT remains sealed.

## Source requalification

In this run the original committed `forecast_utility_lab.js` (`4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873`) was re-executed as source in V8, independently reproducing its 1,344 source-replayed episodes, 82,493 host-like legal action receipts and 16 assertions. The prior model had 720 seeded logistic Adam optimizer updates; no new optimizer run is claimed here. This analysis consumes its committed evidence JSON at Git blob `2be8ebbbce9da1b67b2bbfc36c801b5e5400a1b3` and verifies both identities when invoked through Node CLI.

The experiment computes **reward-conditioned rational handoff thresholds**, comparing continuing a cached route with early route-overlap replanning. It does not assume the experimental future-conditioned cue is present in CyberSand's actual host ABI. The 64-paint limit stays intact for every `native64_proxy` cohort; 128-painted-move results remain separately labelled `shadow128`.

## Goal reward = 1000, early/late mode matched

| Cohort | Cost of eager replanning when gate reopens early | Benefit of replanning when gate stays closed late | Minimum symmetric cue correctness to skip replanning on negative cue |
|---|---:|---:|---:|
| 36 / native64 proxy | 26.000 | 1010.250 | **97.491%** |
| 72 / native64 proxy | 26.000 | 1042.250 | **97.566%** |
| 128 / native64 proxy | 6.000 | 105.375 | **94.613%** |
| 128 / shadow128 | 26.000 | 1097.250 | **97.685%** |

Each source aggregate has 16 DEVELOPMENT seeds per mode. Under balanced early/late, independent symmetric forecast errors, and reward `1000 × goals - 2 × decisions - 3 × planner_calls - backtracks`, replan on the negative signal only when `(1 - p) × benefit > p × cost`, i.e. the critical fidelity is `benefit / (benefit + cost)`.

At **75%**, **87.5%** and **93.75%** reliability the rational expected-utility policy **ignores both signs of the forecast** on all four cohorts. The overall mean remains **623.094** (fixed route overlap); no prospective advantage is shown. At **96.875%** only the 128/native64 proxy uses the negative signal, for a negligible **+0.210** mean utility aggregated across cohorts. At **98.4375%**, all four do and gain **+2.652** mean utility. Perfect future information reaches **630.094**, just **+7.000** above route overlap.

**Censor caveat:** at horizon 128/native64, neither lazy nor overlap completed any of the 16 late-gate goals. Its +105.375 advantage reduces wasted backtracking rather than recovering goals; it must not be advertised as 128-decision native task success. The 128 shadow cohort is not native acceptance.

## Objective sensitivity and interpretation

The threshold is **not universal**. At a lower hypothetical goal reward (200), the 36/native64 threshold drops to **88.995%**; with zero goal reward it is **28.276%**. Such changes alter which errors are worth tolerating, but do not add observable features or predict unseen events.

The finding is a second, distinct negative result: an apparently promising 75%-accurate cue is not economically valuable in these high-penalty synthetic episodes, even if it is statistically predictive. For real deployment one needs an actually authorised, prospective host-observable feature, calibrated reliability on independent worlds and an explicit reward definition—not privileged future labels. Larger LLM/SLM models cannot remove genuine observational aliasing without additional information.

## Reproduce

From the PR checkout with Node.js:

```bash
node experiments/driv3r/training/long_horizon/test_forecast_break_even.js
node experiments/driv3r/training/long_horizon/forecast_break_even_lab.js /tmp/forecast-break-even-003.json
```

The CLI checks the imported upstream Git blob SHA-1 identities before analysing the original receipt-qualified aggregates. The first command checks original reconstruction, replay-count invariants, determinism and evidence corruption rejection. Full new evidence and the labelled reliability frontier SVG are committed beside the source. No checkpoint binaries, bulk logs, workflows, paid resources, native source, other laboratories or external repositories were changed.

**Limitations:** Future cue reliabilities are an analytical expected-utility sweep assuming exchangeability and symmetric errors, **not 7× replicated trajectories** or new observed native events; original JavaScript world remains an independent synthetic approximation, not exact canonical game-side action/observation. Inference latency/RSS was not newly benchmarked.
