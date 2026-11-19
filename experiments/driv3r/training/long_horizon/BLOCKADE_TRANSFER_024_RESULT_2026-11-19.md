# BLOCKADE-TRANSFER-024 — negative single-to-double blockade transfer
19 November 2026 · Long-horizon research programme · **Draft/HOLD**

## Question and empirical method
Do inspection modes chosen to maximise terminal-goal completion without obstructions or under **one targeted closure** transfer safely to **two closures** at 36, 72 and 128 decisions, including the **64-logical-move censor**?

Ran a **new, bounded, deterministic exhaustive fit of a four-mode horizon/cap lookup controller** on existing, already source-replayed [INTERCEPT-BLOCKADE-023](./INTERCEPT_BLOCKADE_023_RESULT_2026-11-19.md) candidate trajectories. Input evidence Git blob `cf4dc3bcd96057c2dcf7fb6dd4446ea584ec45d4`. Independently reconciled 1,080 original 023 records, 360/360/360 paired stress conditions, 36 aggregate goal/call fields, 1,080 frozen-student mode receipts, and original single→double wall prefixes. Source 023 replayed 253,022 synthetic host receipts; **024 executed zero new host rollouts or receipt replays**.

Fit on **18 unsealed DEVELOPMENT world seeds** (160100–160117); diagnose on **12 disjoint unsealed DEVELOPMENT seeds** (160118–160129), keeping all correlated damage forks in their seed cluster. This is intentionally labelled *retrospective exploratory*, not prospective held-out validation: prior experiments had already inspected the same DEVELOPMENT programme.

- **Mild fit:** maximise total terminal goals across zero and one closure; break ties by minimum planner calls.
- **Robust fit:** maximise the *worst* completion count across zero, one and two closures; then total goals, then calls.
- Both choose feasibility for 36/36, 72/64 and 128/128 shadow. Only 128/64 differs: mild chooses **early inspection**, robust chooses **feasibility**. No policy observes the experimental closure category or hidden material damage.

## Results — diagnostic test split
| Stress (144 correlated fork records each) | Feasibility | Mild-fit | Robust-fit | Frozen 021 |
|---|---:|---:|---:|---:|
| No closure | 104 | 107 | 104 | 104 |
| One closure | 101 | 107 | 101 | 101 |
| Two closures | 97 | 91 | 97 | 97 |
| **Total (432 correlated forks)** | **302** | **305** | **302** | **302** |

Mild-fit's +3 aggregate goals cost **136 additional planner calls** (2600 versus 2464). Frozen 021 ties the guard on goals with +24 calls.

At **128 decisions / 64 logical moves**, the diagnostic result is more informative:

| Obstructions | Guard / 48 forks | Mild-fit early / 48 forks |
|---|---:|---:|
| Zero | 12 | 15 |
| One | 9 | 15 |
| **Two** | **6** | **0** |

**Concrete negative result:** mild-condition selection increases completions with no/one obstruction but collapses to **0/48** when two targeted closures arrive, whereas the feasibility guard retains **6/48**. All six lost completions occur in separate seeds with `firstDamage=0, secondDamage=0`, demonstrating action-budget competition even without erasure. Their exact original event sites, ticks, paired receipt hashes, and calls are retained in the JSON evidence. A seed-cluster bootstrap (4,096 deterministic draws over **12** seeds) yields a descriptive 95% resampling range of **[-9, -3]** goals for the -6 severe difference. Do not treat 48 correlated forks as 48 independent worlds.

## Limitations and decision
The robust selector avoids this specific collapse, but sacrifices the mild-case advantage. Retain the conservative feasibility guard as the benchmark, keep both lookup fits experimental, and do **not** promote the frozen 021 student based on 023/024. This is synthetic route and STOP behaviour, not verified native CyberSand ABI or independent host replay. No new neural checkpoint, Node latency/RSS qualification or new world evidence was produced by 024. The next high-value experiment must use *new unsealed worlds*, altered obstacle timing and independent receipt verification rather than further optimisation on inspected 023 cases.

Reproduce **only by explicit invocation**: `node experiments/driv3r/training/long_horizon/blockade_transfer_024.js /absolute/INTERCEPT_BLOCKADE_023_EVIDENCE.json /absolute/output`. The runner verifies the exact 023 input Git blob. Original traceable [023 SVG trajectories](./INTERCEPT_BLOCKADE_023_PATHS.svg) are retained; 024 does not invent new paths. Structural HELDOUT remains unopened. Only the authorised DRIV3R long-horizon directory is changed; no paid compute, game changes, model binaries or workflow triggers.
