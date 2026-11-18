# M17 — causal sensor-map memory and three tiny trained action heads

**16 November 2026 — synthetic research only; NO model promotion, no native claims.**

A self-contained Node.js script executed **720 actual CPU AdamW optimizer updates** (240 each) across local linear, local random-feature, and causal-graph-hinted linear decision heads. It generated 216 original 80-turn maximum, held-out synthetic trajectories and 16750 host-legal action receipts. It independently reran each seeded trajectory for exact result equality. TRAIN (16 seeds), DEV (4 seeds), and EVAL (9 seeds) are disjoint, and EVAL uses nested-gate, comb and partition-wall geometries not generated during TRAIN/DEV.

| Policy | Unique cells | Waypoints / 6 | Retraces | Branch nodes |
|---|---:|---:|---:|---:|
| oracle | 65.296 | 5.889 | 5.778 | 1.444 |
| greedy | 12.037 | 1.074 | 68.963 | 0.481 |
| novelty | 59.778 | 0.444 | 21.222 | 0.444 |
| causal-map | 66.519 | 5.37 | 8.63 | 1.815 |
| local-linear | 43.741 | 0.519 | 37.259 | 1.222 |
| local-rff | 33.407 | 0.333 | 47.593 | 0.481 |
| graph-linear | 65.963 | 5.333 | 9.37 | 1.593 |
| graph-linear-ablated | 48.037 | 0.519 | 32.963 | 1.963 |

Exploratory seed-cluster paired waypoint differences (95% bootstrap ranges):
- causal-map vs greedy: 4.296 [3.63, 4.852]
- graph-linear vs local-linear: 4.815 [4.37, 5.185]
- graph-linear-ablated vs graph-linear: -4.815 [-5.185, -4.37]
- graph-linear vs causal-map: -0.037 [-0.111, 0]

## Scientific limitations

This 20×20 toy host has cardinal moves plus STOP, blocked/protected tiles, single-cell paint, and host-projected reachable waypoints. It is **not** the native CyberSand ABI. The causal mapper sees strictly accumulated 5×5 sensor patches, not whole-world obstacle arrays; the fully privileged oracle is a separate ceiling control. The graph-linear head is provided a **one-hot proposed move from a deterministic route planner**, so performance near the mapper does **not** establish that the neural model learned independent long-horizon planning. Family separation comes from predetermined host waypoints; there is no validated generative texture quality. This is an unsealed research EVAL, not final model selection.

The exact executed script and result ledger (including per-world checksums and example action traces) are committed in this owned directory. Weight fingerprints are 32-bit **noncryptographic diagnostics only**, not cryptographic checkpoint hashes. Checkpoint weights are intentionally **not committed to Git**; no immutable GitHub Actions model artifact was produced, so no model is promotable. This experiment does **not** constitute SmolLM2-135M or other pretrained-backbone tuning. Subsequent work should run bounded, opt-in hosted qualification and test learned compact map representations rather than relying on a direct planner-action hint.

Repository custody: DRIV3R Draft/HOLD PR #1 only; CyberSand and Neurounner untouched.
