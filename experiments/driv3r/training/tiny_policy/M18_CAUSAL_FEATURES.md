# M18 — causal observed-map features, no planner action hint

16 November 2026. **Exploratory synthetic CPU experiment; no model promoted.**

M17's graph-assisted neural head was given an explicit one-hot planner action. M18 withheld that hint completely. Three models were fitted with 720 actual AdamW updates: a local-observation MLP, memory-conditioned linear head, and memory-conditioned MLP. All received privileged BFS *training labels*, but at inference their inputs are host observations plus an explicitly accumulated 9×9 sensed-map/trail neighborhood, with no route recommendation. A deterministic map planner and privileged oracle remain separate baselines.

8166 TRAIN examples, 1796 DEV examples, 270 matched EVAL trajectories across 9 unseen seeds, 3 drawing families, 3 obstacle generators, 80-turn action caps; 18588 action receipts all legal and replayed. Evaluated first-seed trajectories from action sequences in an independent new World instance; all trajectories reran deterministically.

| Controller | Unique cells | Waypoints / 6 | Retraces | Branches |
|---|---:|---:|---:|---:|
| oracle | 64.852 | 5.926 | 6.37 | 1.704 |
| map-planner | 63.259 | 5.333 | 9.741 | 1.741 |
| greedy | 8.667 | 0.667 | 72.333 | 0.444 |
| novelty | 77.778 | 0.259 | 3.222 | 0.333 |
| random-move | 31.185 | 0.111 | 49.815 | 12.815 |
| random-legal | 4.296 | 0 | 1.111 | 0.148 |
| local-mlp | 32.704 | 0.778 | 40.074 | 1.63 |
| memory-linear | 47.704 | 0.704 | 33.296 | 0.556 |
| memory-mlp | 63.259 | 0.407 | 17.741 | 1.407 |
| memory-mlp-ablated | 34.963 | 0.556 | 33.333 | 1.593 |

### Trained heads

| Model | Parameters | DEV teacher-action agreement | Incorrect STOPs on DEV |
|---|---:|---:|---:|
| local-mlp | 1557 | 0.8491 | 0 |
| memory-linear | 1270 | 0.8341 | 2 |
| memory-mlp | 4149 | 0.8363 | 1 |

### Paired exploratory differences in completed waypoints

- memory-mlp minus local-mlp: -0.37, seed-cluster bootstrap 95% [-0.926, 0.259].
- memory-mlp minus map-planner: -4.926, seed-cluster bootstrap 95% [-5.593, -4.185].
- memory-mlp-ablated minus memory-mlp: 0.148, seed-cluster bootstrap 95% [-0.407, 0.667].
- memory-linear minus local-mlp: -0.074, seed-cluster bootstrap 95% [-0.519, 0.37].

**Interpretation:** compare memory models against local, ablated-memory and deterministic map-planner controls. Any improved coverage is not evidence of independent multi-waypoint reasoning. Differences in DEV action agreement are not automatically useful closed-loop behaviour. The families have different host-provided waypoints: low inter-family IoU alone does not prove good line or texture artistry.

**Qualification limits:** Although train/dev/eval seeds and geometry generators differ, this EVAL is not sealed FINAL. The map is engineered causal spatial memory from 5×5 local sensors, not a learned map encoder. No native CyberSand ABI, physics, replay, artistic assessment, or physical Workbench acceptance. This is scratch-head training only, not SmolLM2-135M tuning, pretrained LoRA or stronger-model teacher distillation. No checkpoint binaries have been archived in immutable Actions artifacts; weight FNV32 values are expressly noncryptographic. **No candidate is promotable.**

The script is exactly the function executed to produce the committed evidence. For independent repetition with Node 20+: `node experiments/driv3r/training/tiny_policy/m18_causal_features.js > /tmp/m18-result.json`; compare parsed JSON with the committed `M18_CAUSAL_FEATURES_RESULT.json`. No CI is triggered by a normal PR push.

All modifications confined to the owned tiny_policy directory on DRIV3R Draft/HOLD PR #1. No changes to DRIV3R main, CyberSand or Neurounner.
