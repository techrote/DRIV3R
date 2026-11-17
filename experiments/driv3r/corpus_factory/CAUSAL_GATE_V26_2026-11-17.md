# DRIV3R Corpus Factory v26 — paint-triggered causal route changes and accepted-wrong-branch recovery

17 November 2026. **Synthetic game-policy host, NOT native CyberSand ABI conformance and NOT trained-model evidence.** Data-only lane: `experiments/driv3r/corpus_factory/**`, existing DRIV3R Draft/HOLD PR #1.

## Problem and hypothesis

V25 provided terminal-outcome-proven BACKTRACK versus MOVE_N decisions, but its distal obstruction was a matched *noncausal distractor*. Other labs' RETRACE training did not reliably improve sustained painting: improving prescribed local moves could worsen completed paint and cause premature stops. V26 tests whether a learner can respect a **public live-gate transition caused by its own accepted paint**, rather than treating the initial scanned topology as immutable, and whether it can recover after an accepted but inferior first branch.

This is incremental synthetic data, not a policy trainer or game implementation. It imports pinned, existing v25 geometry rather than editing v25 files. Source identity: `outcome_forks_v25.js` SHA-256 `c879024a7c8ac96408a0a507be8f12bc7299b2bc6f8baf6be918e738e5d01a7d`.

## Host, decisions and labels

- Orthogonal loop-and-branch worlds retain an exit, three material-A/B paintable targets and an exact **two-of-three** quota. Incorrect paint can be accepted, erased, and repaired; protected cells are absent from legal traversable edges.
- The gate is the lower-west loop entrance (`-1,0` ↔ `-1,1`). It is initially open or closed according to a **public** rule in the old remote survey. The first accepted PAINT_A/B **toggles this gate once** and appends an observable `GATE_TRANSITION` receipt. The action's receipt is not silently rewritten.
- A finite-state exhaustive reverse-BFS considers `(position, predecessor, base-3 paint status, gate latch)`. Labels are minimum remaining actions to actual terminal STOP with no wrong paint and quota satisfied. They are not heuristics based on blocked edges.
- Every selected clean case has a unique optimal BACKTRACK or MOVE_N first action and an independently replayed accepted alternative with **at least 2 extra actions** to terminal completion.
- For each clean case, a second episode records the *accepted suboptimal first move* in the initial history and then gives minimum-suffix repair actions. Thus all post-intervention moves/paint and eventual STOP are oracle replayed, not scripted. The shared early remote scan and the deliberate wrong-first action are explicitly scripted prerequisites.
- Eight clean cases per group: four BACKTRACK and four MOVE_N, with four initially open and four initially closed gates. Their current local observations, public goal/remaining budget, legal choices and last 64 history entries/positions match; older remote survey/goal information distinguishes them. Each clean case has a paired recovery episode.
- All samples have the `synthetic_host` label. There are **no model weight/adapter updates, GPU/API calls, or native CyberSand control claims**.

## Full deterministic corpus

| Split | World groups | Clean / recovery episodes | Labelled actions | Unique complete inputs | Duplicate inputs, zero weight in TRAIN | Strict causal clean episodes |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 64 / 64 | 1,984 | 1,744 | 240 | 12 |
| DEV | 4 | 32 / 32 | 1,148 | 1,148 | 0 | 10 |
| HELDOUT | 4 | 32 / 32 | 1,320 | 1,320 | 0 | 8 |
| **Total** | **16** | **128 / 128** | **4,452** | **4,212** | **240** | **30** |

All **256/256** synthetic oracle/recovery episodes terminated successfully. There are **256** public gate transitions, **512** expert paint operations, **0** expert erasures on the selected minimum-suffix trajectories, and **128** accepted suboptimal first-move interventions. The zero expert erasures are a useful negative result: v26 does **not** add wrong-paint correction examples, even though the action and local test support ERASE. V19 remains the correction corpus.

## Causality and leakage audit

To distinguish an actual causal gate from decorative gate-event noise, a separate **frozen-initial-gate** planner was evaluated at every post-trigger state along each clean oracle suffix. These frozen-gate recommendations are controls/diagnostics and are **not model inputs**. Compared against the actual dynamic solver:

| Split | Clean episodes with at least one strictly harmful frozen-gate recommendation | Frozen-gate illegal recommendations | Legal frozen-gate choices with strictly higher actual suffix cost |
|---|---:|---:|---:|
| TRAIN | 12/64 | 3 | 37 |
| DEV | 10/32 | 2 | 38 |
| HELDOUT | 8/32 | 0 | 38 |
| **Total** | **30/128** | **5** | **113** |

The remaining **98 clean episodes are negative causal controls**: gate toggling happened but freezing it produced no strictly harmful choice on the particular oracle trajectory. Do **not** report all 128 cases as causally discriminative. The first-action comparison is balanced within each family and gate-initial condition; initial gate status alone is not a label predictor.

World identity hashes and geometry are disjoint across TRAIN/DEV/HELDOUT. All complete model-input SHA-256 values were checked across partitions; no cross-split exact overlaps or conflicting action labels were found. TRAIN duplicates (240) are retained for trace integrity with `training_weight=0`. Distinct geometries share a small topology grammar, so the split is not proof of unconstrained native generalisation.

## Reproducibility

- Generator `causal_gate_v26.js` SHA-256: `b64925f57b95a20fc1562748a25ba24d7a6cdc01d11787413544aa26451bdd1c`.
- Test `test_causal_gate_v26.js` SHA-256: `6f172afef2e6e6a9f4da90c43151d626c873aea971bfb71ad9fcb60a53275b8b`.
- Full generated manifest SHA-256: `4425b93d05469a2dec1d2211a2b49c81e9417af8343424276b6632ae23c76bb2` (pins six JSONL shard hashes and dependencies).
- Two full default generations on 17 November 2026 were **byte-for-byte identical** across all six shards and the manifest. Observed generation duration 31.16 s / 29.29 s; peak RSS 535,992 / 468,300 KiB, respectively. These measure **generator CPU and memory**, not an SLM.
- `node test_causal_gate_v26.js /path/to/generated` passed all targeted tests plus complete 256-episode observation reconstruction, decision hash and shard verification. Local test duration 25.85 s; peak RSS 545,072 KiB. Targeted suites replay the alternative first move and prove its measured terminal-cost regret.
- Compact consumption contract: reconstruct observation by appending each `events_added` and `positions_added` to the episode's `initial_history`; combine with each decision's `state` in `observation_keys` order and validate `model_input_sha256`. **Never supply target, receipt, first-action regret, world/difficulty metadata, counterfactual results or episode mode as model inputs.** Respect TRAIN weights, and select only on DEV.

## Failure modes and limits

No actual LFM/Sprocket/other named SLM has performed this benchmark; no real model latency/RSS or native game observation/receipt was measured. Root checkpoint availability remains the consumer-side blocker according to the trainer-lane handoffs. Only 30/128 clean episodes measure a strictly consequential gate change on the expert suffix; a tuned model could succeed through easier shortcuts in the rest. Route topology remains v25's small orthogonal grammar, material semantics remain synthetic A/B, and the remote scan/first-error prefix are scripted. No illegal oracle moves or accepted protected-cell incursions occurred. No immutable GitHub Actions corpus artifact yet; regenerate shards outside Git from the pinned source.

## Consumer experiment

Train an actual pinned SLM on TRAIN, select once on DEV, then evaluate HELDOUT in closed loop for **entire game-like trajectories**. Stratify completion, correct paint quota, premature STOP, false BACKTRACK, recovery and realized suffix regret by (i) strict causal gate cases versus negative controls, (ii) initially open versus closed, and (iii) clean versus wrong-first recovery. Compare full old-survey memory, latest-64, frozen-gate and exact symbolic solver baselines. Report raw illegal preferences, accepted host actions, p50/p95/p99 inference latency and peak model RSS. Do not promote or merge based on oracle data alone.