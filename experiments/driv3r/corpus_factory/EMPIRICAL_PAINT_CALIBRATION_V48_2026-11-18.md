# DRIV3R — Corpus Factory v48: accepted-action paint-outcome calibration

Date: 2026-11-18. Scope: `experiments/driv3r/corpus_factory/**` only. Research is **synthetic game-like host**, not native CyberSand. Do not represent as trained SLM game capability.

## Question and causal intervention

v47 provides a literal `TOOL_LAW` mode receipt. A one-line event parser can translate that into the correct paint command. v48 replaces this with **two genuine synthetic-host paint/erase calibrations** at the starting tile. The host physically applies `PAINT_A` using the active mapping; the observed resulting material (`A` or `B`) is saved as a timestamped `PAINT_RESULT` receipt. `ERASE` is then host-accepted, restoring the tile. The first probe is deliberately obsolete and separated from the second by 16 accepted movements and an irrelevant diagnostic. The second probe is followed by another 80 accepted movements; at decision time the model cannot see its effect in the current local tile, recently accepted commands, or current painted state. To choose the correct command for material A/B the policy must use the most recent physically observed result, not the obsolete probe or an oracle law code.

Per world, eight matched arms vary the current tool mapping, obsolete calibration, final vs stale mission, and remote gate status. Probe-positive and obsolete-probe-negative first paint checkpoints have otherwise identical model-visible inputs after PAINT_RESULT ablation. The material-positive first commands differ, the obsolete negative commands agree. Mission and gate navigation counterfactuals are preserved. Four naturally distinct split-preserving challenge types are evaluated (mission, gate, paint outcome, obsolete observation), with stale mission controls additionally retained.

## Reproducible source and leakage protections

- Parent source: unchanged v46 worlds and splits; TRAIN 48, DEV 16, HELDOUT 16 worlds, with eight arms per world. Never mix or relabel source v46/v47 worlds across splits. Group training by `source_world_index` + `source_seed` + `split` if combining v46/v47/v48.
- The host executes exactly 1,280 calibration `PAINT_A` and 1,280 `ERASE` commands; they are part of scripted accepted-action preludes and **not included as policy training decisions**. The emitted PAINT_RESULT is physically measured from the host's material cell state after each command. No `TOOL_LAW` event enters v48 model inputs.
- Trainer input is only the `input` field of the split's `_policy.jsonl`; `action` is the supervision label and `weight` the loss weight; evaluator-only challenge JSONL is sealed. Each TRAIN/DEV/HELDOUT input is checked for duplicate contradiction and cross-split collision. All shipped metadata contains source hashes and scenario provenance.
- All policy labels are computed using *observable* partial graph and historical information (v46 reference, material instruction reconstructed from PAINT_RESULT). The independent audit uses the v46 independent planner plus a separately implemented paint-law interpretation and separately implemented host replay; does not call v48 teacher/rollout/accepted/scenario. Every saved policy decision is compared with an independently reconstructed observation, legal menu and applied as an accepted synthetic-host command. Detailed spatial-observation validation is additionally performed on the first, last, every sixteenth and every PAINT decision of each episode.
- Two full generations resulted in byte-identical manifests and all six JSONL shards. Generation A: 12.63 s, 291,096 KB peak RSS. Generation B: 12.49 s, 273,300 KB peak RSS. Combined full independent audit + deterministic hash comparison: 15.40 s, 842,548 KB peak RSS. These are **corpus/tooling figures, not neural inference**.

## Verified measurements

| Split | Worlds | Successful episodes | Labelled policy decisions | Probe-positive pairs | Obsolete-probe controls |
|---|---:|---:|---:|---:|---:|
| TRAIN | 48 | 384 | 7,814 | 48 | 48 |
| DEV | 16 | 128 | 3,110 | 16 | 16 |
| HELDOUT | 16 | 128 | 4,067 | 16 | 16 |
| **Total** | **80** | **640** | **14,991** | **80** | **80** |

Independent audit: **14,991/14,991** host-applied and independently action-labelled; **640/640** successful synthetic terminal episodes. **80** positive mission comparisons, **80** gate comparisons, **80** physical-outcome comparisons, **80** stale mission controls, and **80** obsolete physical-probe controls. Model-supervised commands include 640 PAINT_A, 640 PAINT_B and 640 ERASE, exclusive of the 2,560 preparatory calibration commands. No detected cross-split geometry/input overlap or contradictory label. No TRAIN duplicates requiring zero loss weight for this generation.

## Negative results, boundaries, and consumer handoff

- v48 still admits a symbolic shortcut: the latest physical PAINT_RESULT is explicit, even though obtained from an actually executed action rather than a privileged TOOL_LAW notification. There is no native rendering/simulation/material sensor noise, and it is **not** evidence that a model learned tool calibration.
- The script chooses when and where to probe. The model is **not** supervised to initiate calibrations; this is a fixed-law-within-rollout challenge, not a successor to v47's within-mission change events. v47 remains valuable as a separate stage-flip benchmark, and v43 remains useful for multi-domain curriculum.
- The partial graph and protected-cell semantics remain synthetic. Neither native CyberSand nor an SLM was executed. No model optimizer steps, frozen forwards or measured model latency/RSS; no GPU or paid compute.
- Other lab LATEST_HANDOFF reports reviewed in this run: `lfm_game_policy` has archived full LFM350 evidence but no locally pinned root for new forwards and no Actions dispatch; `sprocket_game_policy` has no pinned Sprocket 500M root/tokenizer locally, missing dependencies and no dispatch. Their blockers do not authorize an auxiliary-training substitute here.
- For an actual target-model consumer: compare v47 and v48 on topology/group-disjoint held-out episodes, ablate recent/old physical PAINT_RESULT receipts and stale probe receipts, report first paint material correctness, terminal host success, raw illegal model preferences, p50/p95 CPU forward latency, peak RSS and actual checkpoint/model hashes. Adapt target LFM350/Sprocket weights only in the respective trainer lane with pinned assets and opted-in CPU execution.
- Generated dataset shards are retained in this release ZIP; **GitHub Actions immutable artifact retention remains outstanding**. No CI training workflow is installed or dispatched. Git retains only compact source/test/manifest/report/handoff.

**Authority:** CorpusFactory DATA ONLY, DRIV3R Draft/HOLD PR #1. No edits outside this owned directory; no main merge, no CyberSand/Neurounner write.
