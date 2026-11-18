# DRIV3R Corpus Factory v47 — historical material law × mission memory

**Date:** 2026-11-18. **Scientific status:** independent synthetic-host full-trajectory replay PASS; genuine named-SLM research NOT performed. This dataset is **not native CyberSand**. Research is data-only on existing DRIV3R Draft/HOLD PR #1.

## Research hypothesis and improvement over v46

V46 tested old superseded mission receipts and delayed remote edge obstructions but all paint commands preserved one fixed physical meaning. V47 tests **the interaction of delayed mission memory, remote navigation, partial survey, material-law changes, and corrective painting** in a closed-loop synthetic host. The material command which physically applies A/B can reverse while the physical target and legal menu remain unchanged. The model receives only historical `TOOL_LAW` receipts, not a current tool-mode flag. These are symbolic receipts and can be decoded by simple rules; they are *not* evidence of SLM-level reasoning.

Each v46 split-isolated world is reused in its **same original split**. Eight matched host continuations per world: old-mission positive, remote-gate positive, material-command positive, corresponding combined mission and law contrast, stale-mission control and stale-law control. Preparatory survey and event ageing consist of accepted actual actions on the unchanged synthetic v44 graph host. An independent synthetic adapter translates PAINT command A/B under `NORMAL` or `SWAPPED`, preserves client-facing ACKs, and emits a new law receipt after successful first-objective painting. Some worlds flip the law at the midpoint; others do not.

At the initial route fork, the latest relevant mission assignment is at least **256 accepted movements** old and latest material-tool law receipt at least **80 accepted movements** old. At the matched first painting checkpoint, two versions with identical physical state, recent action receipts, legal menu, mission, surveyed topology and painted state differ **only in historical tool-law receipts**, yet require opposite `PAINT_A` versus `PAINT_B` labels. The law-ablated matched-pair conditional accuracy upper bound is 50% under balanced paired evaluation. Obsolete historical law changes are checked as explicit same-label controls. Paired route forks separately enforce mission and remote-gate causality; both route alternatives remain legal.

## Complete qualified results

| Split | Worlds | Successful host episodes | Supervised decisions | Worlds with mid-episode law flip |
|---|---:|---:|---:|---:|
| TRAIN | 48 | 384 | 7,814 | 21 |
| DEV | 16 | 128 | 3,110 | 7 |
| HELDOUT | 16 | 128 | 4,067 | 11 |
| **Total** | **80** | **640** | **14,991** | **39** |

Independent audit (no v47 teacher/rollout/action-application calls): **14,991/14,991** exact policy-input snapshots and independent observed-map/action labels matched; **640/640** synthetic host replay episodes succeeded. **80 each** causal historical mission, remote gate and tool-law matched contrasts; **80 each** stale-mission and stale-law controls. **640 ERASE**, **796 `PAINT_A`**, **484 `PAINT_B`** accepted commands. No cross-split topology/input overlap or conflicting input labels. All 7 output files including manifest were byte-identical across two clean full generations (12.35 and 12.14 sec; peak generator RSS about 291 MiB). Independent complete audit took 18.38 sec with peak about 778 MiB RSS. These performance numbers are *corpus-tooling costs*, NOT LFM/Sprocket inference metrics.

### Reproduction

Use Node 22+ (built-in Node modules only). From `experiments/driv3r/corpus_factory/`:

```bash
node lawful_mission_v47.js /tmp/v47-a
node lawful_mission_v47.js /tmp/v47-b
node test_lawful_mission_v47.js /tmp/v47-a /tmp/v47-b
```

The generator's pinned `V47_DEFAULT_MANIFEST.json` is a copy of the full generated `manifest.json`, including SHA-256 of every input source and each data shard. Never reassign v46/v47 source worlds to different splits. Parent host and policy scripts are retained unchanged and verified by exact SHA-256. All six JSONL files are **generated data**, not intended for Git; use external retained artifact storage. A source-smoke CI pass is not a model evaluation or proof of artifact retention.

### Consumer protocol

Only `input` is a model observation. `action` and `weight` are supervision and must not be present in input; `*_challenge_EVALUATOR_ONLY.jsonl` contains answer keys/identities and must be inaccessible during training or model selection. Use TRAIN only for adaptation, DEV for choice, and one frozen HELDOUT evaluation after target SLM checkpoint pinning. Group/deduplicate jointly by v46 `source_world_index`/`source_seed` because this dataset shares v46 geometry, with no train/heldout reshuffling. Compare (a) durable history, (b) tool-law receipt ablated, (c) mission receipt ablated, (d) gate bulletin ablated, (e) last 32 action receipts only, (f) stale receipt controls. Report separately first material action correctness, two-stage physical paint success, first-move mission/gate accuracy, regret, recovery after wrong paint, invalid raw preferences, p50/p95 **actual SLM** CPU forward latency, and peak RSS. Evaluate **autonomously in closed loop**, not merely teacher-forced next-token accuracy.

### Negative results and limitations

- This is a **synthetic graph host plus adapter**, not native CyberSand's observation/action ABI, dynamic materials or physics. Neither native CyberSand nor Neurounner was modified.
- History and survey preparation is scripted though host-accepted; exogenous material-law transitions are synthetic. Event codes and their order are explicit, so a symbolic parser still solves the action labels.
- Source worlds are intentionally shared with v46 **within the same split**; these versions are not independent HELDOUT suites. Strong performance on v47 alone does not establish transfer or native game progress.
- The corpus only supplies teacher data and evaluator challenges. New LFM350/Sprocket full-model forwards, optimizer updates, native episodes and neural latency/RSS are **zero**. Consumer labs report unavailable pinned model binaries and missing opt-in CPU dispatch here.
- Large shards are local and not yet retained as immutable GitHub Actions artifacts. Do not claim an Actions custody link until its exact artifact and hashes are verified.

**Boundary:** only the five source/report/handoff files for v47 in this lane may be committed to the existing `research/import-consolidated-worm-labs` branch; PR #1 stays Draft/HOLD. No main/other lab/game writes or paid work.