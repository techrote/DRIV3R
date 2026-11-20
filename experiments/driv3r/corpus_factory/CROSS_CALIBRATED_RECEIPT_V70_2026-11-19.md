# DRIV3R Corpus Factory v70 — physically cross-calibrated, out-of-order remote receipts

**19 November 2026.** DATA-only lane; Draft/HOLD corpus-factory programme, `techrote/DRIV3R` PR #3. **Synthetic host extension, not native CyberSand; no learned neural policy was evaluated.**

## Research question and improvement over v69

The v69 controller could trust a timestamped material sensor. v70 makes the **raw sensor readings ambiguous**: each of two physical-surface sensors can be stable truthful or stable inverted. The historical calibration comes from a **co-located local measurement after a host-accepted `PAINT_A` / `PAINT_B`**, when the physically painted cell is present in the normal local observation. Only the report value is emitted, not a privileged `inverted` tag.

Later, each material either has **no remote report** or has a pair of **authentic, out-of-order, source-time-stamped reports**, the most recently delivered being the *older* physical measurement. Remote erosion occurs during a 112-action legally executed detour, long after the grounding calibration and well before the critical route choice. Model-visible current observations are identical across all arms in the same world. Material goals remain physically checked after the continuation.

The agent must combine earlier calibration evidence, source timestamp, material-specific shock history, local inspection when reports are missing, and legal route planning. There are no direct generator-provided sensor-mode booleans in the model input.

## Qualified deterministic synthetic-host corpus

| Split | Exclusive worlds | Counterfactual episodes | Independently replayed host actions | Naive arrival-only failures |
|---|---:|---:|---:|---:|
| TRAIN | 6 | 1,152 | 175,088 | 294 |
| DEV | 3 | 576 | 89,288 | 147 |
| HELDOUT | 3 | 576 | 93,544 | 147 |
| **Total** | **12** | **2,304** | **357,920** | **588** |

All 2,304 conservative teacher demonstrations physically complete their synthetic goals; 16-arm calibrated/report-availability sets are nested in each 192-arm world group. There are **72 matched raw-report collision groups** in which identical raw remote readings permit different correct first actions only after incorporating older physical calibration. All 12 world geometries are disjoint from the 8 v69 geometries, and splits are disjoint.

Control-only measurements: blindly following the last-arriving raw report fails **588/2,304 (25.52%)** episodes; history-aware conservative symbolic continuation succeeds **2,304/2,304**. Mean teacher savings against unconditional A+B reinspection: **3.6528** actions per episode. Mean additional actions over a physically privileged oracle: **3.5972**. These are **deterministic controls, not SLM neural scores**. The fully trusted symbolic controller remains easy enough to write; distinguishing it from an actual learned recurrent model is an outstanding experiment.

## Data and leakage contracts

* The only learner-consumable split is `TRAIN_cross_calibrated_receipt_stream.jsonl`. DEV, HELDOUT, and every `*_certificates_EVALUATOR_ONLY` file must never be used in gradient updates.
* All scripted calibration, painting prelude and 112 detour actions have zero supervised weight. The critical action is weighted; repeated TRAIN follow-up actions are masked by SHA-256 of identical `(input,action)` pairs.
* Calibrations are physically grounded within each source history; hidden sensor inversion bits and physically eroded targets appear only in evaluator-only certificates. The independent auditor rederives calibration from the **observable local sensor reading and actual co-located painted material**, never from a privileged policy feature.
* One geometry per split owner; `geometry_sha256` and exact source-world indices are included. TRAIN, DEV and HELDOUT contain disjoint geometries and disjoint source seeds. v69–v70 geometry overlap is zero.
* All action and input receipts are full source-host reconstructions; negative control forks and privileged-oracle travel differences are scorer-only.

## Reproduction and integrity

Source generator SHA-256: `aa742b764b8ad05df3a252c6fc1dae9a9b1406959e0c94cdbe0fb312558a64b1`.

Independent auditor SHA-256: `d5a89d8b43087955ceae0dcb46d680bca7780ec2b03ed7a1109786ceb7411fc9`.

Host SHA-256: `14d27b999d5eee0a83fe7e5afd9e85ce29670bdd4320024d3465705f7b29f8b6`.

Parent physical extension SHA-256: `ebac51b0be04ae2a05a1f8c7af8c7003152ef604031e2152342dd5f057dc85bd`.

Prior observation extension SHA-256: `c9d2cd11b8ae4de0ebb8ff96ead0b3de1d88b7810aa5f18675eca736ef7423db`.

```bash
node experiments/driv3r/corpus_factory/cross_calibrated_receipt_v70.js /tmp/v70corpus
node experiments/driv3r/corpus_factory/test_cross_calibrated_receipt_v70.js /tmp/v70corpus
```

The generator requires an absent output directory. Reuse the exact committed parent-source hashes shown above. The independent auditor checks file content SHA-256 and byte length, split/sequence/mode allocation, 357,920 host-accepted actions, locally grounded sensor calibration, actual physics after disturbances, chronological remote reads, learner observations, continuation legality, label provenance, masked actions, counterfactual group completeness, and cost-control metrics. SHA-bound `manifest.json` is also pinned as `V70_DEFAULT_MANIFEST.json`. **Full independent replay PASSED**: 2,304/2,304 terminal outcomes and 357,920/357,920 host-accepted actions. Two full generations produced byte-identical outputs; changing an evaluator calibration certificate while updating its manifest hash was rejected by the independent chronological audit. Full replay is local qualification; routine GitHub CI is source-smoke only.

## Important limitations and consumer handoff

This is still a **synthetic v44 graph-host test**. Local synthetic sensor-check and inversion semantics, remote source-stamped receipts, and synthetic physical erosion are **not established CyberSand game ABI features**. Calibration is perfectly stable during each episode and source timestamps are assumed truthful. Most prefixes are host-scripted, while only conservative continuations are oracle-labelled. There are just 12 new geometries; claims of general neural transfer or native performance are unwarranted. No named SLM weights or LoRA adapters were trained in this data lane, no autonomous neural HELDOUT goal completion or CPU latency/RSS was measured, and generated data are not yet an immutable Actions artifact.

**Next consumer experiment — separate PR #2 only:** CPU fine-tune a pinned actual target SLM/adapter on the TRAIN stream only, with prefix loss masks and context long enough to retain physical calibration. Run autonomous HELDOUT completion and travel cost against (1) complete-history / source-time, (2) truncated-to-postcalibration history, (3) arrival-only and (4) unconditional reinspect baselines. Measure p50/p95 CPU action latency, peak RSS and exact checkpoint hashes. Focus on **zero lost material objectives**, then travel savings. Keep diagnostic certificates sealed and do not turn this data lane into a model-training programme.
