# DRIV3R Corpus Factory v59 — balanced observed-feedback intervention fork

**Date:** 2026-11-19. **Scope:** reproducible synthetic-host game-policy data only. No native CyberSand evaluation, no target SLM weights trained and no learned policy benchmark. Owned entirely by `experiments/driv3r/corpus_factory/**` in Draft/HOLD programme PR #3; main and other labs untouched.

## Problem / value

Previous v58 off-policy recovery demonstrations offered a shortcut: **every recorded continuation after the forced intervention begins with ERASE**. This is a defective discriminator for neural recovery, because a model can learn unconditional ERASE without interpreting the accepted physical paint result. v59 adds matched *correct* and *incorrect* host-executed B-goal interventions from the identical pre-intervention history. Both interventions are explicitly zero-weighted: they are external interventions, not teacher targets. The actual host's next observable state determines whether STOP or ERASE is correct.

- Correct B physical paint: the mission completes its second stage, emits physical paint and completion receipts; subsequent legal policy action **STOP**.
- Wrong B physical paint: the host remains at B stage and reports the physically incorrect material; subsequent actions **ERASE, correct PAINT, STOP**.
- Complete observation/action streams, all pre-intervention decisions and all physical receipts are preserved. The synthetic host is independently re-executed from start through terminal acceptance for every generated episode.
- TRAIN data contain no source identities, fork arm, counterfactual certificates or hidden laws; sidecar files for ALL splits are marked `EVALUATOR_ONLY`. DEV/HELDOUT contain no trainable shard.

## Results (local exact-source qualification)

| Split | Parent B fork origins | Correct episodes | Wrong-and-repair episodes | Successful synthetic episodes | Decisions | Duplicate TRAIN prefix decisions additionally loss-masked |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 64 | 64 | 128 | 3,540 | 1,578 |
| DEV | 24 | 24 | 24 | 48 | 1,448 | 0 |
| HELDOUT | 24 | 24 | 24 | 48 | 2,072 | 0 |
| **Total** | **112** | **112** | **112** | **224** | **7,060** | **1,578** |

All 224 post-paint interventions are zero-weighted; all 448 next-action/correction decisions are supervised (focus weight 6). Every correct arm's next decision is STOP (112/112); every incorrect arm's first decision is ERASE (112/112), followed by corrected PAINT and STOP. This produces a **50% ceiling for any fixed unconditional first continuation action** on the paired contrast set, versus 100% for the deterministic teacher. **Those figures are not neural-model performance.** The host may still be solved by simple observation parsing, especially because completed mission and locally painted material are explicitly observable.

Checks: parent v56 file hashes pinned; output manifests file hashes verified; `test_feedback_choice_v59.py` independently checks source references, equal pre-intervention histories, correct/wrong receipts, legal action masks, all 7,060 rows, EVALUATOR_ONLY isolation and 56 distinct world-geometry split owners. `test_feedback_choice_host_v59.js` independently replays **every one of 7,060 host-accepted actions**, matching the exact observation before each command, confirming 224/224 terminal successes. Two full generations byte-identical; single-byte upstream corruption fails closed. No source was changed in other labs.

## Reproduction

Inputs: the exact v56 dataset from its pinned release (parent manifest SHA-256 `b7f4b9f0d76416378c0d38713b4c60df90c149c2ef29533c73cb0e030ec9690e`), repository's v53-v58 synthetic-host modules with manifest-pinned hashes, Node.js 22 and Python 3.

```bash
node experiments/driv3r/corpus_factory/feedback_choice_v59.js /path/to/v56corpus /tmp/v59corpus
python experiments/driv3r/corpus_factory/test_feedback_choice_v59.py --source /path/to/v56corpus --output /tmp/v59corpus --modules experiments/driv3r/corpus_factory
node experiments/driv3r/corpus_factory/test_feedback_choice_host_v59.js /tmp/v59corpus
```

Do not train on `*_EVALUATOR_ONLY_*` records, even when the split is `TRAIN`; TRAIN policy is `TRAIN_receipt_choice_stream.jsonl` only. Keep all data partitions grouped by original world geometry and use HELDOUT only once for final reporting. The generator refuses an existing output directory and detects upstream hash changes.

## Limitations / next consumer

This is a **synthetic graph/material host**. Its explicit material observations, mission-complete signal, binary material law, and asymmetric irreversible A versus repairable B are not verified native CyberSand rules. The pre-intervention trajectories are inherited from v56 rather than new topology, and the continuations are deterministic oracle demonstrations. The 50% unconditional first-action control does **not** establish long-memory difficulty. The v59 dataset corrects v58's unconditional ERASE confound, but an observation-aware symbolic policy may still perform perfectly. No actual SLM training, neural CPU latency/RSS, autonomous held-out game trajectories, GPU spend or Actions binary artifact publication occurred.

**Next consumer:** train an actual pinned target SLM/adapter in its owned trainer lane, using v54 broad sequential TRAIN, v56 paired material-memory, and balanced v59 TRAIN (optionally v58), with both intervention actions zero-weighted. Test untouched HELDOUT with unconditional ERASE/STOP controls, receipt ablation, full sequence success, remote-memory ablation, p50/p95 CPU action latency and RSS. Investigate native ABI differences before game claims. Future Corpus Factory increments should introduce longer recovery chains and varying safe-tool dynamics rather than another 1-step symbolic benchmark.
