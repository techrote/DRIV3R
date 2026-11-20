# DRIV3R Corpus Factory v74 — causal-memory decision pack (19 Nov 2026)

**Status:** synthetic-host DATA product; local source-matched export and independent physical-parent audit PASS. Not a trained policy, native CyberSand test, model-score improvement or GitHub Actions artifact. Keep programme PR Draft/HOLD.

## Why this increment

V67–v73 added physically executed erasure, inverting channels, common faults, costly active witnessing and nontrivial temporal memory. Their full host streams were correct but a poor direct SLM training interface: v73 is ~306 MB of repetitious trajectories, with evaluation certificates carrying oracle costs and hidden outcomes. Trainer workers must not accidentally prompt an SLM with those receipts or poison HELDOUT.

V74 creates **model-consumable legal-action decisions** from independently qualified v73 *without changing physical host semantics or pretending to generate new worlds*. It retains the earliest physical coupon, tariff, channel calibration and subsequent report-history evidence as observed input only; certificates, hidden fault/erosion outcomes, oracle tables and future observations are never included in model messages. It supervises the first active-witness action, the second choice after an A witness (needed to distinguish A-only from BOTH), and the subsequent remote-recovery decision. Every target is an actual source-host-accepted legal action token, not an invented high-level action.

## Reproduce

From a complete v73 release, extract its `corpus/` directory to `PARENT`. Use the matching v73 generator and independent auditor from Git source. First verify v73 with `node test_endogenous_witness_host_v73.js PARENT`; then:

```bash
python3 experiments/driv3r/corpus_factory/memory_decision_export_v74.py PARENT NEW_V74_DIR
python3 experiments/driv3r/corpus_factory/test_memory_decision_export_v74.py PARENT NEW_V74_DIR
```

The exporter checks **all** v73 stream/certificate SHA-256s and each original episode's pre-action receipt hash before writing. It rejects an existing output directory and inconsistent full-memory labels. TRAIN exports only `TRAIN_messages.jsonl` (user observable input and assistant host-legal action). DEV and HELDOUT produce *evaluator-only* action pairs with sealed ablation inputs (`last32`, `current_only`, `no_tariff`, `no_calibration`). Do not train on DEV/HELDOUT or the parent certificates.

## Measured local results

| Split | Parent episodes | Pivots before dedup | Distinct legal-action messages | Repeated exact pairs removed | World geometries |
|---|---:|---:|---:|---:|---:|
| TRAIN | 512 | 1,112 | 684 | 428 | 4 |
| DEV | 256 | 592 | 360 | 232 | 2 |
| HELDOUT | 256 | 576 | 352 | 224 | 2 |

The independent evaluator grouped **raw pivotal examples** by exact current-only observation, then computed the accuracy ceiling of a deterministic modal-class controller (best single action per indistinguishable current state): **42.81% TRAIN, 56.76% DEV, 36.11% HELDOUT**. Every current-only equivalence class contains conflicting legal-action labels (TRAIN 4/4, DEV 2/2, HELDOUT 2/2). This is a counterfactual aliasing diagnostic on this small set of worlds, **not a neural inference result or a general information-theoretic ceiling**. Full-history exact-input labels have zero conflicts after dedup.

All 1,024 v73 parent episodes and 153,776 legal actions passed the original independent physical-host replay. The v74 independent auditor reconstructed every exported decision and ablation from its parent source, checked trainability allowlists and split-disjoint geometry, and passed. Two v74 exports were byte-identical. Two rehashed-manifest attacks (falsified TRAIN legal-action label, fabricated HELDOUT ablation) were rejected by independent source replay. No CPU model weights were updated; latency/RSS for neural inference are still unmeasured.

### Explicit limitations

This is an **offline derivative of v73**, not 8 new maps or a new physical scenario distribution. TRAIN has only four distinct underlying world geometries; some full trajectories share prefixes, and all reference telemetry is synthetic rather than established CyberSand-native semantics. `current_only` deliberately retains any durable state present in the current legal host observation, but not the independently accumulated action/event history. Ablation `last32` truncates historical events and actions, not persistent current host fields. The exported messages support *offline action accuracy*; closed-loop behavior must be evaluated independently by trainer-owned host execution with source-frozen checkpoints. The synthetic oracle labels are not proof of an SLM learning or executing them.

## Consumer handoff

Use pinned v74 TRAIN messages only to update an **actual named CPU SLM or adapter** in the trainer lane. Evaluate DEV first; seal HELDOUT and compare full-history vs last32/current-only/no-tariff/no-calibration. Then evaluate autonomous source-host episodes against always-reinspect, always-skip and the source teacher, measuring terminal goal retention, travel/action regret and CPU p50/p95/RSS. Do not transfer optimizer work, other directories, or GitHub workflow changes to this data lane. The next *new-world* Corpus Factory challenge should broaden TRAIN geometry and introduce shifted nonuniform fault/erosion priors with physically justified observable calibration (rather than adding another trivial symbolic flag).
