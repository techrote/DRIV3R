# RECEIPT-RECOVERY-008 — learned receipt-conditioned repair, with topology transfer

**19 November 2026.** Research-only synthetic CyberSand-style worlds; NOT the official host ABI or native acceptance. Evaluated from DRIV3R Draft/HOLD long-horizon programme. This is an explicit, bounded CPU experiment; nothing runs on PR or push by default.

## Scientific question

Can a small genuinely optimized policy distinguish three contemporaneously observed situations — a rejected/wrong paint material, an epoch that erases a previous painted objective, and an irrelevant epoch — and recover from earlier mistakes over 36, 72 and 128 decisions? Is trained feedback conditioning actually more useful than a literal receipt parser?

The independent DEVELOPMENT split moves passage geometry and adds blocked cells. Host-legal actions include four-way movement, PAINT, ERASE, STOP. The synthetic host injects one wrong-material paint and optionally an earlier-target erasure; a 75% material-change / 25% irrelevant-epoch mix creates a costly false-positive control. A temporary geometric obstruction also forces route replanning. All trajectories are evaluated on achieved final material goals, not local stage counters.

TRAIN: 64 synthetic seeds (1100–1163), 768 balanced receipt-label examples per learner. Two six-parameter logistic classifiers, 1,000 CPU full-batch optimization steps each. DEVELOPMENT: 24 disjoint seeds (3100–3123). Five paired controllers, four horizon/censor conditions, 480 episodes. Structural HELDOUT remained sealed. These balanced TRAIN labels are synthetic imitation targets, not proof that real gameplay has such receipts.

## Exact DEVELOPMENT results

| Controller | Goals / 96 | Mean utility | Planner calls |
|---|---:|---:|---:|
| Direct observed-receipt rule | **51** | **368.0417** | 526 |
| Trained receipt-conditioned classifier | **51** | **368.0417** | 526 |
| Trained, feedback features removed | **51** | 364.9271 | 549 |
| Always repair on any alert | **51** | 364.9271 | 549 |
| Blind (ignore repair receipts) | 3 | -116.9479 | 461 |

The trained model perfectly matches the literal receipt rule on all 96 paired DEVELOPMENT cases. It **does not beat the simple rule**. The no-feedback ablation wastes 23 extra planner calls, losing mean utility by 3.1146 but without losing goals; it shares the always-repair behaviour. The blind controller loses 48 goals and suffers 56 paired utility losses, despite 16 cheap but unsuccessful utility wins. The learned policy's training-label agreement was 100%; no-feedback agreement was 50%.

**Censoring is decisive:** goals with the receipt rule are 7/24 at horizon36/cap36, 20/24 at horizon72/cap64, **0/24** at horizon128/cap64, and **24/24** in separately labelled synthetic shadow128/cap128. Never report shadow128 successes as native 128-move CyberSand achievements.

## Qualification, provenance and limitations

The exact local Python source (SHA-256 `39200da27ff7a135e1fa7216b4d8531c57b7ee5b19d2e6c2b402efb0ca3fc3ac`; Git blob `36af759d52e1834998727433640ca4772a08ebb4`) passed Python bytecode compilation. Explicit execution generated **480 episodes and 29,262 host-legal action receipts**; each receipt stream was checked by rebuilding a new host from the recorded action sequence, plus independent deterministic repeat of each policy rollout. Tampered illegal-action evidence was rejected.

Measured locally (Python 3 process, not Node or native): 680 timed repair-classification calls; p50 **0.681 µs**, p95 **8.533 µs**, p99 **12.939 µs**; aggregate run **2.819 s**; process max RSS **114,952 KiB**. These measurements include Python interpreter overhead and are not end-to-end model or CyberSand runtime latency.

The receipt auditor shares the host implementation with the simulator, so it is **same-source replay, not cross-implementation validation**. The global material-change receipt, wrong-material status, temporary obstruction and protected-cell rules are research assumptions and are **not verified official CyberSand ABI fields**. No real native sessions, paid GPUs, external language models or checkpoint binaries were used. The trained six-scalar coefficients are preserved as text in the evidence ledger, not a deployable checkpoint.

A full evidence JSON and nine-panel SVG are generated on explicit invocation; process timing and RSS fields are run-specific, so their complete output bytes are not deterministic:

```bash
python experiments/driv3r/training/long_horizon/receipt_recovery_008.py /absolute/output/dir
```

Committed compact evidence preserves action distributions, exact source hash, paired negatives, latency/RSS, and representative receipt fingerprints. The historical predecessor experiments and negative cases were not changed.

## Decision

**Retain the literal receipt rule as the conservative control.** Learned signal detection can match it in this deliberately observable benchmark, but no marginal gain justifies deployment. The next useful step is a harder task where feedback is incomplete, stale or partially ambiguous, with real movement/material observation and checkpoint persistence. Before any native-facing claim, independently validate the host action/observation contract against CyberSand read-only source and receipts.
