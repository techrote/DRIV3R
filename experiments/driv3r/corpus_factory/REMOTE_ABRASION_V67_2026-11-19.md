# REMOTE-ABRASION-067 — physical remote erasure, memory and cost-sensitive reinspection

**19 November 2026 · Data lane · DRAFT/HOLD PR #3 · synthetic host extension, NOT native CyberSand**

## Research question

The long-horizon trainer's 19 November BELIEF-UTILITY-010 negative result found that TRAIN-selected erasure belief collapsed to always-revisit: DEV 178/240 goals versus 154 for uncalibrated 0.5, while privileged oracle gained 15.65 mean utility. In response, this corpus isolates the missing causal evidence: calibration of durability using a legal local painted sample, a temporally separated sequence of remote material disturbances, and the action cost of inspecting one/two completed remote goals.

## Physical execution and observable protocol

Start from deterministic novel v44 topology. Paint a sample cell with host `PAINT_A`, physically stress its stored painted state under two law thresholds (erodes after 1 or 2 hits), expose local response in the subsequent legal movement observation, and reset the sample via an explicitly declared **synthetic fixture** intervention. Then navigate, paint A at goal and B at goal2 using host-accepted actions. Return to a remote start, experience 0–3 distributed external abrasion events affecting the actual host `painted` map, move through 60+ additional host-accepted distractor steps, then decide STOP or cost-bearing travel to inspect/repaint missing A/B. All post-critical painting and STOP decisions are independently host-replayed; all 480 completed episodes succeed. The independent auditor additionally forks the exact pre-critical physical state for both immediate STOP and inspect-both baselines and verifies their action counts and outcomes instead of trusting generator certificates.

Model input hides the global `painted` table beyond Manhattan radius 1 but preserves current local state, legal action menu, geometry, movement receipts and incremental abrasion events. Thus no privileged remote painted state appears at the critical decision. `LOCAL_ABRASION_TEST`, `COUPON_RESET`, `DISTANT_ABRASION`, `BACKGROUND_VIBRATION` are **invented synthetic telemetry**, not CyberSand native ABI fields. The fixture cleanup and external abrasion are declared synthetic host interventions, not accepted user paint actions.

## Corpus and controls

| Split | Distinct geometries | Crossed episodes | Replayed legal decisions | STOP-only successes | Mean redundant actions of inspect-both |
|---|---:|---:|---:|---:|---:|
| TRAIN | 12 | 240 | 33,220 | 72 | 7.033 |
| DEV | 6 | 120 | 17,390 | 36 | 9.533 |
| HELDOUT | 6 | 120 | 17,147 | 36 | 7.392 |
| **All** | **24** | **480** | **67,757** | **144** | **7.748** |

**All 20 arms per geometry have byte-identical instantaneous critical observations**, but require distinct first actions (STOP, route to A, route to B). Every arm has the same three event slots, delayed beyond local recent-action history. Both law thresholds cross 10 historical shock patterns including repeated and alternating A/B shocks. Models must retain the local measured durability and count material-specific remote disturbances; the last event alone cannot determine the correct action.

The following are **post-hoc evaluator-only majority-label upper bounds on first-action accuracy**, computed within matched geometry groups, not neural or trainable predictive model accuracy. They use evaluator labels and must not be used to tune on HELDOUT:

| Visible historical information | TRAIN | DEV | HELDOUT |
|---|---:|---:|---:|
| Instantaneous current input | 47.9% | 46.7% | 46.7% |
| Most recent abrasion only | 67.9% | 66.7% | 66.7% |
| Early physical calibration only | 60.0% | 60.0% | 60.0% |
| Early calibration plus last abrasion | 70.0% | 70.0% | 70.0% |
| Complete calibration plus event counts | 100.0% | 100.0% | 100.0% |

STOP-only success is **30.0%** across all splits. Always inspect both remote goals succeeds **100%**, but wastes an average **7.748 legal actions per episode** compared with the labelled history-aware reinspection, under the synthetic benchmark's action-count metric. No inference latency, CPU model RSS, or neural policy success is claimed.

## Quality gates and reproduction

- Independent replay reconstructs host observations before **every legal action**, applies abrasion to the true synthetic painted world, independently validates terminal success, and verifies the delayed observation, action legality, weight masks and local-only painted view.
- Every 20-arm matched group shares an identical current critical observation and contains at least three distinct correct first actions; 24 geometry SHA-256 values are disjoint across TRAIN/DEV/HELDOUT and from 16 separately qualified v65 geometries. Deterministic source-index ranges also differ from historical v63/v64.
- Two full generations are byte-identical; an intentionally corrupted TRAIN stream was rejected by the independent manifest-hash gate. All 960 baseline forks (480 STOP + 480 inspect-both) were executed by the independently written replay, alongside 480 successful teacher episodes.
- Scripted calibration, navigation and disturbance-preparation steps are zero loss weight; post-critical continuation actions are supervised. All DEV, HELDOUT and certificates are evaluator-only. Generated JSONL lives outside Git; only generator, tests, compact manifest, report and lane handoff are committed.

```bash
node experiments/driv3r/corpus_factory/remote_abrasion_v67.js /tmp/driv3r-v67
node experiments/driv3r/corpus_factory/test_remote_abrasion_host_v67.js /tmp/driv3r-v67
```

## Limitations and consumer handoff

This is v44's synthetic grid host with an explicitly added paint-abrasion intervention and local-view filter. It is not native CyberSand game execution or independently confirmed CyberSand material physics. Deterministic thresholds can still be solved by a small symbolic event counter. Full static graph topology is known, and paths are teacher-forced rather than autonomously neural. No LFM/Sprocket weights were updated in this data lane. No paid compute, native observations, neural latency/RSS or immutable Actions corpus artifacts. The next separately owned trainer experiment should train an actual pinned CPU SLM on TRAIN only and measure autonomous DEV and sealed HELDOUT closed-loop utility vs always-revisit, current-only, last-event and memory-reset controls, with model hashes and inference RSS/latency. Retain Draft/HOLD.
