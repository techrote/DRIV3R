# DRIV3R Corpus Factory v66 — chronological sensor recalibration

**19 November 2026. Data-only synthetic-host experiment.** Scope: `experiments/driv3r/corpus_factory/**`. Physical motion and routes are certified by the existing v65/v64 source-host chain; v66 changes only model-visible sensor telemetry, not native CyberSand. Not trained model performance.

## Scientific question and regression addressed

v65's *earliest* historical sensor diagnostic correctly inverted the newest report in 100% of its pivotal cases. v66 makes the sensor change mode once more, **either before or after** the host-measured gate revision. The latest and first diagnostic codes are identical across both schedules, but their causal applicability differs. Applying the first diagnostic to every report or applying the last diagnostic retroactively each fails half the pivotal cases. A measurement-time-aware rule succeeds on this restricted synthetic benchmark. The extra diagnostic is a synthetic event, not a native game ABI signal.

No privileged certificate metadata enters `steps.input`. Only prior observable events and accepted actions are accessible. The late diagnostic is at least 16 host-accepted steps before the critical action, and each world's 64 arms share the exact instantaneous critical input and legal menu while requiring both route labels. TRAIN has counterbalanced causal arms and redundant detours carry zero supervised weight.

## Reproduction and qualification

```bash
node experiments/driv3r/corpus_factory/sensor_fault_v65.js /tmp/v65
node experiments/driv3r/corpus_factory/test_sensor_fault_v65.js /tmp/v65 /tmp/v65.v64_source
node experiments/driv3r/corpus_factory/temporal_calibration_v66.js /tmp/v66 /tmp/v65
node experiments/driv3r/corpus_factory/test_temporal_calibration_v66.js /tmp/v66 /tmp/v65
```

The generator refuses an existing destination and checks SHA-256 of every parent dataset file before transforming it. The independent v66 auditor does not import v66 generator logic: it reconstructs each event transformation and validates every model input, legal action, loss mask, critical input hash, 64-arm causal contrast, split isolation and aggregate metric. The parent v64 audit independently replays true host physical actions; v66 does **not** claim a new independent physical execution, since the modified sensor stream does not affect executed actions.

| Split | Worlds reused from v65 | Episodes | Model-input/action decisions | Pivotal cases |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 8 | 512 | 35,232 | 256 |
| DEV | 4 | 256 | 33,216 | 128 |
| HELDOUT | 4 | 256 | 48,640 | 128 |
| **Total** | **16** | **1,024** | **117,088** | **512** |

Controls on pivotal critical labels: first diagnostic only **50%**, last diagnostic only **50%**, diagnostic effective at gate *measurement time* **100%**. Controls are symbolic deterministic label reconstructions, **not neural accuracy**. Zero geometry overlap across splits; worlds deliberately reused from v65 to isolate the temporal intervention, so this is **not new-topology transfer**. The base v65 source and full datasets reproduced byte-identically; two complete v66 corpus generations produced identical hashes. The independent v66 transport audit passed all 117,088 steps. A separate v64 source-host audit verified 256 physical source episodes / 29,272 source legal actions, 256 successes and minimum two-move wrong-route regret. The full v65 transport re-audit was too costly in this run; archived v65 proof and exact input hashes are retained. No actual SLM was trained or evaluated here.

## Known weaknesses and next consumer experiment

The timestamp-sensitive rule is still implementable using a small symbolic event-history parser. Some v66 arms inherit similar prefixes and routes from v65, intentionally varying the counterfactual telemetry without introducing new physics or task diversity. Consumer trainers should measure **both-correct** counterfactual groups, and compare full-history vs last-8/16 history, first-status-only, last-status-only and sensor-event deletion. Run actual pinned SLM training **only in the separate trainer lane**, using TRAIN only; DEV/HELDOUT and all evaluator certificates are sealed. Report autonomous held-out completion and p50/p95 CPU inference plus peak RSS, with exact model hash. Synthetic game-like host is not native CyberSand. No paid API/GPU, model checkpoint, or immutable Actions dataset artifact was created.
