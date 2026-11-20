# FUTURE-GATE-ALIAS-001 — future material-state observability

**Status:** DEVELOPMENT-only, provisional research. Committed reproducible laboratory code, deterministic optimizer and tamper-checking tests. This is neither a native CyberSand evaluation nor a production action/observation adapter. No structural HELDOUT opened.

## Scientific question

When a material gate closes across an initially cached route, can a tiny learned handoff decide whether to invalidate early, or continue and replan only if the next action becomes illegal? Prespecified matched pairs use the same material state, action menu and numerical input at tick 5. One future reopens at tick 8, before the affected action; the other remains closed until the last horizon tick. The future schedule is **not visible** to the policy. There is a paired irrelevant material edit.

This is an intentional **observation-alias falsification experiment**, not a realistic estimate of gate-reopen frequency. It extends the existing stale-route and gate-oscillation work without modifying those labs. All targets, geometry and intervention schedules are fixed before any outcome measurement.

## Implementation and scope

- Train: seeds 1000–1047, all three horizons × three modes = **432 labelled examples**. DEVELOPMENT: seeds 3000–3015 (no seed overlap).
- Model: genuine **161-parameter NumPy MLP** (8 inputs → 16 tanh → 1 sigmoid); seeded Adam, 180 updates per 36/72/128-decision curriculum stage, **540 optimizer updates** total. Predictions use only observable features. A privileged future-aware selector is explicitly identified as an oracle lower-bound control, not a deployable policy.
- Six matched arms: blind cache, lazy next-action legality, route overlap, every-epoch invalidation, learned gate, privileged oracle.
- One-cell cardinal painting; STOP always legal; BACKTRACK travels retained painted history; deterministic BFS planner; per-action legal menus and material-state hashes replay-checked. This is a **synthetic approximation**, not canonical CyberSand native semantics.
- 64 painted-move censor is maintained in the **native64 approximation cohort**. For the 128-decision research horizon only, a separately labelled **shadow128 synthetic cohort** lifts the research ceiling to observe longer trajectories. No production/native limit is changed or implied to be changeable.

## Local pilot results — 18 November 2026

The prior local prototype completed **1,152 development episodes**, replay-checked **67,290 action receipts**, and ran all 540 optimizer updates. An independent source-only test suite had **8/8 tests passing**, including byte-level deterministic optimizer checks and corruption detection. Its external checkpoint was **not committed**. The prototype run reported approximately 14.5 s total wall time, ~0.043 s optimizer training, ~4.8 ms average episode evaluation time (including simulation and receipt recording), and ~98.7 MiB peak process RSS. Model weights SHA-256: `ea3abd6eaac458ac720e2d694b3bd26b0e0c3f5daa92c19d36905cce738295ba`; external ZIP/NPZ file SHA-256: `fe59dd34a07605d6a1c9c6c7719f25bf5301aec97b57982ea6f5a1ddbc457c84`.

**Provenance limitation:** the pilot's exact local source SHA-256 was `61ce5954f87bc6d9665182b46de187d3800b3b33a29026c29c7758958189b3b1`. The published research source has been reconstructed from that prototype rather than uploaded byte-for-byte. Consequently the above measurements remain **provisional**, not exact-published-source science, until this PR's committed source is executed and replayed with its own source digest. The included CLI regenerates source-specific evidence and an external checkpoint without committing binaries.

### Pilot outcomes (16 DEVELOPMENT seeds per row)

| Horizon/cohort | Scenario | Lazy goals | Route overlap goals | Learned goals | Mean lazy decisions | Mean learned decisions |
|---|---|---:|---:|---:|---:|---:|
| 36/native64 | early reopen | 16 | 16 | 16 | 21.938 | 32.938 |
| 36/native64 | late reopen | 0 | 16 | 16 | 36.000 | 32.938 |
| 72/native64 | early reopen | 16 | 16 | 16 | 47.938 | 58.938 |
| 72/native64 | late reopen | 0 | 16 | 16 | 72.000 | 58.938 |
| 128/native64 | early reopen | 0 | 0 | 0 | 64.000 | 65.000 |
| 128/native64 | late reopen | 0 | 0 | 0 | 100.125 | 65.000 |
| 128/shadow128 | early reopen | 16 | 16 | 16 | 85.938 | 96.938 |
| 128/shadow128 | late reopen | 0 | 16 | 16 | 128.000 | 96.938 |

All irrelevant off-route edits complete in the two shorter native64 horizons. Unconditional epoch invalidation incurs an avoidable extra planning call on those controls.

**Main negative finding:** all **144/144 matched TRAIN early/late pairs** have identical observable vectors with opposing hindsight-optimal handoffs. On 144 native64 DEVELOPMENT gate decisions, the prototype learned policy achieved **96/144 (66.7%)** schedule-specific handoff accuracy: it replanned on all 48 early closures, all 48 late closures, and on none of 48 irrelevant edits. The predicted early and late probabilities were equal within each matched pair, with group mean 0.622406. Eager invalidation rescued late-route failures but cost an average **11 extra decisions** on early reopening (36 and 72), versus lazy control. The privileged oracle avoids that trade-off by inspecting unavailable future information.

This is a causal observability limitation in the specified synthetic family, **not** evidence that larger models or stronger training can predict an unobserved future. It also does not establish any model improvement over the hand-coded route-overlap policy.

## Source-specific reproduction

Run from the checked-out **actual PR HEAD**, after installing NumPy (no model downloads):

```bash
python -m unittest experiments.driv3r.training.long_horizon.test_future_gate_alias_lab -v
python -m experiments.driv3r.training.long_horizon.future_gate_alias_lab --out /tmp/driv3r-future-gate-001 --dev 16
```

The CLI writes a checkpoint .npz, a full metrics JSON including exact source SHA-256 and episode-summary digest, and illustrative replay-checked trajectory traces **outside Git**. Do not add those binaries to the repository. Treat the local pilot as a research lead until the published-source run confirms it.

## Next experiment

Introduce an **observable, host-approved indication of reopening confidence or waiting cost**, separately from a privileged oracle future flag, then retest the same matched scenarios and event horizons. Keep explicit uncertainty and the irreducible alias baseline; do not silently grant the model future state. Preserve source-level evaluation and strict read-only CyberSand/Neurounner boundaries.
