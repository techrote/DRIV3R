# STOP-RESERVATION-020 — terminal action reservation in physical-recovery policies

19 November 2026. Explicit bounded synthetic experiment, reusing frozen 018 TRAIN-only softmax reconstruction. This tests reserving the final logical action for STOP across 36/36, 72/64, 128/64-censored and 128/128-shadow cohorts, with paired hidden damage interventions and shifted obstacles. It is **not** an established CyberSand ABI requirement.

| Policy | Fresh terminal /600 | With reservation | Shifted terminal /600 | With reservation |
|---|---:|---:|---:|---:|
| Feasibility guard | 427 | 427 | 421 | 421 |
| Frozen trained student | 430 | 430 | 431 | 431 |
| Censor-aware | 432 | 432 | 416 | 416 |
| Late physical check | 356 | **427** | 346 | **421** |

**Finding:** late physical checking previously had 71 fresh and 76 shifted *material-complete without STOP* cases. Reserving STOP removes these terminal misses; **not additional material repair**. For censor-aware, fresh material goals drop from 441 to 432 while terminal completions stay 432; shifted material goals fall 428 to 416 with terminal unchanged at 416. Early STOP is a trade-off between material work and terminal recording. Feasibility and the frozen student have identical terminal totals with or without reservation. The 64-move censor still limits 128-decision material objectives.

**Qualification:** 1200 paired conditions, 10800 synthetic policy rollouts, 605160 receipts independently reconstructed by fresh instances of the **same** pinned synthetic host. All source receipts/menu actions/positions/material state checked, 2 corrupted-receipt tests rejected. All reserved arms issued STOP, no paired terminal losses. TRAIN 70 seed clusters/280 labels/1100 updates; DEVELOPMENT 100 new clusters in two geometry groups; 160 selected paired failure records, complete action distributions, pinned source blobs, and SVG trajectories retained. V8 source-science wall 23689 ms. Native Node RSS and p95 inference **not qualified by this V8 run**; the explicit Node script records RSS. 128/128 is shadow only, not native admissible.

**Reproduce manually:** `node experiments/driv3r/training/long_horizon/stop_reservation_020.js /absolute/output`. It verifies the Git blob SHA-1 hashes of the original 016 host and 018 model. No automatic optimizer on push, no structural HELDOUT, no paid compute, no native CyberSand changes.

**Decision:** preserve feasibility as safety control. STOP budgeting corrects terminal semantics in the synthetic evaluator but does not solve hidden material state or 128/64 spatial feasibility. Trained student remains experimental, and source-smoke must not be mistaken for scientific replay/native acceptance.
