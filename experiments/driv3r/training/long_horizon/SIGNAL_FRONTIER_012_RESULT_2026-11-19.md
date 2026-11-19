# SIGNAL-FRONTIER-012 — material-confirmation reliability and transfer

19 November 2026 · Explicit, bounded CPU synthetic research. Draft/HOLD; **not CyberSand native acceptance**.

## Question and methodology

Follow-on to SIGNAL-QUALITY-011: does the efficient revisit policy survive intermediate receipt-quality regimes, rather than only a perfect receipt? Four **hypothetical** channels with matched worlds and material-damage schedules were tested: perfect (100% sensitivity, 0% false alarms/loss/lag); high (98% sensitivity, 1% false alarms, 1% delivery loss, 0–1 tick lag); middle (90%/4%/4%, 0–2 ticks); poor (70%/10%/10%, 0–4 ticks). Each condition independently trains a nine-feature logistic classifier for 850 explicit optimizer updates (70 TRAIN seeds × 36/36, 72/64, 128/64, 128/128 = 280 TRAIN examples). TRAIN-only expected utility chooses the deployment cutoff; **DEV is untouched by calibration**. Independent DEVELOPMENT has 60 seeds × four configurations = 240 matched tasks, evaluated using five controllers: never revisit, always revisit, observed warning, TRAIN-calibrated learned controller, and privileged hidden-state oracle. DEV damage rate changes from 16% to 30%; obstacle geometry also changes. The native 64-logical-move censor is a **synthetic proxy** only. Structural HELDOUT remains sealed.

## Source-replayed DEVELOPMENT outcomes

| Channel | TRAIN cutoff | Learned goals / 240 | Always goals / 240 | Learned utility | Always utility | Learned planners | Always planners | Warning-only goals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perfect | 0.01 | 197 | 197 | 722.6 | 704.2708 | 1010 | 1185 | 197 |
| high | 0.005 | 193 | 197 | 703.6833 | 704.2708 | 1024 | 1185 | 185 |
| middle | 0.02 | 195 | 197 | 709.6292 | 704.2708 | 1052 | 1185 | 185 |
| poor | 0 | 197 | 197 | 704.2708 | 704.2708 | 1185 | 1185 | 163 |

### perfect channel

Cohort goals (always / learned): 36/36: 55/55; 72/64: 48/48; 128/64: 47/47; 128/128: 47/47. Trained policy against always: 0 paired utility losses, 160 wins, and 80 ties. Replayed 44922 action receipts.

### high channel

Cohort goals (always / learned): 36/36: 55/55; 72/64: 48/48; 128/64: 47/45; 128/128: 47/45. Trained policy against always: 4 paired utility losses, 144 wins, and 92 ties. Replayed 44965 action receipts.

### middle channel

Cohort goals (always / learned): 36/36: 55/55; 72/64: 48/48; 128/64: 47/46; 128/128: 47/46. Trained policy against always: 2 paired utility losses, 123 wins, and 115 ties. Replayed 45177 action receipts.

### poor channel

Cohort goals (always / learned): 36/36: 55/55; 72/64: 48/48; 128/64: 47/47; 128/128: 47/47. Trained policy against always: 0 paired utility losses, 0 wins, and 240 ties. Replayed 45960 action receipts.

## Technical checks and limitations

For each DEV episode the same synthetic Host was freshly replayed action-by-action, with additional independent movement/material/menu receipt algebra and deliberate malformed-action and coordinate rejection. All controls have exact matched outcomes across channel conditions. The four mode evidence files retain model weights, thresholds, source-replayed action counts, paired negative cases and episode hashes; the SVG gallery labels matched synthetic paths. An FNV32 replay hash is not cryptographic provenance; the opt-in Node CLI records full-source SHA-256 and process RSS if run under Node. V8 benchmark wall times for this campaign were perfect 2064 ms, high 1987 ms, middle 2075 ms, poor 2065 ms, not inference timings; process RSS and native game acceptance **were not measured in this connector-run environment**. Training does not trigger on routine push/PR. Model weights are embedded as text evidence, not an artifact checkpoint. None of these observation channels is certified as part of the CyberSand ABI. Source replay is same-host plus independent receipt algebra, not a separate Python/native scientific replication.

Scientific interpretation must distinguish **lost goals** from saved replanning calls. No learned model should be promoted on mean utility alone if it sacrifices paired goals under material damage. These are provisional synthetic transfer findings; the next meaningful gate is independent Node-process RSS/latency and stricter policy transfer with real host-observable fields before any game-side proposal.

Reproduce explicitly: `node experiments/driv3r/training/long_horizon/signal_frontier_012.js /absolute/output/directory`.
