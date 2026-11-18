# Corpus Factory v22 — public law pulses, goal retargeting and trajectory correction

**17 November 2026 · DRIV3R PR #1 Draft/HOLD · synthetic host, NOT native CyberSand.** This is a data/label corpus, not policy training.

## Scientific question and experimental design

V21 removed contradictions between publicly observed material recipes and oracle paint labels but left mission goals and material laws stationary. V22 adds **accepted-action-triggered, publicly observed changes** to material laws and authorized workshop goals while painting is in progress. This targets the changing-material and short-memory recovery weaknesses reported by independent DRIV3R training labs.

Each eight-member counterfactual family begins with three causal 0/1 remote receipts followed by 80 neutral receipt/position events. The same current view and last 64 events/positions must support four NORTH and four SOUTH first actions; full causal history is required. All subsequent movement, paint, erase and STOP actions are model-controllable through createHost, observe and apply. The synthetic host protects obstacles and closes dynamic shutters after eight accepted moves.

After 2 accepted paints in TRAIN, 3 in DEV or 4 in HELDOUT, the host generates a **public mission_revision receipt** and modifies the publicly observed recipe and/or authorized goal. On double-pulse worlds, the law changes first and the target workshop flips after 7 accepted paints. This trigger counts genuine accepted paint operations from any policy, not scripted oracle indices. A successful agent must revisit, erase or repaint incorrect cells, remove abandoned-branch paint, avoid protected cells, and finish at the hub with correct STOP.

The oracle reconstructs the mission from public remote receipts, subsequent public goal changes and the visible full recipe, asserting agreement with hidden terminal state. Only its shortest-route planner may use full topology. These material-law changes are synthetic palette swaps, not verified CyberSand physics/chemistry or ABI.

## Generated corpus (default scale 1)

| Partition | World groups | Episodes | Labels | Paints | Corrective erases | Public pulses | Duplicate full inputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 24 | 192 | 12,488 | 1,344 | 384 | 192 | 872 |
| DEV | 8 | 64 | 7,144 | 576 | 256 | 80 | 64 |
| HELDOUT | 8 | 64 | 10,200 | 672 | 352 | 96 | 64 |
| **Total** | **40** | **320** | **29,832** | **2,592** | **992** | **368** | **1,000** |

TRAIN has 96 law-only, 48 goal-only and 48 simultaneous episodes. DEV has 16 each of law-only, goal-only, simultaneous and sequential double changes. HELDOUT has 16 simultaneous, 16 goal-only and 32 sequential double changes.

The oracle's episode-length ranges are TRAIN 42–88 (mean 65.04), DEV 68–140 (mean 111.62) and HELDOUT 146–174 (mean 159.38). Longer held-out corridors and larger detours are confounds; these data do not isolate memory capacity causally.

## Qualification

- **320/320** oracle expert replays reached the exact terminal condition; zero illegal oracle actions and zero protected-cell entries.
- **9/9** standalone Node.js regression sections passed, including public-receipt observability, full split replay, protected-cell refusal, premature STOP and replay determinism.
- All 40 eight-member counterfactual families were balanced and last-64 aliased, yielding a **theoretical** 50% first-action ceiling for last-64-only information. This is not a model accuracy measurement.
- Two complete independently regenerated six-shard corpora and their manifests were byte-identical, with per-shard SHA-256 checks.
- A separate full-dataset hash audit found **zero exact overlapping full model-inputs** between TRAIN, DEV and HELDOUT and no conflicting identical model-input labels within a split. The 872 repeated TRAIN inputs remain replayable but have zero training weight; DEV/HELDOUT repeats retain default weight.
- Default generator wall time **7.78 s**, peak **236,596 KiB RSS**, Node.js v22.16.0. These are corpus-generation measurements, not SLM inference measurements.

Source Git blob: **b6f744216859eb91bec925a42f053153c85a7d0d**. Regression Git blob: **34d2e3d78ec0945741d8fa92b735fbad02ed0527**. Parent source Git blob: **19e8b18139263850930666cae119f58f8b993c82**. Default generated manifest SHA-256: **7cfab215221cd3c4094719c4d0b145f5bc8aba5d6148450f91abfe9c683bcc63**.

Run the v22 test script in the corpus_factory directory with Node.js. Generate shards using the v22 generator with a new empty output directory and optional scale 1 or 2. The six JSONL shards are NOT committed to Git; the manifest pins their expected sizes, row counts and checksums.

## Limitations and next consumer

This experiment **still inherits the two-workshop skeleton, scripted prelude and small XOR rule** from v21. It does not prove that a pretrained SLM learned long-range game behaviour, and a compact symbolic controller could solve its goal logic. It does not test hidden-law discovery, real CyberSand materials, native ABI, or actual model inference. No named model weights were trained. The locally hash-verified shards are not yet an immutable Actions artifact.

Next: train an actual pinned SLM on TRAIN; select on DEV; open HELDOUT once and let the model choose **every post-prelude action**. Compare full causal history versus last64 and law/goal change ablations on motif completion, wrong-branch contamination, corrective erasures, rejected actions, path stretch, model inference p50/p95 and peak RSS. Store source/model/checkpoint hashes and failures, never promote this synthetic result as native game acceptance.
