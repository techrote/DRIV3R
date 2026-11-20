# DRIV3R Corpus Factory v80 — three-lane loop and remote goal memory (19 November 2026)

**Lane:** `experiments/driv3r/corpus_factory/**` — data generation and independent validation only. **Status:** Draft/HOLD, do not merge. **Research host:** newly implemented synthetic graph, **NOT native CyberSand**. No game repository writes, trainer-lane writes, paid compute or SLM training.

## Experiment

Unlike v79's two-corridor gate-risk problem, v80 adds three physically distinct branch choices at two junctions, cross-connected looping passageways, goal changes from paint A to paint B, delayed delivery of contradictory snapshot reports, optional unreported second-gate shifts, attempted wrong-route traversal, legal backtracking and incorrect-paint correction. A 250-action patrol hides the decisive reports for more than 128 actions before the first junction; the model sees only before-action host observations and physically accepted actions. Only one stage gate is open at a time, making the branch label unambiguous. Evidence from the unreachable future is never part of the earlier observation.

The oracle produces legal actions, not an autonomous trained model. Forced bad-paint interventions and repeated hidden-future prefixes are excluded from training loss; `*_EVALUATOR_ONLY_*` datasets and certificate files must never be loaded into policy training.

## Exact-source commands

```bash
node three_lane_loop_v80.js /tmp/corpus_v80
node test_three_lane_host_v80.js /tmp/corpus_v80
node export_three_lane_decisions_v80.js /tmp/corpus_v80 /tmp/index_v80
node score_three_lane_controls_v80.js /tmp/corpus_v80
```

Source SHA-256 generator `d8225fca56d8f3027c4198546a3af42d5c77168dac02be983acc40add3b3d041`; auditor `d2b100cfe3e5a02c78225b59848857436018e99f201336ec53b1e70609288173`; exporter `08d9bbd17fd0769526d8fb800ce0cb4fd9481cacbb04efc1e7eaee896adf9f8e`; scorer `216df461e22a1ba6c13b6dbdca084e26bf6078e439992bfdbd1acf10427b6080`. Generated corpus manifest SHA-256 `7c427554ae3d7a5d0cedfdecd3af24e409c2fc9b8c2312f7cf4c6adfc95f0450` and decision-index manifest `da92e0751b0cb28443da540c7e40db1802ca8181c146a9277d06812983ebda44`. Repeat generation was byte-identical across all seven corpus files. Deliberately changed action text with a recomputed file SHA was rejected by the independent physical replayer. Do not substitute checksums for semantic replay.

## Measured full-corpus physical qualification

| Split | Worlds | Terminal episodes | Accepted host actions | Actual blocked-corridor discoveries | Bad-paint recoveries |
|---|---:|---:|---:|---:|---:|
| TRAIN | 6 | 432 | 135,504 | 216 | 108 |
| DEV | 3 | 216 | 72,828 | 108 | 54 |
| HELDOUT | 3 | 216 | 79,092 | 108 | 54 |
| **Total** | **12** | **864** | **287,424** | **432** | **216** |

Independent auditor reconstructed 287,424 physical moves/paint actions, switch events, observations, legal masks, reports, material states and goals. There are 288 exact-current-observation **three-way** branch contrast groups and 432 two-way terminal material contrast groups, all with different full histories. Twelve unique geometry SHA-256 hashes occur in disjoint train/dev/heldout partitions.

## Compact positive-weight model decision index

TRAIN has 864 indexed landmarks: 216 first-junction, 216 second-junction and 432 material-memory targets. DEV and HELDOUT each have 432 **evaluator-only** indexed landmarks. All exact normalized full-history/action keys were unique. The indexes reference raw chronological stream rows by episode/step; they are **not** a source of oracle certificates.

The current-only Bayes majority ceilings for **HELDOUT** are 33.33% (first gate, 108 targets), 33.33% (second gate, 108), and 50% (material, 216). These are controlled deterministic corpus statistics, **not neural-model accuracy**. They do not prove a neural model will learn the history; a symbolic receipt parser still solves the labelled task.

## Limitations and handoff

Topology is a small synthetic three-lane family with scripted patrol, only one open lane per stage, explicit source-time receipts and no native CyberSand validation. Short models may learn spurious heuristics; test autonomous physical outcomes, not just imitative decision accuracy. No SLM checkpoint/LoRA, real neural heldout rollout, p50/p95 CPU inference, RSS or immutable Actions dataset publication occurred in this data-lane increment. Historical v76 B-specific DEV contrast is still open.

The separately owned trainer should train a pinned real CPU SLM using indexed TRAIN data, reconstruct input history from the raw streams, seal HELDOUT, and report legal closed-loop success, physical recovery, material-goal completion and history-ablation controls. Do not edit the trainer lane from Corpus Factory.
