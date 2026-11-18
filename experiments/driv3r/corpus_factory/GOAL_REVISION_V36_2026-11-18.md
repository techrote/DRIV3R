# DRIV3R Corpus Factory v36 — delayed, causally consequential goal revision

**18 November 2026 · synthetic graph host, NOT native CyberSand · Data lane only · Draft/HOLD PR #1**

## Research question

Can a game-policy data consumer handle an *accepted environmental goal change* during an already-running game, preserve durable evidence of remote painting mistakes, and recover after an additional legally accepted but detrimental movement choice? Unlike v35's trainer-safe re-export, v36 adds a new host state transition and a strictly measured counterfactual. v35's v34 corpus remains a separately useful baseline.

## Host transition and labels

- Builds from the unchanged v32 five-target synthetic four-arm world and v26 host. Each prefix includes **two host-accepted incorrect PAINTs**, an opened paint-triggered gate and long legally accepted movement history. The starting survey legitimately describes `quota=2`.
- The v36 **synthetic host**, at a legal between-agent-actions boundary at the root, emits a public `GOAL_REVISION` event to either reaffirm `quota=2` (negative control) or increase the active goal to **three correctly painted targets**. This is an environmental event, **not a model-selected action**. It changes `w.quota` before computing any teacher labels.
- After the environmental receipt, the host accepts **160 more navigation actions**. Thus the goal update and earlier paint receipts are absent from the latest 64 events. The current public goal remains correctly observable; durable memory is needed for the **remote contamination location**, not for remembering the currently visible quota. A v30-derived compact view is explicitly corrected to the *live* host goal to avoid a stale-survey leak.
- Four paired episodes per family have exactly the same recent/restricted input, legality, public counters and current goal, but older different wrong-material locations. In the stable-goal families the four optimal first actions differ (restricted-history ceiling **25%**); in revised-goal families two of four share `MOVE_E` (ceiling **50%**). These are deterministic **first-action upper bounds**, not learned model accuracies.
- For each starting arm and quota, produce a clean shortest-successful-suffix continuation and an accepted strictly detrimental **first navigation** followed by newly optimal recovery. An exact reverse-BFS suffix oracle supervises subsequent accepted steps and terminal paint cleanup. The intervention has at least **2** action units of certified regret.
- For every revised-goal episode, run a **frozen-quota-2** oracle on each actually visited state; at least one legal frozen choice has **2+ actions of actual regret** (or becomes illegal). Reaffirmation controls generate zero false positive goal-policy effects. The diagnostic oracle never enters the policy input.

## Default reproducible corpus

| Split | Geometry groups | Goal families | Episodes | Decision rows | TRAIN loss weight=1 | Weight=0 |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 6 | 48 | 384 | 9,504 | 8,472 | 1,032 |
| DEV | 4 | 32 | 256 | 12,896 | — | 0 |
| HELDOUT | 4 | 32 | 256 | 17,824 | — | 0 |
| **Total** | **14** | **112** | **896** | **40,224** | **8,472** | **1,032** |

Additional measurements: 448 revised-goal trajectories, **2,344 individual strict frozen-goal regret decisions** (648 TRAIN / 736 DEV / 960 HELDOUT), 448 accepted wrong-first navigation interventions and **1,792 corrective erasures** in completed recoveries. Evaluation-only memory challenges contain 128 first-step records each in DEV and HELDOUT, half at each goal quota. Each selected family has exactly four remote-contamination variants.

All exact full bounded-input duplicates are zero. A conservative near-context key removes **only** the current movement tail (`recent`/`recent_positions`) and step budget (`left`), preserving all durable remote paint/goal receipts, current goal, materials, legal menu and local observations. This key identified 1,032 repeated TRAIN contexts with **zero opposite labels**; those repetitions are assigned loss weight zero. DEV/HELDOUT report 1,904 / 2,800 near repeats but retain all examples for evaluation. Complete-input and near-key contradictions are fatal. No cross-split exact input or geometry collisions were found; geometry groups, not decisions, define disjoint split identity. Do not treat 40,224 as independent topologies.

## Reproduction and validation

From the lane directory, with Node.js and the *exact pinned* v25/v26/v30/v32 sources:

```bash
node experiments/driv3r/corpus_factory/goal_revision_v36.js /tmp/driv3r-v36-a
node experiments/driv3r/corpus_factory/goal_revision_v36.js /tmp/driv3r-v36-b
node experiments/driv3r/corpus_factory/test_goal_revision_v36.js /tmp/driv3r-v36-a /tmp/driv3r-v36-b
```

The generator refuses to overwrite a destination. Its eight JSONL shards and manifest belong in immutable CI artifacts, **not Git**. The source-matched default manifest `V36_DEFAULT_MANIFEST.json` includes source/dependency SHA-256, shard SHA-256/size/row counts, experimental split metrics, label and evidence contracts. The five source files are small, unlike the outputs.

The independent test **reconstructed every one of 40,224 model-visible inputs**, replayed all 896 trajectories through the synthetic host, re-ranked every labelled action against a new shortest-successful-suffix oracle, verified host receipts and terminal correctness, audited all challenge aliases and two complete generations. Two complete datasets and manifests were byte-identical. Local CPU generator times: **17.25s / 17.42s**, peak RSS **147,816 / 149,308 KiB**. Independent audit: **19.92s**, peak RSS **232,512 KiB**. These figures are generator/test, not model inference or training performance.

## Consumer contract and limitations

For ordinary policy training, read `TRAIN_policy.jsonl`; present **only `row.input`** to the model and use `row.action` as the supervised label. Exclude `weight=0`; never train on DEV, HELDOUT or `*_memory_challenge.jsonl`. `*_episodes.jsonl` contains teacher-only episode identifiers, intervention metadata, exact regret, full state histories and replay traces. It must never become a model prompt. Use DEV once to choose a consumer recipe; reserve HELDOUT as untouched evaluation and report per-goal and per-error-stratum results.

**Limitations:** this is a graph simulation with radial corridor motifs, fixed material laws and scripted remote painting; the goal event is simulated by a controlled host transition. Current live quota is visible, so these results **do not prove neural memory of the quota**. No named SLM was trained or evaluated and no native CyberSand ABI, native game behaviour, latency or model RSS was measured. A symbolic interpreter can solve the memory task; first-action ceilings apply only to the specifically ablated identical input groups. Real model execution remains blocked in consumer lanes by unavailable pinned model assets/runtime or opt-in Actions dispatch. No paid GPU/API used.

**Next useful consumer experiment:** a real pinned LFM350/Sprocket adapter trained only on nonzero-weight TRAIN rows, evaluated on four cells: stable/revised goal × expert/accepted-offpolicy intervention, with durable-history vs latest-64 ablation. Require legality, terminal goal completion, residual contamination, realized excess actions, p50/p95 decision latency and model RSS. A separate later corpus should test *truly latent* goal/mapping revisions and dynamic obstructions with distinguishable host-legal causality; do not infer those from v36.
