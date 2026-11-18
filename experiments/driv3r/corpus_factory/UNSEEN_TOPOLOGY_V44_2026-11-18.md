# DRIV3R Corpus Factory v44 — held-out procedural topology and delayed remote gate

**2026-11-18 | DATA LANE ONLY | Draft/HOLD PR #1 | synthetic host, not native CyberSand.**

## Scientific question

Prior v39–v42 causal junction evaluations repeatedly reused a manually designed graph. v43 balanced two useful synthetic curricula, but did not add unseen topologies. Can a game-policy consumer follow a *historically observed but not currently visible* remote edge-gate event, generalize route choices to different graph sizes and structure, then complete a two-stage painting objective with possible wrong-material remediation?

**Do not claim SLM results.** This package supplies actual host-executed labelled trajectories and sealed paired tests. Training and model inference belong to model lanes, not Corpus Factory.

## New generator and host

`unseen_topology_v44.js` deterministically generates 120 distinct connected, procedurally sampled 2-D graph worlds from a seed per `(split, world_index, attempt)`; these are **synthetic** representations with the historical host-legal action vocabulary `MOVE_N/E/S/W`, `PAINT_A/B`, `ERASE`, `STOP`. Graph cells have adjacent traversable edges, inaccessible/protected coordinates, an additional remote mutable edge, two separated painting goals and a fully observed static survey graph. The latest remote gate state itself is not provided in current local observation: it exists in the durable event receipt.

For each world, the generator finds a junction with (i) an OPEN and CLOSED remote-gate intervention causing different legal optimal first moves and **at least two additional accepted actions of regret** when choosing the opposite case's first move, and (ii) a distinct **irrelevant-gate negative control**, in which gate transitions do not alter the optimal initial route. Two gate-state variants are generated for each kind, yielding four complete host-action episodes per world.

Every episode includes **192 host-accepted movement actions** at the local junction after environmental events; the decisive gate event is **144 accepted movements old** at the first supervised decision, outside the `recent` 32-move window. The matched pair shares identical geometry, goals, legal actions, local observation, recent movements, visited positions and current painting; the *old remote edge event* differs. The latest state is preceded by a contradictory earlier event, and unrelated sensor diagnostics occur, preventing a fixed first-event shortcut.

Two out of every three worlds also have a **host-accepted wrong-material painting prelude** at one of the two targets; the actor returns to the junction before gate ageing. The oracle must later ERASE contamination and apply the correct paint. After correctly painting target A, the host emits a `GOAL_REVISION` and reveals target B, requiring new navigation, PAINT_B and terminal STOP. The host rejects illegal movements/paint operations; the oracle labels only accepted actions.

### Split distributions and output

| | TRAIN | DEV | HELDOUT | Total |
|---|---:|---:|---:|---:|
| Worlds / unique graph topologies | 72 | 24 | 24 | **120** |
| Positive remote-gate pairs | 72 | 24 | 24 | **120** |
| Irrelevant-gate negative-control pairs | 72 | 24 | 24 | **120** |
| Paired state episodes | 288 | 96 | 96 | **480** |
| Host-accepted supervised decisions | 5,652 | 2,298 | 2,562 | **10,512** |
| Completed successful synthetic episodes | 288 | 96 | 96 | **480** |

TRAIN worlds use 7–8 grid dimensions with 0.40 random extra-edge probability; DEV uses 9 with 0.32; HELDOUT uses **previously unseen 10–11 grid dimensions** with 0.26. Topology hashes are disjoint across the three partitions. All four movement directions occur as first actions in every partition. Exactly 80 worlds / 320 variants include deliberately contaminated material, which the accepted-action oracle removes.

Trainer records contain exactly `{input, action, weight}`. `input` carries static survey topology, currently legal commands, active painting objective, local view, recent 32 receipts and durable host events. Gate-state truth, optimal-action regret, seed, split identity and full challenge answers stay outside model input. An evaluator-only challenge sidecar contains matched-pair identities, intervention edge, first-action answers, regret, split provenance, input/ablation hashes and terminal trajectory lengths; it is **not training data**. All 10,512 records are unique, weight 1, and source-checkable.

Positive pairs have an **exact 50% first-decision theoretical ceiling without the old remote-gate events**: after removing `EDGE_GATE` durable entries, both paired inputs are byte-identical but require distinct best actions. Irrelevant-gate controls demonstrate that merely noticing a gate event should not always change behavior. These are *theoretical input/label properties*, not measured neural accuracies.

## Exact reproducibility

```sh
node unseen_topology_v44.js /tmp/driv3r-v44-a
node unseen_topology_v44.js /tmp/driv3r-v44-b
node test_unseen_topology_v44.js /tmp/driv3r-v44-a /tmp/driv3r-v44-b
```

Node 24 is sufficient; only built-in `node:fs`, `node:path`, `node:crypto` are used. No paid API, GPU, Python package, external network or model files are required. The outputs are six JSONL shards and a deterministic `manifest.json` with SHA-256s; `V44_DEFAULT_MANIFEST.json` pins the manifest expected from the complete generation. `--quick` is a small generation smoke, not scientific acceptance.

- Generator SHA-256: `14d27b999d5eee0a83fe7e5afd9e85ce29670bdd4320024d3465705f7b29f8b6`.
- Independent auditor SHA-256: `bdfe849d89956702a1129060bb2f4d82bb3f1bb408bac551176123e69f24acbe`.
- Complete generated manifest SHA-256: `705ecb176304054c06e3e0e9d7eb8feda05e313009b36d6b0f93d5522844a052`.
- Two full generators completed in 1.05 and 1.13 seconds; peak RSS 161,084 and 154,884 KiB.
- Full independent dual-source audit completed in 3.12 seconds with peak RSS 496,868 KiB. This includes **all 480 exact-input host-action terminal replays**, independent per-record graph-shortest-distance label verification, 120 causal and 120 negative-control pair validations, six shard hashes, split-topology ownership, trainer-input field isolation and all byte-identical outputs.
- No conflicting labels, trainer/evaluator metadata contamination, or cross-split topology collisions were found.

## Constrained consumer contract

For supervised training, feed **only `policy.input`**; use `policy.action` for loss and `policy.weight` as the sample weight. Keep `*_challenge_EVALUATOR_ONLY.jsonl` sealed until evaluation. To run genuinely closed-loop held-out synthetic gameplay, instantiate `scenario('HELDOUT', world_index)`, call `prepare(world, kind, gate_closed)`, give only `input(host)` to the policy, and return its proposed command through `apply(host, command)`. An independent host-side scorer should record accepted/rejected decisions, correct terminal STOP, painting contamination, travel regret and actual inference latency/RSS.

Compare v43-only, v44-only and mixed TRAIN with **identical pinned model root**, then choose hyperparameters on DEV only. Run heldout worlds once per frozen candidate. Check paired full-history vs event-ablated and truncated-history inputs, causal vs irrelevant-gate controls and contamination/no-contamination cases. The named SLM lanes have their own weight/checkpoint provenance; Corpus Factory must not train their weights or edit their directories.

## Honest limitations / negative findings

- This is **not native CyberSand**. Graph-legal action/event semantics are synthetic and do not implement Godot, Rapier, Workbench, physical terrain, native materials, or the complete canonical runtime ABI.
- The static survey graph is fully visible. Unknown maze geometry, search under hidden terrain, physics-driven obstacles, genuine goal-discovery and exploration-vs-exploitation are **not** evaluated. Partial observability concerns the *aged remote gate state* only.
- There is one procedural graph family despite 120 distinct topologies. DEV and HELDOUT vary size/density, but independent neural performance is unmeasured.
- Oracle route labels are shortest-path demonstrations. A symbolic solver with access to historical events can still solve every case; no neural capability may be inferred from oracle terminal success.
- Wrong-material interventions are deliberately host-executed by the fixture; they are not autonomous policy errors. There is no autonomous learned policy rolled out in this lane.
- No actual LFM/Sprocket weight update, adapter checkpoint, native CyberSand evaluation, model latency or model RSS is produced. LFM handoff GP350-018 reported no new full-model inference/update and branch/retrace shortcomings; Sprocket handoff reported missing pinned root files and CPU runtime dependencies.
- Generated datasets are delivered as a reproducible local archive only; immutable GitHub Actions artifact publication is not yet qualified. No costly automatic push-triggered CI was added.

## Handoff

This is an *additional* OOD branch to consume alongside v43, not a replacement for v43's balanced material-law/interlock curriculum or v38's changing tool laws. The next scientific gate belongs to the LFM/Sprocket trainer owners: pinned actual-SLM TRAIN/DEV fit, frozen HELDOUT autonomous evaluation and matched gate-history ablations with p50/p95 inference latency and RSS. Do not merge or Ready-promote PR #1.
