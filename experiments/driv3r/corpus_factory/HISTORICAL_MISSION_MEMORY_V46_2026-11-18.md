# DRIV3R Corpus Factory v46 — delayed superseding mission memory

Date: 2026-11-18. STATUS: SOURCE + LOCAL REPLAY QUALIFIED; GitHub publication and exact-HEAD CI status should be recorded separately. SYNTHETIC research host, NOT native CyberSand.

## Scientific question

Can a game-policy SLM retain and prioritize a superseding mission receipt after a long physical trajectory when the latest observation no longer repeats the current target? v45 exposed `target` and `stage` in every observation. That allowed a trivial stateless target-following policy; v46 removes both and requires using old historical events. v46 also preserves the v45 partially discovered map and remote-gate counterfactual.

## Game-like scenario

Each episode starts with v45's actual accepted-action survey and wrong-paint preparation on one deterministic, disjoint procedural graph. The research-only synthetic host then receives an initial target-A assignment, executes 24 accepted movements, receives a superseding target-A assignment, then executes 176 accepted movements. At the first evaluated decision, `target` and `stage` are absent. The correct goal is encoded only in durable event receipts. Subsequent A completion produces a B-goal revision receipt; B completion produces an explicit `MISSION_COMPLETE` receipt, permitting a legal STOP. Host legality, painting, mistakes and state progression remain checked through the unchanged v44 host implementation.

Per world, five fully replayed terminal arms establish (1) a mission counterfactual with identical input after mission-receipt ablation but a different first optimal movement; (2) a remote-gate counterfactual requiring another first action; (3) a stale-receipt negative control where the obsolete initial target changes but the latest assignment and correct action stay fixed; plus an additional gate/mission cross arm. All candidate worlds require strict *full-graph* first-action regret for both mission alternatives, while the training teacher only sees the discovered map, legal menu and available events. A deterministic observation-only shortest-frontier teacher supplies labels. The full hidden graph is used to qualify the challenge, not injected into policy inputs.

## Reproducible results

| Split | Worlds | Host-successful episodes | Policy records | Mission positives | Gate positives | Stale negatives |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 48 | 240 | 4,957 | 48 | 48 | 48 |
| DEV | 16 | 80 | 1,970 | 16 | 16 | 16 |
| HELDOUT | 16 | 80 | 2,616 | 16 | 16 | 16 |
| Total | 80 | 400 | 9,543 | 80 | 80 | 80 |

- No cross-split topology/input hash overlaps or contradictory labels in any output.
- 2,747 MOVE decisions have a target node absent from the currently surveyed map.
- Superseding mission receipt age at the evaluation fork: at least 176 accepted movement actions.
- Two clean full generations are byte-for-byte identical across 7 shards and manifest.
- Independent audit reconstitutes v44 host state and checks each of the 9,543 accepted actions and 400 successful terminal completions; labels computed by an independently written observed-map planner.
- TRAIN duplicates are zero-weighted; observed duplicates in this version: zero.
- SHA-256 of the generator, parent v45/v44 scripts and 6 data shards is locked in `V46_DEFAULT_MANIFEST.json`; generator output `manifest.json` is itself the canonical data manifest.

## Consumer protocol

Use **only** the `input` object from `*_policy.jsonl` as the model input. `action` is the teacher label; `weight` is the loss weight. Seal the `*_challenge_EVALUATOR_ONLY.jsonl` files; they contain full source metadata, hashes, counterfactual answers and teacher/host terminal lengths. TRAIN-only tuning; DEV selection; HELDOUT once for a pinned model. Compare (a) complete durable-event input, (b) mission events removed, (c) latest 32 accepted actions without durable events, (d) stale/most-recent-event confusion controls. Report terminal success, paint correctness, legal action preference, first-action mission/gate accuracy, route regret, memory errors and p50/p95 CPU model latency/peak RSS. A command parser/oracle is **not** an actual model benchmark.

## Hard limitations and negative results

This is a **synthetic** graph host, not native CyberSand's physics, painting/material rules or observation delivery. The full prelude survey and mission-ageing moves are scripted host-accepted actions; only subsequent continuation teacher decisions use the restricted observation. The synthetic host's two-material objective is intentionally narrow. Absolute event timestamps, last-receipt selection and explicit MISSION_COMPLETE remain machine-readable symbolic shortcuts. v46 does not itself perform genuine named SLM training or inference. The contemporaneous LFM and Sprocket consumer handoffs report unavailable pinned model binaries / missing opt-in CPU execution in their workers. No Actions immutable corpus artifact custody is proven; no native CyberSand success, neural latency or neural RSS result is claimed. Do not promote this dataset over v43/v44/v45 without demonstrated pinned SLM held-out lift.

## Safety and ownership

Only write under `experiments/driv3r/corpus_factory/**` on existing DRIV3R research branch (PR #1 Draft/HOLD). Read-only CyberSand and Neurounner; no merge, ready, branch creation, force push, paid compute, or out-of-lane writes.
