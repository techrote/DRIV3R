# DRIV3R Corpus Factory v38 — host-accepted sequential material-law changes

18 November 2026. Draft/HOLD PR #1. This is a **synthetic host adapter, not native CyberSand**, and this data lane trains no policy.

## Scientific change

V37 performed public tool-law counterfactuals at a single decision without rerunning the host. V38 instead performs complete game-like recovery episodes via the existing, pinned v32/v26 synthetic Host. A public material-tool mapping starts in either normal or inverted mode. The **first host-accepted corrective ERASE** flips the mapping; the first subsequent accepted PAINT restores the initial mapping. The adapter maps the issued PAINT_A / PAINT_B command to the appropriate physical material, and public receipts record both issued command and actual material.

At every accepted step, the source recomputes the shortest-successful-suffix oracle and checks its successor cost. Paired runs must have **identical underlying physical-state trajectories** and **different issued PAINT commands** at material actions. Both are enforced, not assumed. The underlying synthetic Host does not implement native CyberSand material-law semantics; mapping is implemented in this experimental adapter.

The four-way durable-memory ambiguity survives. The corpus also deliberately executes and recovers from both a detrimental navigation decision and a wrong-material PAINT, rather than merely pricing their regrets.

## Local full-run results

| Partition | Unique graph topologies | Qualified world/recipe families | Episodes | Decisions | Law-paired PAINT command divergences | Four-way memory families |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 32 | 512 | 12,800 | 640 | 64 |
| DEV | 4 | 16 | 256 | 12,096 | 320 | 32 |
| HELDOUT | 4 | 16 | 256 | 16,576 | 320 | 32 |
| **Total** | **16** | **64** | **1,024** | **41,472** | **1,280** | **128** |

The 1,024 oracle trajectories all terminate with no wrong-material cells. There are **2,048 accepted law-change events**, 512 accepted bad navigation decisions, 512 accepted bad PAINT decisions and 1,280 PAINT actions issued while the mapping is inverted. The 256 DEV/HELDOUT memory examples form 64 clean four-way families: removing durable history makes each family identical yet its four optimal first actions differ. The theoretical historyless accuracy ceiling on that defined first-action challenge is 25%; **not** a measured neural-model score.

Data consumer sees only input; action is the label and weight is the loss mask. DEV and HELDOUT challenge shards carry evaluation metadata and must not be fed as prompts. Training rows with injected bad paint are zero-weighted.

Two full generations of seven JSONL shards and their manifest were byte-for-byte identical. An independent full-data audit verified 41,472 inputs and actions, paint semantics, legal menus, teacher separation, SHA-256 shards, absence of conflicting labels, cross-split topology/input separation, and the four-way memory alias. Source hash: 599aaaa8b70007cd8a50be6e569d7e7a2d0e40ef7119b3cb40c889950fe95976; full generated manifest SHA-256: 0d0c3c8f821dc9900863d45c9aac916014e4d66e644f99799c491a82f6554f5f. All seven shards and their individual hashes are listed in V38_DEFAULT_MANIFEST.json.

Run A: 13.11 seconds and 565,140 KiB peak RSS. Run B: 12.90 seconds and 569,652 KiB peak RSS. Independent audit: 1.71 seconds and 401,948 KiB peak RSS. **These are corpus generation/audit measurements, not model-inference latency or RSS.** Local qualification; not an independent GitHub CI replay claim.

## Reproduction

Place closed_loop_law_v38.js and its four dependency sources in the same directory. Execute with Node.js:

    node closed_loop_law_v38.js /tmp/v38A
    node closed_loop_law_v38.js /tmp/v38B
    node test_closed_loop_law_v38.js /tmp/v38A /tmp/v38B

Outputs must be directed to previously nonexistent directories. Local shards are not versioned in Git; publish through opt-in immutable Actions artifacts if required.

## Limits and consumer handoff

Still synthetic radial graphs with a scripted host-accepted historical prelude and deliberately injected mistakes; no dynamic obstruction grammar yet. The mapping adapter is not CyberSand native. No actual LFM/Sprocket weights were trained or run. No paid API/GPU compute, no model latency/RSS, no immutable Actions corpus artifact.

Next consumer experiment: train an actual pinned SLM/adapter on TRAIN, then run complete HELDOUT episodes, stratified by clean/compound intervention and history-full/truncated. Compare visible public tool law with a **comprehensive** law-information ablation (including residual receipt cues, not merely removing one field). Report terminal success, wrong-material rate, realized regret, latency, peak model RSS, exact model hash and data manifest. Next Corpus Factory iteration should prioritize true dynamic obstructions or substantially distinct topologies.
