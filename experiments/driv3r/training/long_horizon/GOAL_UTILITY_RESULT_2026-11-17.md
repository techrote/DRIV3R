# GOAL-UTILITY-001 — robust goal-time advantage, not robust painted-cell advantage

17 November 2026 · DRIV3R long-horizon · **synthetic positive time-to-goal result; no new optimizer run or checkpoint**.

## Method

The original counterfactual painting objective in RECOVERY-DEBT-001 produced inconsistent backtracking labels. Here the same host-legal fork and 11/12 genuinely expended recovery operations were evaluated against two explicit task goals: north exit (53,45) and south spur (54,50).

Eight existing SHA-256-verified suffix checkpoints were restored: two training seeds (811,814) times frozen, always_move, contrastive and outcome-supervised controllers. All policies received the goal using the **experimental** goal feature, not any production CyberSand ABI addition. The initial physical state and action probability distribution were identical within each MOVE_S/BACKTRACK pair. Either first action was forced only if actually legal, then the same greedy neural suffix controller operated up to 128 decisions.

DEVELOPMENT seeds 3001, 3002, 3101, 3102, 3201, 3202, 3301 and 3302 were previously unevaluated; original first-per-family DEVELOPMENT seeds were used to explore the goal design. **This is exploratory research, not blinded or preregistered validation**. All eight are variants of the four existing terrain families, not new structural families. No TRAIN/HELDOUT structural test set was opened.

## Original controlled result

| Metric across 64 controller/world pairs per goal | BACKTRACK first | MOVE_S first |
|---|---:|---:|
| North goal reached within 128 decisions | **64/64** | 57/64 |
| North median first-visit step, among visits | **4** | 20 |
| South goal reached within 128 decisions | 60/64 | **64/64** |
| South median first-visit step, among visits | 14 | **2** |

With unvisited goals assigned capped time 129, **all 64/64 north pairs** favoured BACKTRACK by at least 12 decisions, and **all 64/64 south pairs** favoured MOVE_S by at least 10 decisions. Every one of the eight DEVELOPMENT worlds retained an eight-decision advantage across all eight model-seed/suffix-controller configurations.

This result is about **time to first observed goal occupancy**. Binary target arrival mostly saturates, and painted-cell differences remain inconsistent. There is no across-all-controller strict binary-success winner for either goal. Immediate remaining forward moves were recorded but do not establish global task reachability.

## Greedy neural first-action choice

| Restored controller | Correct for north goal | Correct for south goal | Combined |
|---|---:|---:|---:|
| Frozen | 10/16 | 16/16 | 26/32 |
| Always-move adaptation | 0/16 | 16/16 | 16/32 |
| **Contrastive adaptation** | **16/16** | **16/16** | **32/32** |
| Previous outcome-supervised adaptation | 7/16 | 16/16 | 23/32 |

The existing contrastive adaptation selected the faster initial direction on every one of these DEVELOPMENT cases **without additional training**. Because the geometry is deliberately simple and reused, this is not evidence of general long-horizon goal planning.

## Qualification and provenance

- 256 full independently replay-verified host-legal episodes, 128 counterfactual pairs, **19,144 decisions**, all source-replayed against restored checkpoints, complete material receipts and original neural probabilities.
- Action totals: BACKTRACK 215; MOVE_E 5,167; MOVE_N 4,576; MOVE_S 4,173; MOVE_W 4,835; RELOCATE_START 10; STOP 168.
- Original bounded CPU run **21.151 s**, measured peak process RSS **138,768 KiB**, mean model probability-inference latency **0.050879 ms** (the last excludes World construction and replay). Four focused smoke tests passed.
- Replayed original SHA-256 manifest contains 256 episode JSON files and one full report; original checkpoint and source hashes are in the machine-readable ledger.
- Earlier checkpoint ZIP SHA-256: bb0be90bb1cceab1b42496a5cac66f39d7b2a881de25228fba1a87a7a642ace1. Original audit source SHA-256: a976d4c25bcfc9cebefd2b18c818a81ec99e301a421ac622d6e3abf4ad4b296d.
- This is a **synthetic** 96x96 research World. The synthetic 128-decision horizon does not preserve native CyberSand's historical 64-logical-move restriction. No native actor/Workbench test, new checkpoint, paid API or GPU resource, or sealed structural HELDOUT qualification occurred.
- The full per-decision trajectory archive and existing eight-checkpoint archive are locally retained, but immutable GitHub Actions artifact publication remains outstanding. The compact ledger and original rendered trajectory gallery accompany this research. No automated training on push/PR.

## Next

Train a small goal-conditioned adapter on **multiple TRAIN exit geometries**, explicitly minimizing time-to-goal rather than maximising painting or saturating binary completion. Include genuinely unseen DEVELOPMENT structures, compare source-matched frozen/contrastive/always-forward controllers, and qualify STOP after successful arrival. Only after freezing the protocol should structural HELDOUT be opened.

Scope: only experiments/driv3r/training/long_horizon/** inside DRIV3R PR #1. Draft/HOLD; do not merge or promote Ready. CyberSand and Neurounner remain READ-ONLY.
