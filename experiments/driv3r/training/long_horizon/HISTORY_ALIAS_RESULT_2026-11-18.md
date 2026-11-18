# HISTORY-ALIAS-001 — matched painted-state / different-stack experiment

**18 November 2026 · DRIV3R Draft/HOLD PR #1 · long_horizon only · CPU synthetic research · no native acceptance**

## Finding

**Positive but tightly bounded:** a tiny neural ranker conditioned on **raw, host-observable path-stack depth** distinguishes two otherwise identical painted states where the correct immediate decision is BACKTRACK in one and STOP in the other. The raw-depth controller chose the supervised action in **455/480** DEVELOPMENT episodes, versus **240/480** for the equal-capacity stack-blind controller. Adding an explicitly calculated depth/budget slack improved only marginally, to **456/480**.

This does **not** prove general long-horizon autonomy, robust recovery in CyberSand, or emergent planning. The supervision is analytically constructed from path membership and remaining recovery budget; an explicit rule baseline is perfect by construction. The improvement demonstrates that the omitted information was necessary for the matched alias classification, not that a neural model outperforms an algorithmic stack lookup.

## Controlled procedure

- Generate pairs of distinct self-avoiding paths across the same 4x4 local cell set, with identical origin, final two path cells, current position, heading, ink, materials, protected cells, steps, last action, recovery consumption and full host-legal menu. Path ordering differs.
- Construct a real **8-BACKTRACK** legal prehistory and one more legal BACKTRACK after walking the path, leaving three recovery operations from the fixture's 12-operation budget. In one paired stack the already-inked target is 1–3 backtracks away; in the other it is more than three and cannot be revisited through ordinary ink movement.
- The **goal-only** observation matrix is bit-for-bit identical across each paired fork; later inherited recovery/history feature groups can already leak some stack cues and were therefore treated as a distinct legacy comparator, not as part of the truly blind baseline.
- TRAIN: 36 independent path-pair motifs with lengths 10–12 and synthetic world seed partition 1007–1307. DEVELOPMENT: 40 different motifs with lengths 13–14, synthetic world seed partition 3007–3307. Both use the same clear central 4x4 grid design; this is **length and motif transfer, not structural world generalization**. Structural HELDOUT remains sealed.
- Two independent optimizer initializations (619, 620), matched 1,153-parameter MLPs, 96 optimizer steps per 36/72/128 stage (288 updates per model), balanced BACKTRACK/STOP TRAIN labels and no paid compute. Training labels are source-known task affordances, not independently inferred native outcomes.

## Autonomous DEVELOPMENT results (40 unique pairs × 2 choices × 2 optimizer seeds × 3 nominal horizons)

| Feature configuration | Correct first actions | Recoverable goals reached | Correct immediate STOP on unrecoverable goal |
| --- | ---: | ---: | ---: |
| Goal-only / stack-blind | **240/480** | 240/240 | 0/240 |
| Legacy recovery + 4 history features | 288/480 | 188/240 | 94/240 |
| **Raw stack depth and remaining budget** | **455/480** | **240/240** | 215/240 |
| Raw features + explicit action-specific budget slack | **456/480** | **240/240** | **216/240** |

The blind controller cannot distinguish any matched pair's initial observation: its 50% balanced-label ceiling is structural. Legacy path-preview features help only modestly and can also interfere with positive recovery. The raw-depth comparator is only **one correct first decision** behind the engineered-slack comparator across 480 episodes.

Descriptive case-cluster bootstrap (40 motifs, 5,000 resamples, repeated seed/horizon evaluations kept together):

| Comparator minus blind first-action accuracy | Paired percentage-point gain | Descriptive 95% interval |
| --- | ---: | ---: |
| Legacy | +10.0 | +3.8 to +16.3 |
| Raw depth | +44.8 | +39.8 to +48.8 |
| Explicit slack | +45.0 | +40.0 to +48.8 |

Repeated horizon/seed runs are **not** independent 480-world replications. Intervals describe variation across this chosen set of 40 DEVELOPMENT path motifs; they are not claims about unseen CyberSand worlds.

## Reproduction and qualification

Five focused unittest cases passed, including exact goal-only feature equivalence, matched painted/material state, genuinely consumed recovery operations, balanced teacher labels and budget bounds. Eight checkpoint NPZ files were written and restored with SHA-256 identity. An independent source-matched verifier replayed **all 1,920 DEVELOPMENT episode summaries** and **48 original complete neural-probability / host-action-receipt transcripts**. These are synthetic receipts; **zero native CyberSand runs**.

- Original CPU campaign: **21.953 seconds**, peak process RSS **110,864 KiB**, total **4,225 selected continuation actions**.
- Model-only median inference: **0.008307–0.008599 ms** on this local CPU (excludes host and feature extraction).
- Immutable source hashes, checkpoint hashes, per-case identifiers, sampled trajectory SHA-256 values and action-count distributions are in `HISTORY_ALIAS_EVIDENCE_2026-11-18.json`.
- Gallery `HISTORY_ALIAS_TRAJECTORIES_2026-11-18.svg` is generated from six original DEVELOPMENT path pairs, showing identical painted footprints and different ordered histories.

Run from a source-complete clone with NumPy available:

```bash
python -m unittest experiments.driv3r.training.long_horizon.test_history_alias_lab -v
python -m experiments.driv3r.training.long_horizon.history_alias_lab --output /tmp/history_alias_evidence
python -m experiments.driv3r.training.long_horizon.history_alias_lab --verify --output /tmp/history_alias_evidence
python -m experiments.driv3r.training.long_horizon.history_alias_gallery /tmp/history_alias.svg
```

The experimental source itself is committed to PR #1. Checkpoint binaries and full original episode logs must remain **outside Git**; the complete locally verified archive is available from the associated run handoff. Immutable GitHub Actions artifact publication remains outstanding and must not silently trigger paid training.

## Caveats and next discriminating study

The target is an *already painted, historical* cell, so adjacent ink movement cannot reach it and a deterministic stack-reachability rule solves the toy task. STOP is the task-level choice when a target cannot be reached within the remaining three recoveries; that is not a general instruction to terminate game play. The simulation includes `RELOCATE_START` in its legal menu, but no evaluated controller chose it. No physical/native contract validation was run. Although the nominal feature horizons include 128, these controlled continuation episodes terminate after at most 12 additional actions and do **not** establish sustained 128-action behaviour.

The natural next experiment is **ambiguous history on less regular traversals where neither trivial STOP nor a perfect stack-affordance rule is sufficient**: future-goal switching, externally perturbed actions, multiple reachable ancestor targets, truly differentiated material obstruction, and a held-out map topology. Require independent suffix-controller agreement and autonomous success beyond 72–128 actual decisions before advancing claims. Avoid opening the structural HELDOUT until the experimental protocol is frozen.

**Ownership:** all work confined to `experiments/driv3r/training/long_horizon/**` in DRIV3R's single Draft/HOLD PR #1. No CyberSand/Neurounner or DRIV3R-main writes, no merge, no new workflow or paid API/GPU actions.
