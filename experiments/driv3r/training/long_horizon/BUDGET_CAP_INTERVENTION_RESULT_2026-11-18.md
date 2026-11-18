# DRIV3R — BUDGET-CAP-INTERVENTION-001 (18 November 2026)

## Finding — causal-input diagnostic, NOT native game evidence

**Negative generalization result.** Holding the entire previously executed synthetic trajectory, painted cells, materials, protected cells, path stack, cursor, heading, action menu, steps, recovery consumption, and goal fixed, increasing only the *synthetic maximum permitted recovery count* (a `World.menu(max_recoveries)` parameter) generally made trained neural controllers **less**, not more, willing to BACKTRACK. A deterministic stack-depth-versus-remaining-credit rule remains strictly correct on this constructed reachability task.

This intervention is a valid *synthetic* host-cap counterfactual with unchanged legal menus at the initial decision, **not a real CyberSand configuration change**. The historical models were trained at fixed cap=12; alternative caps and recomputed normalized features are therefore out-of-distribution. No executed counterfactual trajectories and no native acceptance are claimed.

## Full DEVELOPMENT census (two original optimizer seeds)

480 source-matched DEVELOPMENT states × 2 seeds × 3 frozen controllers × 3 cap values (1, 3, 5 remaining credits) = **8,640 action probability vectors** and 2,880 matched three-cap triads. Every triad retains the identical pre-state SHA and action menu. The 2,880 no-intervention anchors reproduce all corresponding original first choices, oracle labels and state hashes.

| Controller | Matched cap triads | BACKTRACK→STOP as credit increases | Nonmonotone P(BACKTRACK) | BACKTRACK count at 1 / 3 / 5 left |
|---|---:|---:|---:|---|
| Multi-budget calibrated | 960 | **14** | **788** | 375 / 371 / 361 |
| Fixed-budget-3 | 960 | **96** | **928** | 600 / 548 / 504 |
| Explicit-budget-feature-masked | 960 | 0 | 0 | 403 / 403 / 403 |

The masked arm is an **invariance control**, not a success at budget reasoning: the intervention changes precisely four channels that the mask removes. It consequently cannot respond to additional recoveries and still makes 139 futile retreats when only one credit is available. The calibrated arm exhibits fewer discrete order inversions than the fixed-budget arm but its P(BACKTRACK) decreases somewhere in **788/960** matched triads. All probability measurements are model outputs from synthetic snapshots, not downstream goal achievements.

| Controller | First-choice correct at 1 / 3 / 5 credits | Futile BACKTRACK at 1 / 3 / 5 |
|---|---|---|
| Multi-budget calibrated | 769 / 795 / 773 | 127 / 28 / 22 |
| Fixed-budget-3 | 672 / 892 / 864 | 288 / 68 / 48 |
| Explicit-budget masked | 773 / 813 / 805 | 139 / 35 / 27 |

The correct oracle outcome is monotone by construction: BACKTRACK if the already-painted target is reachable along the existing history within the remaining credit, otherwise task-specific STOP. STOP is not advice for general gameplay.

## Integrity, provenance and resource bounds

- **Inputs:** original BUDGET-CALIBRATION-001 checkpoint/evidence ZIP SHA-256 `06573cd6d9502f7e132b3b3effd16a9c74dff2654f5668c692c9c9407906c2f9`. Source from exact DRIV3R branch head `cd642b7d319c80526eb97e980e5401d9fb0c74e2` and identical source SHA-256 values in the original campaign ledger. Original first-action episode JSON SHA-256 `e38b26ae35394f051fdc08be547821012f89d62ce9ddc0bebc80d18588a71f33`.
- Original checkpoint hashes are pinned and each of the six `NPZ` models restored without mutation. No new optimizer steps, no new checkpoint, no non-DRIV3R writes, no paid spend.
- **Tests:** six new source tests + seven existing source-backed calibration tests passed. An independent second inference pass reproduced all **8,640** probability vectors, all **2,880** unchanged-cap anchors and their computed metrics. The earlier original calibration run had independently replayed complete action receipts; those old receipts are not newly executed in this study.
- The primary CPU audit took **5.375 s**, **146,212 KiB peak RSS**, model-only **0.010139 ms median** and **0.025448 ms p95** inference (environment-specific; no native or feature-construction end-to-end timing).
- Full 8,640-vector action distributions, checkpoint models, source-matched prehistories, SHA-256 manifest and six-panel original-path witness gallery are retained in the external evidence archive. Git contains source, compact ledger, scientific report and SVG, **not** model binaries/large vectors.

## Scientific implications

The earlier 100 fixed-budget observational inversions cannot be explained away solely by changing prehistory: a controlled cap input produces 96 fixed-budget and 14 calibrated inversions, even with identical original history and ink. **However, this result cannot establish a game-contract-level causal effect**, because `max_recoveries` is a synthetic method parameter and normalized inputs are outside the original fixed-cap training distribution. The calibrated model remains poorly monotone at probability level. The masked model offers a useful exact no-change control, not a budget-aware policy.

**Next:** train with explicit cap variation as a first-class source-compatible task parameter and rerun these frozen states using a held-back DEVELOPMENT motif set. Compare complete autonomous recovery and a deterministic stack oracle; no further 128-horizon claims without 128 actually executed decisions. The sealed structural HELDOUT and native CyberSand boundary remain untouched. Immutable Actions publication of checkpoint ZIP remains outstanding.

Reproduction (requires the original verified local archive, not present in Git):

```sh
python -m unittest experiments.driv3r.training.long_horizon.test_budget_cap_intervention_lab -v
python -m experiments.driv3r.training.long_horizon.budget_cap_intervention_lab --checkpoints /path/to/evidence/checkpoints --original-episodes /path/to/evidence/episodes.json --output /tmp/driv3r-cap-audit
python -m experiments.driv3r.training.long_horizon.budget_cap_intervention_lab --checkpoints /path/to/evidence/checkpoints --original-episodes /path/to/evidence/episodes.json --output /tmp/driv3r-cap-audit --verify
python -m experiments.driv3r.training.long_horizon.budget_cap_intervention_gallery /tmp/driv3r-cap-audit/analysis.json /tmp/driv3r-cap-gallery.svg
```

**Ownership:** DRIV3R PR #1 only, Draft/HOLD, never merge/Ready; source only under `experiments/driv3r/training/long_horizon/`. CyberSand and Neurounner remain read-only.
