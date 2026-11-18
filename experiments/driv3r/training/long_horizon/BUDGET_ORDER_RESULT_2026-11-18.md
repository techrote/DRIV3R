# DRIV3R — BUDGET-ORDER-001 (18 November 2026)

## Independent DEVELOPMENT reanalysis, not a new optimizer run

All 2,880 original BUDGET-CALIBRATION-001 first decisions and 342 sampled continuation receipts were reanalyzed from a source-identified original ZIP with verified archive SHA-256, member hashes and checkpoint hashes. No native execution, no paid compute and no structural HELDOUT access.

**Negative result:** the fixed-budget model shows **100/320 (31.25%) recovery-budget-order inversions** on paired motif/rotation/history-class/optimizer-seed triads: it chooses BACKTRACK with a smaller budget and STOP with a larger budget. Forty-four inversions occur between budgets 1 and 3; 56 between 3 and 5. Neither the multi-budget-calibrated nor the explicit-budget-feature-masked model shows an inversion (0/320 each). Target depth and oracle labels remain consistent and monotone in each triad.

| Arm | BACKTRACK by remaining budget 1 / 3 / 5 | Inversions | First-action correct by budget 1 / 3 / 5 |
|---|---|---:|---|
| Multi-budget calibrated | 40 / 151 / 180 | **0/320** | 256 / 295 / 300 |
| Fixed-budget | 220 / 176 / 120 | **100/320** | 204 / 304 / 272 |
| Multi-budget, explicit budget masked | 56 / 160 / 187 | **0/320** | 272 / 304 / 301 |

An explicit deterministic stack-reachability rule scored 960/960 in the previous original census; neural policies did not beat it. At budget 1, the fixed-budget model made 116 futile retreats, while calibrated missed 64 reachable goals despite avoiding futile retreats.

### Neural distributions and receipts

The 342 **nonrandom** sampled synthetic action-receipt trajectories were checked for legal action menus, probabilities and actual chosen actions. These sampled numbers must not be generalized to population calibration.

| Arm | Wrong first decisions / 114 | Mean confidence when wrong | Maximum confidence when wrong | Five-bin top-class ECE |
|---|---:|---:|---:|---:|
| Calibrated | 24 | 0.674 | 0.870 | 0.094 |
| Fixed-budget | 20 | **0.851** | **0.999979** | 0.118 |
| Explicit budget masked | 15 | 0.637 | 0.791 | 0.050 |

### Scientific caveats

1. **Not a causal budget intervention.** The original fixture changes detour length, painted ink and preceding history when changing the remaining budget. The state SHA changes at every budget. The result is only a controlled observational association.
2. STOP is a task-specific negative label for the previously painted goal, not a safe general-game STOP recommendation.
3. The synthetic host proxy has a 64 logical-move limit. No sustained 72–128 actual autonomous decisions were demonstrated.
4. No new checkpoint was trained. Complete original trajectories/checkpoint binaries are retained in a locally verified ZIP but not immutable GitHub Actions artifacts. Git-only reproduction cannot presently recover the original source archive.

### Resources and next experiment

Original BUDGET-CALIBRATION-001 campaign: 13.959 seconds CPU wall, 168,080 KiB peak RSS, 0.023014 ms median/0.031960 ms p95 model-only inference. Present audit: zero optimizer steps; same original checkpoint hashes and action receipts preserved.

Next: compare controllers against a deterministic stack planner with **identical full prehistories/ink** while independently varying a host-observable budget. Only then test true budget sensitivity and autonomous recovery with complete receipts.
