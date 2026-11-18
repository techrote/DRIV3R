# BUDGET-CALIBRATION-001 — multi-budget supervised retracing under unseen recovery budgets

**18 November 2026 — dedicated DRIV3R Draft/HOLD PR #1 — synthetic World, CPU-only**

## Finding

**Mixed / not yet generalized.** A new multi-budget curriculum (remaining budgets 2, 3, 4; two independently initialized 1,153-parameter tiny MLP rankers) improved first-action classification relative to a matched 3-budget-only training control on novel DEVELOPMENT path motifs at previously unseen budgets 1 and 5. Across all three evaluated budgets, multi-budget training achieved **851/960 (88.65%)** correct first actions versus **780/960 (81.25%)** for fixed-3 training, with **24 versus 132 false BACKTRACK attempts** toward analytically unreachable historical targets. With only one recovery left, the corresponding counts were **256/320 versus 204/320**; with five left they were **300/320 versus 272/320**. Both arms used identical model architectures, initialization seeds, stage count and optimizer-update budgets. Their exposure to unique training examples differs, as intended for a multi-budget curriculum comparison.

**However, this is not strong evidence of learned budget arithmetic.** A model that masked the explicit recovery-budget channels, but retained potentially correlated elapsed/history information, scored **877/960 (91.35%)**, outperforming the unmasked curriculum. The multi-budget-vs-fixed3 advantage is also motif-dependent: a 20-independent-motif clustered bootstrap interval for the first-action accuracy difference was **[-3.33, +18.44] percentage points**, including zero. A deterministic stack-depth <= remaining-budget rule scores **100%** on this deliberately constructed task and remains the stronger controller for it. Generalized neural recovery is therefore unproven.

## First-action results — independent optimizer seeds 731 and 732

| Model arm | All 960 | Budget 1 (unseen) /320 | Budget 3 (seen) /320 | Budget 5 (unseen) /320 | False retreats |
|---|---:|---:|---:|---:|---:|
| Calibrated multi-budget | 851 | 256 | 295 | 300 | 24 |
| Fixed-budget-3 control | 780 | 204 | 304 | 272 | 132 |
| Explicit budget channels masked | **877** | **272** | **304** | **301** | 27 |
| Deterministic history-depth/budget rule | **960** | **320** | **320** | **320** | 0 |

The last comparator is an analytic rule, not trained neural inference. The ablation removes explicit budget features at historical feature indices 18, 30 and 34 and the new raw budget slot, but does **not** eliminate every timing/recovery proxy. Therefore do not interpret its result as information-free evidence.

Pair-accuracy bootstrap intervals, clustered by the 20 distinct underlying path motifs (not by correlated rotation/seed episodes): multi-budget vs fixed3 **+7.40 pp**, descriptive 95% **[-3.33, +18.44] pp**; explicit budget masked vs fixed3 **+10.10 pp**, 95% **[-0.73, +21.88] pp**. Both cross zero. Nine motifs favored each multi-budget arm and eight favored fixed3, with three ties.

## Design and verification

- Novel TRAIN set: the **last 14 length-12 path motifs** from the deterministic path enumerator, disjoint from the earlier history-alias experiment's first 36 motifs. Four legal rotations, recovery budgets 2/3/4 and source-faithful TRAIN seeds 1611–1840.
- Novel DEVELOPMENT set: **20 length-14 motifs at positions 70–89** in the long-path list, not used in the original history-alias (first 40) or history-alias-transfer (first 20) audits; four rotations, **budgets 1/3/5**, fresh DEVELOPMENT seeds 3601–3830.
- State pairs maintain exactly identical ink, terrain/protection, cursor, heading, host-legal menus, recovery counters and elapsed steps while differing in path order. Budgets were reached exclusively through legal out-and-back operations. A source-backed test also verifies that original goal-only features cannot distinguish the matched pairs.
- Six newly trained checkpoint binaries: **2 seeds × (multi-budget, fixed3, explicit-budget-masked)**, each 3 curriculum stages at nominal horizons 36/72/128 and 64 Adam steps per stage, **192 updates per checkpoint**. At inference, action selection always comes from the full synthetic host-legal menu; ROOT relocation is not an evaluation-time oracle or forced first move.
- **7 tests passed**; SHA-256 source and checkpoint identities recorded; independent recomputation of **2,880 first-action episodes**; independent restoration/replay of **342 complete six-action-maximum sampled continuation transcripts**, including complete probabilities, actual host receipts, legal menus, and state checksums.
- From those sampled continuations, reachable-target hits: **82/106** calibrated, **94/106** fixed3, **91/106** explicit-budget-masked. This limited sample shows that better first-action accuracy does not necessarily yield better recovery completion, and sampling was intentionally enriched for difficult near-goal cases; do not extrapolate it to all 960 cases.
- The new source run took **13.959 seconds CPU wall time**, peak process RSS **168,080 KiB**, model-only median inference **0.023014 ms**, p95 **0.031960 ms**. Model-only timings exclude World feature calculation, loading, native latency and GPU.

The original episode-level JSON and NPZ checkpoints are in the separate locally verified evidence archive; Git contains only implementation, tests, small evidence ledger, report and original SVG gallery. Neither checkpoint binaries nor this large local archive are committed to Git. Immutable Actions artifact publication remains outstanding.

## Limitations and next decision

These are synthetic 4×4 path aliases embedded in an artificial 96×96 material world, not physical CyberSand gameplay. STOP is a task-defined response to an unreachable *historical* goal and should not be construed as the best way to continue a wider game. The developmental protocol has no blinded structural HELDOUT results. None of the sampled continuations lasted longer than five chosen decisions, so the nominal 128 stage is **not evidence of sustained 128-step autonomy**.

The next defensible experiment should first test against the deterministic history/budget rule under more realistic observation uncertainties and real read-only game traces; only introduce a larger model if it can improve useful downstream behavior rather than memorizing recoverability labels. Compare complete goal attainment and false-recovery depletion, not only initial classifier accuracy. Keep structural HELDOUT sealed until a candidate and stop rule are frozen.

## Reproduction

```sh
python -m unittest experiments.driv3r.training.long_horizon.test_budget_calibration_lab -v
python -m experiments.driv3r.training.long_horizon.budget_calibration_lab --output /tmp/budget-calibration-evidence
python -m experiments.driv3r.training.long_horizon.budget_calibration_lab --output /tmp/budget-calibration-evidence --verify
python -m experiments.driv3r.training.long_horizon.budget_calibration_gallery /tmp/budget-calibration-evidence /tmp/BUDGET_CALIBRATION_TRAJECTORIES_2026-11-18.svg
```

No game edits, no paid compute, no new PR/branch/workflow, and no merge or Ready promotion are authorized. DRIV3R `main` and all external repositories remain unchanged.
