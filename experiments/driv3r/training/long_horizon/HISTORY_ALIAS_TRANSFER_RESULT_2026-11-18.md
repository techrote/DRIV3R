# HISTORY-ALIAS-TRANSFER-001 — rotation and remaining-recovery stress

**18 November 2026 · DRIV3R Draft/HOLD PR #1 · CPU-only synthetic inference audit**

## Finding

**Mixed: strong rotation invariance, but a material recovery-budget failure.** Two original `HISTORY-ALIAS-001` tiny controllers using raw, host-observable history features retained **832/960 correct first actions (86.7%)** when the same 20 DEVELOPMENT path motifs were rotated through all four cardinal orientations and recovery balances were changed by executing extra or fewer genuinely legal detour/backtrack operations. Their stack-blind comparator obtained **384/960 (40.0%)**. The engineered-slack controller tied raw overall (**832/960**).

However, the raw controller scored **208/320 (65%)** with only two recoveries remaining, versus **304/320 (95%)** with three and **320/320 (100%)** with four. The residual errors are overwhelmingly false BACKTRACK choices when the target is too deep to reach. **The recovery policy is not robustly budget-aware**; its success on the original 3-recovery fixture was not sufficient evidence of generalized recovery reasoning. A deterministic history-depth/budget rule solves this deliberately synthetic classification perfectly; the learned controller does not outperform it.

## Design and results

The 20 path motifs were selected from the **previously examined DEVELOPMENT set**, rather than being new motifs or a sealed holdout. Their world specifications use fresh DEVELOPMENT seeds 3507–3811, disjoint from the original 3007–3307 slice, with four rotations (0°, 90°, 180°, 270°). Each rotated pair has identical material field, protection, ink, cursor, heading, legal action menu, step count, recovery debt, and goal-only feature matrix; only ordered traversal history differs.

Each 2-, 3- or 4-operation remaining budget was created by a legal seven-, eight- or nine-step outward prehistory, fully undone by the corresponding BACKTRACK operations, followed by another legal path and terminal BACKTRACK. No state counters were manually overwritten. The synthetic 64-logical-move proxy was checked at fork entry. These are still deliberately chosen 4×4 path-alias structures, not independent native game worlds. Target labels are calculated from whether ordinary BACKTRACK can reach an already-inked historical goal within the remaining budget; STOP is an artificial task-specific alternative, not a general gameplay recommendation. The synthetic host still offers RELOCATE_START, though ordinary recovery is the only way to reach these targets.

| Controller | All first choices | 2 recoveries left | 3 left | 4 left | Attained reachable historical goals |
|---|---:|---:|---:|---:|---:|
| Stack-blind | **384/960** | 64/320 | 160/320 | 160/320 | 224/384 |
| Raw history depth + remaining | **832/960** | 208/320 | 304/320 | 320/320 | **384/384** |
| Engineered action-specific slack | **832/960** | 224/320 | 304/320 | 304/320 | **384/384** |

Both independently initialized checkpoint seeds yielded the same aggregate counts in each arm (blind 192/480, raw 416/480, engineered 416/480). Rotation-stratified raw accuracy was identical in all four orientations (208/240 each). This is rotation robustness **within a symmetric, clear-center local geometry**, not proof of general world-topology transfer.

At two remaining recoveries, the raw model made **112 incorrect first choices** (all selected BACKTRACK when STOP was the correct task label), even though it reached all 64 reachable goals. This exposes a failure to distinguish worthwhile from futile retreat, not an inability to perform consecutive BACKTRACKs when they are feasible.

## Source-faithful qualification

Five focused unit tests passed. The original campaign restored six frozen NPZ checkpoints (two seeds × three feature arms) from `HISTORY-ALIAS-001`; no optimizer updates occurred. A source-matched verifier recomputed all **2,880** DEVELOPMENT episode results and their summaries, checked source/checkpoint/evidence SHA-256s, and independently re-executed **36 complete sampled neural-probability and synthetic host-receipt continuations**, reconstructing their original legal prehistories from a fresh World. Source hashes and original artifact digests are in the compact JSON ledger. The full original episode and sample JSON files are retained separately in the local evidence archive, **not in Git**.

- **480** scenario states: 20 path motifs × four rotations × three remaining-recovery budgets × two ordered histories.
- **2,880** model continuations, **5,824** actual chosen actions, maximum five continuation actions in any episode.
- Original local CPU campaign: **13.275 s**, peak process RSS **133,260 KiB**; model-only median probability latency **0.022103 ms**, p95 **0.036120 ms** on this CPU. These timings do not include external host, GPU or model-loading latency.
- Provenance: preserved original path generator and frozen historical `d_common`/synthetic World fixtures. This is **not** native CyberSand, an ABI certification or a validation of real physical action limits.

## Research consequence

Prioritize **calibrated budget-conditioned STOP/BACKTRACK decisions** with multiple new histories and deliberately held-out length/budget combinations. Assess *false retreat cost*, target arrival and recovery depletion independently; compare against the perfect stack-rule baseline. Do not claim durable 72–128-decision autonomy from a nominal horizon; every continuation here lasted at most five actions. Structural HELDOUT remains sealed and native CyberSand acceptance outstanding. No paid services or GPU were used. This work changes only files under DRIV3R `experiments/driv3r/training/long_horizon/**`, and the consolidated PR remains Draft/HOLD.

## Reproduction (source-complete clone plus original external checkpoints)

```sh
python -m unittest experiments.driv3r.training.long_horizon.test_history_alias_transfer_lab -v
python -m experiments.driv3r.training.long_horizon.history_alias_transfer_lab --output /tmp/ha-transfer --checkpoints /path/to/HISTORY_ALIAS_original_evidence/checkpoints
python -m experiments.driv3r.training.long_horizon.history_alias_transfer_lab --verify --output /tmp/ha-transfer --checkpoints /path/to/HISTORY_ALIAS_original_evidence/checkpoints
python -m experiments.driv3r.training.long_horizon.history_alias_transfer_gallery /tmp/history-alias-transfer.svg
```

The complete original checkpoint binary archive remains outside Git. An immutable GitHub Actions checkpoint artifact, a structural holdout run and a physical CyberSand acceptance test remain outstanding.
