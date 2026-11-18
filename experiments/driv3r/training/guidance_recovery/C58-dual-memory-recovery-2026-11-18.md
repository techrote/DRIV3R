# C58 — Dual-memory degradation, recovery and personality audit

**Status: RESEARCH HOLD · synthetic-only · 18 November 2026.** Laboratory owner: `experiments/driv3r/training/guidance_recovery/**`, consolidated DRIV3R Draft/HOLD PR #1. No production/native CyberSand changes, and no revised host contract.

## Experiment and controls

This run follows C57's source-verified recovery harness. It erases **both** previously observed local neighbor menus and independently recorded executed-action edges, at 0%, 50%, and 100%. At each recovery step it reobserves only the current legal menu, rejects no host rule, and retains previously accumulated visit counters. Removing historical action entries is not the same as losing all worm internal state, and the new legal menu is deliberately always available. Formerly executed edges, where retained, remain proposals only, subject to current host approval. Remote goals remain genuinely unobserved; observed waypoint targets are explicitly easier.

Train on fresh familiar-family graph identities; test on 24 independently seeded graphs: eight Wilson/Kruskal and sixteen fresh tree/braid/Prim/Prim-braid. No C56 or C57 train/evaluation graph hash recurs in the C58 evaluation. One host-issued recovery cue is applied at tick 9, weak and strong guidance are matched against independently executed no-guidance. Each policy continues within the same world state, RNG state and global clock. Six arms are C18-A0, C34-RESET, frozen non-neural C57 TRACE, non-neural C57 HEURISTIC, newly trained C58 ALIGNED and shuffled-label C58 SHUFFLED.

The **aligned** and **graph-shuffled** CPU models each received 840 AdamW updates on 6,666 host-observed legal candidate examples from 21 contributing training graphs. Both were refitted byte-identically in a second optimizer run. Labels imitate a local transparent route/fallback heuristic; no shortest-path oracle, unseen geometry, or hidden remote target knowledge is supplied at inference. This is **not** proof of learned independent map understanding.

## Independent held-out completion

Each cell gives completed goals / 30 guided episodes, pooled across weak and strong guidance. The same 15 eligible graph identities recur across memory levels and goal types; these are *paired* conditions, not 180 independent graph identities.

| Goal type | Joint historical loss | RESET | TRACE | HEURISTIC | ALIGNED | SHUFFLED |
|:--|--:|--:|--:|--:|--:|--:|
| Observed waypoint | 0% | 23 | 25 | 25 | **25** | 23 |
| Observed waypoint | 50% | 23 | 24 | 24 | **27** | 23 |
| Observed waypoint | 100% | 23 | 23 | 24 | **26** | 23 |
| Remote unknown | 0% | **9** | 9 | 7 | 6 | **9** |
| Remote unknown | 50% | **9** | 9 | 7 | 6 | **9** |
| Remote unknown | 100% | **9** | 9 | 7 | 6 | **9** |

C18-A0 baseline completed 17/30 observed and 8/30 remote goals under every loss condition. The C58-ALIGNED gain at **50% joint corruption was +4 goals over RESET and SHUFFLED** with no goal-level regressions in that cell. A descriptive 15-graph bootstrap gives **0 to +9** additional goals; it is not a confirmatory confidence interval given the adaptive C48–C57 research history. The effect is clustered in familiar families; only one additional Wilson goal and none in Kruskal. At 100% loss the observed-waypoint gain was +3 with one condition regression.

**Negative result:** On previously unseen *remote* goals, ALIGNED completed **6/30** versus RESET **9/30**, with three goal-level regressions at each loss level. The same failing actions recur when historical memory is varied because this goal is not present in the local observed-route graph. The controller does not satisfy a cross-family or distant-goal recovery gate.

## Memory ablation and personality

Pre-recovery observed-route retention fell from **30/30** with intact memory, to **4/30** after 50% joint erasure and **0/30** after complete erasure. Mean retained action entries were 9.00, 4.53, 0.00 respectively, and mean retained neighbor menus were 7.00, 5.67, 1.00 (the contemporary/current node remains observable). The heuristic's observed goal gains under full erasure arise from current-menu local fallback, not reconstructing lost routes.

The controller is intentionally stochastic. Under 50% loss, ALIGNED executed 36 personality-preferred versus 20 opposed turns on observed waypoints, while RESET executed 31 versus 26; on remote goals ALIGNED was 79 versus 76, and RESET was 72 versus 68. These are *descriptive proxy counts* with different trajectory lengths and opportunity sets, **not** evidence that the original meandering/branching personality survives guidance. A matched long-horizon personality acceptance gate remains outstanding.

## Source identity and qualification

- **1,620** independently executed policy suffixes: 270 matched cells × six policies, including 90 independently re-executed no-guidance conditions × six policies (exact movement and painting hash parity).
- Zero executed synthetic-host legality rejections and zero protected-cell changes, with all outcome assertions enforced during execution.
- 18 source-matched gallery trajectories retained; nine independently replayed against reconstructed frozen worlds, prefixed state and action/painting hashes.
- **5/5** local provenance, dual-corruption nonfabrication, legality/personality sanity, deterministic optimizer refit, and independent replay tests passed.
- Full executed online, training, summary and trajectory records preserved losslessly in `C58-executed-ledger.zlib.b64` with SHA-256 per-file verification. `C58-extract-ledger.py` reconstructs them. The rendered gallery depicts actual coordinate traces, not imagined terrain.
- Trained checkpoint binaries are preserved in the owner-evaluation archive, *not* Git history. Immutable GitHub Actions checkpoint artifact custody remains outstanding.

## Scientific disposition

**HOLD; do not promote.** This controlled synthetic intervention shows a small, fragile recovery benefit when the host goal is already local/observed and both historical memories are degraded. The result is valuable for studying fallback from missing memory, but does not demonstrate robust unknown-goal navigation or whole-trajectory personality. A realistic subsequent study needs delayed outcome feedback on distant goals with cross-family positive baselines, plus separate human/native gameplay acceptance; avoid treating further local label gains as the main scientific objective.
