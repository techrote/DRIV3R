# DRIV3R / INK-ALIAS-MEMORY-001 — 36→72→128-action synthetic history alias
**18 November 2026 · DEVELOPMENT-only · Draft/HOLD PR #1 · NOT native CyberSand acceptance**

## Experimental question
Can a small actually trained memory-conditioned decision head distinguish two legal CyberSand-style painting histories that have **identical present cursor, heading, legal menu, current goal, ink count, last eight action labels and radius-two occupancy**, yet require opposite first actions for shortest reachable continuation?

Four original DEVELOPMENT world families (pickets, ridges, islands, zigzag) and two independent DEV seeds each were tested at 36, 72 and 128 preceding host-menu actions. All pairs used the same predeclared local clearing footprint (computed from *both* possible histories before either outcome) and the unchanged synthetic `World.apply` and host-owned legal menu. Both histories end at cursor (48,54), with goal (48,44). The earlier east-ink path makes WEST the optimal first action; the earlier west-ink path makes EAST optimal. Each pair has exactly the same **limited observation**, and 24/24 have opposite labels, giving any deterministic last-eight/local-only policy a 50% pair-balanced ceiling.

**This is an engineered information-necessity positive control, not evidence of generalized gameplay skill.** It deliberately makes two distant prior-ink cells predictive; no end-to-end recurrent state was trained.

## Genuine bounded optimizer run
A three-input, two-output NumPy softmax decision probe uses prior ink occupancy at (50,48) and (46,48), plus bias. It received **240 actual full-batch Adam updates** on 72 balanced TRAIN states (four TRAIN world families × three TRAIN seeds × three horizons × two histories). Mean training loss decreased 0.6931472 → 0.0071685. The trained `.npz` checkpoint SHA-256 is `ea7a0e291a898b2b9215c676e2bdcbd466284305292a42d767dd8906e8c6a981`.

Two previously frozen 120-update goal-conditioned neural checkpoints (seeds 711, 712) were independently loaded, measured, and compared on 48 matched DEV decision states. No tuning or new optimizer updates were applied to them.

## Results (48 DEVELOPMENT cases per controller)
| Controller | Correct shortest-path first action | Reach goal after scripted suffix | Planned suffix steps | Median decision latency |
|---|---:|---:|---:|---:|
| **Trained distant-ink probe** | **48/48** | 48/48 | **528** | 0.118530 ms |
| Last-eight/local-only fixed WEST control | 24/48 | 48/48 | 624 | 0.000711 ms |
| Frozen neural 711 | 14/48 | 48/48 | 624 | 0.181809 ms |
| Frozen neural 712 | 15/48 | 48/48 | 624 | 0.100033 ms |
| Full-state BFS oracle | 48/48 | 48/48 | 528 | 0.000366 ms |

After the **single model-selected action**, the suffix is executed by a deterministic empty-cell BFS planner. Thus *all controllers* reached the goal with scripted rescue; success is not an autonomous neural-policy result. The learned probe's narrow benefit is **24 more optimal first actions** and 96 fewer subsequent scripted BFS actions than the local-only control. Frozen 711 chose EAST 28 times and SOUTH 20 times; frozen 712 chose EAST 30 and SOUTH 18 times, demonstrating poor action agreement on this designed task, not broader deficiency.

## Integrity and computation
Original host-produced receipts were retained for all **240 matched controller episodes / 22,048 applied synthetic actions** and independently re-executed. A separate verifier recomputed **144 actual model probability vectors** from the saved probe and both frozen checkpoint binaries, matched prelude state hashes, checked the full action/legal-menu acknowledgements and original final-state hashes, and rejected forged menu/action receipts in five focused tests. Both full local campaigns reproduced the trained checkpoint SHA-256, compressed receipts SHA-256 and per-case outcome/trace hashes **byte-identically**. Original eight-panel SVG trajectories were rendered from executed 128-action receipt paths.

First completed campaign: 14.5855 CPU-wall seconds, 136,720 KiB process peak RSS; local choice-latency percentiles and action distributions are in the evidence ledger. Recorded receipt archive SHA-256: `f3e80965305e09f9b1e0c325c426e2d85ef4e300d29008c27e503e0ff43979de`. Full 1.3 MiB compressed receipt and checkpoint archives are retained as local handoff evidence, **not yet in immutable GitHub Actions artifacts**.

## Important limits
- The 36/72/128 values count host-menu **decisions in a constructed synthetic prelude**. Because `RELOCATE_START` charges path length, the 72- and 128-decision preludes consumed 136 and 248 synthetic logical moves respectively, **beyond the historical native 64-logical-move limit**. Do not interpret these as native-authorized long sessions.
- Static material/protected cells within the shared synthetic test corridor were cleared *before* trajectories; variation outside the corridor remained, but this is not a blind unseen-topology transfer. Structural HELDOUT families were never accessed.
- The head was given two hand-selected distant old-ink indicators, not a general recurrent history embedding. Training/DEV structural grammar is highly shared. A future learned sequence encoder must face longer and less directly marked dependencies and independent held-out topologies.
- The existing `d_common` fixture is immutable and is not CyberSand's canonical action/observation ABI. No native gameplay, material physics, Workbench, paid API/GPU, other lab, or external repository was modified.

**Disposition: narrow positive-control only; do not promote a DRIV3R gameplay policy.** Next: train a bounded real sequence encoder on nontrivial full legal histories, compare 8/16/32/full-memory windows with matched state-menu controls, then test complete autonomous recovery and source-matched native receipts separately.
