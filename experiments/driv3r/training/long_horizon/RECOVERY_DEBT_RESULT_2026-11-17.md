# RECOVERY-DEBT-001 — legal pre-spent recovery budget is not sufficient for robust backtracking labels

17 November 2026 · DRIV3R long-horizon research · **negative result; no new adapter trained**.

## Question and controlled construction

Can we obtain a substantial, controller-stable benefit from immediate BACKTRACK by making a tempting pocket costly when nearly all recovery operations have been spent? This follows the negative OUTCOME-MARGIN-001 audit, in which 128-step outcome labels varied by suffix controller. The present experiment uses **11 actual legal BACKTRACK operations** in a fully replayable 28-decision synthetic prefix (11 off-path painted moves, 11 BACKTRACK returns, and six eastward progress moves). No recovery counter was injected or rewritten. A final fork exposes exactly `BACKTRACK`, `MOVE_S`, `RELOCATE_START`, and `STOP` in each case; the south cell is a blind pocket, and the preceding corridor cell has a possible north exit. The forced first actions are both on the legal menu.

The underlying 96×96 synthetic material world is composed from the unchanged DRIV3R `World` and `retrace_choice_lab`/`sequential_probe_lab` feature machinery. Fixture terrain is edited **before** legal prefix execution, not during an episode. No native CyberSand observation contract is modified. The first-action difference is causal within this constructed fixture, but the fixture has been designed to expose recovery-debt effects, not sampled from real gameplay.

Two independent checkpoint seeds (811/814), four previously trained suffix arms (`frozen`, `always_move`, `contrastive`, `outcome`), and four terrain families per split were compared. Each matched pair forks from an identical state, forces either `MOVE_S` or `BACKTRACK`, then allows the **same restored model** to autonomously select subsequent actions for up to 128 decisions. Primary delta = new painted cells after forced MOVE minus after forced BACKTRACK; negative values favour retreat. A robust substantial win requires at least eight more painted cells after BACKTRACK **for all eight seed/arm combinations** on the same family. TRAIN and DEVELOPMENT remain separate. Structural HELDOUT remains sealed.

## Results

| Finding | TRAIN | DEVELOPMENT |
|---|---:|---:|
| Distinct terrain cases | 4 | 4 |
| Matched pairwise comparisons | 32 | 32 |
| Controller-sign conflicts | 4/4 | 3/4 |
| Cases with robust BACKTRACK margin ≥8 painted cells | **0/4** | **0/4** |

The closest DEVELOPMENT candidate, `pickets`, favoured BACKTRACK in 7/8 seed/arm combinations, but one frozen suffix was tied (0 cell difference). Thus it fails the agreed eight-cell robust criterion. The DEVELOPMENT `islands` case ranged from a 118-cell BACKTRACK advantage to a 33-cell MOVE advantage, and `zigzag` ranged from a 112-cell BACKTRACK advantage to a 28-cell MOVE advantage. Recovery debt makes sizeable backtracking gains possible for *individual suffix policies* but does not make their training labels stable.

| DEVELOPMENT suffix arm | BACKTRACK wins / 8 | Mean MOVE − BACKTRACK painted cells |
|---|---:|---:|
| Frozen | 2 | −24.125 |
| Forward imitation | 4 | −7.250 |
| Geometry contrastive | 5 | −36.500 |
| Outcome supervised | 3 | −3.875 |

## Qualification and evidence

The experiment restored **eight SHA-256-identified checkpoints** from the prior OUTCOME-FORK-001 evidence ZIP (`bb0be90bb1cceab1b42496a5cac66f39d7b2a881de25228fba1a87a7a642ace1`). It produced 64 matched comparisons, **128 fully recorded synthetic trajectories** and **12,138 host-legal decisions**. Each recorded decision includes the offered menu, complete neural action probability vector and source `World.apply` receipt. An independent verification pass restored every checkpoint and re-executed all trajectories, checking probabilities, offered actions, receipts, state hashes, action totals, and the **129-file SHA-256 manifest**. Three focused tests passed, including sealed-HELDOUT rejection.

CPU execution took approximately 15 seconds for the evidence run on this host; observed peak RSS was approximately 120 MiB. Per-arm measured neural selection latency and full action histograms appear in `RECOVERY_DEBT_EVIDENCE_2026-11-17.json`. These latencies are descriptive, not cross-machine performance measurements. The full original events and checkpoint binaries are kept in a local evidence archive, not Git history. No new optimizer or adapter was used.

Source: `recovery_debt_audit.py` and `test_recovery_debt_audit.py`. Local run: `python -m experiments.driv3r.training.long_horizon.recovery_debt_audit --checkpoint-zip <outcome-fork-evidence.zip> --out <fresh-output-dir>`, then repeat with `--verify`. Input source blob identities were checked against the exact DRIV3R PR branch before development. The checked-in compact JSON ledger reports every case and original checkpoint SHA; the full trajectory gallery is derived from real receipts, not fabricated movement.

## Interpretation and next step

**No TRAIN family passed the robust label gate; do not train another adapter on this result.** The contradiction persists even after genuine legal recovery expenditure, so simple danger proxies or a single frozen teacher remain inadequate. The eight checkpoint combinations are *not eight independent foundation-model families*: at least one adapted arm shares an identical checkpoint with its comparison in seed 811, and all descend from the same tiny policy architecture. These experiments are synthetic and don't imply native CyberSand reliability.

The next discriminating experiment should measure *goal completion and controllable surviving option value*, not merely painted-cell count, and should include several structurally distinct recoverable worlds with strong positive/negative labels. Retain matched legal action menus, explicit prefixes, multi-controller suffix replication, and an untouched structural HELDOUT. Do not change the CyberSand game, immutable historical `d_common`, or another worker's files.

**Boundary:** DRIV3R PR #1 Draft/HOLD only; no new PR/issue/workflow, no merge, no native modification, no production setting changes, no paid compute, and no automatic CI training. Checkpoint publication as immutable Actions artifacts, native evaluation, and sealed HELDOUT qualification remain outstanding.
