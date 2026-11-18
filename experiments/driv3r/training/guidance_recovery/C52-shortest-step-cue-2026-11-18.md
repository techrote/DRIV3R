# C52 — Action-level shortest-step cue calibration · 18 November 2026

**Repository ownership:** DRIV3R PR #1 only; Draft/HOLD. Synthetic research; not native CyberSand validation.

## Hypothesis and implementation

C51 found no meaningful Wilson/Kruskal transfer for frozen C40. C52 tests an alternative
mechanism rather than another frozen-checkpoint replication: a small **newly trained
neural correction head** predicts, from a host-observed local action menu, history and
signed BACKTRACK cue, whether host-legal BACKTRACK is the **uniquely shortest-route
next step**. Full protected-cell geometry is used *only* by the offline BFS teacher
and blind scorer; never at policy inference. A bounded learned delta modifies only
the host-legal BACKTRACK logit, and only while the host's BACKTRACK cue is active.
No mandatory scripts and no novel action types.

Training and blind classification use legally executed mid-trajectory prefixes and
consecutive live synthetic-host suffix states. Training families are DFS/Prim
(tree, braid, prim, prim_braid); blind and online families are independent Wilson
and Kruskal topologies. The same graph identity never enters more than one cohort,
and C51 graph identities are excluded from blind and online generation. Source-
matched history and active cue are observational, not an authority override.

## Actual bounded optimizer and evidence

- Generated training graphs: **64**; 44 produced eligible labeled states. Retained
  **2,392** highly correlated host-visited labeled states,
  including **1,116** positive labels. Effective
  independent sample size is closer to graph count, not state count.
- Separate Wilson/Kruskal blind cohort: **16 graphs**, **502** labeled states.
- Separate Wilson/Kruskal online cohort: **32 graphs**,
  **164** matched graph/cue rows, **984**
  policy outcome records. **574** actual independently
  executed suffix rollouts; **410** none/continue
  policy records reuse A0's physically identical host execution, with independent
  continuation parity checks. Also 18 gallery replay suffixes.
- CPU AdamW **836 updates per head** over 22 epochs;
  separately trained aligned and graph-shuffled controls, with separate label-derived
  class weights. Trained checkpoint binaries remain **outside Git**.
- Two repeated complete fit passes produced byte-identical checkpoint hashes;
  **7/7 unit/regression tests PASS**, including real replay and legal-menu isolation.

## Blind classification

| Model | Accuracy | Balanced accuracy | Brier score (lower better) |
|---|---:|---:|---:|
| C52-ALIGNED | 0.544 | 0.537 | 0.291 |
| C52-SHUFFLED | 0.530 | 0.522 | 0.253 |
| Majority baseline | 0.522 | — | — |

ALIGNED modestly improved raw classification accuracy but worsened Brier
calibration relative to SHUFFLED, failing a preregistered gate.

## Held-out online recovery

**123 guided episodes per policy** across both
out-of-family generators. Goal completions:

| Policy | Goal events | Wilson | Kruskal | Coarse regressions vs RESET |
|---|---:|---:|---:|---:|
| C18-A0 | 8 | 5 | 3 | 1 |
| C34-RESET | 8 | 5 | 3 | 0 |
| C40-PAIR | 9 | 5 | 4 | 1 |
| C52-ALIGNED | 8 | 5 | 3 | 1 |
| C52-SHUFFLED | 8 | 5 | 3 | 1 |
| C52-RANDOM | 8 | 5 | 3 | 1 |

The aligned head equals RESET and SHUFFLED at **8** completed
goals. The paired C52-ALIGNED minus RESET difference is
**+0** goals;
95% graph-cluster descriptive bootstrap interval
**[-3.0, 3.0]**,
with one favorable and one unfavorable graph cluster.
Frozen C40 gains only **+1** goal in this evaluation,
consistent with C51's weak unseen-family transfer. This is a small, underpowered
sample, not a positive efficacy signal.

The aligned head fired during **721**
active-cue states, but its online goal result was unchanged versus SHUFFLED.
All **41** matched no-guidance action and
painting hashes are identical for every policy. There were **zero executed
host rejections or protected-cell mutations**. These are safety and unguided
behaviour checks, **not** full guided personality acceptance.

## Interpretation / negative findings

**REJECT C52-ALIGNED for promotion.** The action-level graph-oracle surrogate
was learnable to a small degree but did not improve actual executed goals.
Offline shortest-step labels do not uniquely determine long-horizon success,
and the local feature set lacks distant branch structure. The graph-shuffled
control attained better probabilistic calibration, while guided condition-level
nonregression failed. Graph-cluster intervals are descriptive and unadjusted
for the C40–C52 research selection series.

Still outstanding: direct native CyberSand ABI/physical acceptance, real dynamic
material reactions, full whole-trajectory personality qualification, external
novel topology family generalization, and immutable GitHub Actions checkpoint
custody. The frozen `d_common` fixture remains historical, not canonical.

## Source and reproduction

- `c52_shortest_step_cue.py`: complete executable Python trainer/evaluator;
  `test_c52_shortest_step_cue.py`: seven tests; `c52_gallery.py`: replay-verified
  SVG from actual executed synthetic trajectories.
- `C52-summary.json`: compact model results, graph-generation retries, gates,
  and hashes; `C52-gallery.svg`: 18 independently replay-matched panels.
- `C52-text-evidence.zlib.b64` and `c52_extract.py`: SHA-256-verified restoration
  of all class labels, optimizer curves, full test/online trajectories,
  normalization and replay receipts. See `C52-evidence-receipt.json`.
- Frozen prerequisites and **two 2.1 KB safetensors checkpoints** are retained
  in the separate complete local archive, not Git history. Even verified
  optimizer determinism is not immutable Actions artifact custody.

**Disposition: Draft/HOLD · do not merge/promote.** No changes outside the owned
`experiments/driv3r/training/guidance_recovery/**` laboratory.
