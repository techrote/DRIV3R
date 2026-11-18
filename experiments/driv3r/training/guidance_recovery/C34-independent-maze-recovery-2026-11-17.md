# C34 — independent maze recovery and repeated host-issued goals

**17 November 2026 — DRIV3R guidance/recovery laboratory — REJECT for promotion.**

This is an external **synthetic CyberSand-like host** experiment, **not** a
native CyberSand agent qualification. Game/Workbench/host ABI code is untouched;
`d_common` is not modified or treated as the authoritative game contract.

## Research question

Does the previous C32-WATCH recovery advantage transfer to genuinely independent
branching topologies, and does re-arming after a verified second goal change or
training an executed-return head improve recovery without shortcut regressions?

## Exact experiment

* Independently generated, seeded randomized **7×7 DFS spanning-tree mazes**
  embedded in a 15×15 protected-material field (97 traversable cells), and a
  withheld **braid** family with 14 extra connectors (111 traversable cells).
  No U-shaped C32 ancestor geometry is reused. Training uses only trees; the
  braid graph family is withheld from optimizer data.
* A 24-step trail is **actually executed** via synthetic `World.apply()` before
  a goal changes to a reachable cell outside the executed history. Scenario
  selection (detour versus forward) uses a graph oracle **only to construct
  fixtures**; the policy never receives the oracle path.
* Single-goal and repeated-goal arms. The second host goal changes at decision
  50; the same signed advice is issued again at the change. Both old and new
  goals are explicitly counted. Episode horizon: **85 decisions** after prefix.
* Matched no advice, weak/strong BACKTRACK, and weak/strong CONTINUE arms;
  two personality signs and two guidance strengths. Four policies share
  exactly the same generated graph and stochastic sampling seed: frozen C18-A0,
  C32-WATCH, C34-RESET (rearm after a host-issued new goal), and C34-HEAD
  (same reset plus trained cue-gated BACKTRACK score correction).
* Neural head: 9 host-visible features, 16 Tanh hidden units, 140 CPU AdamW
  updates on 314 training observations with counterfactual **host-executed**
  eight-step suffix returns, 183 positive return differences and 98 negative.
  The short-suffix teacher uses local observations, not shortest-path labels.
* Two independent evaluation cohorts (A seed 340701, B seed 540704), with
  one world-index per factorial cell, **640 policy episodes/cohort**, 1,280
  total; 128 guided episodes per model/cohort. Cohort size limits statistical
  precision. Full cell-level results are in `C34-summary.json`.

## Results

Guided completed goals, summed over single- and double-goal episodes (up to two
goals per double-goal episode; these are **goal events**, not unique episodes):

| Policy | Cohort A | Cohort B | Combined |
|---|---:|---:|---:|
| C18-A0 baseline | 25 | 27 | 52 |
| C32-WATCH | 32 | 31 | 63 |
| C34-RESET | **36** | 32 | **68** |
| C34-HEAD | 33 | **34** | 67 |

C34-RESET's second-goal completions were **13 (A) and 6 (B)**, versus C32's
9 and 5. The trained C34-HEAD produced 3 fewer combined goal events than
C34-RESET. It demonstrably affected probability calculations in **6,974
cue-gated decisions**, but did **not** demonstrate improved held-out gameplay.
The 140-update checkpoint reproduced byte-identically across independent
optimizer executions. Training MAE was 0.5962; do not interpret it as reliable
return calibration without a separate held-out regression test.

**Strict condition-level nonregression failed.** Against C18-A0, C32-WATCH
regressed in 2/4 condition groups in cohorts A/B respectively; C34-RESET
regressed in 2/3, and C34-HEAD in 2/2. Against C32-WATCH, C34-RESET regressed
in 0/2 condition groups (A/B), and C34-HEAD regressed in 1/2. The strongest
aggregate result is therefore **not** a qualified safe new policy.

All three recovery candidates reproduced baseline action and ink hashes in
**32/32 unguided episodes per cohort**. Synthetic host action rejections: **0**.
Protected-cell modifications: **0**. However, the *uncensored* initial policy
score's top choice was illegal at **15 sampled decisions** across 1,280 episodes (9 A, 6 B); the
host-legal menu masked those before execution. Therefore do not claim zero raw
illegal preferred actions or equate the synthetic action mask to native safety.

## Reproduction and custody

Within the historical laboratory with its existing baseline safetensors:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python c34_independent_maze.py /path/to/archive-root /path/to/results \
  --n 1 --horizon 85 --seed 340701
```

The root requires `prerequisites/guidance_recovery_{cue_calibrated,
 adversarial_adapter,personality_gate,topology_transfer,persistent_feedback,
 signed_feedback,selective_continue}.safetensors` and
`results/C28-TEMPORAL.safetensors`, available in the retained scientific
archive, **not newly added to Git history**. The command computes both
cohorts, with B seed A+200003. The independent evaluations were executed
separately and their full JSON trajectory receipts retained in the local
archive; integrated one-command reproduction was not completed in this run.

`C34-trace-gallery.json` commits 24 exact recorded paths from cohort A;
`c34_gallery.py` renders these, regenerates graph and prefix hashes, and independently
replays each selected host-legal action and ink hash. `C34-summary.json` records grouped results,
checkpoint SHA-256 and exact prerequisite Git blob identities. `test_c34_independent_maze.py`
plus the preserved C32/C33 tests pass **15/15** locally. No paid API/GPU usage.

**Custody limitation:** Checkpoint tensors and full 1,280-episode raw JSON
remain in the SHA-256-manifested downloadable local research archive, not in
Git history and **not yet in immutable GitHub Actions artifacts**. This is
interim custody. The repository itself carries source, selected executed
traces, full condition-level counts and evidence hashes. No automatic training
workflow or new PR is authorized or added.

## Interpretation and next experiment

Independent graph families expose persistent nonmonotonic navigation failures
that remain hidden by average aggregate completions. Simple re-arm logic
sometimes helps the second goal more than the trained head, but its benefits
are not stable across all conditions. Next test: train explicit local-observation
return calibration on different random graph generators and held-out seed
families, with a condition-wise *paired* significance analysis, and reject
any head unless it beats a matched rearm-only control under both honest and
misleading cues while preserving active-guidance personality.

**Status: RESEARCH ONLY · Draft/HOLD PR #1 · no merge/Ready · no native game changes.**
