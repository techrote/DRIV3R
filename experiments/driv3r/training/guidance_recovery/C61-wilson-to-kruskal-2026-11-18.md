# C61 — Wilson-trained remote-goal exploration versus independently held-out Kruskal

**Status: RESEARCH HOLD — no promotion. 18 November 2026.**

Repository ownership: only `techrote/DRIV3R`, Draft/HOLD PR #1, branch `research/import-consolidated-worm-labs`. Authorised write directory: `experiments/driv3r/training/guidance_recovery/**`. This is **synthetic CyberSand-like host science**, not a native CyberSand ABI test and not a gameplay change.

## Hypothesis and pre-outcome protocol

C60 found that a 12-feature on-policy neural exploration head improved familiar-family distant-goal navigation without significantly outperforming RESET on unseen Wilson/Kruskal topologies; a non-neural observed-frontier planner was stronger. C61 tests **family transfer**: train the same bounded stochastic head on **Wilson + four familiar topology families**, hold Kruskal entirely absent from training, and assess new Kruskal graphs alongside unseen Wilson instances and fresh familiar diagnostic graphs. The same actions, legal menu, environment generator, reward function and frozen C60 baseline were retained. The horizon was fixed to 110 before evaluation (C60 originally used 78); **do not compare raw C61 success rates to the earlier C60 run**.

Fixed seeds: `610118` protocol, `611119` familiar training, `612120` Wilson training, `613121` Wilson/Kruskal test, `614122` familiar test. Generated **28 training graph identities** (16 familiar, 12 Wilson) and **64 disjoint test graphs** (24 Wilson, 24 Kruskal, 16 familiar). Full graph-sha exclusion includes all C60 historical and C60 train/test identities. Models and policies never see held-out graphs before fitting. Training produced **15 eligible host-visited remote-goal prefixes**; testing produced **40 eligible prefixes** from the 64 generated topologies. The remainder, including hard unreachable/early-terminal cases, were excluded *by the fixed prior host-prefix eligibility filter*, not by model result. See full generator receipts/skips for every graph.

Policies: C18-A0 unguided, C34-RESET, non-neural C60-FRONTIER, original frozen C60-ALIGNED weights (C60-FROZEN), newly trained C61-WILSON, and wrong-reward-credit C61-SHUFFLED. Each trial uses the same host-executed prefix, random-generator state, goal and current legal menu across policies. Arms are no guidance, weak BACKTRACK 0.35, strong BACKTRACK 1.0. Memory losses are 0% and 50%. Training is real CPU AdamW over executed on-policy actions and delayed discounted reward; logits are clipped to ±1.8 and additive to a nondegenerate stochastic action distribution. Runtime policy does not inspect protected map, complete adjacency, shortest-path oracle, future reward or heldout graph geometry.

**Important training-control asymmetry:** the inherited C60 per-epoch batching produces **12 optimizer steps** for aligned, **9** for shuffled, because their on-policy trajectories have different lengths. The optimizer code is unchanged; this is a real confound that makes the trained-versus-shuffled comparison weaker, not a matched-update result. Keep the outcomes and fix the budget-matching protocol in a future **fresh-seed** experiment; no post-hoc retraining or holdout parameter selection is admissible here.

## Matched held-out remote-goal completions

Each count uses only the **guided** trials (weak + strong); no-guidance results are a distinct safety/parity control. Results below use intact memory (50% erasure yielded the same goal counts; exact visit counts may differ).

| Independent test family | Eligible guided episodes | RESET | FRONTIER | Frozen C60 | C61-WILSON | C61-SHUFFLED |
|---|---:|---:|---:|---:|---:|---:|
| Kruskal (never trained) | 26 | 2 | **6** | 4 | 4 | 2 |
| Wilson (new instances, trained family) | 34 | 5 | 5 | **6** | **6** | 5 |
| Four familiar families combined | 20 | 4 | **9** | 6 | 5 | 5 |
| **All eligible cohorts** | **80** | **11** | **20** | **16** | **15** | **12** |

**Main result:** C61-WILSON gains +2 goals relative to RESET on truly unseen **Kruskal** geometry, but ties **the already frozen C60 controller** on Kruskal and Wilson, and remains below the non-neural FRONTIER policy. Thus introducing Wilson-family training does **not** establish a novel cross-family recovery advantage. The Kruskal +2 comes from two positive graph clusters, none negative; graph-cluster exploratory bootstrap 95% spans **0 to +5 goals**. There are no condition-level guided losses relative to RESET for C61-WILSON on this small selected cohort, but the prior C-series experiments have already adapted the research design; these intervals are not confirmatory evidence.

Critically, the C18-A0 unguided policy unexpectedly completed **6** Kruskal goals versus RESET's 2. This does not mean corrective guidance is always beneficial: the host-issued recovery controller can suppress useful natural exploration in some cases. This failure is retained as a negative signal, not discarded by reporting only stronger guidance.

## Safety, movement character and evidence quality

- **1,440** full synthetic-host suffixes across 240 matched condition rows and six policies. **80/80** no-guidance condition rows had identical action and ink SHA-256 across all policies. No synthetic-host action rejection, illegal material application or protected-cell hash change occurred in any suffix.
- **24** replay trajectories were selected by a source-declared graph-order rule, then independently re-executed and checked against ledger action/ink hashes. SVG draws actual movement segments, visited coordinates, start and published goal only; it intentionally does **not** invent hidden maze geometry.
- Two **independent deterministic CPU refits** reproduced the exact aligned and shuffled head tensors. Aligned/refit optimizer steps: 12; shuffled/refit: 9. The source and final parameter hashes are in `C61-summary.json`, the checkpoint file hashes in the full owner's archive.
- Movement personality proxies, measured only when both matching episodes have 20 actions, show Kruskal C61-WILSON versus RESET: unique-position ratio +0.0612, turning frequency −0.0351, reverse rate −0.0677, direction entropy +0.0230, painting fraction −0.0167; **n=21 selected paired samples**. These shifts demonstrate altered trajectories, not preservation of a particular Meanderer/Curler/Brancher artistic character. Exclusion of early-terminating trajectories creates survivorship bias. No native agent personality acceptance.
- Goal counts are low (Kruskal 2–6 of 26). The study is an early controlled transfer test, not a broad benchmark of general agent recovery.

## Reproduction and storage

Extract the owner archive so `lab/`, `publication/`, `frozen/` and `results/` are siblings. Required Python dependencies: numpy, torch CPU, safetensors, pytest. From archive root:

```bash
PYTHONPATH=frozen/research-source python lab/c61_wilson_to_kruskal.py results
PYTHONPATH=frozen/research-source pytest -q lab/test_c61_wilson_to_kruskal.py
python lab/C61-extract-ledger.py lab/C61-executed-ledger.zlib.b64 full-ledger.json
python lab/c61_gallery.py results/C61-traces.json lab/C61-gallery.svg
```

`C61-executed-ledger.zlib.b64` is a lossless compressed JSON record of the summary, **all** executed policy suffixes (including negatives/no-guidance) and the replayed gallery paths. Its SHA-256 and Git blob identity are verified in the evidence receipt. Checkpoint binaries and the frozen predecessor chain are deliberately **outside Git history**, retained in the local owner archive; immutable GitHub Actions artifact custody remains unqualified. Repo checks/source smoke alone do not qualify scientific reproducibility without the frozen prerequisites and independent host execution.

## Disposition

**HOLD; do not promote C61-WILSON.** C61 offers no lift over frozen C60 in the key never-trained Kruskal family, trails the non-neural frontier planner, and has a shuffled-label optimizer-step imbalance. Whole-trajectory artistic personality, native CyberSand host-legal ABI parity and immutable Actions checkpoint custody remain outstanding. Next experiment should freeze a genuinely matched-optimizer protocol and evaluate all hard/terminal cases in a separate fresh, predeclared test cohort without changing the host ABI.
