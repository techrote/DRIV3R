# C67 — Observation-only weak/strong recovery-cue selector (18 November 2026)

**Disposition: REJECT / HOLD.** This is model-side synthetic-host research only, not native CyberSand qualification. It is not a candidate for promotion.

## Question and protocol

Can a bounded head select *weak* (`backtrack=0.35`) versus *strong* (`backtrack=1.0`) recovery guidance from the **pre-intervention observation**, rather than choosing a frontier controller? C67 reclassifies the previously held-out **C66** 24-graph cohort as **training data**; none of those graphs is claimed as held-out for C67. Two 12-input linear heads (`ALIGNED`, `SHUFFLED`) receive equal 192-step CPU AdamW updates, identical initial parameters and identical training minibatch indexes. The shuffled-credit negative control permutes outcomes within topology-family × memory-loss strata. Features use frozen C63 observation logic with the cue-strength field neutralized; no future menu, unseen map, or test outcome is an input.

The **new** evaluation cohort contains 36 independently generated graph identities over six families, excluded from known C60–C66 identities. For each graph, both guidance intensities were actually executed under intact and 50%-damaged memories using the frozen C60 neural action policy, with C18, C34 and non-neural frontier comparisons and no-guidance parity checks. Learned selections assign one of these previously executed legal policy suffixes. This is counterfactual assignment, **not an additional 72 independent learned-policy rollouts**.

### Actual CPU execution

- Training: 24 C66 graph identities × 2 memory conditions = 48 paired weak/strong cases. Only **2 cases** had a weak-hint goal win; **0** had a strong-hint goal win. Thirty pairs had nonzero shaped reward differences.
- New evaluation: **36 graphs**, **72 matched guided rows**, **720 actually host-executed policy suffixes** including unguided parity controls, and 24 labeled gallery trajectories.
- Local tests **4/4 PASS**, including matched optimizer budgets, refit determinism and non-oracle selection. Two complete CPU runs reproduced **byte-identical** source-derived summary, ledger, gallery and the two trained safetensors.
- All host actions were legal and protected-cell hashes unchanged in executed comparisons. No-guidance hashes were invariant across compared controller families.

## Truly fresh cohort results

Completed remote goals (36 guided episodes per memory condition). The strongest baseline here was to always apply **strong guidance**, not the learned gate:

| Controller | Intact memory | 50% erased | New sites, intact |
|---|---:|---:|---:|
| C18 base with cue | 12 | 12 | — |
| C34 RESET, weak cue | 14 | 14 | — |
| Non-neural FRONTIER, weak cue | 16 | 17 | — |
| Always WEAK, frozen C60 | 17 | 17 | 244 |
| Always STRONG, frozen C60 | **21** | **21** | **451** |
| C67-ALIGNED | 19 | 19 | 310 |
| C67-SHUFFLED | 17 | 19 | 244 |
| C67-COIN | **21** | 19 | 385 |

ALIGNED selected weak cues in 30/36 intact and 28/36 erased conditions; it missed two paired goal completions relative to always STRONG in **each** memory condition. Across the 36 graph clusters × 2 conditions, ALIGNED minus STRONG = **−4 completed goals**, with descriptive graph-cluster bootstrap 95% interval **[−10, 0]**. ALIGNED minus SHUFFLED = +2, descriptive interval **[−2, +7]**. These are exploratory bootstrap descriptions, not confirmatory inference.

### Interpretation

C66 training contingencies weakly favored weak hints, but the fresh C67 cohort favored strong hints (42 vs 34 completions, two memory arms pooled). The gate mostly reproduced its training bias. Even a simple deterministic always-strong rule outperformed C67-ALIGNED in this cohort, while a coin selector matched/exceeded it. This is **distribution-shift failure**, not evidence of successful guidance adaptation. As in C60–C66, increased site coverage alone remains an unreliable surrogate for goal recovery.

The stronger guidance arm also changed movement proxies (mean full-suffix turn rate ~0.553 vs ~0.601 under weak guidance; mean unique-position ratio ~0.546 vs ~0.507). Such proxies do not independently establish Meanderer/Curler/Brancher personality preservation. Safety observations apply to this synthetic host only.

## Evidence and source custody

Exact executed code and tests: `c67_cue_strength_transfer.py`, `test_c67_cue_strength_transfer.py`. Complete machine-readable graph identities, coefficients' SHA-256 receipts, training/evaluation aggregates: `C67-summary.json`. All action/ink hashes, per-case outcomes and selected rendered traces: `C67-executed-ledger.zlib.b64` (base64(zlib(JSON))). True executed trajectories: `C67-gallery.svg`. `C67-evidence-receipt.json` gives file and checkpoint identities. All these are scoped to the named guidance-recovery laboratory. Both trained checkpoint binaries are **outside Git history** in the owner-evaluation archive.

**Critical execution limit:** The source was executed in the local frozen-research fixture layout reconstructed from the C64/C65/C66 owner archives, with PyTorch, NumPy and safetensors. It expects `lab/`, `publication/`, `results/` and `frozen/` beneath that fixture root. This repository source commit does **not** constitute self-contained checkout reproduction or native CyberSand ABI acceptance. The C66 full ledger and frozen checkpoint prerequisites must also be provided. No immutable GitHub Actions artifact custody is claimed.

No CyberSand or Neurounner mutations. No DRIV3R main modifications. PR #1 remains Draft/HOLD, not Ready, not merged.
