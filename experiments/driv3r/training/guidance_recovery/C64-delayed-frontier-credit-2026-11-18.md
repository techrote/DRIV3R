# C64 — Delayed host-observed frontier credit: matched held-out transfer

**Disposition: DRAFT/HOLD — negative learning result; do not promote.**

## Scientific question

Can a tiny stochastic policy trained on **delayed, executed host feedback** learn better recovery actions for remote, initially unobserved goals than the frozen C60 policy-gradient head and a non-neural frontier heuristic, without violating host legality or flattening personality? This is a synthetic CyberSand-like **research host**, not native CyberSand or its authoritative ABI.

## Design and precommitted boundaries

`c64_delayed_frontier_credit.py` imports the original C60 environment/action scorer and C63 historical-cohort exclusion code. The only inference features are *current host legal actions*, host-observed positions, publicly announced goals, and actually visited action menus. No hidden map edges, protected cell map, or shortest-route oracle are consulted for action selection. C60's stochastic softmax remains in control; the learned 12-feature linear head produces only bounded additive biases, not forced actions. Every sampled action passes through the existing host legality gate. The historical `d_common` fixture is not modified.

C64 used fresh graph identities excluding historical C60–C63 identities, and independent generated test identities outside training. A fixed seed (`640118`) defines three modes of goal cue (none, weak, strong), 0%/50% historical memory erasure, and matching policy comparisons. Both trained heads received the same **24 AdamW update budget** and the same training-state schedule. The aligned head received returns incorporating terminal actual goal completion, later observed discovery and small step costs. The control reassigns those reward events to incorrect steps **within each actually executed training trajectory**. This is an imperfect wrong-credit control: it retains episode success and length information, and the two models can execute different on-policy actions; updates were equal, not the number of training actions.

Training: **14 eligible state/graph identities**, 24 optimizer updates per head, 1,199 host actions sampled for ALIGNED versus 1,294 for SHUFFLED. Both heads underwent a second complete deterministic refit, with byte-identical tensors in each mode. Held-out: **15 eligible graph identities**, 90 matched graph/cue/memory rows, **540 actual host-executed suffixes**, 30 unguided parity conditions, 60 action-and-ink-hash matched rendered movement traces.

## Held-out results (intact memory, 30 guided episodes per controller)

| Controller | Completed remote goals | New sites visited | Paired goal losses versus RESET |
|---|---:|---:|---:|
| C34-RESET | 14 | 312 | 0 |
| C60-FRONTIER (non-neural) | 17 | 368 | 1 |
| Frozen C60 neural | 17 | 429 | 0 |
| **C64-ALIGNED** | **17** | **364** | **0** |
| **C64-SHUFFLED** | **19** | **393** | **0** |

ALIGNED therefore gained **three goals over RESET**, but **zero over frozen C60**, and performed **two goals worse than its wrong-credit control**. The graph-cluster bootstrap interval for the aligned gain versus RESET is **0 to +6 goals**, descriptively only. The shuffled control's five-goal gain has interval **+1 to +10** but is *not* evidence of valid delayed credit assignment. With 50% memory erasure, the pattern remains similar. No promotion is warranted.

On held-out Wilson graphs ALIGNED achieved 1/6 versus RESET 0/6, but on Kruskal it only tied RESET 4/6. The sample is small and this is not a demonstration of robust cross-family transfer. Frozen C60 and frontier controllers remained competitive. Measured 20-action turning, reversal, direction entropy, unique-site and painting proxies are supplied in the summary; they **do not** qualify whole-trajectory guided personality preservation.

## Safety and evidence

- All 540 independently executed suffixes reported **zero rejected host actions**. Protected-cell final hashes matched the corresponding saved prefixes across policies.
- The **30 separate no-guidance conditions** had identical action and ink hashes across all six controllers.
- The 60-panel selected trajectory set was **re-executed through the host** with action and ink digest assertions, before drawing the SVG. No unseen obstacles or maze walls were fabricated in the gallery.
- Source SHA-256, pretrained C60 checkpoint identity, newly trained checkpoint SHA-256 and full seed/graph receipts are in `C64-summary.json`. Complete per-episode executed outcomes and trajectory gallery receipts are in the base64-zlib ledger; extractor supplied.
- Six local regression tests verify source identity, train/test separation, no-guidance parity, synthetic-host legality, SHA-matched traces and optimizer/checkpoint evidence. Standalone repo tests operate against the committed ledger; owner archive also checks checkpoint bytes.

### Custody and reproducibility limit

The separately packaged **owner-evaluation archive** retains both trained `.safetensors` checkpoints and frozen C42/C60/C63 prerequisites for independent local reruns; checkpoint binaries are **not placed into Git history**. Immutable Actions artifact custody is still absent. To execute the complete experiment outside this archive, provide the original verified C42 prerequisites and C60/C63 research source/evidence paths under the archive's `frozen/`, `publication/` and `results/` layout. Research scripts are not a production runtime or a claim of native CyberSand reproducibility.

## Disposition and next falsifiable experiment

**HOLD / negative result.** Shuffled temporal credit matches or beats delayed aligned credit. A larger, genuinely held-out recovery cohort with controlled equal action exposure, an episode-success-matched action-credit negative control, and full guided movement morphology is required before interpreting further frontier-policy learning. Do not expend unapproved GPU or API credits, modify CyberSand/Neurounner, or change this PR's Draft/HOLD state.
