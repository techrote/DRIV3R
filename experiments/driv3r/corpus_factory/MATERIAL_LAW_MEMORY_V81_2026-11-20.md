# Corpus Factory v81 — physical material-law memory with source-time rewire receipts

**Date:** 2026-11-20. **Repository:** `techrote/DRIV3R`, Draft/HOLD programme PR #3. **Scope:** `experiments/driv3r/corpus_factory/**` only. **Model:** synthetic game-like host, **not** native CyberSand. No SLM training, paid inference or main-branch updates.

## Scientific question

Can a trainer remember an actual host-executed material calibration, combine it with a delayed out-of-order law-rewire event and a separately revised objective, then select the correct *legal command* after more than 128 intervening actions? v80 depended only on remote gate and objective receipts. v81 adds an independently physical calibration that cannot be inferred from a single receipt. With `NORMAL`, PAINT_A physically makes A; with `INVERTED`, PAINT_A physically makes B. Early `PAINT_A→ERASE→PAINT_B→ERASE` probes physically expose the mapping. Later, an `REMOTE_LAW_EVENT` source-time receipt reports whether the mapping has inverted. The end goal is independently A or B, requiring correct action composition rather than memorising a command.

The host still has three corridors, two gate decisions, hidden later gate obstruction, traversable crossovers, and legal recovery. Only observable input/accepted action history is consumer-facing. The initial law, changed physical law, oracle goal action and gate surprises are **evaluator-only certificates**. Four early probe actions and forced mistakes are loss-masked. Post-error corrective paint has a separate `post_repair_paint` role and is **excluded from clean-goal memory controls**, because a recent failed paint could otherwise reveal the action without long-term memory.

## Qualified full corpus

| Split | Geometries | Host-terminal episodes | Replayed legal actions | Physical gate recoveries | Physical paint repairs | Clean supervised goal decisions | Total indexed positive landmarks |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 6 | 576 | 182,976 | 288 | 144 | 432 | 1,008 |
| DEV (evaluator only) | 3 | 288 | 98,256 | 144 | 72 | 216 | 504 |
| HELDOUT (evaluator only) | 3 | 288 | 106,608 | 144 | 72 | 216 | 504 |
| TOTAL | 12 | 1,152 | 387,840 | 576 | 288 | 864 | 2,016 |

The independent auditor does **not** import the generator or its planner. It reconstructs legal PAINT/ERASE/MOVE/STOP, physical material transformation and reversal, gate transitions, receipt delivery and latency, final physical material, episode hashes, split-specific world geometry, matched ambiguity groups and mask positions. Full replay: **1,152/1,152 terminal-success episodes; 387,840/387,840 accepted actions; 12 split-disjoint geometries; 384 first-route three-way groups, 576 goal contrasts and 288 law-memory contrasts.**

## Held-out adversarial controls — clean goal choices only

| Available information | Deterministic maximum correct on 216 HELDOUT clean decisions |
|---|---:|
| Instantaneous observation only | 108/216 (50%) |
| Last 128 action+observation pairs | 108/216 (50%) |
| Correct material goal alone | 108/216 (50%) |
| Physical witness plus goal; missing rewire event | 108/216 (50%) |
| Rewire receipt plus goal; missing initial physical witness | 108/216 (50%) |
| All three observable factors + correct source-time chronology | 216/216 (100%) |

The fully observable-history symbolic parser independently recovers 216/216 clean HELDOUT actions from **pre-action inputs only**, rather than reading hidden cert fields. These are **deterministic dataset ceilings, NOT neural-model accuracy**. Gate decisions are inherited from v80; the new scientific contrast is material-law memory. The physically forced wrong-paint episodes remain valid host trajectories, but their recent material outcomes are explicitly excluded from the memory contrast.

## Reproducibility and tamper resistance

- `node experiments/driv3r/corpus_factory/material_law_memory_v81.js OUT`
- `node experiments/driv3r/corpus_factory/test_material_law_host_v81.js OUT`
- `node experiments/driv3r/corpus_factory/score_material_law_controls_v81.js OUT`
- `node experiments/driv3r/corpus_factory/export_material_law_decisions_v81.js OUT INDEX`
- Byte-identical second complete generation: **PASS** for all corpus and index output files.
- Rehashed first-witness corruption (stream, episode and manifest hashes updated): **REJECTED** by independent physical observation replay at episode zero, step one.
- The only positive TRAIN index is trainer-consumable. DEV/HELDOUT are sealed evaluator-only. Model inputs must be reconstructed from past `(input,action)` pairs; never expose `role`, `loss_weight`, source certs, or the control report to a trainer.

**Exact generator SHA-256:** `5a1eff69f814cee422b1cebdebf17cf8b50042ba9a8f1973dcca00ff4f324c5b`. **Corpus manifest SHA-256:** `49e3d8126364f9618e9b8089a3571ebea1cc2e621e70e2d391f0de9a83ee8d83`. **Index manifest SHA-256:** `d580cc8cabd94b73fb7f37de38bde7da97bd2de5b616c5b0287c00b7bb6e6066`. The source file embeds its own SHA-256 in generated manifests, and each output file carries its row count, byte count and SHA-256.

## Boundaries and next experiment

No native CyberSand ABI validation, learned SLM weights, autonomous neural rollout or neural CPU latency/RSS. A deterministic physical-witness + timestamp-parser controller can solve this constructed task, so the next consumer-owned experiment should train an actual pinned SLM on TRAIN and report HELDOUT autonomous outcomes, history ablations, p50/p95 CPU inference latency and peak RSS. The prior v76 DEV-B weakness is still unresolved; the present benchmark balances both material commands under law swapping but does not retroactively repair v76's probability-control weakness. No immutable GitHub Actions corpus artifact has been published. Corpus binaries remain out of Git.
