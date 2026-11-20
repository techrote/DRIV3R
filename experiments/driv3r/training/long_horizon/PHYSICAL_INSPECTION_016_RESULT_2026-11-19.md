# PHYSICAL-INSPECTION-016 — physical material verification vs capped feasibility

Date: 19 November 2026. Parent HEAD `4c8c21b146b5734a984f576e515ab6a6a4eed582`. **Synthetic research only; not native CyberSand acceptance.**

## Experiment
Earlier notification experiments 011–015 assumed a hypothetical material-damage notification. This experiment removes that observation. A controller can examine material only when standing on the cell; navigation observes its current position, local legal menu, known static map and remaining move budget. The host triggers an initial offscreen material-erasure opportunity on the same completed prefix-paint receipt in every arm, and for 128-decision worlds can trigger a *second* independently seeded offscreen erasure after the final target paint. No controller sees an event notification. A wrong-material paint at target B must be corrected from its locally observed receipt.

80 TRAIN world clusters × four horizon/cap variants provide 320 labels for a 1,500-step logistic optimizer. The TRAIN-selected cutoff `0.95` minimizes avoidable early inspection losses, then maximizes utility. 50 disjoint DEVELOPMENT world clusters × 36/36, 72/64, 128/64 censored and separately labelled 128/128 shadow horizons × seven strategies yield **1400 explicit CPU policy rollouts and 71899 replayed legal action receipts**. The 200 horizon/cap records are **not 200 independent seeds**.

## DEVELOPMENT results

| Strategy | Goals / 200 | Mean utility | Planner calls |
|---|---:|---:|---:|
| early | 158 | 653.49 | 952 |
| late | 167 | 691.62 | 949 |
| both | 165 | 676.6 | 1022 |
| never | 120 | 493.51 | 750 |
| feasible | 167 | 701.165 | 870 |
| learned | 167 | 691.62 | 949 |
| naive_learned | 158 | 653.49 | 952 |

The feasibility guard inspects physically only if a static shortest-path lower bound leaves time for a repair and STOP. It preserves all **167** late-inspection goals, saves **79 planner calls**, and raises mean utility from **691.620 to 701.165**. This bounded mechanical guard outperforms the trained controller in efficiency. The calibrated neural decision is identical to late inspection (167 goals), offering **no extra benefit**; naive 0.5 calibration reverts to early checks and loses **9 goals**. Never checking is insufficient (120 goals).

| Horizon/cap | Early | Late | Guard | Never | Learned | Naive learned |
|---|---:|---:|---:|---:|---:|---:|
| 36/36 | 39 | 39 | 39 | 32 | 39 | 39 |
| 72/64 | 50 | 50 | 50 | 32 | 50 | 50 |
| 128/64 | 26 | 28 | 28 | 28 | 28 | 26 |
| 128/128 | 43 | 50 | 50 | 28 | 50 | 43 |

## Verification and limits
The bounded optimizer ran explicitly, not on push/import. Two complete local Node.js v22.16.0 campaigns had identical trained coefficients, per-arm actions, goals, negative cases and receipt hashes when excluding volatile wall-time/RSS. Every rollout was reconstructed through a fresh **same-source synthetic host**; legal menu, move algebra, material outcomes, receipt continuity and final material state were checked. Two corrupted receipts (teleport action and forged coordinates) were rejected. The ledger retains each seed/horizon/policy record, action distribution, all 16 paired negative comparisons, and the gallery's matched trajectories. FNV32 hashes aid equality checking, not cryptographic provenance.

An earlier local Node qualification of the equivalent experiment core observed roughly **1.8 s total run time** and **49 MB end-of-process RSS**. This is not a source-pinned machine benchmark of this published exact file. Per-inference latency, exact-source peak RSS, independent simulator replay, native CyberSand acceptance and an actual SLM/LoRA checkpoint remain unqualified. Synthetic pulse physics and local visibility are research analogues, not asserted official host ABI contracts. Structural HELDOUT remains unopened. 128/64 logical-move-censored results remain separate from the 128/128 shadow cohort.

### Explicit opt-in reproduction

```bash
node experiments/driv3r/training/long_horizon/physical_inspection_016.js /absolute/output/directory
```

Scientific decision: a cheap observed-state feasibility guard is justified for further testing; the present learned inspection-timing classifier is **not promoted**. Next: independent replay with host-validated local material observations and real game traces; avoid synthetic notification features until their ABI feasibility is established.
