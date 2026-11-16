# Wave 18 — observation-local loop escape, original Wave-17 shadow evidence

**Research-only offline replay, 16 November 2026. Not a new model inference or CyberSand runtime test.**

The historical [Wave-17 real-model run](https://github.com/techrote/cybersand/actions/runs/43947574492) remains immutable and read-only. The [original archive (artifact 14268756978)](https://github.com/techrote/cybersand/actions/runs/43947574492/artifacts/14268756978) has SHA-256 `dcaf7539d39c7b1a28b4f1be00ad427e71cffe8a6378500b928d123c914db0d9`; **193/193 files** matched its manifest again during this experiment.

## Executed experiment

A new deterministic sidecar gate consumes only host-observed `x,y,heading`, last four host actions, the local 5×5, the current legal action menu, the raw frozen d1 choice and the learned 55-parameter residual's choice. Existing wall shielding is suspended **only** when the same pose has already appeared twice within the last eight observed decisions **and** the last four actions alternate a legal motion and RETRACE. No family label, target world layout, variant name, future state, hidden material, or post-hoc scoring enters the policy.

The offline runner independently loads the original ZIP, verifies every manifest entry, and replays the recorded wall-shielded action/observation stream. **It halts a scenario at the first action that the new policy would change.** Thus the prefix is based on actual model responses and observed host states; any subsequent trajectory requires new inference and cannot be assumed.

| Original synthetic variant | First changed decision | Recorded wall shield → loop-aware candidate | On-policy prefix checked |
|---|---|---|---:|
| east_blocked | Tick 15, `(8,48,heading=1)` | RIGHT → BRANCH | 16 decisions |
| south_blocked | Tick 8, `(7,48,heading=0)` | RIGHT → BRANCH | 9 decisions |
| east_open | None | Same decisions | 36 decisions |
| north_blocked | None | Same decisions | 36 decisions |
| far_open | None | Same decisions | 36 decisions |
| far_blocked | None | Same decisions | 36 decisions |

The known east-blocked RIGHT/RETRACE loop is therefore detectable without consulting future state. The south-blocked alert is another *candidate* escape, not proof that its downstream path improves. All 193 historical payloads were reverified and four source-only gate tests passed. The generated [`wave18-shadow-2026-11-16.json`](wave18-shadow-2026-11-16.json) preserves the exact recorded divergence signatures.

## Interpretation and next qualification

This is **stronger than a proposal**, because it executes a concrete guard on verified real-model inference traces and identifies two falsifiable first-divergence interventions. It is **not** a claim of improved novelty, escape completion, latency, protected-cell safety, or geometry responsiveness. After the first divergence, old model logits and poses do not belong to the counterfactual branch; extending replay would be off-policy fabrication. The hand-crafted criterion is subject to false positives and was informed by the known failure example, so it must be tested on independent held-out seeds/families. The trained *strong* residual remains the performance leader on historical Wave 17; the original wall shield remains a negative result.

Next bounded real-model run should compare Q8 stock, Q4 stock, raw NO2 Q4, strong residual, wall shield and loop-aware guard on at least two disjoint held-out seed/family groups, with first-exposure-normalised geometry response, obstacle collision/recovery, novelty and premature STOP metrics. Training remains confined to a separate fold; no game or production integration and no default CI training.

**Ownership:** Written solely under `experiments/driv3r/frankenlayers_lab/**` of DRIV3R Draft/HOLD PR #1; CyberSand/Neurounner read-only, no paid GPU/API, no merge or Ready promotion.
