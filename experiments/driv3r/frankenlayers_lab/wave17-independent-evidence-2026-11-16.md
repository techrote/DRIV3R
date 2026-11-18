# Wave 17 — independent original-evidence audit (16 November 2026)

**Historical real-model inference, independently reanalysed after transfer to DRIV3R.** This is not a rerun in DRIV3R or native CyberSand acceptance. All external repositories remain read-only.

## Source and evidence custody

- Original read-only [Wave-17 CPU Actions run](https://github.com/techrote/cybersand/actions/runs/43947574492): SUCCESS, source c0a4b66c58783e0385f17f6a19bcd7a81408ee01.
- Original [artifact 14268756978](https://github.com/techrote/cybersand/actions/runs/43947574492/artifacts/14268756978): downloaded ZIP and verified its SHA-256 **dcaf7539d39c7b1a28b4f1be00ad427e71cffe8a6378500b928d123c914db0d9**, matching the Actions artifact digest.
- Independently rehashed **193/193** original SHA256SUMS.txt payloads; zero mismatches, missing, unlisted or duplicated entries.
- Independently replay-audited each original decision file and host event stream: host-legal action, applied action, observation, trajectory position, newly visited cells, non-RETRACE/non-STOP revisit contacts, action counts, median decision latency, and first paired observation/action divergence all agreed with original comparison.json.
- Q8 donor SHA-256: 073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c.
- Learned 55-weight sidecar checksum: 65013426f8074a7bb7785627cd6c52de58318cdc904ba052d2235d38f1cfc8c4. Training-record checksum: 02ec5e5370f5c422a8202de2e8ccaccec0d005c56077d3b00c75d244747b0a27.

The experiment used genuine frozen System One inference and real CPU adapter training, not LoRA or fine-tuning of the d1-2B model. Training: seed 2718 / architecture. Held-out evaluation: seed 4979 / greebles; six synthetic obstacle variants, 36 decisions each. Five same-runner comparison arms.

## Independent result

| Candidate | Decisions / 216 | New cells | Revisit contacts | Premature STOP episodes / 6 | Responsive / exposed pairs | Median decision latency |
|---|---:|---:|---:|---:|---:|---:|
| Q8 stock | 76 | 12 | 60 | 4 | 1/1 | 3.673 s |
| Q4 stock | 216 | 45 | 171 | 0 | 1/1 | 3.368 s |
| NO2 Q4 raw | 216 | 91 | 110 | 0 | 1/1 | 0.478 s |
| NO2 Q4 + trained residual | **216** | **214** | **0** | **0** | **2/2** | **0.478 s** |
| NO2 Q4 + wall-shielded residual | **216** | **168** | **26** | **0** | **2/2** | **0.479 s** |

**New substantive negative result:** The strong unshielded learned residual outperformed the shield by **46 newly visited cells** on this held-out seed while retaining observed obstacle action divergence. A shield superiority claim is unsupported. The raw controller only *exposed* one of the two relevant obstacle pairs; the adapted trajectories exposed two. Do not call this restored raw-model geometry response without normalising geometry exposure.

The hand-coded shield suppressed **26** learned overrides (13 east_blocked; 5 north_blocked; 8 south_blocked). In east_blocked it repeatedly suppressed the learned BRANCH in favour of the raw RIGHT at odd ticks 11–35, feeding a RIGHT/RETRACE oscillation: shield **11 novel cells**, unshielded **36**. The simple wall-visible rule demonstrably reduced local escape/exploration here. The learned unshielded policy made 78 BRANCH, 44 LEFT, 92 RIGHT, 2 RETRACE decisions overall; zero revisits do not establish planning or recovery. Historical Wave 16 found a lost forward-gate response from this same kind of residual on another family, so no cross-seed promotion is justified.

The original all_gates_pass=true only tested its **raw-versus-shield** preregistered inequalities. It did not compare shield with the strong unshielded control. The distant gate was not observed and must be excluded from response denominators. Q8 stock terminated four trials early; latency ratios are per-decision, not useful game throughput. Six correlated interventions (far-open/blocked trace duplication), a single held-out seed, no protected-cell/goal tasks and no native CyberSand test preclude controller acceptance.

## Next falsifiable run

Preserve the original NO2 Q4, strong adapter, wall shield and failure cases unchanged. Test additional disjoint seeds and obstacle families with exposure-normalised response rates, goal/protected-cell checks and actual recovery after collision. Compare against a **loop-aware** guard that only vetoes a proposal when demonstrably harmful under the observed recent trajectory, instead of preserving every raw action near walls. Retain per-decision traces and same-runner Q8/Q4 stock. Never train on routine PR push.

**Ownership:** This newly written evidence is in DRIV3R Draft/HOLD PR #1 only; CyberSand and Neurounner were read only. Import/source provenance does not by itself establish scientific reproduction or checkpoint portability. No merge, Ready transition, paid compute or game code change.
