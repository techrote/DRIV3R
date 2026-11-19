# DRIV3R Corpus Factory v68 — credible observation quality versus conservative reinspection

**19 November 2026** · synthetic host only · Corpus Factory data lane · **Draft/HOLD**, PR #3. Not native CyberSand gameplay; the report codes are deliberately invented hypothetical sensor signals, **not** an observed host ABI. No target SLM weights trained or evaluated in this lane.

## Scientific question and parent evidence

Long-horizon SIGNAL-QUALITY-011 (PR #2, HEAD `64f88ad8fdac201988d7594ead4ffdc5a90ee92b`) found a learned policy with idealized verified damage messages preserved all 292/320 DEV goals while avoiding 240 of 320 revisits; with degraded feedback, the trained policy lost 14 goals. This is NOT evidence that native CyberSand emits verified remote erasure receipts. v67's remote abrasion data were also omniscient at the expert continuation: the oracle knew which goal had eroded, although the agent could not necessarily know.

V68 removes that privileged-information supervision. It tests the information/inspection tradeoff directly. For each physical world×abrasion threshold×shock sequence, four observation modes differ **only** in whether hypothetical material-specific remote checks are verified: NONE, A_ONLY, B_ONLY, BOTH. Both A and B are painted with host-accepted actions; a local sacrificial coupon empirically tests material fragility, two goals are visited, three remote disturbances occur and the agent makes 112 detour movements. A remote material-check report is supplied mid-detour (or explicitly marked unverified) and occurs **at least 67 accepted detour movements** before the first trainable action.

The legal/host-observable teacher policy conservatively visits every material with unknown status; revisits confirmed eroded materials; and skips confirmed intact materials. At the target it uses the host's local material view to repair if needed. No remote physical truth is provided to the policy when no credible observation exists. A privileged omniscient planner is retained ONLY as evaluator-only information upper bound, not a model label. No learned policy is claimed.

## Generated split manifest and qualification

| Split | Disjoint worlds | 80-arm episodes | Independently replayed legal actions | Successful synthetic completions |
|---|---:|---:|---:|---:|
| TRAIN | 8 | 640 | 97,842 | 640 |
| DEV | 4 | 320 | 49,518 | 320 |
| HELDOUT | 4 | 320 | 50,936 | 320 |
| **Total** | **16** | **1,280** | **198,296** | **1,280** |

The **independent host auditor**, which does not import the v68 generator or its teacher, reconstructed every pre-action observation, synthetic paint event, host movement, local sensor receipt, issued label, STOP success and counterfactual travel cost. It confirmed 16 split-exclusive worlds, 16 complete 80-arm crossing groups, 32 unverified-history groups with observationally indistinguishable physical erasures, and 80-arm identical instantaneous critical inputs. Existing v67 and v68 worlds have **zero** SHA-256 geometry overlap. Two full runs regenerated all seven corpus files byte-for-byte identically; deliberate TRAIN stream corruption was rejected. Source-smoke CI is not substitute for this full local host audit.

## Held-out control results, not neural scores

| Observation channel | Episodes | Expert synthetic successes | Saved actions vs always reinspection / episode | Extra actions vs privileged oracle / episode |
|---|---:|---:|---:|---:|
| No verification | 80 | 80 | 0.0000 | 9.4375 |
| A verified only | 80 | 80 | 5.9125 | 3.5250 |
| B verified only | 80 | 80 | 3.7500 | 5.6875 |
| Both verified | 80 | 80 | 9.4375 | 0.0000 |

All held-out labels preserve success by employing *conservative physical reinspection* when feedback is ambiguous. Consequently this corpus does **not** magically solve the trainer's real challenge of extracting safe utility under degraded signals. It demonstrates exactly what additional observation quality is needed to make those savings information-theoretically possible, and retains the unavoidable ambiguity cost rather than leaking its latent physical state.

**Controls:** STOP-only succeeds on 30% of full episodes, irrespective of observation mode. Always-reinspect and omniscient evaluator both succeed 100%; omniscient is infeasible without hidden state. V68 generated guidance averagely avoids 4.0938 actions relative to always-reinspect and incurs 4.0750 extra actions versus the privileged bound across all modes. These numbers reflect deterministic teacher/host behaviour, not SLM inference.

## Integrity, training scope and limitations

- Each 80-arm group crosses two physical abrasion thresholds × ten physical disturbance sequences × four credible-report masks. Physical outcomes are invariant to mask; the decisive local observation is identical across all 80 arms.
- TRAIN is the sole permitted policy training stream; DEV/HELDOUT streams and ALL physical certificates include `EVALUATOR_ONLY` in their filename. Every scripted coupon, painting visit and detour action has zero loss weight. Redundant TRAIN post-critical followups are loss-masked while every critical action retains weight one.
- SHA-256 hashes for generator, v44 physical host, v67 parent and every dataset file are in `V68_DEFAULT_MANIFEST.json` (source snapshot) and the generated `manifest.json`.
- New sensor codes (`REMOTE_MATERIAL_CHECK_A/B`) are experimental. All predecision movement remains teacher-forced; static graph is visible, no hidden future dynamics are learned, and a symbolic three-state per-material memory controller suffices. No actual target SLM has been trained; no neural closed-loop HELDOUT, native CyberSand acceptance, or neural CPU latency/RSS has been performed.
- Split isolation is by newly sampled geometry within v68 and relative to v67, not a proof of globally fresh geometry against every historical corpus. The full corpus is generated explicitly; no expensive automatic model training or paid API/GPU use.

## Reproduction (within `experiments/driv3r/corpus_factory`)

```bash
node reinspection_quality_v68.js /tmp/corpus-v68-new
node test_reinspection_quality_v68.js /tmp/corpus-v68-new
```

The generator refuses an existing output directory, so use a fresh path for deterministic reruns. Consume the generated TRAIN stream alone for policy optimization, not certificates or DEV/HELDOUT. All large JSONL corpora should be immutable CI artifacts, not committed to Git.

## Next consumer experiment

In trainer-owned PR #2, train an **actual pinned target SLM/adapter** on TRAIN-only v67+v68, then evaluate autonomous DEV and sealed HELDOUT with four matched observation masks; record goal completion, unnecessary physical inspections, actions/utility, memory-truncation ablations, p50/p95 inference latency, peak RSS and source/model hashes. Keep idealized verified-report results separate from uncertain-report results, and DO NOT claim native game behaviour.
