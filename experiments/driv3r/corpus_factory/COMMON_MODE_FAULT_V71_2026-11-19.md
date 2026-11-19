# DRIV3R Corpus Factory v71 — correlated dual-channel telemetry fault and conservative physical reinspection

**19 November 2026 — DATA ONLY.** Existing [Draft/HOLD programme PR #3](https://github.com/techrote/DRIV3R/pull/3), current `main` remains read-only. All physical-world episodes use the **synthetic** v44 graph host with the explicitly hypothetical v67 abrasion extension. None of the sensor-bus or receipt events is an established native CyberSand ABI feature. No SLM training is claimed.

## Why this experiment

[Long-horizon trainer PR #2, SIGNAL-CORRELATION-014](https://github.com/techrote/DRIV3R/pull/2#issuecomment-7009937631) found that dual-channel feedback saved travel in an independent-error DEV cohort, but introducing 25% common-mode error cost two completed baseline goals; 50–100% common error cost four. The separate source also corrected a provenance-losing delivery bug. The negative result is causal evidence that channel count alone is not assurance.

V71 therefore makes **sensor fault provenance** a real decision variable in a held-out game-like corpus. Each A/B paint goal has two physical-sensor channels, initially calibrated while the agent is actually standing over a freshly painted cell. A later telemetry-domain event has three distinguishable meanings: `NO_FAULT`, `ISOLATED_SECONDARY` (primary remains trustworthy), or `COMMON_BUS` (both channels may change calibration together). The event says **which domain is exposed, not whether either sensor actually inverted**. Actual inversion is crossed as a hidden physical telemetry factor. The two sensor values sampled at a later timestamp can both agree and still be wrong after a common-bus fault; source-time-old samples can remain historically valid but cannot certify states after further unverified vibration.

The host independently executes threshold-based remote material erasure during 112 legal detour moves. Reports are delivered in reverse measurement order, 22+ moves before the critical decision. The model's **instantaneous observation at the critical decision is identical across each 36-arm counterfactual group**. Retained calibration, source clocks, fault-domain history and genuine local reinspection are needed to act safely. An unverified vibration is ALWAYS treated as uncertainty whether or not hidden erasure actually happened, preventing privileged-state supervision.

## Corpus and host replay

| Split | Exclusive new worlds | Physical counterfactual groups | Successful expert episodes | Independently reconstructed legal actions |
|---|---:|---:|---:|---:|
| TRAIN | 4 | 16 × 36 arms | 576/576 | 86,388 |
| DEV | 2 | 8 × 36 arms | 288/288 | 46,860 |
| HELDOUT | 2 | 8 × 36 arms | 288/288 | 44,604 |
| **Total** | **8** | **32 groups** | **1,152/1,152** | **177,852** |

There are 8 split-disjoint geometry hashes and **1,152 distinct complete input/action trace digests**. In TRAIN, 6,194 redundant downstream continuation labels have loss weight zero; all scripted physical probes, prelude visits and 112 detour actions also carry zero weight. The history remains present in context. Evaluation certificates contain erasure truth, fault flip and teacher target labels and must never enter the model input or training gradient.

## Deterministic controller controls — not model results

An independent scorer recomputed labels from the *model-visible chronological sensor messages*. A conservative historical-validity expert preserves all 1,152 material objectives; unconditional A+B reinspection likewise preserves all, at a travel cost. Mean expert savings over unconditional reinspection: **4.3958 host actions** per episode overall, **5.1250 on HELDOUT**. This is **not a neural SLM score**.

A historical-calibration controller that blindly trusts the last source-time primary report despite the bus fault loses objectives in **66/1,152** episodes: 34 TRAIN, 16 DEV, 16 HELDOUT; all are in the `COMMON_BUS` condition. A two-channel agreement controller using both previously calibrated sensors but ignoring common-bus invalidation has the **same 66 avoidable goal losses**. In the particularly revealing case of a common flip with **both** channels reporting, it loses **32/64** episodes. Independent-secondary and no-fault controls lose **0** under those baseline algorithms. With no remote receipt or a common-bus fault, the conservative expert accepts the uncertainty and revisits material; observed fault domain lets it retain selective savings under an isolated-secondary fault. The raw-report collision search finds 92 matched groups with the same latest raw A/B sensor signature but different correct actions depending on history.

These controls isolate the scientific hazard: **apparent consensus is not independent evidence**. The dataset does not establish that a neural model can learn calibration and conditional reinspection or transfer to real CyberSand.

## Exact-source qualification and negative controls

The independent JS auditor imports the pinned **v44 host only**, not v71's generator or teacher. It reproduces every action and model-visible sensor event from scenario seeds, separate physical paint outcomes and host accepted clocks; independently reconstructs local radius-one visibility, early co-located sensor calibration, two source clocks, fault-domain invalidation, conservative first route, critical target membership and terminal goals. It rejects action/observation mismatches and leaked privileged labels. **1,152/1,152 complete episodes and 177,852/177,852 legal actions PASS.** Two full generations yielded byte-identical **all seven output files**. A certificate was deliberately changed and the file's SHA-256/size in `manifest.json` updated; the independent physical/chronological audit still rejected it. This is stronger than manifest-only integrity checking.

**Reproduce** using exact committed source and pinned parent files:

```bash
node experiments/driv3r/corpus_factory/common_mode_fault_v71.js /tmp/driv3r-v71
node experiments/driv3r/corpus_factory/test_common_mode_fault_v71.js /tmp/driv3r-v71
python experiments/driv3r/corpus_factory/score_common_mode_fault_v71.py /tmp/driv3r-v71
```

Generator SHA-256: `eb892a863cc636ae3e6335e62d4e849dae40403e622848e4ee562949e6061abc`
Independent auditor SHA-256: `1d97c587279f33e598c3fbbb910860b790a9a02a32c386df8da3858ea782e6d9`
Controller scorer SHA-256: `6d7e51c485e4d91877201ef03dfb1b5ffa5cc697ac275c78cb502b64de97e2ab`
Pinned default manifest SHA-256: `149c96c47d2fd1315f9016c42d9579260cc97f017ffd45b3ad4b9b8ef5d8acb2`
Pinned v44 host SHA-256: `14d27b999d5eee0a83fe7e5afd9e85ce29670bdd4320024d3465705f7b29f8b6`
Pinned v70 parent SHA-256: `aa742b764b8ad05df3a252c6fc1dae9a9b1406959e0c94cdbe0fb312558a64b1`

## Validity boundaries and next consumer experiment

This remains a **synthetic v44/v67 surrogate host**, never native CyberSand. The physical erosion thresholds, warnings, two diagnostic channels, source-time receipts and bus-fault messages are hypothetical. The agent's mission prelude and detour are teacher-forced, not autonomously navigated. Only 8 new geometries exist, and the DEV/HELDOUT split is disjoint by world, not a proof of out-of-distribution game play. No target SLM weights, LoRA adapter, quantized checkpoint, neural closed-loop goals, CPU per-action inference latency or peak RSS are produced in this DATA lane. Node generator and auditor process RSS are not model inference RSS. All benchmark scores above are oracle/rule controls.

**Consumer — separately owned trainer PR #2:** train a pinned actual SLM/adapter on TRAIN stream with chronology and loss masks, never on certificates. Evaluate autonomous HELDOUT completion, cost regret and zero-goal-loss constraint against (a) trained full history, (b) history truncated before local sensor calibration, (c) stale-primary, (d) dual-consensus without fault-domain, and (e) unconditional reinspection. Stratify the two fault domains and no-fault controls; measure p50/p95 CPU inference latency, peak RSS and exact model/checkpoint hashes. The model must learn to fall back when both channels share an unverified fault. Retain Draft/HOLD and do not change CyberSand/Neurounner or other DRIV3R lanes.
