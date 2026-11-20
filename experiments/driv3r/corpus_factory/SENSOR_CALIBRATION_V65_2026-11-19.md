# DRIV3R Corpus Factory v65 — sensor calibration under delayed reports

**Date:** 19 November 2026. **Scope:** `experiments/driv3r/corpus_factory/**` only. **Status:** research data, synthetic graph host, NOT native CyberSand. **No neural training in this lane.**

## Defect addressed
v64 was 100% solvable by selecting the highest measurement timestamp; no history of sensor integrity was required. v65 crosses healthy versus invert-calibrated telemetry with old/new physical remote-gate state, stale-last/fresh-last delivery, pivotal/irrelevant gate and newly generated out-of-distribution graph topologies. An early historical `SENSOR_DIAGNOSTIC` code (901 = healthy, 902 = inverted) governs whether to invert the subsequent reported `EDGE_GATE` value. Those codes are **synthetic research symbols**, not native CyberSand definitions. The physical host's gate remains untouched; only the *observable sensor transport* differs, with an independently audited source-host replay.

## Reproduction

```bash
node experiments/driv3r/corpus_factory/sensor_fault_v65.js /tmp/v65-corpus
node experiments/driv3r/corpus_factory/test_sensor_fault_v65.js /tmp/v65-corpus /tmp/v65-corpus.v64_source
```

The generator refuses an existing destination and produces the intermediate independent-topology v64 physical-host base in `/tmp/v65-corpus.v64_source`; it is required for independent source-host qualification. Ordinary CI need only syntax/import-smoke; no expensive push-triggered generation.

## Complete locally audited corpus

| Split | Distinct new graph worlds | Complete successful episodes | Legal sequential decisions | Pivotal cases |
|---|---:|---:|---:|---:|
| TRAIN | 8 | 256 | 17,616 | 128 |
| DEV | 4 | 128 | 16,608 | 64 |
| HELDOUT | 4 | 128 | 24,320 | 64 |
| **Total** | **16** | **512/512** | **58,544** | **256** |

Every world has 32 crossed arms; 16 pivotal and 16 irrelevant. All arms within a world have identical instantaneous model input at the critical route choice, with opposite valid labels according to hidden physical gate. A wrong route has at least two additional shortest-path moves. 322 redundant noncritical TRAIN labels were loss-masked (detours were already zero-weight); all 512 critical labels retained. DEV/HELDOUT and all certificates are evaluator-only.

**Deterministic baseline at pivotal critical actions:** current-only 50%, highest-timestamp report without checking integrity 50%, diagnostic-conditioned measurement-time controller 100%. These are symbolic controls, **not measured neural-model accuracy**. The diagnostic-conditioned rule remains simple and solvable; this is a focused conditional-memory benchmark, not evidence that real gameplay generalization has been achieved.

## Independent qualification

1. The unchanged v64 host auditor independently replayed **256 physical source episodes, 29,272 source decisions** for the new seeds; all success and per-action receipts passed.
2. A separate v65 auditor independently reconstructed both telemetry variants for every source episode, validated all **58,544 observable/action steps**, verified accepted action labels, correct critical input hashes and untouched physical state, checked 32-arm balances and split-only masks.
3. All 16 generated world hashes are new and disjoint from the exact v64 archived 16 world hashes; TRAIN, DEV and HELDOUT have no geometry collisions.
4. Two complete v65 generations were byte-identical across all seven generated files. Deliberately modified TRAIN policy data were rejected.
5. No actual pinned LFM/Sprocket optimizer steps, closed-loop neural heldout, native CyberSand receipt, paid GPU/API compute or immutable GitHub Actions corpus artifacts occurred.

## Consumer handoff
Train a **pinned real SLM/adapter in the separately owned trainer lane**, consuming only the v65 TRAIN stream and no certificates. Compare full-history training against removal of diagnostic history, use of raw highest-timestamp receipt, stale arrival order and memory reset at 16/32/64/128 steps. Score sealed HELDOUT both-correct pivotal pairs, autonomous completion and neural CPU p50/p95 latency/RSS. This corpus is not a policy trainer; keep all evaluator-only artifacts out of model context.
