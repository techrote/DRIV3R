# Corpus Factory v60 — long-delay physical calibration memory (19 November 2026)

**Repository:** techrote/DRIV3R; **programme:** Draft/HOLD PR #3, `research/corpus-factory-postmerge-programme`; **lane:** `experiments/driv3r/corpus_factory/**` exclusively. **Science:** synthetic v53 graph host != native CyberSand. **This is DATA, not SLM training.** No consumer model weights were modified.

## Why

The earlier v59 post-paint continuation could be solved by a one-step parser of local paint and `MISSION_COMPLETE`. v60 instead tests whether a learner can retain the **actual host-executed B-surface material calibration result** through long intervening motion, then choose opposite legal paints for **byte-identical instantaneous observations** at the protected goal.

## Mechanics and ownership

For each split-preserved world, three pairs of v53 host arms differ only in hidden physical B material mapping. A host-accepted safe `PAINT_A` probe generates `PAINT_RESULT`, then is erased. The full causal observation is delivered only once in `events_delta`; durable receipt histories are absent. Just before the final B goal, the host executes **48–256 legal movement actions** along multi-step paths and retraces back to goal. These intentionally forced detours are **zero loss weight**; do not train them as expert optimal navigation. The final goal paint is supervised at focus weight 8. `PAINT_A` maps to material B in one arm, while `PAINT_B` maps to material B in the paired arm. The wrong command is a host-executed non-progressing counterfactual; the correct command completes stage 2, followed by a successful STOP.

Worlds derive from `mission_memory_v46.candidate(split,700+index).w` and v53 `prep`/`ARMS`; all geometry groups are isolated by original split. No changes were made to host gameplay, action tokens, CyberSand or Neurounner. Model-visible files contain episode steps, legal actions and one-shot event deltas; paired identity/law/sidecar metadata are EVALUATOR_ONLY.

## Qualified data

| Split | Worlds | Paired counterfactuals | Complete host-success episodes | Decisions | Forced detour decisions (loss 0) | Duplicate TRAIN supervised labels zeroed | Delay lengths |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAIN | 12 | 36 | 72 | 9,532 | 6,528 | 2,009 | 48 / 96 / 128 |
| DEV | 6 | 18 | 36 | 5,736 | 4,224 | 0 | 64 / 128 / 160 |
| HELDOUT | 6 | 18 | 36 | 8,678 | 6,528 | 0 | 96 / 192 / 256 |
| **Total** | **24** | **72** | **144** | **23,946** | **17,280** | **2,009** | |

Measured at final decisions, the full matched **input/action suffix** (excluding the label being predicted) has length 49–156 decisions in TRAIN, 65–176 in DEV and **98–299 in HELDOUT**. Each paired current input is byte-identical; correct actions are exactly balanced and opposing. A no-memory decision table limited to the current input has a **50% upper bound** on the 144 critical decisions and can never get both members of a pair right. A symbolic probe-receipt historical parser succeeds on **144/144** (pure deterministic control, **not learned accuracy**).

## Independent tests

`test_delayed_calibration_host_v60.js` **does not call v60's generator**. It reconstructs all 24 worlds and host arms from pinned dependencies, replays all **23,946 steps** with **exact equality of every pre-action streaming observation**, checks host legal actions, verifies forced movement was actually accepted and zero-weighted, asserts prior B physical receipts, verifies paired final snapshot hashes and opposite paint commands, and forks the actual host with the opposite command at every critical decision. The wrong B paint does not advance stage 1; the correct B paint yields material B and a successful terminal episode. **PASS:** 144/144 full replays, 72/72 causal pairs, 24 split-isolated geometries.

Two independent full generation invocations produced identical manifest SHA-256 `bbbd9a6b97fdadc1eae8d51ef5adec6cbd76fd9ba8940200d73f85a7c77d3443` and byte-identical TRAIN shards (full manifest pins the six data/evidence shards). No neural experiment was run.

## Reproducibility

```bash
node experiments/driv3r/corpus_factory/delayed_calibration_v60.js /tmp/driv3r_v60_data
node experiments/driv3r/corpus_factory/test_delayed_calibration_host_v60.js /tmp/driv3r_v60_data
```

- Generator SHA-256: `0be8f98e99b8238391951d86c548b948c0c8ade8501adf9b5fad28094034b44a`
- Independent host auditor SHA-256: `caa6a9c499c87524290fcaec881beff7974befa790d50c8570630a64eb8ba5bc`
- Generated manifest SHA-256: `bbbd9a6b97fdadc1eae8d51ef5adec6cbd76fd9ba8940200d73f85a7c77d3443`
- Dependency source SHA-256 and all six shard hashes recorded in `V60_DEFAULT_MANIFEST.json`.

## Limitations and handoff

This is a **synthetic closed-loop teacher/host dataset** with **forced motion**, not evidence of trained SLM/adapter memory, learned planning, or native CyberSand rules. Model learning still needs actual LFM350/Sprocket or another pinned consumer model, with model/tokenizer/checkpoint SHA, untouched HELDOUT, autonomous host continuation, and p50/p95 CPU latency/RSS. The challenge is small: 18 HELDOUT pairs and only 24 worlds. A small symbolic history parser remains perfect. Prolonged backtracking may be artificial and should not be confused with expert autonomous navigation. No immutable Actions corpus artifact or neural checkpoint published; retain reproducible SHA-256 pinned local dataset and do not commit large data/binaries into Git.

**Consumer test requested:** Train pinned target SLM/adapter on TRAIN v54 broad episodes + v56 causal material pairs + v59 recovery, optionally v60 TRAIN as long-memory curriculum. Compare full-history recurrent state to 8/32/64/128-step truncated and reset-state evaluations on v60 DEV, then seal/freeze before first v60 HELDOUT. Score critical *both-correct pairs*, whole-episode host completion and p50/p95 CPU latency/RSS separately. Evaluate against a physical-receipt ablation. Do not train or tune on evaluator-only metadata or HELDOUT.
