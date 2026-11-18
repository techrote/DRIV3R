# DRIV3R Corpus Factory v56 — Physically certified material-memory contrasts

**Status:** Synthetic-host data-only curriculum; local independent audit PASS. This does **not** certify native CyberSand behaviour, training, neural performance, or the counterfactual wrong-command host path.

## Problem corrected

v55 contained 282 matched instantaneous-observation/opposite-action pairs. It assigned 147 pairs to navigation and 23 to mixed action across all splits; these classes may include alternative but equally good movement actions. Treating every opposite teacher label as **causally necessary** would overstate the benchmark.

v56 selects only the 112 **material-law** contrasts and requires, for both actions in a pair:

1. The two **current model observations are byte-equivalent after canonical serialization** and the legal commands differ (`PAINT_A` versus `PAINT_B`).
2. The earlier streamed, host-executed `PAINT_A` result on the active surface is present; its physical material agrees with the chosen legal action. The two histories record **opposite actual physical outcomes**.
3. The next synthetic-host observation confirms **correct physical material at the goal** and a `GOAL_REVISION` or `MISSION_COMPLETE` progress event.
4. Both trajectories share a source world/topology but come from distinct episodes. Complete original episodes are retained; no actions, observations, legal-action lists or source loss masks are changed.

This is stronger evidence than contradictory navigation labels, but the *wrong command from the other history* is inferred to be wrong under the v53 binary mapping. That wrong-command branch was **not replayed here**, so this report does not claim empirical alternative-action regret.

## Reproducible counts

| Split | Complete episodes | Decisions | Material-law pairs | Critical choices | Equal last 8 input+action steps | Equal last 16 | Equal last 32 | Both calibration receipts >=16 decisions old | Both >=32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 128 | 4,268 | 64 | 128 | 4 | 1 | 0 | 1 | 0 |
| DEV | 48 | 1,732 | 24 | 48 | 6 | 1 | 0 | 2 | 0 |
| HELDOUT | 48 | 2,458 | 24 | 48 | 8 | 4 | 3 | 4 | 4 |
| **Total** | **224** | **8,458** | **112** | **224** | **18** | **6** | **3** | **7** | **4** |

All 224 selected actions have a correctly recorded physical goal result and an accepted mission-progress event in the following observation. All 112 pairs have opposite preceding physical calibration outcomes, current observation identity, and distinct complete episodes. TRAIN retains 1,490 zero-weight duplicate-navigation steps with source masks unchanged. Critical TRAIN steps receive 6x weight, but only when their source weight is nonzero. DEV and HELDOUT are never reweighted.

## Memory-pressure measurement

On the 48 held-out critical decisions (24 balanced pairs), a **deterministic stateless same-input policy** is mathematically limited to 50% paired-action accuracy and zero pair-both-correct success. This follows from identical current inputs with opposite labels; it is **not** an observed SLM result. The trivial fixed-`PAINT_A` control scored 24/48 actions (50%) and 0/24 pairs-both-correct. Teacher labels gave 48/48 and 24/24 respectively, confirming scorer behaviour.

More usefully, three HELDOUT pairs have an **identical trailing 32-step observation+action history** before their opposite material decision. A model that only accesses the last 32 steps therefore also cannot solve both members of those pairs. However, TRAIN has **zero** 32-step pairs, so this tier is an out-of-distribution diagnostic, not a trained generalization benchmark with good statistical power. Do not overinterpret a 3-pair sample.

The age of the last relevant physical result and the length of a *completely identical* trailing input+action history are deliberately reported separately. A mere recent unrelated notification cannot substitute for the requisite surface-specific calibration result.

## Dataset contract

`TRAIN_causal_stream.jsonl` has the original full sequential observations, actions and masks plus a lane-local `focus_loss_weight` field for trainer consumers. DEV/HELDOUT streams remain evaluator-only. Pair certificates and provenance are written to separate `*_EVALUATOR_ONLY.jsonl` shards, **never fed into model prompts or recurrent state**. Trainers must group all v53/v54/v55/v56 descendants of the same world into the same TRAIN/DEV/HELDOUT partition; v56 is **not** a fresh independent test split.

Run with Python 3.10+:

```bash
python experiments/driv3r/corpus_factory/material_causality_v56.py \
  --source /path/to/v55corpus --output /path/to/NEW_v56corpus
python experiments/driv3r/corpus_factory/test_material_causality_v56.py \
  --source /path/to/v55corpus --export /path/to/NEW_v56corpus
```

The input must be the **exact v55 source** verified by `V55_DEFAULT_MANIFEST.json`. v56 checks every source shard hash and refuses an existing output directory. Full file provenance and SHA-256 values are included in `V56_DEFAULT_MANIFEST.json`. Two fresh complete local generations yielded byte-identical shards and manifest. Independent code replayed all 8,458 selected labels against original v55 ordered episodes and checked all 224 next-observation physical results, split topology isolation, mask preservation, source hashes and paired-calibration proofs.

Scorer usage (evaluation only):

```bash
python experiments/driv3r/corpus_factory/test_material_causality_v56.py \
  --source /path/to/v55corpus --export /path/to/NEW_v56corpus \
  --split HELDOUT --predictions /path/to/predicted_actions.jsonl
```

Prediction schema is one `{"actions":[...]}` JSONL row per complete eval episode, aligned one-for-one with every step and using only legal action tokens or `null`. The scorer reports matched both-correct and per-action accuracy for history-window and physical-receipt-age tiers. Teacher-forced scoring does not imply closed-loop episode completion.

## Qualification and negatives

- Source v55 manifest SHA-256: `dcd6c65de9732756d43e6538a92837385567f00d577d7cf8235088715427c920`.
- Generator SHA-256: `c2f7ce6ef4437650329cf1fe236cecdce5d200c1fb984650f3d7b535361fafca`.
- Independent auditor SHA-256: `930e8a24b179281086d829acd72cae0a87ead97d06f11c5509c6a3836a6fd8a8`.
- v56 generated manifest SHA-256: `b7f4b9f0d76416378c0d38713b4c60df90c149c2ef29533c73cb0e030ec9690e`.
- All 8,458 selected steps matched the immutable source's original observation, action and weight; every original selected episode terminates with `STOP`.
- Two complete generation runs: byte-identical outputs; independent audit PASS.
- Teacher-forced perfect-label scorer PASS on both DEV and HELDOUT. Trivial fixed-paint negative-control 50% and 0 both-correct pairs on both eval splits.

**Unresolved scientific limits:** the synthetic prelude is scripted; symbolic observation tokens reveal calibration results; a simple symbolic timestamp-aware controller can still solve these cases. No target SLM or adapter was trained, no CPU neural latency/RSS was measured, no native CyberSand evaluation occurred, and immutable GitHub Actions dataset artifacts are not yet published. All results are synthetic-host data validation, **not native-game readiness**.

## Consumer handoff

Use the broader v54 sequential TRAIN streams plus v55 focus curriculum for initial weight updates; optionally use v56 for a strictly physical-causal secondary curriculum. Compare recurrent full-history against memory reset, and 4/8/16/32-step context windows on DEV, then once on HELDOUT. Report both-correct pairs, physical-receipt age, autonomous synthetic-host terminal success, p50/p95 CPU inference latency and peak RSS. Publish model/tokenizer/checkpoint hashes and immutable Actions artifacts separately. Do not optimize on held-out annotations.

**Ownership:** all generated Git changes are restricted to `experiments/driv3r/corpus_factory/**`, on the existing shared DRIV3R PR #1 Draft/HOLD branch. CyberSand, Neurounner, main, other research lanes and all paid computation remain untouched.
