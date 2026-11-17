# D023 — pretrained SLM distillation execution blocker and teacher evidence audit (17 Nov 2026)

**Status: SOURCE/EVIDENCE AUDIT COMPLETED; ACTUAL NAMED PRETRAINED SLM/LoRA TRAINING BLOCKED.**
Only DRIV3R Draft/HOLD PR #1 was targeted. No training step, checkpoint, closed-loop evaluation, or native CyberSand acceptance is claimed.

## Why this is a blocker, not D023 surrogate training

The D020–D022 sequence repeatedly optimized ~4,714-parameter synthetic-policy MLPs without establishing a robust dynamic-trap coverage gain. The next authorised test must use an **actual pretrained small language model**, not another MLP, heuristic, scalar calibrator, or randomly initialized model called an SLM.

- **Selected first actual target:** frozen/partially tuned EleutherAI/pythia-70m-deduped at model revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (from read-only DRIV3R model-zoo Pythia-70M research). An actual pretrained transformer final-block adapter and an equal-budget frozen-backbone head would be named-model training, unlike surrogate-only rows.
- **Alternate larger target:** LiquidAI LFM2.5-350M at revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`; GP350-006 rank-8 q/v LoRA step96 SHA-256 `436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c`, read-only original Actions artifact `14199280586`. A local adapter binary alone would not supply the frozen base weights.
- In this CPU environment PyTorch exists, but **transformers, PEFT, tokenizers and any pinned pretrained model cache are absent**. Hugging Face model downloads and direct GitHub raw downloads failed due to network resolution. Only earlier small MLP NPZ evidence archives were available locally, none containing these pretrained backbones.
- The connected GitHub tools provide read/write source and Actions artifact reads, but **no opt-in workflow-dispatch action**. Dispatching externally to CyberSand or Neurounner is prohibited; training on routine PR/push is also prohibited. No paid compute was used.

**Exact unblock:** an explicitly initiated bounded DRIV3R-only CPU Actions run with the pretrained Pythia-70M revision pinned, tokenizer+transformers available, and immutable artifact upload. Alternatively supply the exact pretrained weights/tokenizer in the local research environment. Do not introduce automatic training on push, another branch, or another PR.

## Source-pinned teacher corpus — newly re-audited

Executed the pure-stdlib `teacher_corpus_audit_023.py` against two previous local evidence archives; SHA-256 and ZIP integrity passed, including ten nested original Frankenbrain source ZIPs. Five legality/normalization unit tests passed. See `D023_TEACHER_PROVENANCE.json` (the machine-readable handoff) and `teacher_corpus_audit_023.py` (the fail-closed replayable validation tool).

| Source | Original Q8 recorded decisions | Distinct student-visible observations | Status |
|---|---:|---:|---|
| D001 original stock/GREYHOUND/BRANCHWITCH extracted in D022 | 144 | 111 | single documented seed 570 |
| D016 source-parent Frankenlayer variants | 483 | 286 | ten nested source archives SHA verified; normalized receipts omit source seed |
| Combined receipts | **627** | **286** | **111/111 D001 distinct states overlap D016** |

All 627 recorded actions obeyed their corresponding host-legal menus and action probabilities were normalized with zero mass on illegal actions. This is **evidence integrity**, not good game-policy performance.

Crucial conclusion: combining these corpora does **not** produce 627 independent game states or independent teacher world-seed coverage. Row-random train/test divisions would leak states and give misleading trait-retention estimates. These records are original inference observations/decisions, **not full source-teacher closed-loop rollouts**. D018/D021/D022 synthetic teacher labels must not be relabelled as original d1/LFM/Sprocket model outputs.

## Next falsifiable actual-SLM experiment

1. Use an authentic source-pinned Pythia-70M pretrained checkpoint plus tokenizer; explicitly record files, revision, adapter trainable parameter names, and hashes. Train a structured-action head with optional real final-transformer-block adaptation (e.g. 48–128 CPU optimizer updates, bounded wall-time/memory), and a matched frozen-head ablation. No fallback to toy models.
2. Assemble mixed supervision with per-row `source_kind`, `teacher_model_sha`, `teacher_artifact`, `world/seed/group`, `legal_menu`, `action_probabilities`, `trajectory_id`, `is_synthetic`, and provenance-hash. Add LFM/Sprocket original inference **only after** their artifacts are independently verified; absent labels remain unavailable, not inferred.
3. Use world/trajectory-grouped held-outs, leave source-parent identities and geometries out jointly; measure raw illegal rate before host masking, closed-loop normal/trap/squeeze painted unique cells, branch and retrace behaviour, oscillation/recovery, p50/p95 CPU inference latency and peak RSS. Require thresholded per-seed trap regret, not just agreement/mean coverage.
4. Upload actual trained adapter, frozen comparator, reports, receipts and galleries to immutable **DRIV3R** Actions artifacts. Keep PR #1 Draft/HOLD. No CyberSand or Neurounner write.

**Measurements for D023:** real pretrained optimizer updates **0**, new pretrained model checkpoint **none**, closed-loop episodes **0**, new runtime latency/RSS **not measured**. This is a precise execution blocker and original-receipt provenance audit, **not a successful SLM distillation**.
