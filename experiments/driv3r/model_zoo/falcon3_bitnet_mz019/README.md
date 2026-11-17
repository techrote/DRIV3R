# MZ-019 — Falcon3-1B-Instruct-1.58bit: pinned pretrained onboarding gate

**Status 17 November 2026:** **BLOCKED BEFORE MODEL LOAD; NO TRAINING.** Scope: DRIV3R [Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1) only.

Candidate: [tiiuae/Falcon3-1B-Instruct-1.58bit](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit/tree/72fd3f95fcd82639c902304919629edda8c6f2b4); immutable upstream revision `72fd3f95fcd82639c902304919629edda8c6f2b4`. Model is labeled 0.8B parameters and has a ~1.36 GB safetensors checkpoint (metadata, not locally measured). Public configuration: Llama-family 18-layer architecture, hidden 2048, GQA 8/4, max context 8192, `is_bitnet_config=true`, `quant_method=bitnet`. **TII Falcon-LLM License 2.0**, not Apache-2.0. License conditions need explicit downstream review. No custom remote code trusted.

**Why this model:** pretrained BitNet-like ternary architecture in the target size range, unlike MZ-016's unsuccessful posthoc full-ternary conversion. This lane tests decision quality, context and latency; Extreme Quant owns kernel development.

`run.py` is an explicit bounded CPU experiment: verifies upstream revision and actual downloaded checkpoint SHA-256, checks BitNet configuration and parameter count, extracts **true pretrained** last-token hidden states, trains a 7-action classification head with 64 real optimizer updates and an equal-update structured controller on identical disjoint synthetic material/obstacle/protected-cell TRAIN seeds, evaluates DEV/TEST and paired goal swaps, majority control, raw illegal preference against host action menus, CPU p50/p95, context length, RSS, plus four matched closed-loop trajectories per policy (trained head, structured, random non-STOP, teacher). Save trained heads and full metrics/traces externally; no checkpoint binaries in Git. The common world/counterfactual curriculum is imported from `mamba130/**`, not modified. Host filtering is **not** model-learned legality.

Manually invoke only with approved public CPU/network capacity:
```bash
python -m unittest -v experiments.driv3r.model_zoo.falcon3_bitnet_mz019.test_run
python -m experiments.driv3r.model_zoo.falcon3_bitnet_mz019.run --out /tmp/mz019-evidence --updates 64
```

**Observed execution blocker:** This isolated CPU environment has torch 2.10.0+cpu and huggingface_hub but no `transformers` module (`ModuleNotFoundError` when attempting to run a source-equivalent offline runner). Direct Hugging Face API metadata access here failed with `ConnectError [Errno -3] Temporary failure in name resolution`; pinned model weights were absent from cache. The real pretrained-model path therefore **cannot run here**. Source-level syntax/preflight is not a substitute. No CPU model latency/RSS, model loadability success, legal-action outputs, optimization update, checkpoint, or held-out game-policy outcome can be claimed. Upstream licence/runtime and BitNet Transformers compatibility remain unverified on this host.

**Next decision:** run source-matched explicit CPU evaluation with real weights/dependencies and publish its original artifact; if BitNet load fails, preserve error/stack, package versions, exact checkpoint pin and reject this runtime configuration without surrogate substitution. No auto training workflows; no paid services or external repository edits.
