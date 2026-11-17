# MZ-022 — Source-pinned Granite 4.0 H-350M hybrid onboarding

**Status, 17 November 2026: environment BLOCKED before pretrained model load.** No weights were obtained; no model-head/adapter optimizer update, gameplay trajectory, latency, or model RSS measurement exists. Do not promote on this evidence.

## New candidate, pinned provenance

- Pretrained model: [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m).
- Immutable upstream revision: [3b17b717b8f2f5d305b0a92c1491e239aeda19c8](https://huggingface.co/ibm-granite/granite-4.0-h-350m/tree/3b17b717b8f2f5d305b0a92c1491e239aeda19c8).
- [Model card at pinned revision](https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/3b17b717b8f2f5d305b0a92c1491e239aeda19c8/README.md): Apache-2.0, instruction-tuned Granite 4.0.
- [Config at pinned revision](https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/3b17b717b8f2f5d305b0a92c1491e239aeda19c8/config.json): GraniteMoeHybridForCausalLM, hidden size 768, 32 layers (28 Mamba2 and 4 attention), approximately 340M parameters.
- **Architectural novelty:** a pretrained *hybrid* combining Mamba2 and attention, rather than another scratch recurrence, BitNet repacking, or existing standalone Mamba/RWKV model.

Card and config are externally checked source declarations, not a verified runtime load. Downstream licence review and host-legal contract verification are still required.

## Real preflight and blocker evidence

Executed on the available CPU container on 17 November 2026:

- Python syntax PASS. Five local deterministic tests PASS: immutable source pin, hard training limits, raw-vs-host-legal action selection, disjoint seed/prompt splits, fail-closed preflight.
- PyTorch, huggingface_hub, safetensors present; **Transformers missing**.
- No pretrained safetensors at the pinned HF snapshot cache path.
- Both huggingface.co and github.com DNS resolution failed with gaierror [Errno -3] Temporary failure in name resolution. Therefore installing the missing runtime and downloading the real pretrained model were not possible.
- Source SHA-256: 9053797251048ddb35a1c3fb35a1d7f3cf2daee7a250229c242c315be00b43f8. Source Git blob: da5180694cbc91df0693399eb7286910db99af78. Test SHA-256: a5998bd485ea7cdc02ad82be1d313c9f352d90af128d55ee6cadd3938d994a59.
- Structured result: preflight-2026-11-17.json. **Zero optimizer steps; zero closed-loop rollouts; zero model measurements.** No fake model/substitute was trained.

## Prepared one-shot CPU experiment (not automatically executed)

Run from repository root in a network-enabled, isolated CPU runner with official compatible Transformers support:

~~~sh
python -m pip install "transformers>=4.56.0" huggingface_hub safetensors torch
python experiments/driv3r/model_zoo/granite_hybrid_mz022/run.py --preflight
python experiments/driv3r/model_zoo/granite_hybrid_mz022/run.py \
  --allow-download --out /tmp/mz022-evidence \
  --train-seeds 6 --eval-seeds 2 --steps 4 \
  --updates 32 --rollout-seeds 2 --horizon 12 --threads 2
~~~

With weights already cached, omit --allow-download. The script refuses mismatched pretrained architecture or silent context truncation. It hashes the actual pinned pretrained weight shards. It trains a **real bottleneck adapter and action head over frozen pretrained features** (not backbone LoRA), alongside a pretrained linear readout and a cheap structured-control arm. The latter is only a comparator, never the primary research result.

TRAIN, DEV, TEST world seeds are disjoint (400+, 900+, 1200+); rollout seeds 1807+. Class imbalance, changed-goal pairs, raw unmasked illegal selections, host-filtered choices, held-out teacher accuracy, goal-reaching, paint count, full-prefix CPU forward p50/p95, actual process RSS and context tokens are recorded. The teacher is not a proven solver and may fail; explicitly compare its outcomes before interpreting model goal-reaching. The external 11x11 micro-world is **not native CyberSand PAINT, Workbench or ABI acceptance**.

The checkpoint produced by the run must be preserved in an immutable Actions artifact, **not Git**. No workflow is created/modified/dispatched by this increment. If the real load fails in a network-enabled runner, record exact dependency and execution error. Do not substitute a tiny scratch model or start quantization-kernel research in this lane.
