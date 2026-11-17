# Wave 20 — whole-layer-lobotomy restart: source ready, inference BLOCKED

17 November 2026. Source owner: DRIV3R PR #1 only. Candidate identities frozen in `flows.py`; 8-, 6- and 4-execution masks each contain **two different source-layer selections**. The existing pinned patch calls `const int il = d1_flow[d1_step]` in the actual llama.cpp forward graph; skipped transformer blocks are not executed in *decision mode*. Embedding/output machinery remains, the frozen Q8 donor weights are not rewritten, and both attention types are guarded by model metadata. This is structural *source preparation*, not proof a rebuilt model will load.

The bounded `wave20_structural.py` driver specifies exactly 10 arms × 2 held-out synthetic W0 episodes × 24 decisions = **480 maximum**. Controls are stock, GREYHOUND, BRANCHWITCH and the known four-layer AERO4_NO2. The experiments use genuine model six-way action choices applied immediately by the host; no movement-script, shield, reranking or smoothing. It retains source- and model-linked receipts, original JSON/PNG, exact predecision SVG polylines, p50/p95, raw legal choices, STOP, revisits and sampled post-ready RSS. The final post-action point is not available from predecision records, so no endpoint is invented.

**No new inference, model build, CPU profiling or SVG is claimed in this run.** The current session has no Q8 donor GGUF, pinned llama-server binary or network-enabled 1.8 GB model transfer, and its GitHub connection does not provide workflow dispatch. Further, the DRIV3R source import omits `experiments/driv3r/backend.py`, `experiments/agent_painter/process.py` and the original W0 runtime used by the inherited infer module. Those are missing **runtime dependencies**, not an excuse to build a toy substitute. Supply them as a pinned *read-only external CyberSand checkout* and record its commit identity. Standalone imported syntax checks are not scientific reproduction.

Manual execution from an exact clean research SHA in a CPU runner:

```bash
# Build pinned llama.cpp 8642541... with the existing lab patch (source-check receipt).
python -m experiments.driv3r.frankenlayers_lab.patch_lab --llama-root "$LLAMA_SRC" --receipt "$RECEIPT"
# After compiling the patched llama-server from exactly that tree:
python -m experiments.driv3r.frankenlayers_lab.wave20_structural \
 --binary "$LLAMA_SERVER" --model-q8 "$Q8_GGUF" --w0 "$W0_FILE" \
 --legacy-root "$READ_ONLY_CYBERSAND_ROOT" --patch-receipt "$RECEIPT" \
 --source-sha "$(git rev-parse HEAD)" --output "$NEW_IMMUTABLE_EVIDENCE_DIR"
```

Do not promote any specimen to Meanderer/Curler/Brancher/Chaotic Artist until original independent trajectory review. STOP-only, startup/attention errors and fixed circles must be retained as negative results. Preserve only ephemeral large model data or immutable Actions artifacts; no Git binary outputs. CyberSand and Neurounner remain read-only. No merge/Ready/auto-merge.
