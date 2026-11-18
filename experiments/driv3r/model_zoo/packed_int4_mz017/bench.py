"""MZ-017: actual INT4-packed storage vs eager unpack-on-demand CPU matvec.

Self-contained source-pinned matrix-level negative/positive kernel audit. Synthetic
inputs are not gameplay rollouts. No training, game imports or repository writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import torch

SCHEMA = "MZ017-PACKED-INT4-MATVEC-v1"
CHECKPOINT_SHA256 = "0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f"
TRAINING_SOURCE = "807ac25373c31649dd5a92a9aaa503098a86183b"
MODEL_SCHEMA = "MZ015-TURNING-MAPS-v1"


def pack_int4(w):
    """Per-output-row symmetric signed INT4 [-7,7], including real uint8 packing."""
    if w.ndim != 2 or w.dtype != torch.float32 or w.device.type != "cpu":
        raise ValueError("expected 2D CPU float32 weights")
    rows, cols = w.shape
    scale = (w.abs().amax(dim=1) / 7).clamp(min=1e-12)
    signed = (w / scale[:, None]).round().clamp(-7, 7).to(torch.int8)
    if cols % 2:
        signed = torch.cat((signed, torch.zeros(rows, 1, dtype=torch.int8)), 1)
    unsigned = signed.to(torch.uint8) & 15
    packed = (unsigned[:, ::2] | (unsigned[:, 1::2] << 4)).contiguous()
    return packed, scale.contiguous(), cols


def unpack_int4(packed, scales, cols):
    rows = packed.shape[0]
    lo = packed & 15
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(rows, -1)[:, :cols].to(torch.int16)
    signed = torch.where(codes >= 8, codes - 16, codes).float()
    return signed * scales[:, None]


def tensor_storage_bytes(t):
    return t.untyped_storage().nbytes()


def sampled_error(reference, candidate, seed=17, samples=96):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, reference.shape[1], generator=generator)
    y = x @ reference.T
    z = x @ candidate.T
    rms_y = torch.mean(y.square()).sqrt().item()
    return {
        "random_input_count": samples,
        "normalized_rmse": round((torch.mean((y - z).square()).sqrt().item() / max(rms_y, 1e-12)), 8),
        "max_absolute_difference": round((y - z).abs().max().item(), 7),
        "note": "IID random activations; these metrics are NOT policy accuracy or trajectory results",
    }


def benchmark(fn, x, *, decisions, repeats):
    with torch.inference_mode():
        for _ in range(16):
            fn(x)
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for _ in range(decisions):
                fn(x)
            samples.append((time.perf_counter_ns() - t0) / decisions / 1e6)
    samples.sort()
    return {"p50_ms": round(statistics.median(samples), 6),
            "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 6)}


def run(checkpoint, *, decisions=64, repeats=9):
    torch.set_num_threads(1)
    torch.manual_seed(17017)
    raw = Path(checkpoint).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA256 {sha}; aborting")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != MODEL_SCHEMA or payload["source_sha"] != TRAINING_SOURCE:
        raise ValueError("Unexpected trained-checkpoint source/schema")
    results = {}
    for family in ("diagonal", "gru"):
        matrices = {}
        for key, w in payload["models"][family].items():
            if not (w.ndim == 2 and "weight" in key):
                continue
            w = w.contiguous()
            packed, scales, cols = pack_int4(w)
            deq = unpack_int4(packed, scales, cols).contiguous()
            assert torch.equal(deq, unpack_int4(packed, scales, cols))
            original_bytes = tensor_storage_bytes(w)
            packed_bytes = tensor_storage_bytes(packed) + tensor_storage_bytes(scales)
            x = torch.randn(cols)
            methods = {
                "float32_matvec": lambda inp, a=w: a @ inp,
                "dequantized_int4_matvec": lambda inp, a=deq: a @ inp,
                "packed_unpack_every_matvec": lambda inp, p=packed, s=scales, n=cols: unpack_int4(p, s, n) @ inp,
            }
            with torch.inference_mode():
                discrepancy = (methods["packed_unpack_every_matvec"](x) - methods["dequantized_int4_matvec"](x)).abs().max().item()
            matrices[key] = {
                "shape": list(w.shape),
                "original_weight_bytes": original_bytes,
                "actual_packed_weight_and_scale_bytes": packed_bytes,
                "storage_ratio": round(packed_bytes / original_bytes, 6),
                "packed_dequant_parity_max_abs": discrepancy,
                "input_output_error": sampled_error(w, deq),
                "latency": {name: benchmark(fn, x, decisions=decisions, repeats=repeats)
                            for name, fn in methods.items()},
            }
        byte_fp = sum(a["original_weight_bytes"] for a in matrices.values())
        byte_packed = sum(a["actual_packed_weight_and_scale_bytes"] for a in matrices.values())
        results[family] = {
            "matrices": matrices,
            "matrix_weight_fp32_bytes": byte_fp,
            "actual_packed_matrix_weight_bytes": byte_packed,
            "packed_matrix_storage_ratio": round(byte_packed / byte_fp, 6),
            "sum_layer_p50_ms": {mode: round(sum(a["latency"][mode]["p50_ms"] for a in matrices.values()), 6)
                                  for mode in ("float32_matvec", "dequantized_int4_matvec", "packed_unpack_every_matvec")},
        }
    return {
        "schema": SCHEMA,
        "trained_checkpoint_sha256": sha,
        "training_source_sha": TRAINING_SOURCE,
        "provenance": "Previously trained MZ-015 models; original immutable Actions checkpoint; no new model training",
        "torch_version": torch.__version__,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "thread_count": torch.get_num_threads(),
        "matvec_batch": 1,
        "decisions_per_repetition": decisions,
        "timing_repetitions": repeats,
        "models": results,
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "limitations": [
            "Matrix-only benchmark: no game rollouts, action legality or policy-quality inference",
            "INT4 unsigned nibbles contain signed two's-complement codes -7..7; row scales are FP32",
            "Packed path unpacks into an FP32 temporary every matvec; NOT a fused native INT4 kernel",
            "Packed bytes are actual torch tensor storage for tested matrices, not full process RSS or checkpoint file size",
            "Latency samples are from one CPU machine, warmed up, 1 thread, batch=1; not Windows acceptance",
            "Synthetic IID Gaussian inputs are not calibrated recurrent activations",
            "No claim of end-to-end speedup, GPU performance or topology generalization",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decisions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.decisions <= 0 or args.repeats < 3:
        parser.error("decisions > 0 and repeats >= 3 required")
    report = run(args.checkpoint, decisions=args.decisions, repeats=args.repeats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for family, metrics in report["models"].items():
        print(family, "packed ratio", metrics["packed_matrix_storage_ratio"],
              "layer p50 sums", metrics["sum_layer_p50_ms"])


if __name__ == "__main__":
    main()
