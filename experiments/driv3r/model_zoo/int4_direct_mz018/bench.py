"""MZ-018: direct packed INT4 batch-1 GEMV versus no-expansion FP32 controls.

Trained matrices only; no policy rollouts, no paid compute, no persistent binaries.
The original MZ-015 checkpoint is verified before torch.load(weights_only=True).
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_SHA256 = "0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f"
SOURCE_SHA = "807ac25373c31649dd5a92a9aaa503098a86183b"
CHECKPOINT_SCHEMA = "MZ015-TURNING-MAPS-v1"
SCHEMA = "MZ018-DIRECT-I4-CPU-v1"


def pack_int4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if w.ndim != 2 or w.dtype != np.float32:
        raise ValueError("2D float32 weights required")
    rows, cols = w.shape
    scales = np.maximum(np.abs(w).max(axis=1) / np.float32(7.0), np.float32(1e-12)).astype(np.float32)
    codes = np.clip(np.rint(w / scales[:, None]), -7, 7).astype(np.int8)
    if cols % 2:
        codes = np.pad(codes, ((0, 0), (0, 1)))
    u = codes.view(np.uint8) & 15
    packed = (u[:, ::2] | (u[:, 1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


def unpack_reference(packed: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    b = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    b[:, ::2] = packed & 15
    b[:, 1::2] = packed >> 4
    codes = b[:, :cols].astype(np.int16)
    codes[codes >= 8] -= 16
    return np.ascontiguousarray(codes.astype(np.float32) * scales[:, None])


class Kernels:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        ptr = ctypes.c_void_p
        signature = [ptr, ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.mz018_i4_gemv.argtypes = signature
        self.lib.mz018_f32_gemv.argtypes = [ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.mz018_i4_gemv.restype = None
        self.lib.mz018_f32_gemv.restype = None

    @staticmethod
    def ptr(a: np.ndarray) -> int:
        if not a.flags.c_contiguous:
            raise ValueError("contiguous arrays required")
        return int(a.ctypes.data)

    def direct_int4(self, packed, scales, x, output):
        self.lib.mz018_i4_gemv(self.ptr(packed), self.ptr(scales), self.ptr(x),
                               self.ptr(output), output.size, x.size)
        return output

    def direct_float32(self, w, x, output):
        self.lib.mz018_f32_gemv(self.ptr(w), self.ptr(x), self.ptr(output),
                                output.size, x.size)
        return output


def compile_kernel(out: Path, source: Path | None = None) -> dict:
    source = source or Path(__file__).with_name("kernel.c")
    # Portable x86_64 and AArch64 scalar C; no -march=native or build beyond temporary directory.
    cmd = ["cc", "-O3", "-std=c11", "-fPIC", "-shared", "-Wall", "-Wextra",
           "-o", str(out), str(source)]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {"command": cmd, "stdout": cp.stdout, "stderr": cp.stderr,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def timed(fn, *, batch=96, repeats=11) -> dict:
    for _ in range(32):
        fn()
    spans = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(batch):
            fn()
        spans.append((time.perf_counter_ns() - start) / (1e6 * batch))
    spans.sort()
    return {"p50_ms": round(statistics.median(spans), 6),
            "p95_ms": round(spans[min(len(spans)-1, int(.95*len(spans)))], 6),
            "window_ms": [round(x, 6) for x in spans]}


def one_matrix(kernels: Kernels, w: np.ndarray, *, seed: int, batch: int, repeats: int) -> dict:
    packed, scales = pack_int4(w)
    rows, cols = w.shape
    deq = unpack_reference(packed, scales, cols)
    x = np.random.default_rng(seed).standard_normal(cols).astype(np.float32)
    x_sparse = np.zeros(cols, dtype=np.float32)
    x_sparse[::7] = 1.0
    y0 = np.empty(rows, dtype=np.float32)
    y1 = np.empty(rows, dtype=np.float32)
    yp = np.empty(rows, dtype=np.float32)
    for test_x in (x, x_sparse):
        kernels.direct_int4(packed, scales, test_x, yp)
        kernels.direct_float32(deq, test_x, y1)
        tolerance = 1e-4 + 2e-5 * max(1., float(np.max(np.abs(y1))))
        if float(np.max(np.abs(yp - y1))) > tolerance:
            raise AssertionError("direct packed kernel differs from predequantized INT4")
    kernels.direct_int4(packed, scales, x, yp)
    kernels.direct_float32(deq, x, y1)
    kernels.direct_float32(w, x, y0)
    relative_rmse = float(np.sqrt(np.mean((yp - y0).astype(np.float64)**2)) / max(1e-12, np.sqrt(np.mean(y0.astype(np.float64)**2))))
    tx, tw, td = torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(deq)
    tp, ts = torch.from_numpy(packed), torch.from_numpy(scales)
    def torch_eager_unpack():
        codes = torch.stack((tp & 15, tp >> 4), -1).reshape(rows, -1)[:, :cols].to(torch.int16)
        signed = torch.where(codes >= 8, codes - 16, codes).float()
        return (signed * ts[:, None]) @ tx
    paths = {
        "torch_fp32": lambda: tw @ tx,
        "torch_predequant_i4": lambda: td @ tx,
        "torch_eager_unpack_each": torch_eager_unpack,
        "native_f32_original": lambda: kernels.direct_float32(w, x, y0),
        "native_f32_dequantized": lambda: kernels.direct_float32(deq, x, y1),
        "native_i4_direct_ctypes": lambda: kernels.direct_int4(packed, scales, x, yp),
        "numpy_unpack_then_blas": lambda: unpack_reference(packed, scales, cols) @ x,
    }
    # Each method gets the same calls/warmup with no per-method per-matrix allocation except unpack baseline.
    timings = {k: timed(fn, batch=batch, repeats=repeats) for k, fn in paths.items()}
    return {"shape": [rows, cols], "weight_fp32_bytes": w.nbytes,
            "packed_nibbles_and_scales_bytes": packed.nbytes + scales.nbytes,
            "packed_ratio": round((packed.nbytes + scales.nbytes) / w.nbytes, 7),
            "direct_i4_to_fp32_dequant_max_abs": float(np.max(np.abs(yp - y1))),
            "direct_i4_to_original_fp32_normalized_rmse": round(relative_rmse, 7),
            "timings": timings}


def run(checkpoint: Path, *, batch: int=96, repeats: int=11) -> dict:
    torch.set_num_threads(1)
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("MZ-015 checkpoint sha256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["source_sha"] != SOURCE_SHA:
        raise ValueError("MZ-015 checkpoint schema/source mismatch")
    with tempfile.TemporaryDirectory(prefix="mz018-native-") as td:
        compile_info = compile_kernel(Path(td) / "libmz018.so")
        kernels = Kernels(Path(td) / "libmz018.so")
        families = {}
        for family in ("diagonal", "gru"):
            matrices = {}
            for i, (name, weight) in enumerate(sorted(payload["models"][family].items())):
                if weight.ndim != 2 or "weight" not in name:
                    continue
                matrices[name] = one_matrix(kernels, weight.detach().numpy().copy(),
                                             seed=181800 + i, batch=batch, repeats=repeats)
            families[family] = {"matrices": matrices, "aggregate_weight_bytes_fp32": sum(m["weight_fp32_bytes"] for m in matrices.values()),
                                "aggregate_weight_bytes_int4": sum(m["packed_nibbles_and_scales_bytes"] for m in matrices.values()),
                                "sum_independent_matrix_p50_ms": {mode: round(sum(m["timings"][mode]["p50_ms"] for m in matrices.values()), 6)
                                    for mode in next(iter(matrices.values()))["timings"]}}
        return {"schema": SCHEMA, "checkpoint_sha256": CHECKPOINT_SHA256,
                "training_source": SOURCE_SHA,
                "kernel_source_sha256": compile_info["source_sha256"],
                "compiler_command": compile_info["command"],
                "host": {"system": platform.platform(), "machine": platform.machine(),
                         "processor": platform.processor(), "python": sys.version.split()[0],
                         "torch": torch.__version__, "numpy": np.__version__, "thread_count": torch.get_num_threads()},
                "batch_1_repeated_calls": batch, "timing_windows": repeats,
                "models": families,
                "peak_whole_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
                "limitations": ["Native scalar kernel is genuinely direct packed inference; packed weights remain UINT8 nibbles + FP32 scales",
                     "Only trained MZ-015 matrices; no policy rollout, no game state, no legal-action or trajectory evaluation",
                     "ctypes overhead included; timings sum independent matrices, NOT full policy inference",
                     "FP32 controls are scalar native and PyTorch BLAS; CPU-only synthetic activations; no deployment or Windows data",
                     "No trained model weight/checkpoint, binary or game runtime edits committed"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--repeats", type=int, default=11)
    a = p.parse_args()
    if not 1 <= a.batch <= 4096 or not 3 <= a.repeats <= 50:
        p.error("bounded benchmark: batch in 1..4096, repeats in 3..50")
    report = run(a.checkpoint, batch=a.batch, repeats=a.repeats)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, family in report["models"].items():
        print(name, family["sum_independent_matrix_p50_ms"])

if __name__ == "__main__":
    main()
