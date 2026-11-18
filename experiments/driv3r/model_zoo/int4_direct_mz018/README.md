# MZ-018 — CPU GEMV operating directly on packed signed INT4

**Research-only incremental candidate for [DRIV3R Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1).** No extra PR, branch, workflow, automatic training, game code or external-repository write.

## Question and specimen custody

MZ-016 established 64/64 clean synthetic game-policy rollouts for each of the frozen diagonal-state and GRU controllers with rowwise INT4 *dequantized to FP32*. MZ-017 then measured genuine packed storage, but unpacking into FP32 tensors **on each matvec** was expensive. Does a tiny C99 kernel that consumes the signed nibbles **without materializing an FP32 weight matrix** reduce that overhead while preserving the same INT4 mathematical operation?

This benchmark uses the ten 2-D learned weight matrices from the MZ-015 source-matched `mz015/models.pt`: SHA-256 `0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f`; source commit `807ac25373c31649dd5a92a9aaa503098a86183b`; original immutable [historical Actions artifact](https://github.com/techrote/cybersand/actions/runs/43942288057/artifacts/14265125044) READ ONLY. Both checkpoint hash and internal source/schema are verified before benchmarking. No checkpoint bytes or compiled libraries are committed.

## Actual executed experiment — 16 November 2026

- C11 implementation `kernel.c`, compiled with `cc -O3 -std=c11 -fPIC -shared -Wall -Wextra` to a temporary directory; direct rowwise signed INT4 GEMV and matched *scalar C FP32* GEMV controls. The kernel is **scalar and not architecturally optimized**. Nibbles are two's-complement codes with per-row float32 scales, odd widths padded to whole bytes.
- Python 3.13, PyTorch CPU, NumPy, one Torch thread, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, Linux x86_64 AMD EPYC virtual CPU. 32 warmups, 11 windows of 96 batch-1 calls each, per matrix and mode. Each mode includes the Python or ctypes invocation overhead. Native C, PyTorch FP32, predequantized INT4, PyTorch eager re-unpack, and NumPy eager re-unpack are all measured on the *same* host with the *same* vectors. No full policy forward is timed.
- Five deterministic source tests passed: signed/odd-width parity, negative-eight decode, no input mutation, shape guards, checkpoint custody. For all ten real learned matrices, C packed output matched predequantized INT4 inference to within 2.87e-6 absolute on a fixed Gaussian input, and parity also passed a sparse binary input. This is **not** parity against original full-precision weights.
- Frozen trained weights: diagonal **827,776 FP32 matrix bytes → 107,744 packed+scales** (13.02%); GRU **939,648 → 124,256** (13.22%). These are measured weight-buffer byte counts only, not entire model/RSS.

| Sum of independently timed matrix p50 values | Diagonal (4 GEMVs) | GRU (6 GEMVs) |
|---|---:|---:|
| Original PyTorch FP32 matvec | 0.019569 ms | 0.024847 ms |
| Native scalar C FP32 matvec | 0.151279 ms | 0.148503 ms |
| PyTorch eager INT4 unpack every GEMV | 0.960853 ms | 0.759720 ms |
| **Native C direct packed INT4 via ctypes** | **0.166840 ms** | **0.192453 ms** |
| Eager NumPy unpack + BLAS | 2.077436 ms | 2.250942 ms |

Direct INT4 is **5.76×/3.95× faster than the corresponding PyTorch eager-unpack baseline**, but **8.53×/7.75× slower than PyTorch FP32 BLAS** for diagonal/GRU here. Relative to the scalar C FP32 control, direct INT4 is 1.10×/1.29× slower. The results are not evidence of acceleration over mature FP32 kernels. Whole-process peak RSS **258.97 MiB** reflects Python, PyTorch and benchmark harness, not a compressed deployment.

Original FP32 vs quantized INT4 GEMV output normalized RMSE across one Gaussian vector per matrix: **10.34%–19.92%** (sample size 1 per matrix). This is not closed-loop policy quality. Full per-matrix timings, p95, output checks, environment/compiler and machine-readable caveats are in [measurements.json](measurements.json), with shorter [measurements.csv](measurements.csv).

## Decision

**Retain the packed format, reject this scalar kernel as a production-speed solution.** It establishes the direct-packed inference path and eliminates repeated full-weight expansion, but optimized SIMD/bitpacked GEMV is required to be competitive with FP32 BLAS. The next justified bounded experiment is an optimized kernel with **end-to-end cached recurrent-state** benchmarking, mask/legal-action checking, and held-out genuine closed-loop trajectories on unseen topologies. Do not treat the MZ-016 prior closed-loop result as newly reproduced here.

## Reproduce explicitly (no automatic CI/training)

```bash
# From this directory. Inputs remain external immutable artifacts.
python -m unittest -v test_bench
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python bench.py --checkpoint /path/to/mz015/models.pt \
  --out /tmp/mz018-measurements.json --batch 96 --repeats 11
```

Assumptions/limitations: synthetic inputs rather than calibrated recurrent activations; one CPU virtual machine; timings are independent matrix p50 **sums**, not wall-clock sequential model layers; Python/ctypes overhead included; no full-policy inference, legal-action statistics, gameplay rollouts, training, GPU, real CyberSand state, Windows or production acceptance. The MZ-015 trained specimens are 209k/237k parameters and therefore mechanism probes, not standalone 20M–1B SLM promotion evidence.
