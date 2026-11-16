# MZ-017 — Actual packed INT4 storage, CPU unpack-cost audit

**Status:** standalone, bounded research increment for [DRIV3R Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1). Never merge or promote without owner acceptance. No game, production-provider, or external-repository writes.

## Concrete question

MZ-016 found that post-training rowwise INT4 substitution preserved **64/64 closed-loop clean episodes** for each trained MZ-015 recurrent policy, but measured only *dequantized FP32 inference* and theoretical packed bytes. Does physically packing those same matrices into INT4 nibbles yield smaller actual tensors, and can a straightforward on-demand PyTorch unpacking path be competitive at single-decision CPU batch 1?

## Source and execution

- Source checkpoint: **MZ-015** `mz015/models.pt`, SHA-256 `0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f`, training source commit `807ac25373c31649dd5a92a9aaa503098a86183b`. The original [historical immutable Actions artifact](https://github.com/techrote/cybersand/actions/runs/43942288057/artifacts/14265125044) is **read-only**. No checkpoint weights were committed here.
- 16 November 2026; PyTorch `2.10.0+cpu`; Linux `6.18.57`, x86-64; one CPU thread. This environment is not representative of physical Windows/NVIDIA acceptance.
- Ten actual weight matrices: four from the trained diagonal recurrent model, six from the trained GRU. Per-output-row max-abs symmetric signed INT4 `[-7,7]`, two signed two's-complement nibbles per `uint8` byte; FP32 row scales. Padding for odd-width rows is included in actual storage.
- Each matrix: 96 reproducibly seeded Gaussian activation samples for normalized output RMSE; three matvec paths (original FP32, INT4 *predequantized* to FP32, and actual packed INT4 unpacked to FP32 *on each call*); 16 warmups, 9 × 64 timed calls per path. Tensor storage bytes use PyTorch's `untyped_storage().nbytes()`; latency uses Python monotonic high-resolution timer.
- Four local unit tests PASS. Source blob hashes: `bench.py` `65b84b63f8e70b5f0ddb9c09bef6c47bb0543620`, `test_bench.py` `80209998b84e7f5731cb601bef04dc63d0230635`. Detailed measured matrix values, p50/p95, parity and output errors appear in [measurements.csv](measurements.csv). The script also emits a fuller JSON report when run. No CI or remote rerun is claimed.

## Results

| Trained matrix family | Actual original FP32 weight bytes | Actual packed INT4 + scales bytes | Packed fraction | Sum of independent matvec p50, FP32 | Sum, predequantized INT4 | Sum, unpack every matvec |
|---|---:|---:|---:|---:|---:|---:|
| Diagonal (4 matrices) | 827,776 | 107,744 | **13.02%** | 0.018585 ms | 0.018746 ms | **0.888685 ms** |
| GRU (6 matrices) | 939,648 | 124,256 | **13.22%** | 0.023613 ms | 0.024049 ms | **0.682604 ms** |

All ten packed/unpacked outputs matched their predequantized counterparts exactly on the checked input (max absolute difference **0**). Against original FP32, random-input output normalized RMSE varied **10.996%–16.646%** between matrices; this is not a policy-level accuracy estimate. Peak *whole experiment* process RSS was 261.3 MiB, not a measurement of a packed-only deployment.

**Decision:** actual nibble packing provides the expected *weight-buffer* reduction, but naïve eager PyTorch unpacking is **~48× slower** than the four diagonal FP32 matvecs and **~29× slower** than the six GRU FP32 matvecs under these local conditions. Reject the *unpack-on-every-matvec implementation* as a fast CPU inference route. Retain packed INT4 as a storage option only, pending a fused native or optimized kernel that directly consumes packed weights and a full-policy benchmark. The MZ-016 closed-loop INT4 result is not invalidated; this run does **not** rerun those rollouts.

The p50 sums combine **independently timed matrices**, not a coherent recurrent-forward pass. They are not end-to-end decision latency, and no deployment/energy advantage is demonstrated. Random Gaussian inputs do not represent calibrated recurrent activations or unseen game topologies. Four-bit accuracy on genuine CyberSand gameplay remains untested.

## Reproduce (explicit, bounded)

From this directory, after obtaining the immutable source checkpoint (and verifying its SHA-256):

```bash
python -m unittest -v test_bench
python bench.py --checkpoint /path/to/mz015/models.pt --out mz017-full-metrics.json --decisions 64 --repeats 9
```

No automatic push/PR training workflows, paid inference, or branch creation. Only `experiments/driv3r/model_zoo/packed_int4_mz017/**` is changed by this increment. Further work should benchmark real int4 GEMV kernels or selectively mixed-precision packed kernels against the cached-state full-policy baseline, using topology-diverse held-out worlds.
