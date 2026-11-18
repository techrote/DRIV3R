# MZ-020 — RetNet-derived 20.57M CPU model, scratch-init negative onboarding

**Status:** REAL WEIGHT TRAINING COMPLETE / NOT PROMOTED. 17 Nov 2026. Only DRIV3R Draft/HOLD PR #1.

**Named architectural reference:** Microsoft's [TorchScale RetNet](https://github.com/microsoft/torchscale/tree/4d1e0e82e5adf86dd424f1463192635b73fc8efc), exact upstream commit `4d1e0e82e5adf86dd424f1463192635b73fc8efc`, MIT reference implementation. This candidate is a **scratch-initialized, simplified PyTorch RetNet-derived policy port**, not the official TorchScale package and **not a pretrained language model**. Its recurrent retention has exponentially decayed outer-product state, rotary Q/K and gated multi-head outputs; six 512-wide layers (20,568,072 trainable parameters). No upstream model weights were loaded. This is an initial architecture execution/training check, not evidence about pretrained RetNet or an architectural comparison with MZ-015.

## Bounded concrete experiment

Original synthetic 21×21 two-fork material/painting host with protected cells, movement, RETRACE, PAINT and host-legal masks; eight mirrored/rotated frames and disappearing cue/inversion. This is NOT native CyberSand and differs from MZ-015's original three-station host. TRAIN seeds 1000–1031 (128 cue/invert episodes), held-out teacher seeds 5000–5001 (eight episodes), closed-loop seeds 9000–9001 (eight episodes). CPU torch 2.10.0; 32 genuine AdamW full-model updates, batch 1, 12-step prefix, random seed 2020 and sampler seed 2520. This budget is too short to establish model potential, particularly on long delayed-memory tasks.

Held-out classification: 109/188 approximate teacher decisions (57.98%), 4/16 mineral forks correct (25%), 41 raw illegal proposals before host masking. Closed-loop: **0/8 clean; 0/8 successful; 62 protected collisions; 0 paint operations; 232 raw illegal proposals**. Training minibatch loss first 2.13997, last 0.71196, highly nonmonotonic. More loss optimization did not establish useful game behavior. Cached/full prefix numerical parity max absolute difference: 0; cached 16-step CPU p50 5.6153 ms/decision, p95 6.2791 ms/decision; whole-process peak RSS 717.0 MiB. CPU and RSS are environment measurements, not production benchmarks.

**Decision:** reject promotion. This *does not* reject RetNet architecture or the official pretrained models: only a shallow 32-update synthetic scratch policy. It is substantially slower than the historical small-recurrent controls, though evaluation host, architecture and CPU circumstances are not matched. Future useful test requires pretrained RetNet-family weights or a real comparable training budget, identical common host/curriculum and independent test/topology families; no further scratch-port polishing without those prerequisites.

## Exact source and artifacts

The executable source is stored losslessly in `run.py` as a zlib/Base64 capsule because the current connector cannot transfer a container file directly. **This is a transfer wrapper, not opaque model weights.** It asserts the SHA-256 of the full decoded Python source `145cdf8270a85e7cef36be75a38f947f77a7701fcc6f911ed9cd2d9b66971071`; recover readable source by running:

```bash
python experiments/driv3r/model_zoo/retnet_port_mz020/run.py --extract-source > /tmp/mz020_decoded.py
python /tmp/mz020_decoded.py --out /tmp/mz020-evidence --updates 32 --train-seeds 32 --test-seeds 2 --batch 1 --prefix 12
```

The physical source is intentionally preserved byte-for-byte, although this unusual capsule format should be replaced with plain-text source before any more substantial research. `evidence.json` records the preflight metrics, exact split and checkpoint hashes. The locally trained 82 MB checkpoint is **not committed**; immutable GitHub Actions artifact publication was unavailable. No claim of exact-HEAD GitHub Actions qualification is made, and the GitHub research import by itself is not model reproduction. The source file was trained in isolated CPU environment before publication; DRIV3R execution of this commit remains to be tested.

Hard boundary: synthetic host != CyberSand game/native ABI. Never edit CyberSand or Neurounner. Never merge or promote PR #1.
