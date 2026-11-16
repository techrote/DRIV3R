# SPROCKET W27 — guarded material/frontier arbitration, 384-decision held-out evaluation

**16 November 2026 | Local CPU research | Draft/HOLD, NOT PROMOTED.**
Owned source: [`worm_train_v27.py`](worm_train_v27.py); invoke `python worm_train_v27.py --print-source > /tmp/worm_train_v27_readable.py` for exact uncompressed source. The wrapper verifies SHA-256 `4e29e0f1b4126578011ce452783ee34403804001b47a3d57226f453564a5ca54` before execution. It imports the existing `worm_train_v25.py` from this same laboratory. No extra repository, branch, issue, PR or workflow created.

## Experiment actually executed

The model-side experiment trains a **rank-24 auxiliary action-conditioned gate plus residual** over two frozen experts (W24 wave 3 spatial and W25 wave 1 material/fusion). Both parents remain frozen; the network is not a Sprocket-500M LoRA adapter. Two curriculum waves of 96 genuine AdamW steps each, 192 total, with different material/frontier/KL weights; train observations 4,608 and paired material counterfactuals 1,800. Validation: 3 new seed blocks × 3 worlds; independent test: 5 new seed blocks × 3 worlds. No test data were used for optimizer steps or checkpoint selection.

Train seed blocks `21001,21013,21037,21059`; validation `22003,22019,22027`; test `23011,23029,23047,23059,23083`. All three synthetic families: greebles, pipework, architecture. W24 original ZIP SHA-256 `c1dde980b6eb2b9e2979a7f84168ff7c0e7a3a04529a94538e4c9b37375dd0b6`; W25 original ZIP SHA-256 `2ffbc9530a78f15b37b298f5e8ace44ffdc93ba260cff008e9b9ee17cba5a0d1`; original W0 synthetic host SHA-256 `81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e`. The material/protected overlay and W22 obstacle fixture are *external approximations*, not actual CyberSand material authority or native host observation ABI.

Validation selected **W27 wave 2** on the preregistered minimum material-pair retention and 256-decision coverage ratio relative to frozen W24 wave 3: validation W24 spatial 201.78 mean unique cells, W27 wave 2 203.78; validation selection score W27 wave 1 0.99945, W27 wave 2 1.0. The untouched test opened after selection.

## Held-out test (15 worlds)

| Controller | Material pairs BOTH correct / 2,200 | Mean unique @256 | Retraces @256 | Mean unique @384 | Retraces @384 |
|---|---:|---:|---:|---:|---:|
| Frozen W24 spatial | 316 | 199.73 | 334 | 264.73 | 766 |
| Frozen W25 fusion | 289 | 192.87 | 374 | 250.27 | 843 |
| W27 wave 1 | 307 | 200.47 | 339 | 270.33 | 737 |
| **W27 wave 2, greedy** | **350** | **218.87** | **179** | **281.80** | **616** |
| W27 wave 2, T=0.7 (3 draws/world) | not evaluated | 195.00 | 1,056 across 45 trials | 284.27 | 1,788 across 45 trials |

**Point-estimate dual retention gate: PASS**, under the specified requirements material-pair correctness >=95% of W24, exploration >=103% of W24 at both 256 and 384 decisions. Relative greedy improvements versus frozen W24 are +9.58% unique cells at 256 and +6.45% at 384, with 46.4% and 19.6% fewer total retraces. These are held-out descriptive improvements, not a proof of universal benefit.

**Important uncertainty / negative finding:** only 8/15 world-family combinations improved at 256, and only **6/15** improved at 384; improvements in some worlds offset serious regressions elsewhere. Seed-block bootstrap (5 independent seed clusters, all 3 families retained within each): 256-decision mean delta +19.13 unique cells, 95% CI **[-10.73,+44.08]**; 384-decision delta +17.07, CI **[-33.73,+59.60]**. Both CIs include zero. Test seed 23059 is a substantial regression (-34.0 cells at 256, -73.0 at 384 averaged across three families). The T=0.7 stochastic arm has only three hash-independent draws/world and does not demonstrate a robust improvement.

## Custody and integrity

192/192 optimizer steps independently verified; checkpoints contain complete optimizer state plus PyTorch and Python RNG. At wave boundary 96→192, all eight adapter trainable tensors changed while **72 frozen expert tensors** remained invariant and both experts matched their original admitted weights. The separate read-only audit checked 210 original 256/384-decision trajectory JSON records, **67,200 applied synthetic host actions**, zero illegal choices/protected-cell contacts, and checkpoint/optimizer counters. `root_forwards=0`, `root_updates=0`, `LoRA_steps=0`, paid API/GPU spend=0.

Local original evidence bundle: `sprocket-w27-original-local-research.zip` SHA-256 `bbd48a0cfbbd0e4debb0dd6ec0ccf8ba8c6135672f6904a5432f48ee556c94ff`, 15 payload files plus SHA256SUMS. ZIP CRC and every payload SHA-256 passed. It includes readable exact trainer source, immutable-on-disk model+optimizer+RNG snapshots, original W0 source, synthetic 384-decision painting gallery, all 210 raw trajectories, loss history, source/seed provenance, separate numerical audit, two-run reproduction crosscheck and deterministic heuristic baselines.

- W27 wave 1 checkpoint SHA-256: `051970cbd46342adc996497fdf8ed073a9abeeef6970428a247057bbdb36492c`.
- W27 wave 2 checkpoint SHA-256: `0d311bd2d9d7713dea8e140caef2dae0e2dbff3e1069de3f17857f247c793ae2`.

**Custody gap:** this evidence bundle is available locally to the task owner, **not yet uploaded to an immutable DRIV3R Actions artifact**. No claim of hosted reproducibility or native physics acceptance. Exact readable source is recoverable from the hash-checked repository wrapper. No new CI workflow added; no automatic training on ordinary pushes.

**Disposition:** retain W27 wave 2 as the most promising *local synthetic* W24/W25 arbitration result but **HOLD promotion** until independently hosted checkpoint custody, larger seed-blocked evaluation, motif/persistence measures, and untuned Sprocket-root/d1/tiny baselines. Avoid treating passing point thresholds as statistically established superiority. CyberSand and Neurounner remained read-only; no production ABI/provider/game changes, merge, Ready promotion or modification of DRIV3R main.

## Post-run independent controls and rerun

A second genuinely separate local CPU execution reproduced **both model+optimizer+RNG checkpoint files byte-for-byte** at 96 and 192 steps, the complete 210-trajectory JSON, loss-history JSON, and original PNG gallery. The complete summary is also identical after excluding elapsed wall-clock duration. Exact per-file digests are included as `evidence/reproduction_crosscheck.json` in the revised companion ZIP. This is strong **local replay determinism**, not hosted Actions provenance.

A further zero-optimizer heuristic control was evaluated on exactly the same 15 held-out test worlds using the identical source W0 fixture and host-legal action mask:

| Fixed non-neural control | Mean unique cells @128 | Mean unique @256 | Mean unique @384 | Retraces @384 |
|---|---:|---:|---:|---:|
| Material-greedy teacher | 56.13 | 77.33 | 79.40 | 2,206 |
| 5×5 frontier planner | 112.00 | 213.67 | **311.33** | 1,105 |
| W27 wave 2, greedy | not evaluated at 128 in this campaign | **218.87** | 281.80 | **616** |

**Crucial limitation:** the deterministic frontier planner **outperforms W27 by 29.53 unique cells at the 384-decision horizon**, even though W27 retraces less and beats its frozen neural ancestors. It is therefore incorrect to call W27 the best spatial policy overall. These controls are simple hand-coded synthetic policies; they do **not** substitute for missing untuned Sprocket-500M, d1 or tiny-neural baseline forwards. Scripts and raw control JSON are in the revised local evidence ZIP.
