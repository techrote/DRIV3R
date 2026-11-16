# SPROCKET W26 — route-memory / long-horizon negative result (16 Nov 2026)

**Local CPU result; HELD / NOT PROMOTED.** Ownership: `techrote/DRIV3R` consolidated Draft/HOLD PR #1 only. Historical CyberSand/Neurounner remain read-only. No game-side edits, paid services, new workflows or branch/PR creation.

## Genuine training
Three waves of 120 AdamW steps (360 total), 6,137 training observations, 1,800 training material-counterfactual pairs. A 32-feature route-memory auxiliary adapter (324→48→6) was trained against a frozen W25/W24 fusion-policy lineage. Four trainable tensors changed in each wave; 36 parent tensors were frozen. Full model/optimizer/torch-RNG/Python-RNG checkpoints saved **locally**, not in Git or yet in DRIV3R Actions. Two independent runs produced byte-identical checkpoints. No 500M Sprocket-root inference, LoRA updates or d1/tiny baselines.

- W0 host source SHA-256: `81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e`.
- W24 original ZIP SHA-256: `c1dde980b6eb2b9e2979a7f84168ff7c0e7a3a04529a94538e4c9b37375dd0b6`.
- W25 original ZIP SHA-256: `2ffbc9530a78f15b37b298f5e8ace44ffdc93ba260cff008e9b9ee17cba5a0d1`.
- W25 source SHA-256: `4d431e543457d9edc6492b48d1c5e56e524d4736c27fc8107f61094579727b85`.
- Frozen W25 wave1 checkpoint: `46c5c915bec4155ca73928aa62e33371506e40d6368c153dfb9b3b7e6986f6e5`.
- W26 checkpoint SHA-256 at step 120: `4a9cecdef3e7898ed98d91e5c0094591ed36a78969f9b09c3bcc22fa15bcf20c`; 240: `d1ae0de77742e823a14c6ba212cca985b9d577e98696c53fbace3705692d218f`; 360: `b51b063282d5d4f79dc6b9c12e51dffc06a4bbaf70f022ed8dea66f81efbf164`.
- Exact *readable* training source SHA-256: `1e27211f8011202bdb519cab401aa6720d0aefde50e636ff71ac6c338df53d85`; separate held-out evaluator: `04b7c9c69730d087b0cf0d2576712156eb8c016517ac3f03769b09cfadc2afa1` (see exact downloadable artifact manifest for authoritative evaluator hash).

## Predeclared validation selection and held-out test

Train seed blocks 18013/18029/18047/18061; validation 19001/19009/19031; independent test 20011/20021/20047/20063/20089, each with all three synthetic world families. W26-wave2 was selected **using validation only**: validation 256-decision mean unique cells W24-wave3 197.89, W26-wave2 200.22; material pairs both correct on validation W25-wave1 243/1,500, W26-wave2 254/1,500. This apparent improvement did **not** generalize.

| Held-out controller (15 synthetic worlds) | Material pairs both correct /2,200 | Unique @128 | Unique @256 | Unique @384 | Retraces @384 |
|---|---:|---:|---:|---:|---:|
| Frozen W24-wave3 spatial reference | 370 | 118.73 | **214.87** | **307.27** | 398 |
| Frozen W25-wave1 parent | 348 | 105.87 | 164.93 | 210.33 | 1,189 |
| W26-wave1 | 307 | 111.20 | 188.67 | not tested | — |
| **W26-wave2 selected** | **352** | **110.07** | **162.80** | **193.67** | **1,349** |
| W26-wave3 | 347 | 103.60 | 187.20 | not tested | — |
| W26-wave2 sampled T=0.7 | n/a | 112.40 | 215.40 | 312.67 | 389 |
| W26-wave2 memory-ablation | n/a | 107.27 | 161.53 | 200.00 | 1,334 |

The independent material-retention gate passed; **256-step exploration, 384-step exploration and retrace gates failed**. Paired seed-block bootstrap 256-step unique-cell delta W26-wave2 minus W24-wave3: **−52.07**, 95% CI [−75.60, −28.53]; at 384 steps: **−113.60**, CI [−176.27, −50.93]. Every one of five held-out seed blocks regressed at 384 steps. Zero invalid synthetic host-legal actions or protected-cell contacts.

Temperature 0.7 recovers comparable trajectory coverage, but each world was sampled only once under a deterministic hash-derived draw: *not proof of a superior stochastic policy*. Zeroing learned memory improves 384-step coverage slightly, so persistent memory has **not** demonstrated a meaningful long-horizon benefit.

## Reproduction and limits

Original source/optimizer checkpoints, W0 source, paired 15-world trajectory JSON, 384-step painting gallery, per-file SHA-256 manifest and independent evaluator summary are in the local companion archive `sprocket-w26-original-local-research.zip` (SHA-256 `d3f120468b0f867555c7af0bce0328531193c4ee10db36e1d1c26c048a39d7c9`). Archive CRC/payload hashes passed. Two separately executed evaluations returned identical metrics and trajectories; NumPy inference logits matched PyTorch within 2.87e-6. **This archive is not an immutable DRIV3R Actions artifact and must not be called hosted reproducibility.**

W0 with W18/W22 synthetic overlays is an external game-like fixture, **not** the authoritative CyberSand action/observation ABI or physics. Local material substitutions are not persistent physical material interventions. No production adoption, merge, Ready promotion, or Sprocket-root improvement claim.

**Decision: retain as a substantive negative experiment; do not promote.** Next discriminator: repeated-seed temperature study plus on-policy trajectory-level frontier objectives with explicit held-out 256/384-step gates. Keep research inside this lane and PR #1.
