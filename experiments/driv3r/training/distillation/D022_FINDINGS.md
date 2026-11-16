# D022 — Iterated on-policy curriculum versus trap-hard negatives (16 Nov 2026)

**Research only; [DRIV3R Draft/HOLD PR #1](https://github.com/techrote/DRIV3R/pull/1).** No CyberSand or Neurounner writes. External synthetic game, not native CyberSand.

## Hypothesis and budget

D021's compact 4,714-parameter student passed permissive non-regression gates, but trap coverage was seed-sensitive. Test whether a **second student-state training pass** improves that stability; compare uniform on-policy replay against deliberate oversampling of trap action disagreements. Fixed starting D021-mild checkpoint, **two genuinely trained students, 120 CPU Adam steps each (240 total), batch 288**, no paid inference/GPU. Predeclared disjoint training seeds 61025–61040, validation 66001–66008, test 67001–67016, confirmation 68001–68016 and stress 69001–69016. Validation chooses the student, then three held-out blocks are evaluated without reselection.

The new corpus contains **11,948 student-state synthetic observations**; **568 policy disagreements (244 in dynamic traps)**. Both curricula also use 60,366 offline D018 rows and 18,291 earlier D021 student-state rows. The uniform arm replicates each new disagreement once; the trap-hard arm additionally replicates new trap disagreements twice and all new trap states once. The teacher is **frozen D017 synthetic D007+branch gate**. Original Q8 Frankenbrain model observations from seed 570 supply **style identity only**; no original parent forwards on new game states.

## Results

Validation chose **D022-uniform** over the trap-hard arm. Both satisfied permissive D017 comparison constraints on the 8 validation seeds, but trap-hard showed much worse validation trap-tail robustness against D021 (minimum seed delta -8 versus -0.75 cells). D022-uniform also passed the old *mean-loss ceiling* on the three independent test blocks, but this **does not prove improvement**, and no model should be promoted.

| Delta per 80-decision episode | Test 67001–67016 | Confirmation 68001–68016 | Stress 69001–69016 |
|---|---:|---:|---:|
| Normal coverage vs D017 | -0.83 | -0.33 | +1.31 |
| Dynamic-trap coverage vs D017 | -1.48 | -0.81 | -0.95 |
| Dynamic-trap branches vs D017 | +1.08 | +0.19 | +0.84 |
| Protected-squeeze coverage vs D017 | +0.06 | -0.08 | +0.09 |
| Dynamic-trap coverage vs D021 | **-1.83** | 0.00 | **-1.72** |

Trap-coverage 95% seed-block bootstrap CIs **versus D017** are [-4.61,+1.47], [-5.14,+2.61], [-3.02,+1.20] across test, confirmation and stress; none establishes superiority. The paired trap-coverage deltas **versus D021** have approximate separately computed 95% seed-block intervals [-4.44,+0.56], [-1.53,+1.47], [-3.91,+0.64]. Thus there is no established improvement over either comparator, despite additional normal branching. Test worlds show five seed blocks with trap-coverage losses of at least five cells versus D017; confirmation three, stress two. Relative to D021 these counts are four, one and three. These are meaningful asymmetric-tail risks concealed by permissive mean acceptance guards.

**Disposition:** reject D022 as a replacement for D021 or D017. The branch/coverage stability problem remains. D007 remains the gameplay-coverage champion; D017 remains the selective-branching reference. Avoid more indiscriminate disagreement oversampling. Future experiment should penalize worst-case per-seed trap regret rather than choosing solely on mean non-regression.

## Evidence and limits

- Source/checkpoint SHA-256: D007 `1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f`; D017 gate `715dff35c08a4d19bb7f523925eace020f35bdcaa3bc7bfe44bf2aa67606f11f`; Q8 original `140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522`; D021-mild `9238f6ca4b6d54a099914cd6f61388acebf5acf743938918fe928c5f90a05f55`; D018 offline corpus `336bd906446de4443c68e0696091c581d35ac0ea1b2ece27beb745d7dbaf649f`; earlier D021 on-policy corpus `69aa4febe5e72005a4a02f9bc306371a8a3c8de78d897162a72785e2786bcf9a`.
- Both trained D022 checkpoint SHA-256: uniform `2aea7320a7c5c32e4357c9742331ccfd24307dbfabb2069aa5ded1b0f0a78d5a`; trap-hard `7a9a5fc50a98a80334ac8f957715f4293f050fe6fc1168710d727313fd0e771f`.
- Separate replay audit: **1,728 synthetic episodes**, **109,994 decisions**, **2,394 matched action receipts**, **36 byte-identical PNGs**, **zero illegal actions**, all recorded metrics and evidence manifests checked; four new unit tests PASS. Audit shares the archived simulator fixture; it is not an independent native-game implementation.
- Source from the previous D021 local evidence bundle was used to train. Its archived `branch_retention_020.py` blob matched DRIV3R's imported source, but archived `lab.py`, `repair.py` and `source_parent_branch_gate.py` blobs differed from the current PR branch; **exact-current-source replay remains unverified**. Imported Git source identity is not scientific reproducibility.
- Trained NPZ checkpoints, corpus, complete JSONL traces, trajectory galleries and reproducible runner inputs are preserved **outside Git history**, in a locally hash-addressed evidence bundle. No immutable GitHub Actions artifact was created or implied; no per-experiment workflow or automatic training on pushes added.
