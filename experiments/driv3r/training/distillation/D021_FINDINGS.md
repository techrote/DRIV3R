# D021 — student-state correction after compact branch retention (16 Nov 2026)

**Scope:** Research only; DRIV3R Draft/HOLD PR #1; no CyberSand/Neurounner writes; external synthetic game only.

## Research question
Does DAgger-style correction on *states actually reached by D020* repair the compact policy's out-of-distribution action drift without losing the BRANCH selectivity of frozen D017?

## Experiment
18,291 new synthetic on-policy observations (928 D020-vs-D017 disagreements) across training seeds 61001–61024, four styles and normal/trap/protected-squeeze cases. D018 offline corpus contained 60,366 observations. Two 4,714-parameter standalone D020-initialized student candidates: mild (1x agreement + 1x disagreement), strong (1x agreement + 5x disagreement). Each received exactly 120 genuine CPU Adam updates, batch 288, and legality-aware KL plus branch margin and anchor penalty. Total **240 optimizer updates**. Validation seeds 62001–62008 selected mild. Test seeds 63001–63016 and independent confirmation 64001–64016 were withheld from optimization/selection; exploratory fresh-seed diagnostic 65001–65016 was added **post hoc** and was not a selection criterion.

## Held-out outcomes
The mild model satisfied the original D020 hard mean guards on both predeclared splits and the post-hoc diagnostic. These are permissive loss ceilings, **not evidence of dominance over D017**.

| Mean delta vs D017 per 80-decision episode | Test | Confirmation | Post-hoc third |
|---|---:|---:|---:|
| Normal unique cells | +1.13 | -0.77 | +0.06 |
| Trap unique cells | +0.70 | -1.33 | -1.03 |
| Squeeze unique cells | +0.09 | +0.28 | +0.03 |
| Normal branch count | +1.64 | -0.05 | +0.63 |
| Trap branch count | +1.09 | -0.50 | +0.33 |

Trap-coverage seed-block 95% intervals: test [-1.66,+2.92], confirmation [-3.56,+1.00], post-hoc [-3.42,+1.13]. Thus there is **no established coverage gain**, and D020 occasionally exceeds D021 on untouched worlds. Strong correction was less stable and failed the post-hoc trap coverage threshold (-2.19 cells). Do not replace D007 or D017 with D021.

## Verification and provenance
Three source unit tests PASS; independent replay: 2,304 synthetic episodes, 147,178 recorded decisions across three test splits, 3,008 exact action receipts, and 48 byte-exact PNGs; all replayed actions were legal. These numbers are from a local archived D020 simulator snapshot, not a qualified execution of current DRIV3R branch dependencies or native CyberSand.

Source: D007 SHA-256 `1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f`; D017 gate `715dff35c08a4d19bb7f523925eace020f35bdcaa3bc7bfe44bf2aa67606f11f`; original Q8 `140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522`; D020 `74eca11283b821d5a032486d7daa42791772520f87a21e025bcdb337d2857713`; D018 offline corpus `336bd906446de4443c68e0696091c581d35ac0ea1b2ece27beb745d7dbaf649f`. Selected checkpoint `9238f6ca4b6d54a099914cd6f61388acebf5acf743938918fe928c5f90a05f55`; stronger alternative `ca6891900544ad1882e47ed959bf404ae49c680c8f27392280e25346942522f9`.

The archived source snapshot's `branch_retention_020.py` Git blob was byte-identical to the imported DRIV3R PR #1 blob `15da11ab4f93154e4f7715f154941fd2bacdee28`. The local archived `lab.py`, `repair.py` and `source_parent_branch_gate.py` blobs were *not* identical to the imported branch's corresponding blobs; selected inference functions were manually checked, but full exact-head runtime equivalence is **unverified**.

This PR commit preserves source and scientific findings. Trained checkpoint NPZs, complete action receipts, pictures, and corpora are retained in a separate local, hash-addressed evidence bundle and intentionally are **not committed to Git history**. An immutable GitHub Actions artifact is **not yet published**, since the only existing workflow is a source smoke job without artifact publication. New per-experiment CI was intentionally not created. No stock d1 / Frankenbrain forwards were performed on new game states. Original Q8 seed 570 only provides a style prior.

## Disposition
**D021-mild: research-qualified in an archived synthetic simulator, not promoted.** Further source-matched exact-HEAD replay and genuinely independent parent-game trajectories remain necessary.
