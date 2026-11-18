# C65 — independent archive verification and CPU rerun (2026-11-18)

**Status: independent synthetic-host replay PASS; byte-identical checkpoint reproduction FAIL; source/artifact publication INCOMPLETE. DRAFT / HOLD, no promotion.**

This is a supplementary verification of the provisional C65 research record, not a new trained controller. All work is model-side and synthetic-host-only. No native CyberSand code, production ABI or game assets were modified. Original evidence and negative results remain unchanged.

## Material independently inspected

Owner archive: DRIV3R_C65_paired_delayed_credit_2026-11-18.zip (provided as a conversation attachment; not an immutable GitHub Actions artifact). Its nine ZIP entries include executable source, 6 regression tests, executed 72-row/432-suffix evidence, 36 trace samples, a rendered gallery and two safetensors checkpoints. All eight manifest SHA-256 entries match their corresponding archive bytes. The C65 executable source SHA-256 is 0c08683bbe7033ca6ab7920643cd14be2cf0eb634bd2ef10cadc34f6a21d09c6. The C65-ALIGNED and C65-SHIFT original checkpoint SHA-256 values are 4d603774064e858fe7635e9286f4efd60513e7c425b74cc64ebbc613699fc469 and 2290edabeafdffe25fb0e60c206b22d675dd4f6c9c6a3a271a21b64153a145af.

Independent local executable environment was assembled from the frozen C63/C64 predecessor evaluation archives; test_c65_paired_delayed_credit.py passed 6/6 checks. Full C65 CPU training and fresh synthetic-host evaluation were then rerun using the original archived source, with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PyTorch 2.10.0+cpu, NumPy 2.3.5, and Python 3.13.5. This constitutes a real optimizer rerun, not just archive inspection.

## Independent outcomes

The rerun independently consumed the same 4,184 host-executed offline training actions and performed 48 AdamW updates per arm. All **72 matched condition rows and 432 policy suffix outcomes** were identical to the original evidence, including action SHA-256, painting SHA-256 and goal completion. All **36 sampled trajectory replay records** were identical; generated gallery SVG bytes were also identical.

Per intact-memory/50%-erased-memory condition, respectively: RESET 10/24 and 10/24; non-neural FRONTIER 12/24 and 12/24; frozen C60 12/24 and 12/24; C65-ALIGNED 10/24 and 10/24; C65-SHIFT 10/24 and 10/24. Zero host rejections, zero protected-cell state divergence, and all 24 unguided parity conditions were verified. The aligned model does not improve completed remote goals over RESET or wrong-credit SHIFT.

**Important numerical reproducibility caveat:** The two rerun checkpoints were *not* byte-identical to the supplied owner checkpoints. Maximum absolute weight differences were approximately 8.20e-8 (ALIGNED, 7/12 coefficients differing) and 1.94e-7 (SHIFT, 10/12 coefficients differing). Optimizer traces differed at floating-point roundoff only, while all 432 executed outcomes and all 36 trajectory records matched. This supports behavioral/result reproducibility in this synthetic task, **not cross-environment bitwise tensor reproducibility**. Do not overwrite or silently relabel original checkpoints.

## Verification and custody

Run from the authorized laboratory with the supplied ZIP alongside it:

    python C65_archive_qualification.py DRIV3R_C65_paired_delayed_credit_2026-11-18.zip
    python C65_archive_qualification.py DRIV3R_C65_paired_delayed_credit_2026-11-18.zip /path/to/independently-executed-C65-output

The independent verifier checks each original manifest and checkpoint hash, the complete guided-goal aggregates, protected and unguided invariants, 36 replay hashes, and (when supplied) exact equality of independently rerun executed outcomes. It requires only Python's standard library; running the full research source separately needs PyTorch, NumPy, safetensors and the frozen predecessor suite.

**Publication gap:** The original C65 executable source, full compressed episode ledger, gallery, JSON summary and checkpoint artifacts remain in the supplied owner archive; they are **not yet committed to this PR or stored as immutable GitHub Actions artifacts**. This report and verifier do not substitute for that custody. Do not claim full independent source qualification against DRIV3R Git, native CyberSand validation, or model promotion.

**Scientific disposition:** HOLD. Equal action exposure did not recover delayed-credit goal advantage. Subsequent work should tackle directed return-to-goal rather than optimize novelty alone, and use a larger independent topology cohort with explicit meandering/branching/return style evaluation.
