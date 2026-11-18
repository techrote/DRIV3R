# C66 — Fresh-graph transfer audit of the frozen C65 delayed-credit heads

Date: 18 November 2026. **DRAFT/HOLD — negative; do not promote.**

## Research question and custody

Following C65's zero learning advantage over a shifted-credit control on 12 test graphs, run an *additional*, disjoint graph cohort rather than tuning on the original failures. Freeze the original C65 aligned and shifted-credit checkpoints and their hashes; **no optimizer updates or checkpoint selection were performed in C66**. The same host-legal synthetic game harness from C60/C65, not the authoritative CyberSand native ABI, executed all episodes.

Original frozen C65 checkpoints (owner archive only, not committed to Git):
- ALIGNED SHA-256: `4d603774064e858fe7635e9286f4efd60513e7c425b74cc64ebbc613699fc469`
- SHIFT SHA-256: `2290edabeafdffe25fb0e60c206b22d675dd4f6c9c6a3a271a21b64153a145af`

The exact **C65 training source** and regression suite have now been restored at their native laboratory paths on this PR branch. The C65 source SHA-256 is `0c08683bbe7033ca6ab7920643cd14be2cf0eb634bd2ef10cadc34f6a21d09c6`. Both C65 and C66 source files were verified against the locally executed Git blob identities.

## Executed protocol

Seed `660118`. Excluded 100 previously identified C60–C65 graph identities and generated disjoint new graph candidates. Selected the first four eligible test states from each of **braid, prim, prim_braid, tree, Wilson and Kruskal**, before inspecting outcomes: 24 distinct remote-goal graph identities. Tested weak (0.35) and strong (1.0) backtrack guidance, at 0% and 50% remembered-menu loss. Tested the no-guidance arm with intact memory only, because earlier controlled tests demonstrated the no-guidance parity invariant. Each row independently executed six policies.

Executed: **120 matched condition rows, 720 real synthetic-host suffixes**, including 24 no-guidance parity rows and 96 guided rows; **36 source-matched, independently re-executed trajectory records**, 30 of which appear in the SVG gallery. Six C66 local evidence checks passed. No host-action rejections, protected-cell modifications or no-guidance movement/painting divergence.

| Policy | Goals / 48 guided, intact | Goals / 48 guided, 50% erased | New sites, intact |
|---|---:|---:|---:|
| C18-A0, ungated base | 18 | 18 | 348 |
| C34-RESET | 23 | 23 | 366 |
| Non-neural FRONTIER | 26 | 26 | 537 |
| Frozen C60 neural | 27 | 27 | 584 |
| **C65-ALIGNED** | **27** | **27** | **609** |
| C65-SHIFT, wrong-credit | **29** | **29** | **620** |

**C65-ALIGNED gained no goals over frozen C60.** Its wrong-credit control completed two *more* goals on each memory arm. Per-graph descriptive bootstrap interval for ALIGNED minus SHIFT: **−5 to 0 goals**, on both memory arms (10,000 graph resamples each). The two additional SHIFT wins arose on two graphs; no graph favored ALIGNED. This interval is exploratory, not a registered confidence claim.

New site coverage increased from C60 to ALIGNED, but goal completion did not. The non-neural frontier and frozen C60 references remain important controls. The *tree* family remained unsolved by all controllers in these selected episodes; Wilson remained particularly difficult. The extra two SHIFT successes must **not** be interpreted as scientifically established superiority of corrupted credit, because the small graph cohort and prior model-selection history do not support that conclusion. They do refute a claim of clear aligned-credit advantage.

## Qualification and limitations

The test runner `c66_holdout.py` and `test_c66_holdout.py` are committed in this laboratory. Their verified local SHA-256 values are `b2c80227596feca929e947b8f4a654c5c8bc8d48a89132640d3c54f5abfd4932` and `d4827a8ec83e191ce3ae025383acf862298d4638c25209f4dd3b7e1f0e75002b`, respectively. The original C65 source is archive-layout-oriented and requires its frozen C60–C64 prerequisites and owner checkpoint archive for rerunning C66; merely importing this Git PR is not sufficient to execute it. No production game files, canonical ABI or historical d_common fixture were modified.

Full executed C66 ledger, complete summary JSON, trajectory gallery, and log were produced locally in the owner-evaluation archive but **are not yet stored in this Git PR or immutable Actions artifacts**. Their SHA-256 hashes:
- C66 summary JSON: `9da6e83551c9bd2f8472576a6c391bd13effb862eea6a50bb53c119b029f6df0`
- C66 complete compressed ledger: `fbaee7a28a2dd6f8e51ed41acf665f40f4b4efba1386c2cbd42d5093b80802d6`
- C66 executed SVG gallery: `5b96e66b1f9c946fca1948ae33e87ba76abc6a1e5fa2c34d9099b465de265188`
- C66 owner archive ZIP: `91ed14282ea786e04d2c805b96d5dcad15721d3d91bece049aacc5b2a78bb81b`

Those missing remote evidence transfers are **publication blockers**, not completed custody. The checked-in report conveys results and limitations, not an independently verified Git-hosted ledger. No standalone native CyberSand validation or human worm-personality assessment was performed. No paid compute was consumed.

## Disposition

**HOLD / no promotion.** Two independent cohorts now fail to demonstrate aligned delayed-credit benefit over the shifted-credit control. The next productive step is to close complete ledger/checkpoint custody and test genuinely goal-reaching corrections under larger topology-balanced holdouts rather than optimizing only exploratory site coverage.
