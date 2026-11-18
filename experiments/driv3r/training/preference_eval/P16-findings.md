# P16 — Fresh-seed protected-topology transfer and frozen preference-ranker falsification

**16 November 2026. Research-only. DRIV3R Draft/HOLD PR #1. Locally executed and independently source-reproduced; not Actions-qualified.**

## Experimental question

P15 showed that an exit-aware heuristic could modestly increase construction, but it reused P14-held-out seeds. P16 tests the same frozen **142-feature W16 auxiliary PyTorch head**, fixed three hypothetical protected-position geometries, and original host-legal W0 action contract on **12 fresh seeds** absent from the P11, P12–P15 cohorts: 2903, 2917, 2927, 2939, 2953, 2969, 2971, 2987, 2999, 3011, 3023, 3037. Policies were predeclared before running: original late-STOP W16, position veto, P15 one-step exit-aware veto, and an exploratory **two-ring local geometry veto**. The two-ring policy uses an orientation-agnostic geometric occupancy proxy, *not* predicted native future legal moves.

Across 3 families × 12 seeds × 3 fixed masks × 4 policies, **432 real closed-loop model-head trajectories, 54,064 frozen-head forwards, 108 matched policy quadruplets** and source-hashed W0 PNGs were produced. There were **243 distinct trajectory hashes**. All 432 were host-legal; no filtered policy entered the hypothetical protected-position mask. Six archived P11 control drawings were independently reproduced byte-exactly. There are **zero** archived original PNG comparisons for the **new** seeds; fresh-world drawings have no prior archival counterpart.

## Held-out results

| Policy | Mask contacts | Novel positions (mean) | Goals >=48 novel /108 | Branches (mean) |
|---|---:|---:|---:|---:|
| Original late-STOP W16 | 694 | 106.17 | 105 | 18.97 |
| Position veto | **0** | 99.72 | 105 | 17.63 |
| Exit-aware veto | **0** | 100.53 | 103 | 17.57 |
| Two-ring veto | **0** | **102.69** | **105** | **18.12** |

The predeclared two-ring policy retained **2.97 more novel positions** on average than plain veto, with a whole-seed-block bootstrap 95% percentile interval of **[+0.61, +5.69]** in an independent post-hoc audit (12 resampling blocks, each containing all family/layout conditions). The one-step exit-aware policy gained **0.81** positions with an interval spanning zero; it also lost two completions against plain veto. This is a bounded synthetic-W0 positive for geometry awareness, **not** a production-safe algorithm or human-aesthetic preference improvement. A reduction in retrospectively counted stagnation-to-recovery events does not independently prove improved recovery: it could mean fewer stagnations or more missed recoveries.

The identical outcomes and **all 432 trajectory SHA-256 plus 432 PNG SHA-256 values** were independently reproduced by separately authored P16 replay source. The primary source committed here passed nine local unit/integration tests against the immutable P11 producer. No new GitHub Actions qualification or paid GPU run was performed.

## Frozen ranker transfer: negative result

The previously trained P14 seven-feature linear preference proxy was **not retrained** on P16. On 520 new non-tied comparisons labelled exclusively by P14's deterministic proxy priorities, the frozen ranker agreed **464/520 (89.23%)**, whereas the explicit deterministic baseline agreed **510/520 (98.08%)**. These are **synthetic priority agreements, not owner/human ratings**. Further reward/preference optimisation remains unjustified.

## Provenance, receipt and limitations

- Original P11 immutable producer SHA-256: `9a9f61038bb1a4f68395816a8296fb8957a8a8f166610921c2a9db6502308d9c`.
- Frozen W16 auxiliary model checkpoint SHA-256: `f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9`.
- P14 frozen ranker SHA-256: `fece7f68288df0c1fc5777b10c8c68694206842f48d49fae0533e84bb1db91f9`.
- Executed primary source Git blob: `1c3d32bbad07ff0f4eaf407199b64a14515cbe1a`; nine-test Git blob: `076de445319681ec5aacb1c230bb16d65d36c6ee`.
- Source-matched local archive SHA-256: `3b893b8df8e35971c95f0a10605ae70ea40a7fd8509c1d768bb06a712e1ce82c`. Includes original host-generated PNGs, action receipts, independent P16 source, pinned P11 evidence, provenance and bestiary. This is a *local* archive, not an immutable hosted Actions artifact.
- Primary aggregate P16-results.json SHA-256: `a529e73b5d4191b07f56fdb441b7c7e0c3250ba023de056cce4c776c080b7bff`; matched pairs SHA-256: `b98b9b0f13fb0442321b50233f3eff0219e6f253ea3bef08b9c7323e0f2d7abd`.

**Scientific boundary:** W0 is an external synthetic host, the frozen decision head is not the root Sprocket language model, and these position masks are **not** canonical CyberSand protected material write sets. Zero mask contacts cannot establish native paint safety. Three P15 mask topologies were reused; only the seeds were new. Family/mask observations within a seed are correlated. No owner/human preference labels, no native paint receipt, no game-side changes, no model optimiser updates, no merges. **DRIV3R PR #1 stays Draft/HOLD.**

**Next substantive gate:** canonical CyberSand read-only paint-action write-set traces, tested against a held-out protected-cell verifier, before contemplating any production safety claim; separately acquire blind owner comparisons before creative preference optimisation.
