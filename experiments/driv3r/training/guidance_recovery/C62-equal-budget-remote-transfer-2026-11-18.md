# DRIV3R Guidance & Recovery — C62 equal-budget remote-goal transfer

**Status:** Draft/HOLD research evidence; do **not** promote, merge, or treat as native CyberSand acceptance. 18 November 2026. **Owner:** `experiments/driv3r/training/guidance_recovery/**` in techrote/DRIV3R only.

## Research question

C61's Wilson-trained controller and scrambled-return control had unequal update budgets (12 vs 9). Would equally budgeted real on-policy CPU policy-gradient training achieve robust recovery on independently generated, never-trained Kruskal and Wilson maze geometries, without erasing worm personality?

## Frozen protocol and custody

- Synthetic game-like host with current **host-legal** action menu, position/goal, executed action history, observed frontiers and a weak (0.35) or strong (1.0) BACKTRACK cue. No access to hidden maze connectivity, protected-cell map, material oracle, shortest-path planner or future trajectories at inference.
- Fresh C62 seed `620118` and fixed world-construction retry algorithm. All C60 historical exclusions and all C61 training/testing graph identities excluded **before** outcomes. Frozen import sources and checkpoints from the verified C61 owner research archive; see source hashes inside `C62-summary.json`.
- Wilson + familiar-family *training*; independent Kruskal, Wilson and familiar *test*. 20 Wilson and 32 familiar training graph identities generated; 16 OOD and 8 familiar test graph identities generated. Host-prefix eligibility left **34 training** and **16 test** remote goals (5 Kruskal, 6 Wilson, 5 familiar). The important small effective samples are reported without pretending all generated graphs contributed.
- Identical learning architecture, initial weights, training prefix identities, two guidance strengths, AdamW steps (40 per policy), batch size (256) and update epochs (5 × 8) for C62-ALIGNED and C62-SHUFFLED. The latter deliberately shuffles delayed credit within each epoch. Executed on-policy trajectories differ after parameters diverge; action counts cannot be made identical without ceasing to be on-policy.
- Both new policies are a bounded stochastic action-logit bias (maximum `1.8 × cue_strength`) on top of the existing legal host menu. No hard override or direct planner access. Frozen C18-A0, C34-RESET, C60-FRONTIER and C60-ALIGNED are independent controls.
- 76-tick overall horizon, one host-prefix tick (9), two damage arms (0%, 50%), three cue arms (none, weak, strong), six controllers. **96 fully matched conditions and 576 independently executed suffixes**, including **32 no-guidance conditions**. Strong and weak guidance outcomes are pooled only within each predeclared family and damage stratum. Exact full episode records, graph hashes and negative results retained in the compressed ledger.

## Primary outcome: independent held-out goal completion

Values are completed remote goals **out of eligible guided episodes**, with weak+strong guidance pooled; duplicated 50% damage findings must not be counted as independent evidence. Rows below show **0% erased**; results at 50% erased were identical for goals in this sample.

| Controller | Kruskal (10) | Wilson (12) | Familiar (10) |
|---|---:|---:|---:|
| C18-A0 | 2 | 0 | 0 |
| C34-RESET | 0 | 0 | 0 |
| Non-neural C60-FRONTIER | **4** | **1** | 1 |
| Frozen C60-ALIGNED | 1 | 0 | 1 |
| **C62-ALIGNED** | **1** | **0** | **3** |
| C62-SHUFFLED | 1 | 0 | 0 |

C62-ALIGNED has **zero discernible benefit on the unseen topology families** versus its shuffled-credit control: 1/10 Kruskal, 0/12 Wilson. It remains far behind non-neural FRONTIER (4/10 Kruskal and 1/12 Wilson). This is **negative transfer evidence**. Familiar-family completion improves to 3/10 versus RESET's 0, frozen C60's 1 and SHUFFLED's 0, but only **five familiar graph identities** are eligible. Its familiar graph-cluster descriptive 95% interval for goals over RESET is 0 to +7; it is not an inference of reliable generalization. In the eligible Kruskal cohort FRONTIER completed four goals while C62-ALIGNED completed one.

## Personality and legality

- An additional full-executed-segment analysis records turn frequency, reversals, direction entropy, unique-visited-position ratio, ink fraction, and executed-turn count for each controller and condition. Mean paired full-path differences against RESET, by graph family, strength and damage, are included in the summary/ledger.
- Relative to RESET on intact Kruskal states, C62-ALIGNED had higher direction entropy (+0.218), fewer reversals (−0.252), fewer turns (−0.145) and higher unique-position ratio (+0.115), based on 10 matched guided comparisons. It also executed 13.8 more steps on average. **Unequal stop/horizon duration confounds whole-trajectory proxy comparisons**; these are descriptive and do not certify Meanderer/Curler/Brancher artistic personality preservation.
- All 576 suffixes: **zero synthetic-host rejections** and no protected-cell changes relative to their saved host prefixes. All 32 no-guidance matched conditions retained identical action and ink hashes across all six controllers.
- All 18 fixed-selection gallery traces were cold re-executed using the saved source, frozen host prerequisites, identical graph seeds, checkpoint weights and guidance/memory arms. Path, action hash and ink hash matched exactly. Both heads independently refit to **byte-identical safetensors SHA-256**:
  - ALIGNED: `17ce24e6014d1cfb95b1c641e25fe576647085c40f91ad110d4a0ec616fde47d`
  - SHUFFLED: `0940656683babdc9d894842e968c1b34e6fead392b57f011a8214ca6200d23e0`
- `pytest` laboratory integrity tests: **6/6 passed**, including graph exclusion, complete action/ink parity, immutable source identities, actual replay-gallery hashes and hidden-graph access review.

## Methodological limitations

1. Only 5 Kruskal / 6 Wilson / 5 familiar test graph identities produced eligible remote-goal prefixes. The experiment has low power to detect modest cross-family differences; no confirmation-level statistical claim is appropriate.
2. 50% menu erasure did not change goal completions in the tested cohort. This is likely because policy decisions depend mainly on current legal menu and visit history; independent evidence is not doubled by the two damage arms.
3. C62-ALIGNED and C62-SHUFFLED received exactly 40 optimizer updates but had different post-divergence on-policy rollouts, an unavoidable distinction. Their training episodes/initial conditions, not their action-stream lengths, were matched.
4. Only the historical synthetic `d_common` fixture is exercised. Native CyberSand legal ABI, material-specific hazards, protected-cell paint refusal in native, actual physics, Workbench and physical movement are not qualified.
5. No full guided artistic-personality acceptance; measured movement proxies cannot substitute for rich long-horizon visual inspection of Meanderer, Curler, Brancher and Chaotic Artist wormpaths.
6. The user-owned archive includes frozen prerequisites and tiny checkpoints; binaries **must not enter Git**. Immutable GitHub Actions artifact custody still needs bounded explicit, opt-in execution if it becomes available. No automatic training on routine push, no paid API/GPU used.
7. This source uses the owner-evaluation extracted layout (`lab/`, `frozen/research-source/`, `frozen/predecessor_C42/`, `publication/`, `results/`, `results62/`); the committed source is readable but is not a self-contained Git-only training environment. Reproduce from the exact archive, not by assuming repository checkout contains model checkpoint dependencies.

## Reproduction and custody

Extract the C62 owner archive preserving paths, enter its root, run `python lab/c62_balanced_remote_transfer.py results62`; the script uses source-hash-bound resume files for interrupted local simulations and refuses stale training caches. To recompute from scratch, use a *fresh* results directory (do not reuse checkpoints). Verify with `python -m pytest -q lab/test_c62_balanced_remote_transfer.py` and independently re-execute/refit with `python lab/c62_independent_replay.py`. The committed `C62-extract-ledger.py` expands the losslessly compressed records. Gallery SVG is generated solely from the 18 source-matched executed traces; no invented maze artwork or concealed graph used.

## Disposition

**HOLD / reject C62-ALIGNED for promotion.** Equalizing optimizer budgets corrects C61's confound but does not solve unseen-family recovery. Prioritize a separately frozen, larger eligible Kruskal/Wilson evaluation with an explicit sample-size floor, improved causal recovery outcome coverage and matched personality baselines. The non-neural FRONTIER controller remains a stronger test baseline. No CyberSand, Neurounner or DRIV3R main writes are authorized.
