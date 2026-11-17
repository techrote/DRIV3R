# P21 — original pretrained LFM350 128-turn matched creative-niche audit

17 November 2026. DRIV3R preference_eval lane, existing Draft/HOLD PR #1. Historical executed root-LFM output, **not newly run model inference, model-weight training, native CyberSand, owner ratings or a source-disjoint aesthetic preference trial**.

## Actual source evidence

Independently downloaded three original historical CyberSand Actions ZIP artifacts read-only: GP350-012 stock #14248333879 (ZIP SHA-256 `0eed5828082862d74c9dd796b7a45a6110f2eb02326a6a7d5f508e91501072cd`), GP006 #14244806761 (`88dfeee5360ae16d03de99d6bdfb2dbe79a819613af215373f6dcfb9fdcf0ae6`) and GP010 step24 #14245172930 (`fa6296ad600176b366b367965b86118fc432a92a63f55c8e0b5a44c488258920`). Original runs: stock #43910803355; GP006 and GP010 #43902984081. The pretrained root is LiquidAI/LFM2.5-350M revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`, with adapter digests pinned in `P21_EVIDENCE.json`. The embedded model manifest is a producer-reported model-file digest, **not a fresh independent rehash of the weights**.

Audited **36 genuine model-applied 128-decision episodes (4,608 host-legal decisions)**: 3 model stages × 12 matched synthetic W0 worlds (2 seeds × 2 grid sizes × 3 material families). Original ZIP digests and CRC pass. Each episode's decisions, menus, before/after positions, event and trajectory alignment, source PNG, summary PNG digest and finite model logits were checked. All 4,608 recorded applied choices are legal in their recorded menus. This is synthetic W0 host legality, not current CyberSand native ABI compliance.

## Outcome: high coverage is not creative movement

| Historical real LFM350 stage | Mean unique cells | Mean 90° turn rate | Longest straight, mean | Branch actions, total | Mean graph cycles | STOP actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untuned stock | 124.67 | 4.13% | 36.17 | 0 | 0.33 | 0 |
| GP006 step96 adapter | **128.50** | 4.92% | 36.08 | 0 | 0.50 | 0 |
| GP010 step24 adapter | 124.33 | **8.86%** | 34.92 | **9** | 1.17 | 0 |

Under matched **greedy** decoding, GP010 step24 differs geometrically from GP006 in **4/12** worlds; 8/12 are visually identical full-horizon paths. Stock differs from each trained stage in all 12. All three avoid 2-step reversals but mostly execute long straight segments. GP010 records branch actions only in three Greebles worlds and sometimes loses unique coverage. A geometric junction or cycle proxy can arise without a BRANCH action; do not count one as proof of intentional artistic branching or curl quality.

There are **36 A/B pairs, of which 28 have different full trajectories**, and 8 exact visual ties suitable for blinded attention checks. Unlike P19's STOP-prefix confound, all these pairs have the same 128-action horizon and none chose STOP. These are **not** 28 independent experimental units: all pairs share only 12 source worlds and just two underlying random seeds. Matched OOD worlds were selected by the original producer, not newly reserved by this lab. Do not use these highly correlated specimens as independent held-out taste-training labels.

## Four artistic niches remain distinct

- **MEANDERER:** none of the three greedy cohorts establishes irregular traveling zigzag artistry; straight-run and turn-rate profiles are conservative.
- **CURLER:** graph cycle rank alone cannot establish hooks or partial spirals; original SVG inspection and owner assessment needed.
- **BRANCHER:** GP010 shows sparse Greebles-only BRANCH choices, but not broadly generalised branching.
- **CHAOTIC ARTIST:** no two-cell oscillation or early STOP in this sample, but neither are surprising paths broadly demonstrated; nondegeneracy alone is insufficient.

Anonymised deterministic A4 landscapes show the **original applied polyline data**, with exact source mapping retained separately. Blank owner four-niche comparisons contain **zero** inferred human ratings. No aesthetic ranker was trained or promoted. In particular, this is new source verification of earlier actual pretrained trajectories, **not** new pretrained forward passes or an experiment that separates effects of adapter training from matched decoding changes.

## Next research gate

Obtain GP350-018 real *changing-observation* greedy/sampled **matched-decoding** rollouts and source-verified Tiny/Population/Franken source outputs, with predeclared seed/world-family heldouts, the same observation/action host, and blind owner judgments. Explicitly compare reachability, recovery, long straight segments, loops and stopping. A learned ranker only merits training when independent, appropriately blinded labels exist; train on source/world blocks and evaluate on disjoint model families. Feed the GP010 branching/coverage tradeoff and GP006 conservative coverage control to other lanes in `LATEST_HANDOFF.json`, without editing their directories.

Local source tests: 4/4 PASS; ZIP and PNG source hashes checked. Complete original ZIPs, source-derived A4 SVGs, episode descriptor catalogue, blank ballots, individual evidence hashes and executable generator preserved in P21 downloadable local archive. No new Actions artifact is claimed. No paid compute, external repo writes, game changes, new PR or merge. Draft/HOLD maintained.
