# DRIV3R EXPRESSION-RECOVERY-001 — independently audited 40-decision Q8/Q4 screen

**Date:** 2026-11-18. **Exact model-execution source:** `431e5bbde82abb07a27562a67473fed7e3766127`. **Run:** [Actions #44380220822](https://github.com/techrote/DRIV3R/actions/runs/44380220822) (successful, 2026-11-18 20:08 UTC). **Original research ZIP:** [artifact 14465663492](https://github.com/techrote/DRIV3R/actions/runs/44380220822/artifacts/14465663492), `driv3r-expression-q8-q4-44380220822`. Retain independently of branch HEAD.

## Independent artifact verification

- Downloaded the actual completed-run ZIP, checked all **559 of 559 included files** against `SHA256SUMS.txt`; **0 mismatches / missing / unmanifested**. Inspected all 60 original per-case JSON, PNG signature, original W0 trajectory, unsmoothed SVG, case summary and corresponding original full HTTP model responses.
- Verified exact source SHA, pinned llama.cpp commit `8642541b255bfb5dc39cd351a178be09013a55d6`, patch receipt flags (stock completion graph unchanged, decision graph only, weights unchanged), per-run patched/binary provenance and both vendor download receipts. Official Q8 `073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c`; official Q4 `adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b`. Weight-file bytes are NOT present in the ZIP; acquisition and full-model hash verification were run in CI and attested by the original receipts, not independently re-downloaded here.
- **60/60 cases executed, 1,331 raw System One HTTP-200 decisions audited, 0 raw-probability/model-argmax/host-application mismatches**. Independently recomputed each recorded stochastic draw from seed/mode/tick, temperature 1.5 and 25% legal uniform mixture: **263 sampled decisions; 203 host-applied choices differed from model argmax**. This is explicitly host-side stochastic exploration, NOT model-origin greedy action. All applied actions were legal and actual W0 actions.
- **30/30 Q8-Q4 first-turn comparisons** matched identical model-visible state, observation and question. **30/30 same-model, same-world greedy-versus-sampled initial-call probability vectors** matched exactly. At first turn, Q8 vs Q4 argmax differed for 11/15 world/mask pairs (22/30 including decoder duplicates). On-policy later paths are only paired in *initial conditions* because a divergent action changes later observation and legal menus.
- A4 blind sheets `trajectories/blind-a4-01.svg` ... `-04.svg` contain 60 raw-path cards with ID-only labels; `trajectories/blind-identity-reveal.json` and `trajectories/blank-owner-ballot.json` are separate. Human aesthetic ratings do **not** exist. This audit visually inspected all four rendered sheets. Source ZIP also contains each real original W0 PNG and raw SVG.

## Controlled outcomes: *greedy* model-origin actions, 3 W0 world/seed groups per quantisation

| Mask | Q8 result | Q4 result | Interpretation |
| --- | --- | --- | --- |
| stock 24 | 3/3 full 40, 7–14 new cells, 2/3 period-8 cycles | **3/3 STOP at first decision**, 0 new | Q8 clearly has different first-action ranking; Q8 still poor spatial variety |
| GREYHOUND 13 | 3/3 full 40, **33 new cells each**, no detected 2–16 cycle | **3/3 exact period-4 square loops**, only 3 new cells | Huge Q8/Q4 difference on same initial prompts, but still a tiny 6×8-footprint motif |
| BRANCHWITCH 20 | 3/3 full 40, **40 new cells each** | 3/3 full 40, **36–40 new**, no detected cycle | Strongest local cell-coverage proxy, yet visually very similar symmetric repeat motifs; Q8 produced the *same complete 40-step trajectory* in all three different world groups |
| AERO4_NO2 4 | 3/3 full 40, **30–33 new** | 3/3 full 40, **31–33 new**, no detected cycle | Most reliable quantisation-robust exploratory candidate, but spatial area only 48–49 bounding-box cells |
| GREY3_CUT 3 | 3/3 full 40, **2/3 period-4 loops** | 2/3 full 40, 1/3 early STOP, 1/3 period-2 loop | Unstable severe ablation; one Q4 architecture example genuinely reached 34 new cells and area 72 |

**Stochastic decoder negative:** all **30/30** Q8+Q4 sampled arms STOPped before 40. Raising sampling entropy and uniformly mixing legal STOP did not recover continuing long worm art in this design. Sampled paths are preserved as negative evidence, not art successes.

**Non-independent-looking output despite distinct worlds:** obstacle hashes for the three worlds are distinct, yet initial five-by-five local observations are identical, and some greedy mask trajectories remained identical across all 3 groups before ever reaching different surrounding content. Hence counts of 3/3 reflect execution reproducibility, **not three independent creative styles**. Repeated rose-like geometries are not automatically good Meanderer/Curler/Brancher/Chaotic-artist work. Cycle detection 2–16 found 8 cases, including 5 period-four square cases; 34 of 60 total cases STOPped early (30 stochastic, plus 3 Q4 stock and 1 Q4 GREY3). No native CyberSand PAINT assertions.

## Decision: one bounded long-horizon experiment justified, no winner declared

The screen shows credible, model-origin, nonperiodic 40-step continuation for **AERO4_NO2 in both vendor quants on all three W0 worlds**, with high cell novelty but very small spatial extent. GREY3_CUT has both extreme negative controls and one divergent Q4 exploratory case. Under the pre-registered gate, continue **only AERO4_NO2 and GREY3_CUT**, both quantisations, both decoders, four predeclared W0 worlds including unseen 9601/Greebles, 96-decision limit. Exact first screen artifact ID `14465663492` is attached to the new request. This is an extension/robustness test, **not a confirmation of recovered artistic quality**. Do not rerun the 40-step screen. Do not introduce force-turn heuristics or modify weights. Do not trigger expensive work on routine source pushes.

**Native CyberSand qualification: NONE. Owner Meanderer/Curler/Brancher/Chaotic-Artist preference: NOT RATED. PR: Draft/HOLD.**
