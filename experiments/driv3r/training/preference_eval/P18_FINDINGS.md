# P18 — original-producer source re-verification and four-niche blind comparison pack

**17 November 2026 · techrote/DRIV3R · existing Draft/HOLD PR #1 only.** Research for prospective CyberSand game behaviour; W0 is an external synthetic host, not native CyberSand acceptance. Historical CyberSand sources and Actions artifacts were accessed read-only.

## Source recovery and original-model evidence

P17 treated 14 historical paths as *secondary curated* because the original producer JSON bytes were unavailable. P18 independently downloaded the 12 original Actions artifact ZIPs, verified all 14 original JSON entries against their archived SHA-256, reconstructed recorded action/position/ink histories, and compared every reconstructed path to the P17 fixed SHA-256-pinned source manifest. **14/14 originals matched**: **1,096 applied decisions** (72 historical native, 512 W13 LFM, 448 d1/mutagen, 64 Frankenlayers); no observed applied host-menu violations. Native receipts additionally checked cursor, candidate IDs and continuous world hash chains; W0 checked tick, position and action/observation legality. These are source-integrity results, not new native physics, paint-safety or current host-ABI acceptance.

W13 LFM four original 128-step paths contain 73 BRANCH and 77 RETRACE actions. Four Franken 16-step originals contain 42 BRANCH and 7 RETRACE. Original stock d1's 128 LEFT decisions show why turn-rate rewards encourage degenerate 4-cell loops. Historical W13 free/native style is an aesthetic reference, not evidence of an effect from current LFM350 model training.

## Real trained pretrained-LFM350 adapter: a negative visual result

Independently retrieved historical immutable GP350-006 Actions artifact **14199280586**, original ZIP SHA-256 **1db95314355fb4ea0439b63825a0f829b14bb0a0f8b01284fd614446aca85a47**, with the actual rank-8 LiquidAI/LFM2.5-350M step-96 LoRA adapter. Adapter SHA-256: **436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c**.

This source contains **nine real matched pretrained-root inference pairs**, stock versus step-96 LoRA, from three seeds (9503, 9521, 9539) across architecture, greebles and pipework, 24 decisions per arm. **9/9 applied action sequences and 9/9 original trajectories were identical**, although **216/216 paired ticks had different six-action logit vectors**. The 432 archived model decisions were host-legal under their source W0 menus. The adapter changed logits but did **not** change the greedy-decoded drawing in these specific 24-step controls. Do not extrapolate to longer horizons or other decoders; GP350-018 sampled changing-observation execution remains unperformed. No new inference or optimizer update was run by this lab.

## Blind original-path four-niche trial pack

The 14 reverified historical specimens produce **10 nonidentical original A/B trajectory pairs across only three same-world/horizon blocks**. Three further identical-image source-original controls from trained-versus-stock LFM350 are separately declared as **attention checks**, not independent aesthetic comparisons. Output: **13 anonymous pairs, seven A4 SVG pages and 52 blank questionnaire slots**, with each pair assessed separately for MEANDERER, CURLER, BRANCHER, CHAOTIC_ARTIST. An initial catch-trial identifier risk was detected and repaired before publication: the blind IDs now have the same anonymous prefix for all pairs, so catch-trial status is not displayed. Blind sheets conceal model identity, catch status and scores; source reveal mapping is separated. The SVGs display the original applied paths with start/end, pen-lift and a common scale *within* each A/B pair.

There are **zero owner/human preference ratings** and **zero model-derived aesthetic ratings**. A ranker was **not trained**: 10 visually distinct pairs concentrated in three source-world blocks and nine identical trained/stock paths cannot supply independent world/model-family-disjoint held-out preference supervision. Neither 52 questionnaire blanks nor three visual-tie controls are labels. No scalar creative optimum is claimed.

## Reproduction, evidence and limitations

- Full exact local Python auditor, p18_reverify_ballots.py, Git blob **2cff3ff9a8311bd849ba29ae75be7ca8e842396a**; exact adversarial tests test_p18_reverify_ballots.py, Git blob **be79d397bf813336bedaebaff1b76d70449ffed2**. **7/7 local tests PASS** (original-byte tamper rejection, host legality and historical model-logit/path negative).
- P17 original curated manifest SHA-256: **873d208e370bec2d26586af1c86545957deef4b01fc5a9521f3630fd1920a88d**. Producer JSON IDs and SHA-256 preserved in P18_EVIDENCE.json; original bytes and 78,088-byte curated manifest are in local bundle.
- Local output audit SHA-256 **63e02c42195fb70b27154db34940d86c3ba98fd93100a368633dd06594a3d8b4**, ballot CSV **8513a9cb82edffa967f72d9af39b411070cfbd7c56c1cf6f15fcd6a74bc70620**, separate reveal JSON **a65b8fa8068f3218fb42604057cb9aa45fc28fb638c35a54aa2ea04d9f2e9cd1**.
- Original-input local bundle: 8,179,398 bytes; SHA-256 **b6651d1747b7bbe18469d98a08c7147f8ccc61ea54506cce5488918fb6ac13de**; contains all original producer ZIPs, the manifest, exact tested source, tests, 7 original A4 SVGs, blank ballot, reveal mapping and checksums.
- Reproduce by obtaining the 12 historical CyberSand read-only Actions artifacts plus GP350-006 artifact **14199280586**, the historical manifest, and running the committed source with arguments --manifest, --archives, --lfm-archive, --output. Original-input tests assume their archives and manifest are available; do not enable automatic push/PR training or claim that source-smoke CI reproduces this research.
- **Qualification:** local audit + locally passed tests; **not** a new immutable DRIV3R Actions artifact. Original ZIPs, weights, galleries and manifest are not embedded in the Git commit. Original-source provenance does not prove native CyberSand material/ABI equivalence.

## Next cross-lane consumer gate

Obtain genuine changing-observation GP350-018 sampled versus greedy pretrained LFM350 rollouts, Tiny pretrained/Population actual SLM output and new full execution Franken structural mutants. Hold world family and seeds source-disjoint; preserve six-action/host-menu receipts. Obtain owner blind A/B/tie/skip judgments across the four separate niches, reveal and audit identical-image attention checks after collection. Then evaluate transparent baselines versus bounded trainable rankers on held-out model/world families. Provide hard negatives (short-loop stock d1, early STOP, 2-cell oscillations, and visually identical post-LoRA paths) to Population and Distillation via this **lane-local** handoff only. Do not edit their directories.

**Safety:** DRIV3R PR #1 only, Draft/HOLD, unmerged; no paid compute, new PR/workflow, external repository write or CyberSand game change.
