# FRANKENLAYERS — complete official-Q4 GREYHOUND deletion trilogy (18 November 2026)

**Disposition:** all three owner-approved bounded CPU tiers are concluded and their original Actions ZIPs independently audited. No fourth tier or retry queued. **Real compiled-model inference in a synthetic W0 host; not native CyberSand.** This is a frozen scientific record, not a merge/Ready recommendation.

## Exact evidence custody

| Stage | Original run | Original artifact ID | ZIP SHA-256 | Manifest | Independent raw-to-host legal |
|---|---|---:|---|---:|---:|
| Upper | [44202739826](https://github.com/techrote/DRIV3R/actions/runs/44202739826) | 14381663329 | `7091428bbedb92e486e8b41925277660a894feb6c19387fe3641f201c787ecea` | 84/84 | 60/60 |
| Middle | [44253735083](https://github.com/techrote/DRIV3R/actions/runs/44253735083) | 14405820902 | `ceefd282f64e14bf05fc9b7da4188fdd5269009473058b77872222422ab25688` | 84/84 | 146/146 |
| Extreme | [44307129291](https://github.com/techrote/DRIV3R/actions/runs/44307129291) | 14429156993 | `ad481236a3d47aface28485771db0b2ab8ef65eff366db40a8b40db045672fd5` | 70/70 | 98/98 |

**Total 238/238 original manifest entries and 304/304 HTTP-200 raw model choices independently verified as legal and identical to the action applied by the host.** Each choice was checked against the contemporaneous legal menu and observation, including normalized six-way or five-way probabilities. 22/24 staged cases completed inference; two 2-block cases failed context initialization and are deliberately retained as negative evidence.

Original checkpoints and traced outputs are retained in the named immutable GitHub Actions artifacts. Each successful mode/fold includes `*-choices.json` (original HTTP responses/probabilities), `*.json` host events, `*.png`, unsmoothed `observed-path.svg`/`observed-path.json`, `result.json`, server/process records and memory samples. Failed cases include error and server log. All three ZIPs contain `SHA256SUMS.txt`, graph receipt and model/runtime provenance. The candidate lookup table is [Q4-TRILOGY-CATALOGUE-2026-11-18.json](Q4-TRILOGY-CATALOGUE-2026-11-18.json).

Frozen official Q4_K_M GGUF: `LiquidAI/d1-2B-GGUF` revision `4a181094d1b6addfaa3a0956f78e7ab2aa406a0c`, 1,099,629,888 bytes, SHA-256 `adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b`. Pinned llama.cpp commit `8642541b255bfb5dc39cd351a178be09013a55d6`, compiled server SHA-256 `251cf82f5e26f0db8107e24d95d9f0ac27e659a6a301c03306afa73c0521aa75`.

Stage-specific source commits: upper `e704fb8b4f450becf0b7613a26c70ad42b2f1f12`; middle `cd642b7d319c80526eb97e980e5401d9fb0c74e2`; extreme `fa98269f8d2db827eed4df984a23ecea9515bc1d`. The compiled graph selector omits execution of entire unselected blocks, **but retains all GGUF weights in memory**. Do not misdescribe execution skipping as physical removal of checkpoint tensors.

## Paired genuine-model results

Both worlds used seed 7703/pipework and 8807/greebles with identical starting states and a maximum 24 decisions; trajectories diverge legitimately under host-applied actions. Values separated by `/` are pipework / greebles; median latency values are seconds/decision.

| Tier | Candidate | Surviving blocks | Decisions | Newly visited cells | p50 latency | Empirical outcome |
|---|---|---:|---:|---:|---:|---|
| All | Unchanged Q4 | 24 | 1/1 | 0/0 | 2.335–2.545 | STOP tick 0; unusable stock game baseline |
| All | GREYHOUND | 13 | 24/24 | 3/3 | 1.395–1.511 | 24 LEFT decisions per fold, square-loop collapse |
| Upper | GREY10_CUT | 10 | 5/3 | 2/1 | 1.125/1.023 | BRANCH/RETRACE then STOP at ticks 4/2 |
| Upper | GREY8_CUT | 8 | 1/1 | 0/0 | 0.820/0.785 | STOP tick 0 |
| Middle | GREY6_CUT | 6 | 24/24 | 3/3 | 0.611/0.615 | RIGHT square-loop collapse, 21 revisit contacts/fold |
| Middle | **AERO4_NO2** | **4** | **24/24** | **21/16** | **0.397/0.390** | BRANCH 21/14, RETRACE 3/5, compact novel trails |
| Extreme | **GREY3_CUT** | **3** | **24/24** | **10/18** | **0.274/0.278** | Mixed LEFT/BRANCH/RETRACE, loops/oscillation risk |
| Extreme | GREY2_CUT | 2 | 0/0 | — | — | llama-server context startup failure: lacks full/sliding attention ballast |

### Extreme-tier full detail and negative evidence

**GREY3_CUT** executed source blocks `[0,4,10]`, skipping 21 blocks. In pipework it made 15 LEFT, four BRANCH and five RETRACE choices (10 new cells, nine revisit contacts, p50/p95 `0.274/0.281` s, post-ready sampled peak RSS `1,875,692` KiB). In greebles it made 12 LEFT, nine BRANCH and three RETRACE choices (18 new cells, three revisit contacts, five two-step position oscillations, p50/p95 `0.278/0.289` s, sampled RSS `1,876,040` KiB). All 48 decisions were valid HTTP responses and actually applied by the synthetic host. Prefill-rate medians were 771.9 / 769.7 input tok/s; **not generation output tok/s**.

**GREY2_CUT** executed proposed blocks `[0,10]`. Both startup attempts exited code 1 before the decision endpoint became available. Original server logs say `Experimental flow lacks full/sliding attention ballast`. This is an architectural prerequisite failure of the mask, not a usable zero-action personality and not a valid inference latency measurement. Retain both negative case folders and do not retry unchanged.

The stage-matched untouched Q4 control STOPped immediately in both extreme folds; GREYHOUND repeated LEFT for 24 steps in both. Their p50s were `2.413/2.407` s (stock) and `1.446/1.441` s (GREYHOUND). The approximately 8.7x model-decision speed difference from stock to GREY3 is not a demonstrated improvement in useful gameplay because the stock baseline terminates immediately and all episodes are short.

## Interpretation and handoff

**AERO4_NO2 Q4 is the best balanced verified survivor of these short official-Q4 comparisons; GREY3_CUT is the fastest continuing 3-block architectural probe.** Neither has independently validated goals, protected-cell discipline, native CyberSand acceptance, multiple held-out world families, or an approved expressive personality. Labels *Meanderer, Curler, Brancher, Chaotic Artist* are selection hypotheses only; no niche is certified by this run. GREY3's oscillations especially prevent an automatic Curler claim.

Peak RSS is sampled **after server readiness**, roughly 1.87 million KiB across cases, not lifetime peak and not evidence of memory reduction. The binary loads the original full Q4 model. Output-token generation is zero (System-One choice probabilities); recorded throughput is input prefill. The source host is `cybersand.worm.observation-action/0-experimental`, not the authoritative native CyberSand ABI. Source images and paths are unsmoothed originals in artifacts; any labelled contact sheet is a derived viewer, not replacement evidence.

**For Preference Lab:** examine AERO4_NO2 and GREY3 original per-fold paths blind, with loop penalties and held-out geometry variants, then label only after independent review. **For Extreme Quant / Distillation:** use their exact masks, model/runtime receipts, and source-matched original artifacts as starting points, not an unconditional success claim. No source edits or commits were made in those other laboratories.

The owner-approved upper→middle→extreme Q4 sequence is complete. **Stop this particular recurrence; do not re-download the same Q4 or create another tier on schedule.** Future structural model experiments require a separately justified owned-lane request and should include a legal non-STOP stock reference that actually continues under the chosen task protocol.

No external repositories, native game files, DRIV3R main, or merge/Ready state were altered.
