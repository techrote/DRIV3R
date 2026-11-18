# GREYHOUND Q4 upper tier — independently checked original evidence (18 November 2026)

Status: **REAL COMPILED Q4 INFERENCE COMPLETED; SCIENTIFICALLY NEGATIVE**. This report is a custody/metrics recheck of original immutable GitHub Actions artifacts, not a new inference or native CyberSand run. Branch at review: `ef95f721a1b5911db71450816fcf4a33237c093a`. Do not treat Actions success as gameplay success.

## Original execution and custody

- Original first attempt: [run 44202385937](https://github.com/techrote/DRIV3R/actions/runs/44202385937), source `cc65e2878aa61993c50df7e5cfca809699ef3e04`, **FAILED at pinned llama.cpp build** during optional UI asset provisioning. Failure artifact ID `14381134532`, archive SHA-256 `a134a3dc2a276d97170862239fcbea3113327e533274292ba34bff61aec446d4`; all **7/7** listed artifact payloads independently checked against its `SHA256SUMS.txt`. No inference on that attempt.
- One material repair retry: [run 44202739826](https://github.com/techrote/DRIV3R/actions/runs/44202739826), source `e704fb8b4f450becf0b7613a26c70ad42b2f1f12`, request nonce `repair1-20261117-q4-upper-no-webui`, **COMPLETED** with build, real HTTP decision calls, W0 trajectories and upload steps successful.
- Success artifact ID `14381663329`, artifact name `driv3r-q4-greyhound-44202739826`; original ZIP SHA-256 `7091428bbedb92e486e8b41925277660a894feb6c19387fe3641f201c787ecea`. Independently downloaded and checked **84/84** original manifest entries, **0 missing/corrupt/unlisted** payloads. Eight per-case raw response JSON files, W0 host traces, original PNG and unsmoothed observed-path SVG/JSON are included. Retention listed by GitHub through **15 February 2027**.
- Official donor Q4_K_M (not Q8-requantised): `LiquidAI/d1-2B-GGUF` revision `4a181094d1b6addfaa3a0956f78e7ab2aa406a0c`; byte count **1,099,629,888**; verified source receipt SHA-256 `adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b`.
- Patched llama.cpp pinned commit `8642541b255bfb5dc39cd351a178be09013a55d6`; compiled `llama-server` SHA-256 `251cf82f5e26f0db8107e24d95d9f0ac27e659a6a301c03306afa73c0521aa75`; patched `src/models/lfm3.cpp` SHA-256 `ba9d907f7b546958dd059353f2d3ca4534e6e3da63ff86a61881ec1eb380221a`. The retained patch changes the source-layer computation sequence, not tensors or post-hoc policy actions; all skipped blocks remain stored in the original GGUF. Output/head remain intact. CPU only, same compiled runner and GGUF across cases.
- Historical W0 host Git-blob SHA-1 `8979b6f0296e81051194bacc907992d38e006605`, SHA-256 `81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e`; **synthetic stdlib host, NOT native CyberSand**.

## Original paired live-game-like results

Each mode used two matched initial W0 seeds/worlds, 7703/pipework and 8807/greebles; 24 decision budget per episode. All 60 actually emitted raw decisions were HTTP 200 and matched the applied host-legal action (60/60), and the recorded normalized available-choice probability vectors summed to 1. Exposed menu has five choices at initial state, and a sixth (RETRACE) where host permits. Subsequent states diverge legitimately.

| Actual executed blocks | Variant | Pipework decisions / novelty / p50 / p95 | Greebles decisions / novelty / p50 / p95 | Behaviour |
|---:|---|---|---|---|
| 24 | stock Q4 | 1 / 0 / 2.545 s / 2.545 s | 1 / 0 / 2.533 s / 2.533 s | **STOP at tick 0 in both** |
| 13 | GREYHOUND | 24 / 3 / 1.508 s / 1.535 s | 24 / 3 / 1.511 s / 1.557 s | 24 consecutive LEFT per fold; repeated four-cell square, 21 revisit contacts each; not a Curler discovery |
| 10 | GREY10_CUT | 5 / 2 / 1.125 s / 1.168 s | 3 / 1 / 1.023 s / 1.116 s | 2 BRANCH + 2 RETRACE then STOP at tick 4; 1 BRANCH + 1 RETRACE then STOP at tick 2 |
| 8 | GREY8_CUT | 1 / 0 / 0.820 s / 0.820 s | 1 / 0 / 0.785 s / 0.785 s | **STOP at tick 0 in both** |

Novelty is host-reported `new_steps`, not goal accomplishment. Thus stock, 10-layer, and 8-layer **all STOP in both folds**; GREYHOUND survives 24 decisions but collapses into a four-cell loop. These are **negative behavioural findings**, not viable Meanderer/Curler/Brancher/Chaotic Artist candidates. All eight source/runner cases finished without inference errors. No native-control acceptance, goal completion, geometry responsiveness, independent character review or held-out-generalisation result is established.

Measured `input_tokens_per_second` (prefill proxy, median per case): stock **71.5–71.8**, GREYHOUND **138.6**, GREY10_CUT **182.9–183.1**, GREY8_CUT **222.0–231.8**. Because /v1/systemone reports **zero output tokens** for these decision calls, this is **not** measured generative tok/s. Comparative p50 latency is confounded by 1-decision STOP episodes for stock and GREY8. Real post-ready process sampled maximum RSS (KiB): stock **1,870,492–1,870,608**; GREYHOUND **1,875,256–1,875,340**; GREY10_CUT **1,873,952–1,875,520**; GREY8_CUT **1,870,380–1,870,516**. This is not peak startup memory and **graph-skipping does not shrink the Q4 model file or materially reduce resident weights**.

Caveats: result path's `observed-path` omits the final post-action endpoint; use original host events/positions for motion accounting. The STOP-only stock Q4 baseline under these exact prompts makes speed and survival superiority claims especially weak. Treat 1-sample p95 values as descriptive only. All original artifact files remain the source of truth; no smoothing, re-ranking or post-hoc path creation used in the experiment.

## Next gate and scope

**No further model request was submitted during this review**: last model campaign was started 2026-11-17 **23:49:47 UTC**; six-hour cadence forbids a new request before **2026-11-18 05:49:47 UTC**, even though the repaired upper campaign is complete. On the next eligible invocation recheck current branch, request blob, running/queued Q4 jobs and prior artifacts; then consider one request-file-only `upper`→`middle` transition, same donor/runtime/seeds/budget/single parallel worker. `middle` specifically compares GREY6_CUT and AERO4_NO2 against stock/GREYHOUND. Do not silently skip negative cases or claim recovered function. No other repository, main branch, workflow or other worker's files were changed in this review.

The original zip is retrievable via the linked successful Actions run and its artifact; SVG/PNG and raw JSON are retained as individual files inside that ZIP. Pass only archive references, not unsupported personality labels, to Preference Lab and Extreme Quant/Distillation.
