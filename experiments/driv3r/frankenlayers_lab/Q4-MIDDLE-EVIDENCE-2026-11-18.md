# GREYHOUND Q4 middle tier — original inference artifact independently audited (18 November 2026)

**Status: real Q4 compiled execution-graph inference COMPLETE; mixed structural results, not native acceptance.** This is a custody and recomputation review of one already-completed Actions run, not a new model inference or model-weight training step. PR #1 was Draft/HOLD at review; shared branch HEAD before this report: `a07caf83576cfcb4a214224674cd8dda0aff99cf`.

## Source and immutable evidence

- [Real CPU inference run 44253735083](https://github.com/techrote/DRIV3R/actions/runs/44253735083), successful, source commit `cd642b7d319c80526eb97e980e5401d9fb0c74e2`, request nonce `middle-20261118T0643Z-six-four-ablation`. Run created **2026-11-18T06:45:47Z** and completed **06:52:15Z**. Single Actions artifact ID **14405820902**, `driv3r-q4-greyhound-44253735083`; zip SHA-256 **`ceefd282f64e14bf05fc9b7da4188fdd5269009473058b77872222422ab25688`** (matches GitHub's artifact digest). Retention currently listed through 2027-02-16.
- Independently downloaded the **original** ZIP. Exactly **84/84** entries verified against `SHA256SUMS.txt`, zero missing, corrupted, or unexpected unlisted payload files (85 ZIP members including the manifest).
- Original model receipt: vendor-published official d1-2B `Q4_K_M`, **not Q8-requantized**, model file bytes **1,099,629,888**, SHA-256 **`adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b`**, revision `LiquidAI/d1-2B-GGUF@4a181094d1b6addfaa3a0956f78e7ab2aa406a0c`.
- Compiled patched llama.cpp commit **`8642541b255bfb5dc39cd351a178be09013a55d6`**; `llama-server` SHA-256 **`251cf82f5e26f0db8107e24d95d9f0ac27e659a6a301c03306afa73c0521aa75`**; patched `lfm3.cpp` SHA-256 **`ba9d907f7b546958dd059353f2d3ca4534e6e3da63ff86a61881ec1eb380221a`**. Graph-patch receipt SHA-256 **`831a366f6911c644724f61890b12814c5476acd604140e0361bd57dac8bdb74b`**. Graph receipt describes actual skipped source-layer computation, preserves stock output and embedding machinery, and reports base weights unchanged. Unexecuted blocks remain in the original GGUF: **runtime computation skipping, not a physically smaller checkpoint or tensor deletion**.
- W0 host SHA-256 **`81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e`**, source blob `8979b6f0296e81051194bacc907992d38e006605`. Python-standard-library synthetic host; NOT native CyberSand or a revised host contract.

## Genuine on-policy, host-applied evidence

Each arm started from the same initial 7703/pipework and 8807/greebles conditions; episodes limited to 24 decisions per arm/fold, with subsequent world state free to diverge. Independent per-event comparison across `*-choices.json` and original host `events` checked **146/146** actual HTTP 200 /v1/systemone actions: raw model choice = parser candidate = submitted choice = recorded raw host choice = host-applied action; all in contemporaneous legal menu; probability vectors normalized; no mismatch or invalid action. Original `observed-path.json`, `observed-path.svg`, case PNG, model responses, host traces and RSS receipts are retained in the artifact, not reconstructed as invented trajectories.

| Actual executed source blocks | Variant | Pipework: decisions, new cells, p50/p95 (s) | Greebles: decisions, new cells, p50/p95 (s) | Behaviour |
|---:|---|---|---|---|
| 24 | unmodified Q4 stock | 1, 0, 2.336/2.336 | 1, 0, 2.335/2.335 | STOP at tick zero in both worlds |
| 13 | GREYHOUND (`0,1,2,3,4,6,8,10,14,18,20,22,23`) | 24, 3, 1.395/1.411 | 24, 3, 1.396/1.414 | LEFT 24/24 per world, four-cell loop, 21 revisit contacts each |
| 6 | GREY6_CUT (`0,2,4,8,10,20`) | 24, 3, 0.611/0.619 | 24, 3, 0.615/0.625 | 23 RIGHT + 1 STRAIGHT (pipework); 24 RIGHT (greebles); four-cell loop, 21 revisits each |
| 4 | AERO4_NO2 (`0,4,10,20`) | 24, 21, 0.397/0.402 | 24, 16, 0.390/0.402 | Pipework: 21 BRANCH + 3 RETRACE, 0 revisits; greebles: 14 BRANCH + 5 RETRACE + 5 RIGHT, 3 revisits |

Both GREY6 and AERO4_NO2 are 24/24 nonterminal on both folds, unlike the immediate-STOP stock. **However GREY6 is a repetitive 4-cell RIGHT loop, not a useful Curler.** AERO4_NO2 offers substantial territorial novelty and a branch/retrace character in these 24-decision synthetic trials, but does not demonstrate goal completion, exposed-obstacle causal sensitivity, recoverability, painter safety in native code, or independently reviewed artistic niche classification. Reconcile its trace shape with longer held-out native evidence before any positive personality claim.

Median synthetic input prefill-rate proxy across decision calls by fold (input tokens/s): stock 77.9/77.9, GREYHOUND 150.0/150.0, GREY6 348.4/347.0, AERO4_NO2 549.4/544.9. This **is not autoregressive output tok/s**: response usage reports zero output tokens. The stock's 1-decision STOP episodes are a severe confound for comparative latency/survival. Measured post-ready peak RSS KiB (pipework/greebles): stock 1,870,708/1,870,676; GREYHOUND 1,875,416/1,875,528; GREY6 1,875,532/1,875,536; AERO4_NO2 1,878,104/1,878,236. Peak startup RSS was not measured; execution skipping does not reduce resident model weights. The path summary uses *predecision* positions and omits final post-action endpoint; original host events preserve the applied result.

## Research decision and next authority gate

The middle experiment is **valid and completed, with one promising but not yet qualified 4-layer behaviour and two repetitive/STOP controls**. The scientifically interesting result is 4-layer AERO4_NO2's host-applied branch/retrace exploration, not the nominal speed of STOP-only stock. Carry AERO4 trace references as **candidate for blind inspection**, not a certified Brancher, to Preference Lab; carry graph and exact Q4 model source identity to Extreme Quant and Distillation; no external files or repositories may be mutated by this lane.

The user's progression rule allows **one** request-only `middle`→`extreme` transition after independent artifact inspection and the six-hour rate limit. This review verifies middle evidence but **does not itself authorize skipping the 2026-11-18T12:45:47Z cooldown** (six hours after the 06:45:47Z middle campaign start). Before any extreme request, re-check the exact branch and request blob, absence of queued/running Q4 runs, same donor SHA/bytes, runtime, folds, budget and `max_parallel=1`; only update the lane-owned request JSON. Extreme compares 3- and 2-layer real executed skeletons against stock and GREYHOUND, retains STOP-only/corrupt-output negatives, with no forced motion logic.

Scope: techrote/DRIV3R PR #1 Draft/HOLD only; CyberSand and Neurounner read-only; no merge or paid compute.
