# GREYHOUND-Q4-EXTREME-001 — reusable source-pinned CPU evaluation

**Owner direction: 17 November 2026.** Only techrote/DRIV3R's existing Draft/HOLD PR #1 may be written. The legacy CyberSand research and native Workbench are READ-ONLY. No merge, promotion, native painting, paid GPU/API or automatic training on ordinary source changes.

## Model identity and experimental method

Use the **official published** LiquidAI/d1-2B Q4_K_M GGUF, NOT an unknown third-party file and NOT a lossy requantisation of the historical Q8 specimen. Pinned revision \`4a181094d1b6addfaa3a0956f78e7ab2aa406a0c\`, exact SHA-256 \`adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b\`, 1,099,629,888 bytes. The untouched stock Q4 runs on exactly the same compiled pinned b11960 llama.cpp as mutated *decision-only* graphs. Original GREYHOUND is a baseline, not a claim that Q4 and Q8 produce identical tokens.

The full-body Q4 weights remain unchanged and only **executed attention/FFN blocks in the compiled System-One computation graph are skipped**. The original embedding/head and normal completion path remain intact. Do not call this a pruned-weight checkpoint or a physically smaller GGUF; the Q4 file still contains the original tensors.

A strictly nested progressive deletion campaign:

| Stage | Mode | Executed original layers | Omitted of 24 |
|---|---|---|---:|
| Historical anchor | GREYHOUND | 0,1,2,3,4,6,8,10,14,18,20,22,23 | 11 |
| upper | GREY10_CUT | 0,1,2,4,6,8,10,14,20,23 | 14 |
| upper | GREY8_CUT | 0,1,2,4,8,10,20,23 | 16 |
| middle | GREY6_CUT | 0,2,4,8,10,20 | 18 |
| middle | AERO4_NO2 (frozen existing identity) | 0,4,10,20 | 20 |
| extreme | GREY3_CUT | 0,4,10 | 21 |
| extreme | GREY2_CUT | 0,10 | 22 |

GREY8_CUT and AERO4_NO2 reuse existing flow definitions and preserved historical Q8 evidence. GREY3 and GREY2 are highly speculative; the runtime full/sliding-attention guard may reject them, and that rejection is a valid negative result. Earlier STOP-only skeletons must not be concealed or retried using fake movements.

Each tier runs **stock Q4 + GREYHOUND Q4 + two successively thinner variants** in two fixed independent initial W0 families (7703/pipework and 8807/greebles), at most 24 decisions per case = 192 maximum per tier. No controller-specified move patterns: each choice comes from the actual /v1/systemone model and W0 alone decides host legality. Branch and retrace may move in nontrivial ways; preserve full W0 record, not a fabricated end point.

## Reusable and scheduled triggering without touching main

Workflow: \`.github/workflows/driv3r-q4-greyhound.yml\`. **Only a change to \`experiments/driv3r/frankenlayers_lab/q4-greyhound-request.json\` on \`research/import-consolidated-worm-labs\` triggers expensive execution via push.** Other PR pushes only run existing cheap source-smoke. Workflow also declares \`workflow_dispatch\` for eventual manual UI dispatch after the workflow is admitted to the default branch; GitHub does not serve normal workflow_dispatch for a workflow absent from default.

A scheduled DRIV3R Frankenlayers research automation (external scheduler, not GitHub cron) updates the one request file on a bounded cadence, only after checking the prior Actions run and its original artifact. GitHub's native \`schedule\` is deliberately NOT added: schedules run only from the default branch and this PR must remain Draft/HOLD. Each request update is an explicit source-controlled push with distinct nonce and is reviewable. No routine-source push can start a costly model job.

### Source-only test and manual request

\`\`\`bash
python3 -m unittest experiments.driv3r.frankenlayers_lab.test_q4_greyhound -v
\`\`\`

A user can explicitly start a tier by changing the request's \`tier\` and \`nonce\`, committing and pushing the single request file to the existing research branch. New source/metadata commits are NOT execution triggers. A tagged sample nonce is informational; actual provenance always binds the exact triggering \`GITHUB_SHA\`.

Every actual runner must first validate the request, load historical \`backend.py\`, \`pins.json\`, \`process.py\` and W0 **from immutable read-only CyberSand commit URLs with exact historical Git-blob hashes**, acquire and SHA/size-verify official Q4, source-build the pinned b11960 runtime with exact patch receipts, and retain original responses/World-free W0 PNG/JSON, unsmoothed SVG, p50/p95 and sampled RSS. SHA256SUMS and all original failures are uploaded even when inference fails. No model binaries enter Git or the downloadable artifact.

## Stop gates and scientific boundary

A source-smoke PASS is **not an inference PASS**. A completed result may still be STOP-only, repetitive or architecturally invalid. The scheduler must read *actual* original artifact outcome and not advance to the next tier merely because a build succeeded. Require at least a working original stock+GREYHOUND control, evidence hashes and on-policy full budget or explicitly acknowledged scientifically informative negatives. Fail closed on missing artifacts/runtime, and do not spam duplicate unsuccessful jobs. Owner-rated Meanderer/Curler/Brancher/Chaotic Artist character is separate from counts of turns, loops and branches. Never claim native CyberSand material validity.

This setup deliberately does not assume that the executing chat environment holds 1.1 GB model bytes or can invoke GitHub's workflow-dispatch API.
