# P24 — Independent P23 source reproduction and pretrained-model execution gate

17 November 2026 · Preference Lab · **DRIV3R PR #1 only, Draft/HOLD**.

## Verification executed

Reopened the original P23 local 15.7 MB evidence archive (SHA-256 `a46c425c43f973b2373dfb6ba8b291b03cacb9dbd63b510519a2d6fb4f0be3b6`). Independently reran its pinned source `p23_matched_niche_gate.py` (SHA-256 `b198d664af90203038df0456014784d98a29094ba2f1da63bc9a464811277b53`) on all three exact GP350-012 producer ZIPs and the independently archived P18 originals. With fixture paths configured, **6/6 adversarial tests passed** (including ZIP corruption rejection and source stability). A pair of fresh complete source runs reproduced **13/13 output files byte-identically**.

Comparison to the published P23 evidence ZIP reproduced **11/13 files byte-identically**: all **8 blind A4 SVGs**, the blank owner ballot, source-matched candidate catalogue and blind identity reveal. The remaining `P23_RESULTS.json` differs **only by added provenance/qualification text and static hash fields** absent from the runnable generator; `SHA256SUMS.json` then differs only for that JSON digest. No trajectory, identity reveal, owner ballot or artwork discrepancy was observed. This is an **evidence packaging/source reproducibility limitation**, not an invented model behavioural change. Preserve both original records; never silently rewrite archived evidence. Exact checks in `P24_REPRO_RECEIPT.json`.

## Cross-lane execution gate, source-checked against live DRIV3R handoffs

- GP350-018 has a qualified source-smoke/frozen decoder but **zero new full-model sampled closed-loop episodes**; pinned LFM350 base/tokenizer weights unavailable locally and no connector workflow dispatch.
- Frankenlayers Wave20 structural candidates are prepared but **not executed**; original Q8 GGUF/llama.cpp backend is unavailable to this runner.
- Tiny M19 and Population P20 SmolLM2-135M real trainers are source-ready but have **zero new pretrained root forwards, zero adapter updates, and no checkpoint**; pinned model/tokenizer and transformers/PEFT runtime unavailable locally.
- Distillation D023 likewise lacks pinned Pythia-70M root weights and pretrained training execution.

The lab cannot honestly claim a new cross-model held-out preference training result. The eligible original P23 sample remains **14 nonidentical matched pairs**, including only **two independent GP350 seed blocks**, with **zero blind owner/human ratings**. P19's GPT-6-derived preference labels and STOP-truncation pairs are not substitutes. No additional proxy ranker, heuristic optimiser, or manufactured ratings were produced.

## Precise consumer handoff / resumption conditions

1. **Provide actual pinned pretrained base model/tokenizer and a network-enabled free CPU host, or an explicit owner-approved opt-in DRIV3R-only Actions execution route**; do not dispatch routine push-triggered training or incur paid GPU/API charges.
2. Obtain original full six-way host-legal action receipts, PNG/SVG trajectories, frozen model/checkpoint/source SHA-256, latency p50/p95 and RSS from **GP350-018 matched greedy versus sampled decoding** and genuinely executed Tiny/Population/Franken outputs on independent unseen worlds.
3. Collect the existing **P23 blank blind four-niche owner ballot** (64 rows including hidden identical controls). Keep the reveal separate. Predeclare source-disjoint holdouts before fitting a learned preference model; compare each niche against transparent movement controls rather than maximizing scalar coverage or turning.

**Disposition:** target-SLM/owner-preference work is currently externally blocked. Preserve the P23 outcomes and their negative caveats; suspend recurring proxy-only increments until an authentic model artifact or owner ratings appear. No native CyberSand inference, no write to CyberSand/Neurounner, no GitHub workflow/branch/PR creation, and no model/adapter training on this run.
