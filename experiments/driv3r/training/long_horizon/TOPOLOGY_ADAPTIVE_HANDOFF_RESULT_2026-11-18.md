# DRIV3R — TOPOLOGY-ADAPTIVE-HANDOFF-001 (18 November 2026)

**DEVELOPMENT verdict: encouraging but not independently generalized.** A *single planner intervention at the second scheduled goal reversal* reached **42/64** final goals (65.63%) while making **64** bounded prefix-planner searches and preserving **1,896** neural-selected actions. This exceeded neural-only (27/64), the previous fixed tick-24 handoff (36/64, 122 searches), and the restricted planner at every stage (40/64, 192 searches) on these *same* four synthetic topologies. It must **not** be promoted as a general optimal policy.

## Controlled design

Two immutable goal-conditioned tiny checkpoints (seeds 711/712), four DEVELOPMENT background families, four prespecified central overlays (opposing gates, enclosed courtyard, horizontal switchback, protected slalom), two opposing goal orders, and seven prespecified control arms. There are 64 matched scenarios per arm and 448 autonomous synthetic episodes. Goals reverse at tick **12** and again at tick **30**, with a strict external **64 logical-move** gate. Protected and material cells cannot be painted. No game code or historical d_common fixture was changed.

The key second-switch arm follows the frozen neural model until tick 30, performs exactly one restricted prefix search for the *final* target, then executes that cached plan. Other new arms: search on both switches; search when post-switch Manhattan budget slack <= 10 or progress stalls four decisions; and wait for three stalls/decision 36 after the second switch. Reused comparators neural-only, absolute tick 24 and full-stage restricted planning are source-identical to the preceding topology-goal-churn experiment.

## Measured results

| Controller | Final targets reached | Planner searches | Neural actions | Mean decisions |
|---|---:|---:|---:|---:|
| Neural only | 27/64 | 0 | 3,849 | 60.14 |
| Fixed tick-24 handoff | 36/64 | 122 | 1,536 | 44.03 |
| Restricted planner each stage | 40/64 | 192 | 0 | 44.25 |
| **Planner at second switch** | **42/64** | **64** | **1,896** | 46.63 |
| Planner at each switch | 36/64 | 128 | 768 | 42.44 |
| Post-switch budget slack <= 10 / stall-4 | 37/64 | 66 | 1,980 | 45.95 |
| Second-switch stall-3 / tick 36 | 41/64 | 64 | 2,280 | 49.44 |

Paired second-switch-vs-neural: **15 rescued, 0 regressed**; vs tick-24: **6 rescued, 0 regressed**; vs full-stage planner: **6 rescued, 4 regressed**. The mask-clustered descriptive 95% bootstrap difference intervals, respectively, are **[+9.38,+37.50]**, **[0,+28.12]**, and **[-14.06,+18.75]** percentage points. There are only *four independent design masks*, so these ranges are exploratory and too fragile for inferential superiority claims.

Per-mask second-switch successes were 11/16 (opposing gates), **8/16 (courtyard)**, 12/16 (switchback), and 11/16 (slalom). The courtyard still fails half the time. Notably, neural-only succeeded 6/16 there, indicating that timed sparse planning is not sufficient for reliable transfer.

## Source identity, receipts and falsification

**448** traced episodes, **21,304** archived legal action receipts and **12,309** neural probability vectors passed a complete source-matched replay, plus **three focused tests** covering split guards, trigger exclusivity, forged menu/receipt/probability/switch tampering. All 192 reused baseline trajectories for neural-only, absolute24 and planner_each_stage had **identical final-state and trace hashes** to the prior archived experiment. The 8-panel SVG is drawn from recorded action-receipt cursor transitions; it is not a simulated approximation.

Frozen checkpoint SHA-256: seed 711 `64643cc987d095a94e3dfcd9b5681377f361c2e0df5363f3027bacbe2b8db7df`; seed 712 `111dba8a604c00497d5c5e5726023f67fc06ea62be77a63001122f9bdca5a9a3`. Exact dependency and main-lab source SHA-256 are in the companion JSON ledger. Complete local archive SHA-256: `bc10e55eece69cc6b1b54bac3dabb832d3574bf91173db89368ee1b89ab3acb2`.

**Runtime caveat:** the campaign was completed by checkpointed/resumable local execution. Its final resumption took 28.426 seconds and reported 99,600 KiB peak RSS, but that is **not** the total campaign time. A fresh uninterrupted rerun was bounded and timed out after 260 seconds with 318/448 traces; therefore no independently completed monolithic rerun is claimed. Reported median choice latencies mix neural inference, cached planner actions and expensive first searches. No GPU or paid inference was used.

## Limitations and next falsification

Reusing the same four synthetic masks is not structural HELDOUT and does not resolve the previous transfer failure. All tested terrain masks are static; scheduled goal reversals and early-arrival idle ticks are research-only conveniences. The backtrack-prefix planner does not solve unrestricted native game state transitions. All new arms were prespecified, but the best was **selected after inspecting DEVELOPMENT scores**; the selection must be frozen before any future independent validation. Goal-conditioned neural observations expose engineered Manhattan features. No native CyberSand receipts, new optimizer updates, immutable Actions checkpoint artifacts or model-size transfer are claimed.

**Next:** freeze the one-second-switch intervention rule and test it without tuning on genuinely new topology generators, late goal changes, reactive material states and stronger complete-state planning controls. Preserve failures, resource costs, executed-path galleries and strict owner-gated Draft/HOLD status. No CyberSand/Neurounner writes.
