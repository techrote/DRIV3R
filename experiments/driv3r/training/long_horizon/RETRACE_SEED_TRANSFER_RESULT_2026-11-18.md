# RETRACE-SEED-TRANSFER-001 — cross-optimizer continuation audit

**18 November 2026 · Synthetic DEVELOPMENT, historical source-matched · no new training · Draft/HOLD research only**

## Question

Five states in RETRACE-REALISM-001 showed substantial goal-arrival advantages for immediate adjacent BACKTRACK under the three suffix policies trained with seed 620. Does that advantage survive replacing the *continuation* with independently trained policies from seed 619 while preserving the original seed-620 prehistory, geometry, goal, legal menu and first action?

## Result — bounded positive diagnostic, not autonomous learning

**All five previously identified material BACKTRACK advantages survived all six suffix controllers (seed 619 and 620, frozen/sham/switch).** Each of the five cases reached its target after an adjacent BACKTRACK-first action, whereas the best forward-first candidate did not reach the target within the same 32-decision continuation. Their distinct start-state hashes are in the evidence ledger.

This supports **suffix-controller transfer for five already discovered synthetic states**, not general recovery competence. All five still arise from prehistories generated with *seed 620*. None of 39 eligible seed-619-prefix states has a material six-suffix retreat advantage. Moreover, these five correlated cases use only three world seeds and two families, with normal/trap variants sharing map structure. They must not be counted as five independent held-out replications.

### Cross-seed matrix

| Measure | Previous DEVELOPMENT | Fresh DEVELOPMENT | Combined |
|---|---:|---:|---:|
| Eligible original states | 31 | 56 | 87 |
| Original seed-matched material BACKTRACK wins | 2 | 3 | 5 |
| Material BACKTRACK wins under **all six** suffix policies | **2** | **3** | **5** |
| Strict (possibly small) six-suffix retreat advantages | 3 | 4 | 7 |
| States with conflicting first-action preferences among six suffixes | 3 | 6 | 9 |

Seed-619 prefix: **0/39** six-suffix material retreat wins. Seed-620 prefix: **5/48**. There were **1,788 candidate-action/suffix forks** and **57,216** host-legal synthetic decisions recorded, with **552 fork-goal contacts**. All 1,788 forks exhausted the declared 64-logical-move proxy, and no STOP was observed. Synthetic relocation was suppressed **494** times by the research policy; it remained in the full observation/host-menu evidence when present and was never executed.

The executed action distribution was BACKTRACK 1,196, MOVE_E 14,693, MOVE_N 19,320, MOVE_S 11,266, MOVE_W 10,741. Every natural initial model vote for the 87 states was a MOVE. **No trained controller autonomously selected BACKTRACK at these forks.** Counterfactual advantage is therefore not a demonstrated learned policy improvement.

### Outcome definition and unavoidable caveats

A material advantage requires BACKTRACK to reach when best forward does not, to arrive at least four decisions sooner when both reach, or to end at least eight cells closer when neither reaches. The best legal MOVE comparator is selected independently within each suffix controller; it is not an intentionally weak alternative. No independent model architectures were evaluated; all six suffixes are related 961-parameter policy variants from two optimizer seeds.

The DEVELOPMENT cases have been inspected and selected during previous experiments. They are **not sealed structural HELDOUT**. The evaluation remains a synthetic CyberSand-inspired environment with an imposed local 64-logical-move budget, a manually disallowed synthetic `RELOCATE_START` continuation action, and no native CyberSand ABI/physical receipt. No 128-decision autonomous improvement, recovery budget persistence beyond this proxy, or real-game success is established.

## Qualification and provenance

- Six saved checkpoints were SHA-256 verified and restored; their hashes and all ten source-file SHA-256 pins are in `RETRACE_SEED_TRANSFER_EVIDENCE_2026-11-18.json`.
- The prior 87-case evidence archive was manifest-verified *before* candidate replay. Generation recorded full legal menus, probability vectors, transition receipts, goal-hit times and action traces for every fork.
- **Five focused tests passed.** After generation, a separate source-matched sampled verifier replayed all five material-positive cases plus seven predeclared index-stratified controls (**12/87** states), matching all fork outputs and source hashes, and verified all **89** output files against a SHA-256 manifest. A full 87-state second replay was attempted but did not finish within the execution limit; **do not represent the 57,216 recorded decisions as independently replay-verified.**
- Model-only probability-vector median latency ranged **0.00784–0.00792 ms** per call (256 calls per model, warm-up excluded); this omits world stepping and file I/O. Observed peak process RSS: **252,820 KiB**. The generation was interrupted and resumed, so no valid uninterrupted total campaign wall-clock time is claimed.
- No new weights, native execution, paid compute, automated training trigger, alternate PR/branch or workflow were created. Original checkpoint binaries and full traces are retained in the local archive rather than Git; immutable Actions evidence publication remains outstanding.

## Interpretation and next experiment

The assumption that every observed retreat benefit was merely a *suffix seed-620 artifact* is **falsified for these five states**. The remaining limitation is **prefix/domain dependence**, which is likely to matter more than suffix differences: all successful cases originate from seed-620 prehistory, and all model first votes remain forward.

Next, freeze the five discovered seed-620 examples as diagnostic references and generate new **TRAIN-only** alternative-prefix recovery cases, controlling prefix-state geometry, recovery budget and target positions. Train an explicit conditional BACKTRACK-vs-MOVE policy only after obtaining a balanced TRAIN signal; evaluate on unused DEVELOPMENT configurations without altering this report's case set. Maintain separate native-contract handoff and structural HELDOUT barriers.
