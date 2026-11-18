# DRIV3R RETRACE-REALISM-001 — no-root-relocation, budget-censored retracing audit

**Status:** BOUNDED DEVELOPMENT POSITIVE CANDIDATES, GENERALIZATION UNPROVEN. Synthetic-only evidence, not native CyberSand qualification or a newly trained policy. DRIV3R Draft/HOLD PR #1 is the only intended research publication target. Do not merge.

## Question and controlled method

The earlier POST-SWITCH-REGRET-001 oracle preferred synthetic `RELOCATE_START` in 63/95 eligible switches; this fabricated a shortcut unavailable as an independently qualified native movement. Can *ordinary adjacent BACKTRACK* produce useful, repeatable, goal-directed advantage without that shortcut?

Restore the six unchanged 961-parameter checkpoints (`frozen`, fixed-goal `sham`, goal `switch`; optimizer seeds 619/620). Construct two cohorts of autonomous frozen-policy DEVELOPMENT prefixes (32 previously evaluated prefix configurations and 64 fresh within-family seed configurations), with two target variants. Test only the first switch at decision 32; exclude nine prefixes in which the frozen model had already performed synthetic root relocation. All 87 surviving states are unpainted-target states. Every initial candidate must be present in the unchanged `World.menu(12)` and be `BACKTRACK` or a cardinal `MOVE`; `STOP` and `RELOCATE_START` are not initial comparator choices. Exactly one legal initial action is forced per fork; subsequent decisions are checkpointed-policy argmax restricted *only in the research controller* against `RELOCATE_START`. Full original menu, raw neural probability vector, and both raw/conditional choices remain recorded at every decision. `World.apply` is never called on an invented action. An explicit 64-synthetic-logical-move proxy terminates the fork; 32 decisions remain after the prefix, with all executed backtracks adjacent and priced as one move. This proxy is not the canonical CyberSand native movement contract.

Primary goal metric (declared in source): first goal hit, with earlier hit preferred; if neither fork reaches goal, prefer smaller final Manhattan distance. A **material** BACKTRACK win requires at least four decisions saved when both hit, eight cells of final-distance gain if neither hits, or a hit when the best available MOVE does not. Require improvement under all three suffix models at the *same prefix*.

## Measured DEVELOPMENT results

- **96** candidate prefixes: **9 excluded** for pre-switch synthetic relocation, **87** evaluated (31 prior configurations, 56 fresh configurations). No structurally held-out families opened.
- **894 matched counterfactual forks**, **28,608 host-legal synthetic decisions**, **six restored original checkpoints**. All forks reached the 64-logical-move proxy after 32 decisions; none STOPped. This does not establish sustained native 128-decision play.
- Across all three suffix models, **8/87** states showed *any* strictly favourable BACKTRACK outcome; **5/87** met the predeclared material margin. The older DEVELOPMENT cohort had **2/31** material successes, fresh DEVELOPMENT **3/56**.
- **All five material cases occurred with optimizer seed 620; zero occurred with seed 619.** They occur at pickets (old seed 3002; new seed 3008, both normal/trap) and ridges (new seed 3108, normal). These are correlated procedural structures, not independent cross-seed replication.
- None of the three models naturally selected BACKTRACK at the fork: first-choice counters contain only cardinal MOVE. All five material-positive cases therefore identify a narrow missed opportunity for an oracle-assisted retreat, not an achieved improvement by a trained autonomous policy.
- The no-relocation controller prevented **274 attempted relocation argmaxes** across suffixes (83 frozen, 97 sham, 94 switch), with the raw probabilities preserved. Since this is a policy intervention, it must not be presented as the original controller's autonomous behaviour.
- Executed action counts per suffix are recorded in the compact evidence JSON; 226/182/181 BACKTRACK operations respectively occurred across 9,536 steps per suffix. There were no early STOP decisions in the restricted continuations.

**Interpretation:** Adjacent BACKTRACK can have a substantial downstream goal-arrival benefit in a small number of synthetic switched-goal configurations, even without relocation. But candidate reliability is poor: all five robust cases depend on a single optimizer seed and overlap in structural geometry. No new adapter should be trained on these labels as if generally valid. The exposure of true candidate pockets is a *diagnostic* positive; the current policies' learned recovery ability remains unproven.

## Source-matched qualification

Four focused tests passed: split guard, action-menu and mask semantics, outcome/margin definitions, and deterministic checkpoint-backed replay. Independent verification reconstructed all 87 states and replayed **894 forks / 28,608 decisions**, checking full legal menus, neural probabilities, material receipts, states, original checkpoint SHA-256 and the 89-entry evidence manifest. Eight inherited dependency files matched their exact DRIV3R Git blobs before the run. The generated local source/trajectory ZIP retains the six checkpoint binaries, original full receipts and SHA-256 manifest; checkpoint binaries were **not** committed to Git or published to immutable Actions artifacts.

The CPU research campaign used **23.449 s wall time** and **150,540 KiB peak RSS**. Model probability-vector-only median inference latency was **0.0078 ms** (frozen), **0.0078 ms** (sham), and **0.0078 ms** (switch); p95 **0.0080/0.0083/0.0088 ms** respectively. These measurements exclude synthetic world stepping, serialization, native simulation and rendering.

## Limitations and next decision

The study deliberately reuses familiar procedural world families and a known target schedule; the fresh DEVELOPMENT seeds are **not** structural heldout. Policy suffixes are three related models, not independent model families. 32-step budget truncation, after a 32-decision prefix, can disadvantage delayed but eventually successful paths. The presence of synthetic protected cells/materials is source-faithful *to the lab fixture*, not verified against current native gameplay. No actual native receipts, no immutable Actions checkpoint artifacts, no cross-model-family comparison and no newly trained neural weights are asserted.

**Next experiment:** construct TRAIN cases in which backward recovery *actually improves target arrival under multiple independently seeded continuation models*, while separating geometry correlation; then train a bounded goal-conditioned retrace discriminator and evaluate against frozen/sham baselines on untouched DEVELOPMENT geometry. Leave structural HELDOUT sealed until the candidate actually improves on DEVELOPMENT.
