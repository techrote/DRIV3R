# DRIV3R C56 — Host-observed memory erasure and distant-goal transfer (18 November 2026)

**Decision: RESEARCH HOLD; do not promote C56-AUGMENTED.** This is a bounded, executed synthetic-host study of how previously observed route memory affects guidance. C56 retains C55's architecture, trains a separate dropout-augmented guidance head, and tests it against frozen C55, shuffled labels, non-neural ROUTE and RESET, and unguided controls. It does **not** demonstrate transferable distant-goal navigation, zero regression, or native CyberSand compatibility.

## Source and research boundary

- Owner repository: `techrote/DRIV3R`; writable path only `experiments/driv3r/training/guidance_recovery/**` in existing Draft/HOLD PR #1. No native CyberSand or Neurounner mutations.
- Exact C55 predecessor: `c55_observed_waypoint_distillation.py` Git blob `cb965fbab3a08c2d8cc0c9ce7e300fc36a6c0fd8`, already in PR #1. Its frozen learned-head checkpoint SHA-256 is recorded in the C55 report; C56 loads that exact frozen checkpoint from the predecessor owner archive. C54 dependency blob: `0ec2b4eecb1deec62a7e621a07ee840f852ff23b`.
- C56 study source SHA-256: `3aa71cd04f1921a6a91c080b7837500f74e784bc74e9cd5fc950338cfffb7e91`. Exact executable source and test sources are committed beside this report.
- Historic C55 training, blind, online and external-online graph identities are excluded before sampling C56 train/online worlds. Split is graph identity, not reused prefixes. A further 32 new training graph identities and 24 independent online graph identities were generated: 4 each of Wilson, Kruskal, tree, braid, Prim and braided Prim among the online set.

## Controlled intervention

The experiment creates actually host-executed nine-tick prefixes and **forgets either 0% or 50%** of deterministic-key-selected previously visited local action menus. At every subsequent visit, a fresh host-legal menu is read and can refresh memory. Only remembered menus are removed, never added. The controller never reads remote protected cells or the maze oracle; every sampled action passes the synthetic host's `World.apply` legality checks. The same erased prefix and RNG state are supplied to every policy, with independent execution rather than copied controls.

There are two separately analyzed target types: **OBSERVED**: a frontier waypoint deliberately chosen from a route already available before erasure; and **REMOTE**: the independent host-issued goal, excluded when already observed. These are fundamentally different difficulty distributions; do not pool goal rates. Under 50% erasure only 2 of 18 OBSERVED prefixes retained a usable original route, compared with 18 of 18 under 0% erasure. Neither of 19 REMOTE-prefix goals had an originally observable route.

The new C56-AUGMENTED and matched graph-label-shuffled heads were trained from scratch on 1,296 host-executed legal action examples from 18 contributing training graphs. Both received nine deterministic epochs / **189 CPU AdamW updates**, using identical architecture, training inputs, initialization and optimization. The route-step teacher uses remembered host observations, never global geometry; it cannot teach how to follow a completely absent path. A clean independent refit reproduced every tensor exactly, and 6/6 regression and independent-path-replay tests passed.

## Independent online goal completion

Numbers below are completed goals in **36 OBSERVED guided episodes per policy per memory level** or **38 REMOTE guided episodes per policy per level**; weak and strong host cues are separate matched arms.

| Controller | OBSERVED 0% erasure | OBSERVED 50% erasure | REMOTE 0% | REMOTE 50% |
|---|---:|---:|---:|---:|
| C18-A0 (unguided) | 24 | 24 | 16 | 16 |
| C34-RESET | 26 | 26 | 16 | 16 |
| C55-ROUTE non-neural | **28** | **29** | 16 | 16 |
| Frozen C55-ALIGNED | 26 | 28 | 16 | 16 |
| **C56-AUGMENTED** | **27** | **26** | **16** | **16** |
| C56-SHUFFLED | 24 | 26 | 16 | 16 |

On intact OBSERVED memory the augmented model gains **1 goal** over RESET, 3 over SHUFFLED, and 1 over frozen C55; its descriptive graph-cluster bootstrap interval vs RESET is **−6 to +8**. With 50% erasure it ties RESET and SHUFFLED, while losing two goals to frozen C55 and three to non-neural ROUTE. Its 50%-erasure vs RESET interval is exactly **0 to 0** because no matched cluster differs in completed goals. All controllers tie on REMOTE goals; no learned route-transfer advantage survives this goal change.

At 0% erasure C56 has **three condition-level regressions** against RESET, so the strict no-regression gate fails. At 50% erasure C56 is inert with respect to completed goals; this is not a robustness victory. The initial observed route is absent for 16/18 heavily erased prefixes, and the learned head is structurally gated on an available known route. This is a causal information bottleneck. The counterintuitive small gains for frozen C55 and ROUTE under heavy erasure are exploratory, driven by stochastic trajectories and weak controls, not evidence that forgetting improves planning.

## Qualification and limitations

- **222 independently evaluated scenario rows × six controllers = 1,332 synthetic-host suffix executions**, plus separate training-world rollouts. Of these, 148 scenario rows per controller are guided (74 OBSERVED/REMOTE total per erasure level) and 74 are no-guidance.
- **No host action rejections, no protected-cell mutations**, and exact across-controller no-cue action + painting hash parity in all 74 no-guidance scenario rows. Each controller executed these separately.
- **24 gallery trajectories fully re-executed** with exact source, seed, prefix and action/painting hashes. SVG is evidence drawn only from executed paths, not a generated illustration.
- The front-loaded world factory includes an earlier host-executed initialization; the study validates only its own nine-tick continuing prefix and subsequent suffix. This is not uninterrupted native CyberSand replay.
- The 50% memory deletion is an artificial diagnostic intervention, not a validated model of naturally decaying memory. It was not selected on observed outcomes; however this long sequence of C18–C56 studies has substantial adaptive selection, so all confidence intervals remain **exploratory**, not confirmatory.
- No guided-personality equivalence test exists beyond preference-turn counts and no-cue hash parity; no native CyberSand runtime evaluation was performed. GitHub Actions immutable checkpoint custody remains unavailable. The newly trained binary checkpoints are retained in the owner evaluation archive, **not Git history**.

**Disposition:** HOLD/REJECT C56-AUGMENTED for model promotion. Subsequent work should test whether the host can provide minimal, legal, non-oracular *route restoration* or truly memory-independent local exploration, with a frozen preselected evaluation corpus and separate guide-strength/personality gates. Do not use hidden maze information in deployed policy features.
