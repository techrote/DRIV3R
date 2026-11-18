# C32 — Off-stack, nonmonotone recovery after host feedback

Date: 17 November 2026. Scope: `experiments/driv3r/training/guidance_recovery/**` in DRIV3R Draft/HOLD PR #1 only. **Synthetic host research, not CyberSand native acceptance or checkpoint promotion.**

## Research question

C31-DIRECT successfully traversed visited, locally goalward cells but required its new goal to be present on the executed navigation stack. Can a host-cued worm recover when the goal is **not on its history**, the next useful action temporarily moves away from the goal, and locally attractive shortcuts can instead be deceptive blind alleys?

## Intervention and controls

The C32 generator first executes a 27-action U-shaped source path through `World.apply()` with independently recomputed legality and protected-material checks, then changes the goal to a reachable branch outside that history. Seven test conditions include three off-stack branch positions, a genuinely direct six-step shortcut, and two-, three- or four-step goalward dead-end lures. Offline BFS verifies reachable goals and shortest paths of 30, 26, 23, 6, 30, 30 and 30 legal steps respectively. No BFS distance, future layout or scenario label enters live policy input.

Live C32 policies consume **only current host-legal action menus, recent executed history, goal position, observed nonprogress and a previously issued host BACKTRACK cue**. They may retain a bounded stochastic recovery intent beyond the cue's nine-turn life, but re-check action legality every step and release on explicit cancellation, goal change or a 105-decision limit. Sampling retains at least 12% probability for alternatives to a boosted probe and 1.5% alternatives during retracing. No cue means no policy intervention.

- C18-A0: frozen untrained synthetic baseline.
- C31-DIRECT: previous non-neural visited-goalward search control, evaluated on exactly matched worlds.
- C32-SAFE: preserves a preexisting fresh goalward option, starting retrace only when the local menu is already blocked.
- C32-WATCH: initially leaves an available fresh goalward route untouched, then initiates retrace if the host-executed movement stops producing fresh goalward choices.
- Exploratory C32-RULE and C32-LEARNED: immediately probe fresh branches. The trained head adjusts bounded probe logits but cannot override host legality.

## Training

C32-LEARNED's CPU-only 6→12→1 head was trained with AdamW on 320 current-menu feature examples; the targets came from **legally executed, fixed eight-step counterfactual suffixes**, not from a hidden shortest-path label supplied to the network. There were 96 successful suffixes and 140 optimizer updates; in-sample binary accuracy was 85%. Two fresh, independent 20-epoch optimizer executions yielded byte-identical `safetensors` checkpoints. This is a small auxiliary score head, not a trained end-to-end CyberSand worm or a vision-language model.

## Withheld evaluation

Two held-out seed cohorts use shifted, reflected and rotated worlds not used during fit. Each policy has 224 guided and 56 unguided trials per cohort, with matched weak/strong BACKTRACK and CONTINUE cues, two hint strengths, two personality polarities and seven layouts. A total of **2,240 host-executed evaluation episodes** were evaluated in the primary four-policy matrix.

| Policy | Cohort A: guided successes | Cohort B: guided successes | Group-level regressions against C18-A0 |
| --- | ---: | ---: | ---: |
| C18-A0 | 31/224 | 32/224 | Reference |
| C31-DIRECT | 31/224 | 32/224 | 0 / 0 |
| C32-SAFE | 61/224 | 66/224 | 0 / 0 |
| **C32-WATCH** | **81/224** | **85/224** | **0 / 0** |

C32-WATCH recovered off-stack early/mid/late branch goals **30 times in cohort A** and **34 in B**; C18-A0 completed zero in those categories. It also recovered false-shortcut lure cases **20 and 19 times** respectively, while C18-A0 recovered zero. Legitimate shortcut completion remained unchanged at 31/32 in cohort A and 32/32 in B; matched CONTINUE performance was also unchanged. Note that the one unsuccessful legitimate shortcut in A was already a baseline failure, not a regression.

All **112 unguided paired trajectories per C32 variant** matched C18-A0's action and ink hashes exactly. Across evaluated models there were zero host-rejected actions, zero raw illegal top-choice actions and no protected-material modification. The displayed gallery plots **actual sampled legal moves**, not predicted plans. Training-replay checkpoint identity and seven targeted unit tests passed.

## Limitations and disposition

This is a significant **synthetic state-machine/recovery-architecture improvement**, not evidence that the neural head learned a better recovery policy. C32-WATCH uses no new trained head; C32-LEARNED's in-sample 85% classification accuracy **did not change a single action hash** relative to C32-RULE across 280 paired exploratory episodes (140 per seed cohort). Both obtained 39/112 and 41/112 guided completions and both lost one legitimate-shortcut success in the second cohort. The classifier's bonus saturated the common sampling cap; this is a negative neural result, not evidence of learned strategic improvement. The environments remain small, narrow, handcrafted corridors. Rotation/translation and held-out RNG seeds do not establish broad topological generalisation. Hidden materials and actual CyberSand engine dynamics are absent; host legality here is the synthetic `World` fixture, not evidence of CyberSand's canonical native ABI.

**Retain C32-WATCH as a research control only. Do not promote checkpoints, merge PR #1, change DRIV3R main or write to CyberSand/Neurounner.** Active-guidance personality auditing, broad unseen topology, native ABI equivalence, full repository-based scientific reproduction and immutable GitHub Actions checkpoint custody are outstanding. Current archive is interim local custody, not an immutable Actions artifact.

Next: test C32's WATCH controller under misleading cues with changing goals and more irregular multibranch layouts, and test whether a learned head provides measurable held-out recovery benefit beyond the deterministic controller before contemplating promotion.
