# GOAL-NATIVE-GATE-001 — trained goal conditioning under a 64-logical-move synthetic host gate

**18 November 2026 · DRIV3R Draft/HOLD PR #1 · DEVELOPMENT-only, synthetic experiment.** Original source pinned before edit to `9420cc6cae9c316c6efe76fe1532817d324f2f41`. No CyberSand or Neurounner writes. Reuses unmodified source-faithful `world.py`, `policy.py`, `train.py`, `goal_conditioning_lab.py`, and `goal_observability_lab.py`.

## Research question and result

Can a genuinely trained tiny model follow sequential source-generated spatial goals when an *external synthetic* 64-logical-move gate is applied, compared with an identically sized and identically trained goal-masked control?

**Yes on this DEVELOPMENT fixture.** Two fresh training seeds, **711 and 712**, trained matched 769-parameter `GoalRanker` heads through **36 → 72 → 128**, 40 real Adam updates/stage = **120 updates/checkpoint**. Four new checkpoints, SHA-256 listed in the ledger. Same TRAIN-generated teacher examples, optimizer budget and model capacity within each seed; one arm receives six engineered goal-input channels, the goal-blind arm receives zeros for those channels.

| Controller | Sequential goals attained / offered | Episodes completing all waypoints | 64-gate censored | Premature STOP |
|---|---:|---:|---:|---:|
| Goal-conditioned neural | **443/488** | **37/48** | 11/48 | 0/48 |
| Matched goal-blind neural | **2/488** | **0/48** | 48/48 | 0/48 |
| Goal-aware BFS/stack heuristic | 366/488 | 24/48 | 2/48 | 22/48 |
| Persistent-random legal baseline | 18/488 | 0/48 | 46/48 | 2/48 |

**Outcome is real sequential synthetic execution**, not just first-action classification. Across 192 episodes (12 fresh DEVELOPMENT worlds × two optimizer seeds × ordinary/legally perturbed × four arms), there were **9,858 executed host-legal synthetic receipts** and **5,342 checkpoint-recomputed neural action distributions**. No protected/material edits occurred. Goal-conditioned ordinary **243** successes and perturbed **200**; blind **2** and **0** respectively. Neural seeds 711/712 attained **222/221** goals, respectively.

### Paired confidence and counterfactual sensitivity

For conditioned minus blind, the total success-rate difference is **+90.37 percentage points**, with a **descriptive** 12-source-world clustered bootstrap interval **[+81.63, +97.54] points** (10,000 deterministic resamples). These are *not* an independent structural-family generalization interval. Conditioned minus BFS/stack is **+15.78 points**, cluster interval **[-1.79, +30.74]**, containing zero.

An identical initial World state was supplied contradictory cardinal goals (N/E/S/W) in all four DEVELOPMENT geometry families and both optimizer seeds: **8/8 goal-conditioned states** chose the commanded N/E/S/W actions; **8/8 blind states** gave identical MOVE_E for all goals. Host legal menus and painted/protected/material state were identical across each goal change. Goal conditioning is intentionally aided by engineered Manhattan one-step improvement features: this is **not evidence of emergent planning or broad natural-language understanding**.

## Qualification and resources

- Test suite: **5/5 passed** with local archive (4/4 source-only, one optional full-archive test). Tampered receipt rejects. An independent replay pass validated all **192/192** original episodes with exact legal menus, state transitions, forced interventions and checkpoint probabilities; no estimated/native receipts were credited.
- Run CPU wall time **19.970 s**, peak RSS **111,844 KiB**. Median per-episode neural decision selection: **0.10432 ms conditioned** vs **0.09247 ms blind**; not native hardware-inference benchmarks.
- Action counts, checkpoint SHA-256s, family breakdown, scene seeds, source hashes and representative trajectories are in the JSON ledger. Full per-step receipts, probability vectors, archived checkpoints and SHA manifest stay outside Git history in the attached ZIP.

## Interpretation and limitations

The external limit is a **synthetic analytical censor**, not CyberSand native enforcement. This is 64 logical moves per episode, not 72/128 moves of accepted native gameplay. Host legal actions were preserved, including STOP and synthetic, unvalidated `RELOCATE_START` (zero usages by trained neural arms, eight by random control). Native World epoch/paint acknowledgements are absent.

The BFS/stack heuristic is **not a planning upper bound**: it backtracks into painted goals and sometimes exhausts recovery budget or stops; its 366/488 must not be portrayed as an optimal planner score. Some generated waypoints become inaccessible after divergent painted paths. Legally perturbed controllers experience policy-dependent alternatives at ticks 7/22, not identical exogenous states, so perturbed arm scores are descriptive, not fully controlled counterfactual evidence. TRAIN and DEVELOPMENT geometries have the same four structural families; sealed `archipelago` and `staggered_vault` HELDOUT families were never inspected. d1-2B, LFM2.5-350M, Sprocket-500M and native CyberSand remain untested.

**Next falsification:** train/evaluate on goals and geometry requiring obstacle detours, budget-aware backtracking and recovery after fully identical externally specified wrong moves; compare against a *complete* graph-search oracle and a truly independent structural final holdout only after explicit qualification. Publishing checkpoints as immutable Actions artifacts remains unfinished. No paid API/GPU charges.
