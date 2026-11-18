# DRIV3R TOPOLOGY-GOAL-CHURN-001 — development-only obstacle-geometry transfer

**Verdict: negative for reliable transfer of previously successful sparse intervention.** The previous two-switch corridor experiments found perfect final-goal completion for one delayed planner handoff. A new, prespecified set of **four distinct central obstacle masks**, rather than the previously shared trap corridor, reduced delayed-handoff final-goal completion to **36/64 (56.25%)** and neural-only completion to **27/64 (42.19%)**. A restricted planner invoked at every goal stage reached **40/64 (62.50%)**. No arm reached all targets; none warrants native promotion.

## Experimental design and boundaries

- Source: `topology_goal_churn_lab.py`, direct imports of unchanged `world.py`, `goal_conditioning_lab.py`, `goal_observability_lab.py`, `detour_recovery_lab.py`, `policy.py`, and `train.py`. All dependency Git blobs must match the pinned remote repository head when the result is published.
- Evaluation: DEVELOPMENT only, four training-family seed backgrounds (`pickets`, `ridges`, `islands`, `zigzag`), four prespecified topology masks, two opposing initial goal orders, two frozen goal-conditioned tiny-model checkpoints (711 and 712), and five controller arms = **320 episodes** (64 per arm). No optimizer updates and no HELDOUT reads.
- The 31×31 central research arena is cleared of seed-specific obstacles and protected exclusions *before the first action*, then a deterministic mask is installed: **opposing vertical gates**, **enclosed courtyard**, **horizontal switchbacks**, or a **protected-cell slalom**. The surrounding original DEVELOPMENT world remains. This deliberately isolates geometry but severely limits any claim of diverse *world-level* transfer: most decisive variation is the four masks, not the four background families.
- Goal A `(67,48)` and B `(46,64)` reverse at synthetic decision ticks **12 and 30**. Each policy sees only the current objective. When a goal is reached early, the environment externally idles until its next scheduled goal; this is **not** a native host action. The same initial state, target schedule, legal menu, and obstacle map is used across matched arms. The second switch is only observed if the agent has not stopped earlier.
- A **64 logical-move ceiling** is enforced externally using the existing synthetic `strict_step`. Protected/material painting is always illegal. Neither synthetic geometry changes nor goal replacements are proposed as native CyberSand interfaces.
- Planner comparator is exact only within **backtrack-prefix then unpainted empty-cell paths**, not arbitrary native state-space optimal control. It does not anticipate future goal reversals and cannot be treated as an optimal oracle.

## Results

| Controller | Final target completion | Planner searches | Neural-selected actions | Observed both switches |
| --- | ---: | ---: | ---: | ---: |
| Frozen neural only | 27/64 (42.19%) | 0 | 3,849 | 64/64 |
| **Fixed handoff at tick 24** | **36/64 (56.25%)** | 122 | 1,536 | 58/64 |
| Stall four steps, one search per stage | 25/64 (39.06%) | 68 | 3,074 | 64/64 |
| Neural STOP rescue | 27/64 (42.19%) | **0** | 3,849 | 64/64 |
| Prefix planner each stage | 40/64 (62.50%) | 192 | 0 | 64/64 |

**Effect size:** fixed handoff recovered a net **9/64 = +14.06 percentage points** against frozen neural-only (not the earlier +29.17 points). Repeated planner intervention produced +13/64 = +20.31 points. These are descriptive paired differences, not statistically independent 64-world samples. A STOP-triggered rescue never activated because this frozen policy did not propose STOP in any tested trajectory; this control does *not* establish a general trigger failure.

The effect is extremely layout-sensitive. Fixed handoff successes: opposing gates **11/16**, enclosed courtyard **2/16**, horizontal switchback **12/16**, protected slalom **11/16**. The more intensive restricted planner achieved 8/16, 8/16, 16/16 and 8/16 respectively. The courtyard is a particularly clear falsification of using a fixed tick-24 handoff as a general rule.

**Goal-order asymmetry** is substantial. For order A→B→A, fixed handoff reached 14/32 and full-stage restricted planning 8/32; for B→A→B, they reached 22/32 and 32/32. This reflects path dependence and painted history under scheduled retargeting; it is not evidence of model semantic preference for any named target.

### Integrity and resources

- **320** autonomous synthetic episodes; **16,633** exact legal action receipts and **12,308** rounded neural probability distributions, recomputed by in-run trace verification. All captured trajectories are archived with state, menu, action, receipt, and checkpoint identities.
- Four focused tests passed, including fixture separation, TRAIN/HELDOUT rejection, legal-menu/receipt/probability/switch tampering, and deterministic trace replay. Full archive verifier separately checks every episode and the archive SHA256 manifest.
- Campaign CPU wall time **29.580 seconds**, peak process RSS **98,828 KiB**, median per-episode policy choice latency approximately **0.098 ms** for neural-only. The planner's completed cached-action latency is not the cost of its first search; its search counts are reported separately.
- Two frozen local checkpoints identified by SHA-256 in the JSON ledger and immutable-independent ZIP. Checkpoints are **not** written to Git. Native CyberSand receipts: **zero**. No paid computation.
- The gallery `TOPOLOGY_GOAL_CHURN_TRAJECTORIES_2026-11-18.svg` plots actual per-action cursor transitions, with distinct colors for goal phases, terrain/material cells, protected cells, initial position, final cursor and target positions. No path smoothing or inferred steps.

### Limitations and next falsification

This controlled study is still an engineered synthetic world, not actual Workbench or native worm gameplay. Clearing the center makes four background families partly redundant, so the effective topology diversity is four. Fixed retargeting schedules and goal-conditioned Manhattan-distance features are engineered conveniences. The prefix planner sometimes executes an infeasible STOP after a stage change; it is not an optimal upper bound. No exogenous changes of materials were evaluated during episodes. Structural HELDOUT remains sealed. Actions' source-smoke/compile checks, if run later, cannot substitute for full local replay.

**Next:** split intervention decisions by topology and remaining reachable budget, then evaluate a lightweight *goal-switch-aware* trigger against an offline complete state-space comparator. Test moving materials and true native host observations only through separately approved read-only interface evidence; do not modify the game. Preserve negative results and counterfactual receipts on DRIV3R PR #1 only.
