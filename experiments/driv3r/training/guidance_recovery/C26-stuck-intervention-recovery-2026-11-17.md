# C26 — Intervention-time guidance after induced mistakes (negative replication)

**17 November 2026 · DRIV3R guidance_recovery laboratory · Draft/HOLD PR #1 only**

**Disposition: Reject C26-TRIGGER and C26-FULL. Retain frozen C12 as trained reference and untrained C18-A0 as the principal attestation control.** Neither policy satisfies the pre-registered per-condition recovery nonregression gate; no native game acceptance or provider promotion is claimed.

## Research question

Can a small trainable recovery head use **stuckness observed before a host-issued cue** to respond selectively, without obeying misleading advice, overwriting host permissions, or erasing personality? C25 tried post-cue executed history; the new intervention is deliberately delayed until *after* a controlled, host-applied mistake prefix. The baseline and all candidate policies receive exactly the same pre-cue state.

## Host information boundary and experiment

- The synthetic research host executes five attempted pre-cue steps in the `badturn` arm (two deliberately worsening legal moves followed by alternating baseline and nonprogressing choices), or a baseline-only `neutral` arm. Some episodes terminate early on success/STOP; the actual prefix and event count are recorded. This is controlled mistake injection, **not naturally occurring worm error**.
- C26 appends four *past-executed, current-goal-only* features to C25's `(10,36)` observation: trailing consecutive nonprogress count, recent observed-position revisit fraction, signed net goal-distance progress, and observed event count. The resulting menu is `(10,40)`. Neither hidden topology nor desired cue correctness enters inference. Pre-existing seeded trail records do not masquerade as new executions. Goal changes invalidate old events.
- An intervention trigger requires at least three observed steps and either at least three consecutive nonprogressing steps or at least one-third observed revisits. Under an active cue and the existing C18 historical-goal gate, C26-TRIGGER can modify **only the BACKTRACK logit** when the trigger is satisfied. C26-FULL uses the same head and features but omits the trigger condition. Both are residuals on frozen C18-A0; cue-off logits equal frozen C12 exactly. Training and inference never change the legal menu or protected material cells.
- Both heads are `12→16→1`, output bounded to ±0.85 logit, with stochastic legal-menu sampling at temperature 0.8. An AdamW group-balanced counterfactual-return objective uses 848 source-matched training menus (420 triggered), 1,696 host-executed forced-choice suffixes and 14 epochs/196 optimizer updates **per head** (392 updates total). Training advantages: 112 positive, 118 negative, **618 near-tied**. The two optimizer reruns for separate evaluation seeds produced byte-identical checkpoints; this establishes local deterministic repeatability only.
- Training sources use hairpin/staircase and original shortcut/bridge fixtures; evaluation uses withheld hook/zigzag and rotated/shifted shortcut/bridge geometry. Each goal/personality/arm receives no cue, weak 0.35 BACKTRACK or CONTINUE, and strong 1.0 BACKTRACK or CONTINUE. Cases cover both legitimate and deliberately misleading cues. The exact same disturbance prefix is reused between models in each case.

## Two independent held-out cohorts

Goal completions **under active advice only**, out of 768 and 960 episodes per model:

| Policy | Cohort A, 36-decision cap | Fresh-seed B, 40-decision cap |
|---|---:|---:|
| Frozen C12 | **606/768** | 746/960 |
| C18-A0 untrained attestation | 604/768 | **751/960** |
| C26-TRIGGER | 601/768 | **751/960** |
| C26-FULL | 601/768 | 747/960 |

C26-TRIGGER has four condition-level regressions in A and two in B against C18-A0. C26-FULL has two in A and three in B. The primary failure concentration was **CONTINUE advice**—including misleading CONTINUE when a reversed goal actually warrants retreat. A gain in one cohort/condition did not replicate in the other. Both candidates fail the pre-registered *zero regression in any scenario/strength group* requirement.

In the induced-stuckness subset, C18-A0 vs TRIGGER goals were 225/316 vs **221/316** in A, and 308/428 vs **308/428** in B. The time-to-first net distance reduction was usually **one step** for every policy and therefore largely nondiscriminative in this fixture; mean steps to successful completion under triggered guidance were 9.03 vs 9.23 in A, and 8.31 vs 8.49 in B. These are synthetic per-episode diagnostics, not native game physics timings.

## Personality and refusal/legality

A separately reconstructed **same-state policy audit** evaluates all four models on identical intervention menus, rather than comparing the number of drawing opportunities on divergent trajectories. Under active triggered cues, the probability mass assigned to personality-favoured INK actions was:

| Model | A (64 opportunities) | B (100 opportunities) |
|---|---:|---:|
| Frozen C12 | 31.814% | 32.918% |
| C18-A0 | 30.428% | 31.238% |
| C26-TRIGGER | 30.156% | 30.818% |
| C26-FULL | 30.463% | 31.215% |

TRIGGER marginally reduces personality-favoured INK probability relative to A0 in both cohorts. No assertion of personality preservation beyond those matched observation menus is justified. The active-cue decision entropy remains nonzero and actions remain sampled; the learned head is not a deterministic obedience override. The more effective diagnostic is the **failure to improve recovery**, not an absence of stochasticity.

Across **8,640 synthetic held-out episodes** (3,840 A + 4,800 B, including all guidance arms and models), no raw illegal top-choice commands, host-rejected actions or protected-material modifications occurred. Each candidate's no-guidance action sequence and ink hash matched C12 on all 192 (A) + 240 (B) paired cue-off cases. Host legality is enforced independently by the synthetic executor, so this demonstrates bounded synthetic safety, not real CyberSand safety certification.

## Provenance and repeatability

- Tested against the C25 source blob `a4336183ebeb73ca2c821a8c1fcddaad18048b60`, confirmed byte-for-byte against DRIV3R PR #1 during preparation. C26 source, checkpoint weights, exact model prerequisites, metrics, every held-out host execution, matched policy audit, replay-verified SVG gallery and a SHA-256 manifest are in the accompanying **external research archive**. Model weights and full trajectories are intentionally absent from Git history.
- C26-TRIGGER checkpoint SHA-256: `8c91c68b4190901cb06d9cd095caca4681f00f68a8d1e56adf36aedb154f1290`.
- C26-FULL checkpoint SHA-256: `656fd9e6948487687d792b4ca8607a7082d4af863f4e8134d5a0b3a01a78ec65`.
- 174 local source-level regression tests passed. Both evaluation cohorts had source-identical optimizer outputs. Full exact-HEAD scientific CI reproduction and immutable GitHub Actions checkpoint custody remain outstanding. This synthetic contract is not a replacement for CyberSand's canonical host ABI.

**Next research:** stop optimizing near-tied one-step BACKTRACK advantages. Construct scenarios with genuinely delayed recovery, longer traps, and consequential branch selection where a mistake must be identified over several verified host steps. Separate *detecting a mistake* from *choosing whether to obey a recovery cue*. Apply the same no-guidance, personality, illegal-action and adversarial-advice nonregression gates before any policy promotion.
