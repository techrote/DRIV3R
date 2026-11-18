# C28 — Executed-trail temporal recovery option

**Date:** 2026-11-17  
**Disposition:** research milestone; **VETO is a qualified synthetic control, not a promoted game model**.  
**Repository scope:** `experiments/driv3r/training/guidance_recovery/**` in the one long-lived DRIV3R Draft/HOLD PR #1. Never modify the CyberSand game, its host ABI, production settings, or either historical repository.

## Question and falsifiable hypothesis

C27 established 0/64 goal completions in longer deep-reverse traps, despite an offline legal BACKTRACK witness. Its nine-turn advice expires long before the 26–28 steps of retracing required. Can a small trained **persistent stochastic intention** preserve a host-attested recovery objective beyond a cue's TTL, while refusing the same instruction when local legally traversable geometry offers a plausible shortcut?

C28 specifically tests persistence versus a TTL-only option, unrestricted persistent commitment, and a conservatively vetoed persistent option. It is *not* general navigation planning. Backtrack preference is learned, while invocation and conservative abstention are evidence gates.

## New evidence quality: actually execute the prior trail

Previous C27 fixtures populated `history` and `inked` fields directly. C28 derives the same protected-cell geometry but starts at the other end and **executes each INK movement through `World.apply` and host-recomputed legality** before issuing the new goal. The prefix is synthetic and scripted, not produced by an independently trained worm. Every target is reached by a host-legal recorded event. Material protection is fixed before executing that prefix; no model writes protected material.

Training uses a two-row switchback: a **26-step** shortest legal retrace to a goal six Manhattan cells away. Withheld evaluation uses a three-row switchback: **28 legal moves** versus eight Manhattan cells. The held-out *tunnel* has the same history but a real eight-step forward shortcut; the *decoy* offers a tempting, legal, locally improving **dead end** while retaining a 28-step shortest path. Additional near/short history goals test unwanted interventions. All five geometries are rotated/mirrored; outcome and guidance assignments are matched by seed.

Only host-observable executed trail and candidate legality, goal displacement, cue sign/strength and short-term progress are exposed to the option. Oracle BFS is used **only for offline test fixture audits**. No unseen protected-grid map or ground-truth route enters inference. `d_common` remains untouched and is not represented as the canonical CyberSand ABI.

## Real optimizer and heads

The new 5→1 torch linear head adjusts only the legal BACKTRACK logit. Supervision comprises **2,080 host-executed retrace decision menus**, using an explicit *soft* backtrack target (0.985), not a hard action script. Two CPU AdamW repetitions each performed **352 updates** over 16 epochs. Measured epoch objective went from **0.155850** to **0.148823**. Both resulting `.safetensors` files were **byte-identical**, with SHA-256:

`c84ac73d8738783597d4dbd9800730c1e955e902a6b86b829c75c95902760632`

At inference the option recomputes the host-legal action menu on *every* step. It adds a bounded learned BACKTRACK increment while the same goal remains on the host trail. The cue may expire without cancelling the intent, but explicit cue cancellation, goal change, missing history/legality, or bounded intent age terminate it. **At least 1.5% action probability remains on other legal choices** at each active-option decision. STOP, painting, and host material permissions remain unchanged in the underlying policy; only BACKTRACK scoring is modified.

The `VETO` rule refuses to create a persistent option if the first host-legal view contains a fresh forward move reducing Manhattan distance. Abstention is episode-long until another cue; this is intentionally conservative and sacrifices recovery from some genuinely misleading dead ends. `PERSIST` lacks that veto; `TTL` cannot survive cue expiry.

## Matched held-out results

Two independently seeded cohorts; 72-decision post-cue horizon; **2,000 total policy episodes**, including cue-off controls. Each guided cohort has 160 episodes per model, including 32 deep-reverse episodes. Guidances are no host cue, weak or strong BACKTRACK, and weak or strong CONTINUE, crossed with low/high local soft hints and opposite personalities. Only the two-row training geometry is used in optimization.

| Policy | A guided | B guided | A deep-reverse | B deep-reverse | A/B tunnel completion |
|---|---:|---:|---:|---:|---:|
| Frozen C12 | 84/160 | 86/160 | 0/32 | 0/32 | 24/32 · 23/32 |
| C18-A0 | 85/160 | 86/160 | 0/32 | 0/32 | 24/32 · 23/32 |
| TTL-only trained option | 78/160 | 76/160 | 0/32 | 0/32 | 17/32 · 12/32 |
| Persistent option, no veto | 129/160 | 129/160 | 24/32 | 21/32 | 22/32 · 21/32 |
| **Persistent option + VETO** | **109/160** | **107/160** | **24/32** | **21/32** | **24/32 · 23/32** |

**First replicated nonzero long-horizon trap recovery:** the guarded policy completed **45/64** withheld deep-reverse episodes versus **0/64** for the frozen trained baseline and C18-A0. Across all guided conditions VETO completed **216/320** versus C18-A0's **171/320**, a gain of 45 completions. This result is conditional on the deliberately narrow, host-attested corridor family.

The need for true persistence is independently demonstrated: TTL-only recovered **0/64** deep reversals. Unrestricted persistence sometimes completes false-cue tunnels but incurs dramatic delay: in completed tunnel episodes, **42.0 and 46.0 mean moves** versus C18-A0's **15.33 and 14.52**; it also regressed in two tunnel condition-groups per cohort. VETO exactly preserves C18-A0's tunnel and dead-end-decoy action/ink hashes on all 64 guided matched pairs of each topology across both cohorts. Its own mean tunnel steps match C18-A0 exactly. The cost is **0/64 decoy recoveries** for VETO; unrestricted PERSIST recovers 46/64 of those genuine traps. This **false-negative abstention** is an important negative result, not a victory in general-purpose recovery.

## Safety, stochasticity and personality constraints

- **Zero host-rejected commands**, **zero raw illegal top choices**, **zero protected-material modifications** across all evaluated policies and recorded traces.
- VETO's host-cue-off action and ink hashes matched C12 on **80/80 matched episodes**. An additional 20 independent **zero-local-hint, zero-host-guidance** episodes per model also matched C12 exactly; no changes are made to unprompted drawing personality.
- In VETO's guided recovery option, **657 of 2,750 option-exposed actions were not BACKTRACK**. The measured BACKTRACK probability never exceeded 0.985. This remains stochastic policy selection, not direct action replay or an instruction to blindly obey the host.
- On active misguided guidance with a fresh forward route, VETO abstains and preserves the C18-A0 policy byte-for-byte. **Personality preservation during active, branch-rich recovery remains unproven**: retracing narrow corridors leaves very few meaningful ink/turn opportunities. Cue-off parity is necessary, not sufficient.
- Eight targeted regression tests passed, including host-executed history, corridor and decoy oracle audits, cancellation and goal-change abort, entropy floor and unguided exactness. Torch 2.10.0+cpu; NumPy 2.3.5. `train_inputs_sha256=708fdb24f54c0c6106ad27b7afdad6987e28a8da916a92dfb9ce7e80bbc3cbb7`.

## Adversarial findings and limits

1. The local fresh-step veto works for the simple forward tunnel, but it **refuses needed help** in one-step false-progress dead ends. Repeated failed local probes followed by a *new* verified intervention would be a meaningful next scientific test; do not simply remove the veto.
2. Retrace feasibility depends on an executed trail with a goal actually on the history stack. Other long-horizon planning, off-trail recovery, dynamic material simulation, true host-issued recovery semantics, and third-party physical/natural levels remain unvalidated.
3. The data are synthetic. Creating the initial path via host execution improves the evidence, but does not prove a natural worm would independently construct that path. The tiny head is supervised on successful host-executed paths; **C28 is not on-policy reinforcement learning** and does not prove novel route discovery.
4. The source and test data are committed in the one DRIV3R Draft/HOLD PR; model binaries remain outside Git in the hash-manifest archive. **Immutable GitHub Actions checkpoint custody and native CyberSand acceptance are still outstanding**. Source-smoke CI is not scientific reproduction.

**Research disposition:** keep C12 as preferred *trained* baseline and C18-A0 as primary untrained control. Preserve C28-VETO as a promising *external, scoped research checkpoint* with strict condition-level nonregression on these exact held-out fixtures. **Do not promote to gameplay or claim personality acceptance.** The next run should try re-intervention after verified decoy failure, with fully independent branch-rich geometry and active-guidance painting opportunities.
