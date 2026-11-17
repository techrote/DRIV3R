# C29 — Host-verified probe and conditional reconsideration

**Date:** 17 November 2026. **Disposition:** promising *synthetic-only* guarded checkpoint; **not promoted**. Single DRIV3R Draft/HOLD PR #1, `experiments/driv3r/training/guidance_recovery/**` only. No CyberSand, Neurounner, or default-provider writes.

## Hypothesis and experimental controls

C28-VETO recovered 45/64 held-out long reversals but **0/64** deceptive one-step spurs, because the conservative veto latched permanently when a fresh locally promising movement existed. C29 tests whether the worm may **actually execute bounded legal probes**, then reassess misleading progress without turning host advice into forced action sequences. The same C28 temporal BACKTRACK head remains frozen, with separate no-guidance C12, C18-A0, and C28-VETO controls. All policies are evaluated with identical world/personality/guidance/seed pairings, a 72-decision horizon, and fresh held-out three-row geometry.

## Real optimizer and inference

A four-input probe head and four-input reconsideration head were trained with **54 CPU AdamW updates** over **96 matched menus** and **64 real host-executed legal probe steps** from two-row training worlds. Inputs comprise present host-legal fresh goal-improving actions, counts of such actions, verified trail detour/stuckness and executed probe depth. Supervision is whether a legal probe exists and whether the *next executed observation* still contains fresh improvement; it is **not** trained against full-horizon counterfactual returns. Loss sum per epoch fell from **2.049045 to 1.063946**. Both heads reached 100% accuracy on those very small, *in-sample* training labels; this is **not** independent generalisation evidence.

Two inference configurations share the same trained `.safetensors` head and immutable C28 temporal head:

- **PROBE** gives a bounded stochastic boost to a single freshly legal goal-improving candidate for up to three steps, then reconsiders. At an observed dead-end it can activate the long-horizon stochastic BACKTRACK option; after three forward improvements it returns to the frozen C18-A0 policy. Its exploratory candidate probability is capped at 0.88, and the C28 persistent-backtrack probability at 0.985. Neither candidate is directly executed by the head; `World.apply` revalidates each host-legal action.
- **GUARD** allows this probe only when the **existing host candidate feature 22** reports 75% adjacent obstruction, the goal is at least 20 verified trail steps behind, the detour ratio is at least 2.5, and the original host cue is just issued. Otherwise it conservatively abstains, preserving C18-A0 behaviour for a plausible shortcut. This additional hand-authored sensor gate is *not* itself a learned capability and is a deliberately narrow hedge against PROBE's shortcut regressions.

No host cue means exact frozen-policy behaviour. Explicit advice cancellation and goal changes terminate active recovery. Inference uses only already-executed history and current locally host-advertised legality/adjacency, not unseen topology, BFS distance, protected-grid omniscience, hidden scenario labels, or the offline success oracle. Paths, protected cells, legal menus, and painting changes are audited through the synthetic host executor.

## Independent held-out evaluation

Two fresh-seed cohorts, **160 guided episodes per model per cohort**, with 40 additional no-cue controls per model per cohort (2,000 policy episodes total). Training uses two-row geometry; evaluation holds out three-row geometry. Reused C28 control records were **source-matched by seed and prefix hash**; the tested C29 policies were newly executed. Results below show completed guided goals, separately from no-cue controls:

| Policy | Cohort A | Cohort B | A/B one-step decoy | A/B legitimate tunnel |
|---|---:|---:|---:|---:|
| C12 | 84/160 | 86/160 | 0/32 · 0/32 | 24/32 · 23/32 |
| C18-A0 | 85/160 | 86/160 | 0/32 · 0/32 | 24/32 · 23/32 |
| C28-VETO | 109/160 | 107/160 | 0/32 · 0/32 | 24/32 · 23/32 |
| C29-PROBE | 123/160 | 122/160 | 23/32 · 23/32 | **15/32 · 15/32** |
| **C29-GUARD** | **132/160** | **130/160** | **23/32 · 23/32** | **24/32 · 23/32** |

**Replicated gain on one-step decoys:** GUARD recovers **46/64** versus C28-VETO's 0/64, while exactly matching C18-A0/C28-VETO tunnel completion. Across 20 guidance/outcome/strength groups in each cohort, **GUARD incurred zero condition-level goal-completion regressions** against C18-A0. PROBE recovered the same 46/64 one-step decoys but regressed on **3 tunnel groups in A and 4 tunnel groups in B**. Both new variants retain 80/80 exact no-guidance action and ink hashes against C12 in standard fixtures. All protected-material SHA checks, host-executed legal moves and raw-illegal-top-choice checks passed (zero host rejections and zero raw illegal top choices). Maximum measured persistent BACKTRACK probability: 0.985, leaving alternatives stochastic.

### Important hidden weakness in the previous challenge

The C28 one-step decoy was **already distinguishable from a real tunnel by a current host-supplied sensor**: adjacent-blocked fraction of the legal fresh-progress candidate was 0.75 for the one-cell dead end versus 0.50 for the tunnel. GUARD exploits this precise *local* observation. This proves usefulness of modest local evidence gating, **not** robust learned deception recognition; the apparent gain must not be mistaken for recovery from genuinely ambiguous multi-step traps.

## Independent adversarial two-step decoy

To test that limitation, a new **withheld, host-legal two-step false-progress spur** was constructed on odd withheld rotations. Both the actual tunnel and the first step of the two-step decoy advertise **identical 0.50 adjacent-blocked local sensor values**. The second step is a real legal movement which becomes a dead end; a 28-step legal goal retrace remains available in every fixture. Offline BFS is used solely to verify the fixture's shortest legal path; no BFS truth enters policy inference.

| Model | Held-out cohort A | Fresh-seed cohort B |
|---|---:|---:|
| C12 / C18-A0 / C28-VETO | 0/32 | 0/32 |
| C29-GUARD | **0/32** | **0/32** |
| C29-PROBE | **8/32** | **11/32** |

**Negative result:** The safe GUARD does not transfer to a genuinely ambiguous two-step trap. The more exploratory PROBE demonstrates **19/64** recoveries, but sacrifices tunnel success, failing the required nonregression gate. A better policy must preserve shorter paths while continuing evidence collection long enough to detect deeper deception. There is no qualified general solution.

## Integrity and disposition

Both CPU AdamW replicas produced a byte-identical trained checkpoint, SHA-256 `ff8a6ba370865f948f347a2d6c66c37b07f62b21fc97f33268c67ca1ce51c56a`. The C28 frozen checkpoint and all seven prerequisite model hashes are preserved in machine-readable receipts. All 5 targeted regression tests passed, including executed path legality, obstruction-feature confound, no-cue exactness, optimizer updates and cue cancellation. The 16-panel SVG/HTML gallery reconstructs only actually recorded executed paths. Full raw matched trajectories (standard plus two-step adversaries) are preserved in the external hash-manifest archive, **not** committed as large Git data blobs. Model binaries are not committed to Git.

**Disposition:** preserve C29-GUARD as a scoped, promising *synthetic-control result*, and C29-PROBE as the explicit high-recovery/high-regression control; **do not promote either**. Active-guidance painting/personality preservation, independent broad natural geometry, native CyberSand ABI compatibility, and immutable GitHub Actions checkpoint custody remain unqualified. No paid compute was used.

**Next falsifiable experiment:** train an uncertainty-aware probe controller on two- and three-step false branches as well as real tunnels, using host-executed return labels rather than just immediate legal-degree labels; demand condition-level tunnel parity and replicated deep-spur recovery simultaneously.
