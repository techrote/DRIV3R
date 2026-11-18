# DRIV3R POST-SWITCH-REGRET-001 — common-prefix, cross-suffix causal audit

**Status:** executed negative result / **not** a new trained model. Synthetic DEVELOPMENT research only; no native CyberSand validation or structural heldout. The DRIV3R Draft/HOLD PR #1 is the sole publication target.

## Question and predeclared method

Does goal-switch curriculum training improve the *first decision immediately after a target changes*, compared with a frozen checkpoint and a matched fixed-goal sham, on **identical states**?

Restore the six checkpoints from GOAL-SWITCH-CURRICULUM-001: seeds **619 and 620**, arms **frozen**, **sham/fixed-goal** and **switch-trained**. Their training contained equal 36 → 72 → 128 curricula. Construct 32 actual autonomous **frozen-policy** DEVELOPMENT prefixes (two seeds × four terrain families × normal/trap challenge × two target variants), without forced actions or hidden state changes. At decision indices **32, 72 and 112**, hold a single true synthetic host state fixed. Query all three rankers on that *same* observation and exact legal action menu. For each offered action, force exactly that first legal action on a deep copy and then let each of the three checkpointed suffix policies select every remaining action for a 16-decision window, with the new target held constant. This makes 3 suffix controllers × every available legal first action, without changing the action set itself. A pre-existing painted target is excluded from goal-regret estimates but retained in accounting.

Predeclared measures: 16-step final Manhattan distance to the **new** target, whether/when the target is reached, recovery use, first-action softmax probabilities, and lexicographic first-hit/final-distance oracle. **Regret** denotes the final distance of the selected first action minus the minimum final distance across counterfactual first actions *under the same suffix*. The oracle is diagnostic hindsight, not a deployable controller or an honest model score. Model comparisons use identical prefixes and a common suffix, and a descriptive 2,000-replicate paired bootstrap clustered by seed/world/challenge/variant.

## Main DEVELOPMENT findings

- **96 reachable switch states; 95 eligible** (one next goal had already been painted). **1,563** actual counterfactual forks and **20,122** source-replayed host-legal decisions. Six original checkpoints were restored; 98 local evidence files including their full per-decision receipts were SHA-256 manifested.
- The switch-trained policy selected the same first action as the frozen policy at **82/95** eligible switches. Fixed-goal sham and switch-trained selected identical first actions at **94/95**, a strong indication that the curriculum changed very little about post-switch choice.
- With matched suffixes, the switch-trained choice reduced 16-step final distance by **0.1158 cells on average** versus the frozen choice. The cluster-bootstrap descriptive 95% interval for the *switch-minus-frozen final distance* is **−0.4492 to +0.1597** cells. This does **not** establish a meaningful gain; the interval spans zero.
- Across all suffixes, mean distance regret against the full synthetic menu oracle was **9.2596** cells for frozen and **9.1439** cells for switch-trained. Under its own suffix the frozen model had **74/95** nonzero-distance-regret cases and the switch model **77/95**. Only **20/95** selected first actions reached the newly assigned target within the 16-step window under either policy's own suffix, although some alternative first actions did better.

### Important confound: synthetic `RELOCATE_START`

The full-menu, frozen-suffix lexicographic oracle chose `RELOCATE_START` in **63/95** eligible switch states. In **34** of those cases, the history length exceeded 64 cells. The test harness models relocation as a return to the history root, incrementing its synthetic `logical_moves` counter by the historical path length **in one action**. It is not a demonstrated executable native CyberSand movement and might conflict with native command ceilings; *do not translate these oracle successes into native training labels*.

For this reason a second, **analysis-only** comparison excludes `RELOCATE_START` from the candidate oracle (without suppressing it from any model's legal menu or changing any executed state transition). The frozen controller's self-suffix distance regret falls from **8.6737 to 2.2947** cells, showing that most of the apparent missed opportunity came from the synthetic relocation shortcut. The switch controller's non-relocation regret is **3.2234** cells for the **94** cases where its own first action was not relocation. In the frozen suffix, the no-relocation lexicographic oracle still selected **BACKTRACK in 9 cases**, but the actual frozen policy selected it once. That is a narrower, real synthetic research target, not evidence that all backtracking should be encouraged.

## Qualification and provenance

- **5 focused tests passed**: split guard, real host-menu and probability checks, deterministic counterfactual replay, fixed oracle semantics, and the synthetic relocation accounting caveat.
- Independent verifier **reconstructed all 96 common-prefix switch states** from restored checkpoints and independently replayed **all 1,563 forks and 20,122 decisions**. This included every legal menu, neural probability vector, action receipt, material edit, cursor movement and final state hash. Source SHA-256 fingerprints were checked against eight exact dependency files, matched to their Git blob identities on the DRIV3R branch before modification.
- CPU campaign approximately **19.9 s**, peak RSS **134,680 KiB**. Model-only probability latency measured over 3,072 real host-menu matrices: median **0.007821 ms**, p95 **0.008853 ms**; excludes world stepping, serialization and native rendering.
- Full per-decision JSON, six checkpoint binaries, SHA-256 manifest, and original six-panel 2D trajectory gallery are in the local verified evidence archive. The compact repository ledger includes checkpoint/source hashes, policy probability vectors and a case-by-case fork-outcome matrix. Raw checkpoints are **not committed to Git**.

## Interpretation and decision

**Negative for the intervention:** the existing switch-goal curriculum offers no demonstrated action-quality improvement. **Positive for diagnosis:** common-prefix evaluation identifies which choices differ, separates training from suffix-controller confounding, and reveals that the synthetic relocation law can dominate a naive hindsight label generator. These worlds remain familiar procedural DEVELOPMENT families, not a structural or real-native generalization test.

**Next bounded experiment:** construct fresh TRAIN recovery-goal cases in which ordinary *BACKTRACK* (not root relocation) wins under multiple independently trained suffix controllers; incorporate an explicit native-contract movement-limit comparator before proposing labels. Preserve the existing DEVELOPMENT and sealed HELDOUT partitions. Do not train on this frequently examined DEVELOPMENT set.

No paid GPUs/APIs, no new workflow, no changes to CyberSand/Neurounner/native code, no main update, no merge or PR promotion. Source and compact evidence belong solely to DRIV3R's existing Draft/HOLD PR #1.
