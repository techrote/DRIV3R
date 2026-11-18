# DRIV3R — NATIVE-CEILING-CONTINUATION-001 (18 November 2026)

**Scientific verdict: negative for native-compatible 72/128-step evidence, while positive for genuine *synthetic* executed continuation.** Two newly trained tiny neural policy seeds executed 36/72/128 decisions with complete original action receipts. On all 24 ordinary+perturbed neural DEVELOPMENT episodes at 72 decisions, and all 24 at 128, the trace exceeded the **historical native M3 logical-move limit of 64**. The 64 limit was *not* enforced by the synthetic World; applying it as an analysis-only censor reduces every 72/128 neural trace to a 64-decision prefix. No game-side changes, native receipts, or claim of native 128-move acceptance.

## Experimental controls and source identity

- Source branch: `techrote/DRIV3R` Draft/HOLD PR #1, `research/import-consolidated-worm-labs`; exact remote HEAD inspected **`13893b492ca023ba13694c7416b704243c0f58e2`** before source preparation. Verified SHA-1 Git blobs of reused `world.py`, `policy.py`, `train.py` against the remote HEAD: `c7e33b7`, `c61573b`, `0e6da7d` respectively. Complete SHA-256 identities in evidence ledger.
- The original `World.menu(12)` and `World.apply` are unchanged. All offered menus retain `STOP`, optional `BACKTRACK` and the **non-native synthetic** `RELOCATE_START` when legal. Every applied action was accepted by the original fixture and replayed to the same state digest.
- TRAIN is four world families, seeds 1000–1999, three genuine 36→72→128 optimizer stages, 64 Adam steps/stage, two independent policy seeds 1051/1052, no paid compute. 6 checkpoint `.npz` files, none committed to Git. DEVELOPMENT is 12 geometries (8 ordinary + 4 trap-corridor), seeds 3000–3999. The two structural HELDOUT world families were **never opened**.
- Four controls per geometry and horizon: two trained neural seeds, deterministic teacher mixture, and persistent host-legal random. Each ran ordinary and legally perturbed conditions. Perturbations consist of an externally requested MOVE_E for the first six actions in trap corridors, plus alternative host-legal MOVE choices at positions 18, 45 and 80 where available. They are documented interventions, not policy outputs or hidden bypasses.
- Total **288 synthetic DEVELOPMENT episodes, 22,001 applied legal actions, and 11,242 actual neural probability vectors**. Direct source replay and independent probability re-inference covered all episodes. No rejected actions or edits of protected/material cells.

## Genuine decisions versus native-limit evidence

| Nominal decisions | Neural reached the synthetic horizon (ordinary / perturbed) | Neural traces exceeding 64 logical moves | Mean strict-64 prefix decisions |
|---|---:|---:|---:|
| 36 | 24/24 · 24/24 | 0/48 | 36.0 |
| 72 | 24/24 · 24/24 | **48/48** | **64.0** |
| 128 | 23/24 · 21/24 | **48/48** | **64.0** |

The 128-decision synthetic continuations are real sequential model choices and actual synthetic World transitions, not parameter-labelled horizon caps or copied terminal scores. However, **all** 128 neural traces become unsupported beyond step 64 when compared with the historical native logical-move ceiling. 128 synthetic steps are not 128 host-accepted native decisions.

At 128 ordinary, teacher reached 12/12 nominal horizons, neural 23/24 and persistent random 7/12. With legal perturbations, teacher 12/12, neural 21/24 and persistent random 6/12. This does **not** establish a better game policy: the teacher is privileged handcrafted guidance, random is intentionally untrained, and simple forward painting is not proof of complex goal pursuit. Neural 128 ordinary averaged 125.2 newly painted cells versus 120.8 in perturbed rollouts; those extra synthetic cells beyond logical move 64 cannot be credited as native-proven gameplay.

`RELOCATE_START` appeared **63 times overall** across the persistent-random baseline, and not in the neural/teacher runs. This synthetic action is a source-visible shortcut with a logical-move charge equal to the path length; the analysis-only censor correctly stops before an event that would cross the 64-move threshold. Do not describe it as a native CyberSand reset or shortcut.

## Qualification and custody

- Local focused tests: **4/4 passed**, including strict source partition, legal forced corridor entry, no mutation of the native-limit predicate, original host replay, and saved checkpoint inference.
- Separate opt-in verifier: all **288/288** episodes and **22,001/22,001** receipts independently replayed, all **11,242** neural probability vectors re-inferred from 6 restored checkpoint binaries, all referenced checkpoint hashes verified, and 64-logical-move prefixes reconstructed. Full original action/probability ledger and checkpoint tensors remain in a locally hashed ZIP, not Git.
- Bounded CPU-only campaign: **24.142 seconds** (environment-specific), peak RSS **159,404 KiB**, median measured action-selection latency **0.158 ms**. This timing includes feature generation and Python overhead, not native/SLM runtime latency.
- The six-panel SVG compares original 36/72/128 cursor histories on a single DEVELOPMENT pickets geometry, ordinary versus legally perturbed. Solid cyan traces show at-most-64-logical-move prefixes; orange dashed segments show synthetic-only continuation. Not a native rendering.
- Immutable GitHub Actions checkpoint artifact publication remains outstanding; a local ZIP is not an immutable research artifact. No new Actions workflow or routine-push training was installed.

## Decision and next experiment

**Do not use 72/128 nominal synthetic continuations as native-long-horizon acceptance.** Future qualification must separately report (a) executed decision count, (b) host-legal logical move count, (c) independently replayable action/observation receipts and (d) native acceptance boundary. Workbench/native changes are strictly outside this research branch. If a future game-side multi-epoch/continuation contract is desired, submit it as a read-only interface handoff proposal for owner consideration instead of modifying CyberSand.

For model science, the next defensible experiment should measure stateful goal changes, material avoidance and recovery **inside each explicitly host-bounded decision session**, with a deterministic task-specific comparator; it should not merely increase synthetic horizon labels. Current evidence says long *synthetic* continuation is possible, but native-compatible 72/128 autonomous control remains unproven.

**Ownership:** Only `experiments/driv3r/training/long_horizon/**` in DRIV3R PR #1. No mutation of DRIV3R main, CyberSand, Neurounner, their PRs or workflows. PR stays Draft/HOLD.
