# FT-022 — Sequential legal-world transfer of FT-021 synthetic policies

**16 November 2026. Draft/HOLD research only.** Eight genuine Adam-trained 20×24×1 NumPy policy checkpoints. This is an **external synthetic 16×16 host**, neither native CyberSand nor a pretrained SLM. Run explicitly on local Linux CPU. No workflows, game-side changes or paid compute.

## Custody and reproducibility boundaries

- Executable ft022_sequential_transfer.py SHA-256 4ad3aaf182a5e70e8d31165d6ad505003f11af67ab7a8302275a8b5e6098fc82.
- Five source tests in test_ft022_sequential_transfer.py, SHA-256 68df635406d884aeb7f45dd4d89548f80ab88ab07bb14d594b73e0cbd28cb62d, passed locally. Exact source-hash evidence and parent SHA guard are enforced.
- Parent source: FT-021's original local ft021_local_original_evidence.zip, SHA-256 719d651596d03f60a54d6a557dbd7111ba7cd6ffca46d328345770c81f30cdfb; four original 900-update checkpoints loaded from its ft021_long/one_resident/ directory and checked against individual SHA-256s embedded in the executable. The FT-021 parent ZIP is **not an immutable GitHub Actions artifact**. Its preservation and availability on other runners are not guaranteed.
- Original FT-022 local evidence ZIP SHA-256 f2c461aa37e949024eaa3ebebc430e2abc8f6621cce2b5560ec81bb0b2866cb0. 58/58 ZIP entries and 57/57 SHA256SUMS entries checked. 8 new checkpoint binaries, exact JSON report with all 288 episodes, and 24 replay-checked trace JSON/SVG pairs. This ZIP is **only a local session artifact**, not a hosted Actions artifact. Do not infer hosted reproducibility.
- Each of four niches ran 360 Adam updates in each of two continuation arms, totaling **2,880 actual updates** and **eight new unique checkpoints**. No optimizer training of full SLMs. Initial FT-021 weights were reloaded with fresh Adam moments, not an exact continuation of their earlier optimizer state.

## Fixed controls, host legality and disjoint worlds

- Three arms: frozen, route (shortest legal two-goal demonstrations), and route_recovery (same source world seeds, with scripted detours and subsequent legal corrective demonstrations). Same 360-update budget for both trained arms, but not the same per-arm example distributions; recovery augmentation adds and perturbs demonstrations. Both use the same sixteen training-world seeds per niche.
- 96 **matched held-out world identities per arm** (4 niches × 2 domains × 12 worlds), 288 autonomous evaluations. Independent split keys for training and each held-out domain. Every selected world has two reachable spatial goals. Harder guarded heldout adds protected barriers and a higher obstacle density.
- The host legal menu is 4-way cardinal movement, RETRACE **only to the actual immediate trail parent**, and STOP. STOP is controlled by an exact final-goal host rule. First-goal arrival changes the goal without ending the trajectory. The host checks no protected/blocked cells were painted. Original first trajectory per niche/domain/arm includes action-menu, before-state and after-state SHA-256 receipts and deterministic replay verification. 100-decision ceiling per episode.
- The 20 features are a **synthetic approximation** of FT-020/21's feature names; the feature representation, authoring rules, materials and RETRACE are not verified against CyberSand's authoritative action/observation ABI. No native runtime claims.

## Results (same 96 held-out worlds per arm)

| Controller | Final-goal completions | Goal switches reached | Mean painted cells | Mean RETRACE actions |
|---|---:|---:|---:|---:|
| Frozen FT-021 | 5/96 | 18 | 7.44 | 33.00 |
| Direct legal-route Adam | **39/96** | **55** | **17.11** | 24.64 |
| Route + recovery detours | 22/96 | 39 | 12.55 | **34.90** |

Matched route-versus-frozen: **34 wins, 0 losses, 62 ties**. Route-versus-recovery: **19 wins, 2 losses, 75 ties**. These are deterministic selected-seed comparisons, **not an independent population estimate**.

| Held-out family | Frozen | Legal-route | Route + recovery |
|---|---:|---:|---:|
| Open, 48 worlds | 4 | **24** | 15 |
| Guarded, 48 worlds | 1 | **15** | 7 |

All three controllers were evaluated with the same exact-goal STOP rule, removing the previous terminal-predicate confound. Neither trained arm introduced illegal painting in the original run. The recovery arm retraced more often, but **lost 17 final-goal completions versus ordinary legal-route training** and covered fewer cells; the recovery curriculum should not be treated as an improvement. These drawn trajectories are not aesthetic-quality evaluations.

## Research decision and hold gates

The strongest result is transferable **sequential behaviour** from legal route demonstration into four independently trained tiny policies, with held-out goal-switch and obstacle/layout changes. The negative result is that naïve recovery detour rehearsal increases retracing but damages completion. This is not an FT-019 checkpoint continuation, and cannot establish native CyberSand game competence.

Next: matched training-world count, additional fixed heldout seeds, trajectory-style action legality against authoritative read-only CyberSand evidence, a recovery-only intervention at dead-end states rather than arbitrary detours, and immutable hosted checkpoint/artifact custody using only an existing minimal opt-in DRIV3R mechanism if one is available. Do not create a new workflow per experiment, and do not run training automatically on push/PR.

**Repository policy:** research files and ledger only inside experiments/driv3r/fleet_training/; DRIV3R PR #1 remains Draft/HOLD; no CyberSand/Neurounner mutations; no merge or Ready transition.


## FT-022B — fresh-seed guarded validation (same source/checkpoints, no retraining)

After the original run, four frozen and eight trained policies were evaluated against **96 additional guarded-world seeds** (four niches × 24 worlds), all disjoint from the original 96 world identities. These reuse the *same synthetic guarded generator*, so this is a fresh-seed validation, **not a new environment family**. The second check involved **288 further matched autonomous episodes, zero optimizer updates**, and no game-side writes.

| Policy | Final-goal completions | Goal-switch arrivals | Mean painted cells | Mean RETRACE actions |
|---|---:|---:|---:|---:|
| Frozen FT-021 | 6/96 | 16 | 6.36 | 33.59 |
| Direct legal-route | **31/96** | **51** | **14.72** | 27.48 |
| Route + scripted recovery | 19/96 | 36 | 10.80 | **36.38** |

On these matched worlds, direct-route versus frozen: **25 improved, zero regressed, 71 tied**; direct-route versus recovery: **15 improved, three regressed, 78 tied**. The principal FT-022 result survives this fresh-seed check: legal-route continuation substantially improves two-goal completion over frozen FT-021, whereas naïve recovery detours produce more retracing but lower goal completion.

The full local FT-022B audit JSON SHA-256 is **6f677edb648e1e66718348e1043608397bc2694f35a84e642e5507b8334b47e9**. It includes all 288 episode outcomes, original eight child-checkpoint hashes, policy/source identity and the explicit independent seed label UNSEEN-VALIDATION-2; this audit remains a local session artifact and is **not an immutable hosted artifact**. Legality and protected-cell assertions executed in all episodes, but independent model-logit recomputation and native CyberSand validation were not performed. Neither FT-022 nor this extension resolves the hosted artifact-custody HOLD gate.
