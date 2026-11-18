# C30 — Host-executed multi-step probe: preliminary result (17 Nov 2026)

**RESEARCH / DRAFT-HOLD / NOT QUALIFIED / NOT PROMOTED.**
Owner-authorised laboratory: `experiments/driv3r/training/guidance_recovery/**` in techrote/DRIV3R PR #1 only.

This is a **provisional evidence ledger**, **not** a source-complete published experiment. The locally executed source and checkpoint have not yet been committed to DRIV3R or placed in immutable Actions custody. It must not be treated as reproducible from this commit alone.

## Experiment

C30 relaxes C29-GUARD's initial conservative veto and permits bounded, stochastic, legal forward probes following an actual host-issued guidance cue. An 8→8→1 neural evidence gate is trained to recognize host-observed exhaustion of a branch. The frozen C28 persistent rollback and C29 probe heads are reused; all actions are drawn from freshly authorized host menus. A probe interrupted by stochastic BACKTRACK is released unless a directly observed one-step dead end remains under investigation.

Training: 216 host-observed local-prefix menus (48 positive observed-failure labels), 336 CPU AdamW updates, BCE 0.6836643 → 0.0014378. Two independent same-seed optimizer executions produced byte-identical safetensors, SHA-256 `32f82ad4078ec48ced02bfacb7305dc18a63cb862121cefe579f6fd6e4baef0c`.

**Causal limitation:** the training labels are largely separable from local legal-menu features. Improvements cannot be attributed uniquely to learned long-horizon reasoning; changing the legal probe/release policy is a substantial intervention.

## Paired withheld two-step challenge

Training fixture indices 0–11; evaluation indices 12–13, two independent random-seed cohorts, 90-decision horizon. 1,920 matched complete trajectories across four controls/candidates:

| Policy | Cohort A guided goals /192 | Cohort B guided goals /192 | Two-step decoy A /32 | B /32 |
|---|---:|---:|---:|---:|
| C18-A0 | 90 | 88 | 0 | 0 |
| C29-GUARD | 146 | 146 | 0 | 0 |
| **C30-ADAPT** | **163** | **166** | **16** | **20** |

No condition-level goal-completion regressions against C29-GUARD or C18-A0 in either of these two cohorts; cue-off action/ink hashes matched C12 in 96/96 cases. No host rejections, raw illegal top choices, or protected-material changes.

## Independently constructed three-step U-corridor

New withheld map family: an actual host-executed 21-step detour to a goal seven Manhattan cells away; a false branch improves distance for three moves and then dead-ends. The matched legitimate tunnel reaches the goal in seven steps. Offline BFS verifies both paths and is never an inference feature.

640 matched trajectories across two independent cohorts:

| Policy | A goals /64 | B goals /64 | 3-step decoy A /32 | B /32 | Real tunnel A /32 | B /32 |
|---|---:|---:|---:|---:|---:|---:|
| C18-A0 | 24 | 20 | 2 | 2 | 22 | 18 |
| C29-GUARD | 24 | 20 | 2 | 2 | 22 | 18 |
| **C30-ADAPT** | **45** | **39** | **22** | **20** | **23** | **19** |

**Hard nonregression gate FAILED:** the candidate lost one success in true-tunnel weak-BACKTRACK cohort A and one in true-tunnel weak-CONTINUE cohort B. Aggregated improvements do not erase those safety regressions. All 32/32 OOD cue-off action/ink hashes remained identical; zero synthetic host rejections or illegal raw top choices.

## Provenance and outstanding

- C30 executable source SHA-256 `5eeae396217a24b60f3d08b0d436300d1205fc41ba80257559dfabb47999dba1`.
- Independent U-corridor validator SHA-256 `aa9d533d501e9a4ffdc418c3a8c7e733e808cb1f3f590b39e73f459097cc88c8`.
- Locally verified full metrics SHA-256 `821f59da74ee2371ec390bb41ec5ee6556d399f4426b1d7e3b4c2c59538e5d6c`.
- Cross-geometry metrics SHA-256 `372222c114d3b2b3271ac20fc31eae5538bd233a549f8b94a0cd1dcc13cb822c`.
- Five new targeted local tests passed; the replay-checked SVG gallery contained eight direct comparisons.
- Scientific confidence remains limited by synthetic host fixtures and historical prerequisite checkpoints. Native CyberSand validation and active-guidance personality audit remain outstanding.

**Disposition:** retain C30 as experimental evidence only. Do not promote. Source/checkpoint custody and full source-matched repository qualification remain explicit blockers. No action is authorised against DRIV3R main or any external repository.
