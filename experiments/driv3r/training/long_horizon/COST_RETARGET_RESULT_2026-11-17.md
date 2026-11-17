# COST-RETARGET-001 — cost-sensitive long-horizon game policies

17 November 2026 | DRIV3R Draft/HOLD PR #1 | **negative synthetic DEVELOPMENT result**

## Research question and boundaries

Does adding a bounded, cost-sensitive teacher to an otherwise matched neural training curriculum improve autonomous painting, recovery, survival and surprise-goal responsiveness? **Not in this experiment.** The cost-conditioned controller changed its behaviour but lost substantial goal responsiveness, particularly on seed 622.

This uses the existing synthetic CyberSand-style world and host-owned legal action menu with actual material, obstruction, protected-cell and painted-cell state transitions. It does **not** constitute native CyberSand execution or ABI qualification. In particular the synthetic RELOCATE_START action does not establish native 64-move legality. Read-only historical d_common is unchanged. No sealed structural heldout families were inspected or evaluated.

## Training and controls

Two independently initialized seeds, **621 and 622**. Per seed:

- Three stages of genuine Adam optimization at **36, 72, 128 decisions**, 60 updates each, produce one frozen 961-parameter goal/recovery ranker (**180 updates**).
- The frozen model is copied twice with identical weights and optimizer state. Both adapted arms receive **20 additional updates per stage** (60 more, **240 total**), with identical minibatch indices, sample counts and legality masks.
- Each stage supplies **256 frozen-student-visited TRAIN states**, shared exactly between imitation and cost-mixture arms. Legal perturbations exercise recovery. The imitation arm uses the existing teacher label.
- The cost teacher deep-copies the current TRAIN world and evaluates up to **five legal actions** from each offered candidate, followed by the existing imperfect teacher. It does not reveal future objective switches. Its scalar score is: 1.5 × newly painted + 0.65 × current-goal distance gain + 5 × target contact − 1.75 × recoveries − 9 × early STOP. The adapted treatment mixes **70% scored labels and 30% ordinary imitation**.
- Cost-to-go labels are only training supervision. They are never passed to the neural controller as extra input, and the DEVELOPMENT rollout is fully autonomous except for identical forced *legal* deviations at decisions 19, 57, 91 and 131.

TRAIN cost labels disagreed with imitation on **680/1,536** sampled states across the two seeds and three stages; mixture gating actually changed **486/1,536** labels. Thus the treatments are genuinely different, not duplicates.

The evaluation reuses **eight familiar DEVELOPMENT specifications** (normal and trap variants), two horizons (**128 and 160 decisions**) and both static and surprise-retargeting instructions. This is a matched DEVELOPMENT comparison, **not** a blind structural holdout.

## Surprise-retarget DEVELOPMENT (32 episodes per arm)

| Measure | Frozen 180 updates | Imitation 240 updates | Cost mixture 240 updates |
|---|---:|---:|---:|
| Mean unique painted cells | **138.97** | 128.78 | 129.50 |
| Targets reached while instructed | **78** | 76 | **49** |
| Early STOP episodes | **3/32** | 5/32 | **9/32** |
| Painted cells in 8-step post-switch windows | **743** | 693 | 712 |
| Total recovery actions | 90 | **37** | 87 |
| Mean 8-step goal distance gain | 3.10 | 3.02 | **3.96** |

### Static-goal cross-check (32 episodes per arm)

| Measure | Frozen | Imitation | Cost mixture |
|---|---:|---:|---:|
| Mean unique painted cells | **139.84** | 129.69 | 126.16 |
| Targets reached | **32** | **32** | 20 |
| Early STOP | **1/32** | 5/32 | 9/32 |

**Result:** The cost mixture improves a local eight-step distance-gain metric and some post-switch painting relative to matched imitation, but loses 27 instructed goal contacts and increases premature STOP. A higher local directional-gain score is not a satisfactory long-horizon outcome.

Compared with equal-update imitation, the cost mixture changes mean ink by only **+0.719 cells** per retarget episode (paired wins/ties/losses **5/5/22**), with descriptive episode-bootstrap 95% interval **[−13.56, +16.41]**. In static-goal evaluation it changes mean ink by **−3.531 cells** (wins/ties/losses **13/1/18**), descriptive interval **[−19.62, +12.85]**. Worlds and horizons are reused and not independent, so these intervals are descriptive, not independent structural generalization evidence.

The frozen arm fares better than either adaptation on survival and painting, but it has **fewer optimizer updates**. Causal matched-budget comparison is therefore specifically imitation versus cost mixture; frozen is a useful checkpoint-retention control.

## Retarget action distributions, both seeds

| Offered host action executed | Frozen | Imitation | Cost mixture |
|---|---:|---:|---:|
| BACKTRACK | 72 | 12 | 12 |
| MOVE_E | 1317 | 1426 | 1158 |
| MOVE_N | 1028 | 785 | 1032 |
| MOVE_S | 1066 | 832 | 1016 |
| MOVE_W | 1036 | 1078 | 938 |
| RELOCATE_START (synthetic) | 18 | 25 | **75** |
| STOP | 3 | 6 | 10 |

More frequent synthetic relocation and altered heading preferences do not amount to better recovery strategy or preserved motif character.

## Source-matched qualification

- **Six real restored NumPy checkpoints**, each 961 trainable parameters; no d1-2B, LFM2.5-350M, or Sprocket-500M adapter training was performed.
- **192/192 autonomous DEVELOPMENT trajectories replayed**, containing **25,854 actual legal decisions**. Probabilities, requested action, intervention, offered menu, material edits, state hashes and aggregate results were recomputed against restored models.
- **208/208 SHA-256-manifested evidence files verified**, 104 per seed, including original traces, training counterfactual scores and checkpoint bytes. No checkpoints or binary model weights are committed to Git.
- **15/15 focused and predecessor tests PASS**, including legal candidates, deterministic scoring, TRAIN-only sampling, checkpoint replay and tamper rejection. Run pytest with import-mode=importlib.
- Seed 621 CPU campaign: **31.831 seconds**, peak RSS **130,676 KiB**. Seed 622: **32.984 seconds**, peak RSS **134,060 KiB**. Measured neural decisions are approximately **0.118–0.123 ms** each in this local Python harness, not native Workbench latency.
- The complete local evidence ZIP is source-hash matched and independently checked. **Immutable GitHub Actions checkpoint-artifact publication is still outstanding**; the Git source import or source-smoke CI must not be misrepresented as checkpoint qualification.

### Exact checkpoint SHA-256

| Seed | Policy | SHA-256 |
|---|---|---|
| 621 | frozen | ab121cf25a4969dbbd6162b9068816c87fc67db93cde9d3d4ef3205cbc25c21d |
| 621 | imitation | 5153d4a0d782db132cdb25331b61e167129f1798e7d70fe2a5bd22833e14c5ae |
| 621 | cost mixture | 6bd6644ca718fc7601522ae2f8fb870dcc4ebe493917ac7d468f8d5b5ae1d5e6 |
| 622 | frozen | 309f98951ce2609bdce485166e4d075ee647c0ccfa894e9548ecde290864a4c4 |
| 622 | imitation | 853de548eb701c7a0ec2c15074098ac38ceaba251f1f3928407e148d4230b59b |
| 622 | cost mixture | ed50e0e2dd851d643a051dfe8297427f665ee17dc99b6df55894016e98b73011 |

Local original model source file Git blob SHA: **f6cf01de2b3d7abdb6470803d0e2ce5221af539e**. This is independently compared with the published DRIV3R blob.

## Scientific interpretation / next experiment

This is a concrete **negative** result. Five-step supervised rollout rewards appear too myopic to protect future instructed-goal attainment; the observation is not proof of that causal mechanism. Before extending lookahead or claiming competence, compare true delayed-goal reward and memory-aware recovery costs against a stable matched imitation baseline. Separate retained motif/branching character, goal attainment, post-switch painting and survival, and use fresh precommitted structural families when authorized.

Only experiments/driv3r/training/long_horizon/ is writable for this programme. No changes to DRIV3R main, CyberSand, Neurounner, production settings, native gameplay or other workers. No paid API/GPU compute. **PR #1 remains Draft/HOLD: do not merge or promote Ready.**
