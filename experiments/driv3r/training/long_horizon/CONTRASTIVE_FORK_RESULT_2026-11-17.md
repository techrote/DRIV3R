# CONTRASTIVE-FORK-001 — Four-direction risk/open counterbalancing

17 November 2026 · DRIV3R long-horizon lab · **Positive local discrimination; unresolved continuation/transfer**

## Research question

RECOVERY-FORK-001 taught near-exhausted worms to back out of a deterministic dead end, but 26/32 open-route controls were also unnecessarily retreated. Does a balanced contrastive curriculum distinguish trapped-vs-open at the **actual legal first choice**, across cardinal orientations, without discarding goal-directed continuation?

## Source and scientific boundary

- Research-only 96×96 one-cell synthetic worlds. **NOT** CyberSand native host ABI evidence, physical evaluation, or proof of production history/relocation semantics. The 12-recovery mechanism and 128-decision episodes exceed the old 64-logical-move native reference limit. Host menus, legality, material edits, protected cells, and recovery counts are enforced by the unchanged synthetic `World` fixture.
- Starting exact remote branch HEAD inspected before any edits: `079ddd1440a8cea4d8af367af97c1fcddb479141`. Existing laboratory predecessor source and seven dependencies were Git-blob SHA-matched against the live DRIV3R branch. No changes outside this lab; no workflows introduced.
- Sealed structural HELDOUT families **not opened**. DEVELOPMENT seeds and fixture families were previously used within this research program; these results are *development*, not blind generalization. Models are tiny NumPy rankers, not d1/LFM/Sprocket adapters.
- Two independent model seeds **691** and **693** were selected after checking geometry/counterfactual-witness validity and **before** observing the model outcomes. Training and development seeds remain partitioned, with DEVELOPMENT synthetic parent seed **782**. Other attempted geometry seeds were rejected only because the greedy legal continuation witness failed under some orientations, introducing scenario-selection bias.

## Controlled procedure

1. Derive a near-exhausted trapped corridor using 11 actual legal BACKTRACKs and one legal RELOCATE_START, and a matched open corridor. Reconstruct each into 0°, 90°, 180° and 270° rotations by rotating the **initial synthetic materials and protected cells**, then replaying each transformed **legal** action prefix. No history, ink or cursor assignments are used.
2. Certify at every state that the goal-forward first move in the trap becomes forced STOP after **five painted moves**, whereas its opposite permits at least 20 further legal moves. Both open-corridor first moves permit at least 20 legal moves. The 24-move deterministic viability witness is **not** an autonomous policy.
3. Compare four equal-initialization policies: **frozen**, **goal-forward imitation** (forward on both challenge types), **universal retreat** (back on both), and **contrastive** (back in trapped, forward in open). All *adapted* arms train on **identical host-legal input matrices and masks**; only the supplemental labels differ. Each stage contains 64 sampled natural TRAIN examples plus **32 supplemental examples** (16 trap, 16 open); the same 16 synthetic family/rotation pairs recur in each curriculum stage. No DEVELOPMENT labels or outcomes are used to fit.
4. Genuine Adam updates: frozen base 18×3 = **54** updates; each adapted model receives 32×3 = **96 additional**, total **150**. Each checkpoint has **961 parameters**. Stages are 36 → 72 → 128. Checkpoints restored before evaluating.
5. Autonomous evaluation per seed: four DEVELOPMENT source families × four orientations × trapped/open × four controllers = **128 independently replayed 128-decision episodes**. These are 128-decision *budgets*, not always 128 executed decisions because the real synthetic host offers forced STOP.

## Autonomous DEVELOPMENT outcomes

| Scene / controller | Correct first action | Forced early STOP | Mean new painted cells | Mean decisions | Mean neural choice latency |
|---|---:|---:|---:|---:|---:|
| trap / frozen | **0/32** | 32/32 | 5.00 | 6.00 | 0.100 ms |
| trap / goal_imitation | **0/32** | 32/32 | 5.00 | 6.00 | 0.100 ms |
| trap / universal_retreat | **32/32** | 12/32 | 112.75 | 113.12 | 0.100 ms |
| trap / contrastive | **32/32** | 20/32 | 89.34 | 89.97 | 0.108 ms |
| open / frozen | **32/32** | 23/32 | 91.88 | 92.59 | 0.101 ms |
| open / goal_imitation | **32/32** | 17/32 | 94.38 | 94.91 | 0.102 ms |
| open / universal_retreat | **0/32** | 21/32 | 77.50 | 78.16 | 0.102 ms |
| open / contrastive | **32/32** | 27/32 | 85.06 | 85.91 | 0.098 ms |

**Key comparison:** the contrastive model chooses the certified escape on **32/32 traps**, while also choosing goal-forward on **32/32 open controls**. Its universal-retreat counterpart gets 32/32 traps but **0/32 open controls**. The goal-forward and frozen controls get all 32 open cases but **0/32 trapped**.

Despite that impressive *first-decision* separation, contrastive continuation is weaker: trapped post-fork painting averages **89.34** cells vs **112.75** for universal retreat, with forced early STOP **20/32 vs 12/32**. In open corridors contrastive paints **85.06** vs frozen **91.88** and goal-forward **94.38**. All observed early STOPs were host-forced after the menu became exactly `STOP`; they were not voluntary early termination.

The first-choice result is not robust proof of recovered spatial planning. The open corridor often exposes more legal moves than the trap; nearby geometry, menu composition and orientation features can supply simple classification cues. Since both open first moves survive the 24-step witness, forward is a **goal-selection** label rather than an objective survival requirement. These rotated copies share one corridor topology; they are not 32 unrelated structural environments.

## Independent recovery-feature mask

Without retraining, re-run the first neural choice with the existing 8 recovery features zeroed. Contrastive trap choices remain correct on **32/32** cases and open controls on **31/32**. Thus the first-choice discriminator **does not require the live recovery features**. This supports local geometric/action-menu recognition over a learned general recovery-memory mechanism; see independent audit JSON for actual masked actions.

## Evidence, qualification and caveats

- **8 genuine checkpoints** across two CPU campaigns; 2×128 = **256 replay-verified trajectories**, **18,133 model decisions** (9,312 + 8,821). Each original JSON trace includes full offered legal menus, requested actions, neural probabilities and source-replay state hashes.
- Checkpoint restoration, complete original host trajectories, TRAIN legal prefix and 24-step counterfactual witnesses, seed/goal state hashes and per-file SHA-256 manifests were checked. Both campaign manifests contain 166 files each; no posthoc artifact edits or checkpoint weight changes. Five focused guardrail tests passed, including evidence tampering.
- **No native receipts, no paid API/GPU, no structural blind heldout, no production/game-side changes.** Binary checkpoints remain in locally archived evidence ZIP, **not yet immutable GitHub Actions artifacts**. Do not claim durable source-matched binary publication from this source-only commit.
- Average full campaign wall time 33.446/32.489 seconds; process peak RSS reported 127,552/126,484 KiB respectively. Latency is CPU-local, not hardware-comparable to d1 or other architectures.
- All false-success and negative data are retained: universal retreat fails open-route first choice universally, and contrastive overfits the initial classification without reliably improving longer-run painting or survival.

## Checkpoint SHA-256 ledger

| Seed | Arm | SHA-256 | Parameters | Updates |
|---|---|---|---:|---:|
| 691 | contrastive | `1f7004c3a5524ea8859eb01cdc10cdb3d74c359dac9d10648b03853c1405433c` | 961 | 150 |
| 691 | frozen | `60fb8ff1d0020741202e6666a3256ca07bdb2c02f2b83a1a20ecdcda48eea2fa` | 961 | 54 |
| 691 | goal_imitation | `0b8be2050882f0d0bc40f12e5724d536c1df65c2924e03fba408ee06eafc2334` | 961 | 150 |
| 691 | universal_retreat | `567480bf0851cbe59f6c2689aab8a912c6bcabca61b52c84768059ec9b8eb5d9` | 961 | 150 |
| 693 | contrastive | `8d940162d5113087e3ae26f5e1d407fd4b880741096298ddb38fd5ccf706657c` | 961 | 150 |
| 693 | frozen | `8d1dfd0ef654a0ffd56d99f967b2fbd732f7b0abd1609e7c7c5bf991a70f5da3` | 961 | 54 |
| 693 | goal_imitation | `ecf45718f816503c0a2899a990877d51ee82cd75fddd66863b2962175d50d21e` | 961 | 150 |
| 693 | universal_retreat | `1c64f71b3f6a9e9ed661b0d85751b330ac7b189c0f891c4606dcdd28a9b2c4bf` | 961 | 150 |

## Source SHA-256 ledger

- `contrastive_fork_lab.py`: `d40020055820f0084868c0840363b3086f88d1cf990e3d5c7b5445935043241e`
- `goal_conditioning_lab.py`: `6e00bfdfbaecc8195a86cb626bbbcbb65184a449adc5b7272d5642553cf02659`
- `goal_recovery_lab.py`: `7a8bf710a644b2a42c7e3fb032036f7aa246665e995683891ce9db5fc3caa290`
- `goal_retarget_lab.py`: `88ba7013f3bc8eb1ec8f728c5d704aeeb6aafd4ebc6cf964f534dc14e7f8b14b`
- `policy.py`: `f3e6469b592e68c0d3410dca88e107b215f1bdb1c16f7ebd73232ea9470cbb82`
- `recovery_fork_lab.py`: `97a569eec43b2a73bd390cc31012c719e33a73af9de7316a6d75356860abfb70`
- `train.py`: `9ed2c9ac49fa397c12f0b8256e4d4439bd3a9294b002b54276b702a1e3ddcacb`
- `world.py`: `6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a`

## Next decisive experiment

Test unrotated **structurally different** recoverable situations without modifying the predeclared sealed families, and explicitly match offered-menu cardinality between trap/open controls. Add a longer post-fork continuation objective and compare recovered trajectories under controlled risk gradients; do not add more reward-proxy tuning until these transfer controls pass.

Repository boundary: only this laboratory source, tests and result file belong in DRIV3R PR #1. PR remains Draft/HOLD. No merge, Ready promotion or external writes.
