# AUTONOMOUS-CONTINUATION-001 — matched fully neural continuation

16 November 2026 · DRIV3R PR #1 · DEVELOPMENT-only synthetic worlds · **NEGATIVE / MIXED**.

## Method

- Source-qualified imported `World.menu(cap)` / `World.apply` host-like legal actions. Two train seeds (611/612), three genuine optimizer curricula 36→72→128; 64 Adam steps per stage per continuation model, plus 96 matched extra adaptation updates; 64 Adam steps per stage for a learned 85-parameter nonlinear side-exit fork.
- Shared initial 24-hidden-unit / ~625-parameter continuation MLP. `imitation` receives additional offline teacher replay; `on_policy` receives identical-count updates on student-visited TRAIN states labelled by the frozen source teacher (DAgger-style). No teacher queried during learned-arm DEVELOPMENT continuation.
- 160-decision DEVELOPMENT episodes in two directions, four variable-depth/dead-end configurations, three recovery capacities per seed. Forced initial 6/8/10-step wrong corridor move. Same host actions (cardinal moves, BACKTRACK, RELOCATE_START, STOP) and hard protected/material exclusions. Deterministic legal deviations at positions 37, 79 and 109 when possible.
- Explicit controls: teacher continuation behind the learned fork, and analytic counterfactual fork behind both neural and teacher continuation. The analytic fork uses 12-step future rollouts and is **not a deployable policy or evidence of learned planning**. Teacher controls have goal proximity guidance; learned action model receives no goal-coordinate feature, an important intentionally exposed limitation.
- All 160-decision traces and every neural legal-menu probability recorded. Source SHA-256, checkpoint hashes, per-file manifest, process maximum resident set, per-decision latency, original PNGs. Original episode replay uses restored checkpoints and deterministic teacher/control logic; final metrics and every host transition checked.

## Controlled DEVELOPMENT outcome

| Arm | Goals | Early STOP | Mean ink | Mean branches | Mean post-fork ink |
|---|---:|---:|---:|---:|---:|
| imitation | 20/24 | 23/24 | 34.5 | 6.542 | 26.5 |
| on_policy | 20/24 | 21/24 | 36.125 | 7.208 | 28.125 |
| teacher_control | 20/24 | 23/24 | 33.708 | 6.708 | 25.708 |
| oracle_fork_neural | 20/24 | 21/24 | 36.125 | 7.208 | 28.125 |
| oracle_fork_teacher | 20/24 | 23/24 | 33.708 | 6.708 | 25.708 |

**Interpretation:** Online-labelled adaptation produces a small seed-dependent improvement in mean painted cells but does not solve early termination or demonstrate sustained 160-decision gameplay. The goal count is identical, and all arms still have frequent early STOP. Learned fork choices happened to match the analytic comparator in all these DEVELOPMENT fixtures, so oracle-fork arms do not distinguish an additional causal benefit. This finding cannot be generalized to sealed structural families, native CyberSand, true 64-move receipts or larger SLMs. These were small NumPy MLPs, not trained d1-2B/LFM/Sprocket adapters.

## Verification

- Seed 611: 60 episodes, 2560 source-replayed decisions, 3 real checkpoints, 192 fork Adam updates, 288 continuation Adam updates; 5.939 CPU seconds, peak RSS 103588 KiB.
- Seed 612: 60 episodes, 2523 source-replayed decisions, 3 real checkpoints, 192 fork Adam updates, 288 continuation Adam updates; 5.766 CPU seconds, peak RSS 103640 KiB.
- Four focused tests: split bounds, exhaustive host-menu probability legality under capped recovery, proof no teacher call in learned continuation, checkpoint/trace-tamper rejection.
- Final matched source, test suite, original evidence and output hashes are in the downloadable local artifact. No Actions artifact has been produced, no CI has been dispatched and no structural HELDOUT worlds were instantiated.

**Research next step:** condition the student on a *host-observable* spatial goal (not hidden future rewards), and evaluate whether goal reaching and sustained horizon both survive variable geometry with paired no-goal controls.
