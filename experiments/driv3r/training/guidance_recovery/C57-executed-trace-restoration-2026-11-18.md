# DRIV3R Guidance & Recovery — C57: action-trace memory restoration

**18 November 2026 · external synthetic-host research · Draft/HOLD · reject model promotion**

## Question and intervention

C56 found that removing half the stored local host-menu observations reduced usable observed routes sharply, and that its neural repair head did not recover lost memory. C57 tests a narrower host-legal alternative: restore **directed edges actually traversed** in the preserved executed-action history, then compare (a) using restored routes alone, (b) a transparent exploration fallback, and (c) a freshly trained tiny imitation head against matched shuffled labels and the frozen RESET controller.

The experiment's erasure is **specifically a local-menu memory loss**. It preserves the action log. No inference is permitted about reverse edges, unseen cells or present-day legality. The host revalidates every candidate through its current legal menu, then through `World.apply`. The restorer does not read remote `allowed`, `protected`, hidden maps or geometry. Recovering proof from a retained action log is *not* recovery when both the log and local-menu memory are lost.

This is an **independent synthetic-host study, not CyberSand's authoritative native ABI**. The C57 episode generators, goals and cue mechanisms are inherited from the published C55/C56 research; the observed-frontier target is deliberately easier than a distant unobserved goal.

## Design and controlled work

- 16 fresh training graph identities, after excluding 190 historical graph hashes; 3,140 teacher-labelled legal action examples. Labels use only current host-legal actions, host-issued target, observed or trace-restored edges, local Manhattan-to-goal and previously visited positions. The teacher is a **transparent local heuristic**, not a hidden shortest-path oracle.
- Two new deterministic `13 → 20 → 1` CPU heads, C57-ALIGNED and C57-SHUFFLED, each trained for **400 AdamW optimizer updates** (eight epochs); matched initialization and batch ordering, with 1,614 changed shuffled labels. Independent refit matched their weights exactly. Both checkpoint SHA-256 digests and the training-array digests are in the compact summary.
- 24 unseen graph identities (Wilson/Kruskal and fresh familiar families), disjoint from C55/C56 and C57 training. Host-executed prefixes, fresh legal action menus, continuous RNG and global time; separately execute all six controllers on each available suffix.
- Compare no guidance, weak BACKTRACK (0.35), strong BACKTRACK (1.0), for **0% and 50% stored-menu erasure**; observed-frontier goals and remote goals are reported separately. The 174 matched condition rows total **1,044 independent controller suffix executions** (696 guided, 348 no-guidance).
- Policies: C18-A0 baseline, C34-RESET, restored-trace-only C57-TRACE, restored-trace with transparent local fallback C57-HEURISTIC, C57-ALIGNED, and matched trained C57-SHUFFLED. All neural and non-neural recovery bonuses are soft stochastic logit adjustments, never hard commands. No unapproved external/GPU spend.

## Independent goal results

Completed goals per controller; values pool the two guidance strengths. Observed goal has 28 guided episodes per controller per erasure level; remote goal has 30.

| Host-assigned goal / erased menus | RESET | TRACE | HEURISTIC | ALIGNED | SHUFFLED |
|---|---:|---:|---:|---:|---:|
| Observed, 0% | 21 | **23** | **23** | 21 | 22 |
| Observed, 50% | 21 | 21 | 22 | 21 | 22 |
| Remote, 0% | 11 | 11 | 12 | 11 | 11 |
| Remote, 50% | 11 | 11 | 12 | 11 | 11 |

C57-TRACE raised observed-goal completions by two only with memory intact. Erasing half the local menus reduced usable trace guidance and eliminated that advantage. HEURISTIC gained a single goal on erased observed tasks and a single remote goal regardless of erasure, but **it also incurred condition-level regressions** (two in the half-erased observed cohort). The neural ALIGNED controller **never beat RESET** in any pooled cohort; SHUFFLED equalled or exceeded it on observed goals. No promotion-level effect was established.

Soft trace guidance was offered during 176 steps by the non-neural HEURISTIC on intact observed tasks, but only 58 steps after 50% erasure; under erasure it instead used the local fallback on 119 steps. On remote goals, almost all guidance came from the local fallback (248 steps); trace guidance occurred only eight times. Thus direct trace restoration is a limited instrument, not a solution to remote navigation or erased unknown endpoints.

Goal comparisons remain exploratory: the research sequence has repeatedly adapted tasks to previous failures, the evaluation has only 24 graph identities, and guided personality retention is not established. Per-graph paired unadjusted bootstrap summaries and all policy actions are retained in the evidence ledger.

## Safety, source identity and replay

- 58/58 no-guidance matched conditions independently reproduced identical **action and painting SHA** across all six controllers.
- 0 synthetic-host action rejections; protected-cell hash invariant on every executed suffix.
- The selected 18-trajectory gallery uses **real executed prefixes and continuations**, not reconstructed geometry; 12 selected trajectories independently re-executed from their original seeds, prefixes and frozen head hashes.
- Five targeted tests passed: directed-only restoration and nonmutation, graph isolation, legality/no-guidance parity, byte-identical optimizer refit, and source-matched replay.
- The trained checkpoint binaries are preserved in the local owner archive **not Git**, with SHA-256 receipts. Immutable GitHub Actions artifact custody has not been established. This source import alone does not establish scientific reproducibility without its frozen dependencies.

## Disposition and next test

**HOLD — reject C57-ALIGNED, C57-TRACE and C57-HEURISTIC for deployment.** C57 shows that a retained action history can partially reconstruct paths without reading hidden geometry, but selective imitation has not improved gameplay, and the transparent fallback produces small mixed outcomes including regressions. A decisive next experiment should use genuinely remote goals in independently fresh maps, enforce loss of both the menu cache and parts of action history, and compare exploration policies with explicit zero-regression and guided-personality gates. Do not change CyberSand, its host interface, or unrelated research repositories.

**Publication boundary:** only `experiments/driv3r/training/guidance_recovery/**` in existing DRIV3R Draft/HOLD PR #1. No PR/branch/workflow creation, no automatic training, no changes to DRIV3R main.
