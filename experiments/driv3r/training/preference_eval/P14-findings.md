# P14 — Protected-cell overlay and goal-conditional continuation

**16 November 2026 · DRIV3R Draft/HOLD PR #1 · research-only, local CPU-qualified, not hosted on DRIV3R Actions.** No writes to CyberSand, Neurounner or any game path. No owner/human preference votes, root-language-model inference, native simulation, protected-paint actions, training on live Workbench, or production provider changes.

## Scientific question and precise boundary

Does a bounded external protected-cell **position** veto materially change authentic frozen W16 auxiliary-head trajectories, and what is the cost to a simple creative-coverage objective? Can a lightweight pairwise ranker provide meaningful discrimination beyond a transparent physical-safety/goal rule?

The pinned **synthetic W0 host does not implement painting, real materials, protected-cell readback or protected-state authoring**. Two fixed-before-inference rectangular masks (with deterministic seed-dependent ±2 vertical offset) therefore represent a *hypothetical forbidden-position overlay*, not a CyberSand protected-material feature. Model `BRANCH` follows the unchanged W0 priority; an external candidate filter rejects moves into those cells, then submits only actions already legal under the unchanged W0 host. The model still observes W0's original 142-input observation contract. Thus the host and original frozen model code are not modified, but overlay legality and canonical native CyberSand semantics remain **unverified**.

## Original source and actual execution

Pinned immutable source: P11 hosted artifact SHA-256 `9a9f61038bb1a4f68395816a8296fb8957a8a8f166610921c2a9db6502308d9c`; nested original W16 producer SHA-256 `a7b756a3b0c2d639e5e48dad36d518dfd0e6b752deadca8b04ac11326717eafb`; frozen auxiliary PyTorch head checkpoint SHA-256 `f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9`. Original execution source HEAD `e9c7d9ca2a018c2da591b3fcb3429991c907fdfa` is validated by the P12 reader. The producer is evidence from CyberSand **read-only history**; this programme publishes solely to DRIV3R.

Executed **432 closed-loop frozen-head W0 rollouts** (18 discovery, 9 validation, 27 held-out family/seed worlds × 2 ex-ante mask layouts × 4 policies), comprising **312 byte-distinct trajectory/PNG pairs**, **40,989 real PyTorch-head forwards**, and **108 byte-exact original P11 baseline PNG reproductions**. All reported applied choices were verified host-legal. Every emitted original-host PNG, trace, overlay mask and original source trace has a SHA-256 identity in the separate local evidence archive. No new optimizer updates on the original W16 head. All three cohorts contain the Architecture, Greebles and Pipework families; seeds are disjoint across cohorts. The two masks share each family/seed and **are correlated measurements**, not 54 independent worlds.

The four precommitted policies are: original late-STOP W16; original W16 STOP with extra protected-position veto; extra veto plus exact completion STOP at 48 novel cells; extra veto plus STOP 16 ticks after first completion. The original historical STOP gate is retained only in the first two. The latter two STOP after a verified, *synthetic* goal, without learning to stop. When there is no safe candidate, host-legal STOP is available; the actual held-out veto policies required zero such forced stops.

## Held-out consequence (27 worlds × 2 masks = 54 matched condition instances)

| Policy | No protected-position contacts | 48-novel-cell goal completed | Total protected-position contacts | Mean novel positions | Mean decisions |
|---|---:|---:|---:|---:|---:|
| Original W16 late STOP | 25/54 | 54/54 | **407** | 107.52 | 125.11 |
| Protected-position veto, late STOP | **54/54** | 52/54 | **0** | 104.09 | 125.13 |
| Veto, immediate goal STOP | **54/54** | 52/54 | **0** | 47.94 | 56.74 |
| Veto, goal plus 16-tick grace | **54/54** | 52/54 | **0** | 62.00 | 71.93 |

The protective veto prevents **all 407 synthetic position contacts** observed under matched late-gated W16 while costing **two held-out coverage successes**. Those failures were Greebles seeds **2729** (47 new cells) and **2819** (46 new cells), both `near_origin` masks. In the matched unshielded originals both passed 48. These are substantive counterexamples: extra legal restrictions can preserve one kind of safety while reducing goal reachability. The grace period recovers post-target creative continuation and more structure, but does not fix those two reachability failures. Mask exposure differs substantially by family and layout; none of this establishes a native protected-paint safety rate.

## Actual lightweight ranking experiment: result is effectively a negative

From real rollout outcomes, created **601 non-tied pairwise comparisons** across matched policies. No rows are assigned aesthetic or human preference labels. A predeclared, explicitly subjective **algorithmic priority** ranks zero protected contacts before contact count, then goal completion, novel cells, branches and shorter duration. The ranker does **not** control any live moves or STOP actions: its inputs are *completed-trajectory outcome metrics* and can only support retrospective evaluation.

A seven-feature anti-symmetric linear Bradley–Terry-style logit ranker underwent **500 actual CPU Adam updates** on **202 discovery pairs**, with 101 validation pairs and selection at step 500. On **298 held-out pairs**, it achieved **294/298 (98.66%)** pairwise agreement and 0.01544 Brier. However a trivial transparent heuristic agreed on **the same 294/298**. The four missed cases all arise from the same two Greebles reachability failures with indistinguishable coverage/protection flags and small retrace/duration tradeoffs. The benchmark's near-perfect label agreement primarily demonstrates that simple programmed priorities can be learned, **not** any acquired owner-preference, creative-aesthetic or controllability insight. The exact source priority oracle is tautologically 298/298 by construction, not a meaningful learned baseline.

**Decision:** do not promote this head or the synthetic safety veto into actual gameplay. Hold learned reward optimization until actual paint/protected-state host evidence, meaningful goal/status interfaces, and blind owner/human paired judgments are available.

## Reproduction and audit

Within `experiments/driv3r/training/preference_eval/` and with the unchanged P11 hosted ZIP supplied externally:

```sh
python -m unittest -v test_p14_protected_overlay
python p14_protected_overlay.py --p11-artifact /path/to/immutable-P11-hosted-43918309938.zip --out /tmp/p14 --images
python p14_bestiary.py --rollouts /tmp/p14  # optional Pillow: produces HTML, CSV and annotated comparison sheet
```

Seven local tests passed including an exact frozen-head original-baseline replay and protected-action projection. The expensive and byte-verified integration test skips if the immutable P11 source ZIP is absent, rather than pretending the imported source establishes model reproducibility. Python files and this ledger are committed to DRIV3R; generated `.pt` checkpoint, PNGs, JSON receipts and bestiary remain **outside Git**. The local archive is independent of immutable DRIV3R Actions qualification; no hosted P14 replay/CI claim is made.

## Narrow game interface proposal, not a game change

A future game-side handoff should expose authoritative, per-command paint-cell write sets and protected-cell veto receipts, a host-owned goal-completed condition and a reason-coded STOP (`voluntary`, `goal_completed`, `no_legal_moves`, `budget_exhausted`). Restrictive movement proxies must never be silently treated as painting protection. Until a native CyberSand owner has accepted that interface, P14 is exclusively an external synthetic-risk study.
