# GAL-015 — Observation-limited return-to-frontier scaffold (16 November 2026)

**Scope:** external synthetic material corridors only; no CyberSand or Neurounner mutations. This experiment belongs to DRIV3R Draft/HOLD PR #1 and was developed solely in the Game Action Language laboratory.

## Hypothesis

GAL-012 demonstrated long-horizon transfer failures when a flat policy had to learn painting and branch retrieval together. Here, separate the decisions: (a) use a trained tiny linear head to choose among host-legal, newly painting actions; (b) when painting is not available, deterministically follow the shortest route through *observed* material/accepted receipts to a remembered frontier, using only host-leased LIFT actions. Never lift directly onto the unpainted frontier cell, as the host cannot paint under the cursor afterward. START is excluded from the objective, as specified by the synthetic task. This is a task-policy guard, not a host legal-action rule.

## Design / no hidden map access

48×48 host, blocked cells, protected boundary, material ink, 32/48/64-cell paint objectives; 4 families including previously withheld comb and deep_tree. Each candidate action ID/path and lease is from `World.offer()`; the model and scaffold read radius-4 `visible_material` crops and `Ledger` of accepted receipts only. The host executes or refuses actions and validates protected cells. Bounded training: 48 path/tree worlds (32/48 cells), 826 real online cross-entropy updates. No paid inference or external GPU. Held out: 12 fresh seeds × 4 kinds × 3 targets × 6 policies = **864 synthetic episodes**, not 864 independent world seeds.

## Matched outcomes

| Policy | 144 episode completions | Mean decisions | Model-calls proxy | 
|---|---:|---:|---:|
| Frozen GAL-011 stochastic | 82 | 41.37 | 41.37 |
| Frozen GAL-011 greedy | 101 | 44.67 | 44.67 |
| Constrained scaffold, no trained head | **144** | 40.36 | 21.88 |
| Constrained scaffold, trained head | **144** | **40.17** | **21.88** |
| Unguarded LIFT-to-frontier ablation | 88 | 29.94 | 17.49 |
| Observation-limited routing teacher | **144** | 40.36 | 0 |

**Main result:** constrained routing improves completions 101/144 → 144/144 relative to frozen greedy, without privileged map reads, while reducing policy-selection calls on these synthetic worlds from 44.67 to 21.88 (a *proxy* for approximately 51% fewer possible inference invocations; no SLM ran). Paired seed-cluster bootstrap (12 distinct seed clusters, 5,000 resamples): completion advantage **+23.61 to +36.11 percentage points** (95% exploratory interval); model-call difference **−24.81 to −20.92** per episode. Over 32/48/64 targets each constrained scaffold achieved 48/48.

**Important negative result:** the newly trained linear head adds no completions versus an untrained longest-paint-rule head and changes mean decisions only 40.36 → 40.17. Seed-cluster bootstrap 95% interval of trained-minus-untrained decisions is **−0.507 to +0.069**. This does not establish that the trained weights are useful; the deterministic memory/route scaffold is the dominant intervention.

**Failure mechanism:** permitting a host-legal pen-up LIFT to end *on* the unpainted target caused 56/144 failures; the cell cannot later be painted under the cursor. Host legality alone does not capture user objective semantics. Guard is task-local and does not require changing the canonical game ABI.

## Qualification and provenance

- Ten unit/host-contract tests passed locally; one full synthetic evaluation of 864 episodes completed.
- An independent rerun of 24 held-out 64-cell cases (every mode, four morphologies, one seed) reproduced exact result rows and receipt/material SHA-256 values. **Not** an independent native execution, and not a full second replay of all 864; the attempted exhaustive second replay exceeded the local execution bound.
- Original full 864 rows, selected original host receipts, material PNGs, and a deterministic six-panel SVG gallery are preserved in the associated local evidence bundle. Core metrics, compact gallery and audit manifest are tracked beside this source in DRIV3R.
- The policy checkpoint is **not committed** to Git. The trained head is a six-weight JSON tiny linear model, not a LoRA, resident SLM, neural art-quality evaluation or real token-cost measurement. Immutable Actions artifact publication remains outstanding and is not claimed.
- The prior frozen GAL-011 baseline checkpoint is from the archived research artifact, not reproduced by import alone. This is a source-matched synthetic continuation, not CyberSand physical acceptance. `d_common` is untouched.

## Next falsification

Test in world families with non-tree geometry, disconnected apparent frontiers, variable paint permissions, materials changing under simulation and pen-up cursor semantics that differ from this host. Compare a limited-memory routing scaffold and a policy head trained to predict branch targets without a deterministic BFS. A game-side interface proposal should follow only after the canonical host exposes task objective semantics under owner authorization; no game edit is requested now.

### Source and checkpoint hashes

- `route_scaffold.py` SHA-256 `b830fb07cd8b099c4c3f44a21e6e0700ebad86f06c82beecd6d4d7e77664a05f`
- `test_route_scaffold.py` SHA-256 `6cc22af261a4bcbbebd2aa5d8ade262a9ec4fe3660a28c00cf5f3e2e1b36587d`
- `gal015_metrics.json` SHA-256 `e55dcef647fa0888304143c19fed56c15b614a760e69334e9886be95637d62a5`
- `gal015_bootstrap.json` SHA-256 `4c6ad1d209a74e2cf58844d2072ca211dda760f1202c4df1209814a96e6a51d2`
- `gal015_replay_audit.json` SHA-256 `b48e65fc5fc0f046ebe6e70cfab61e8098936f3efd713ba81f8e4cfb8328a911`
- `gal015_gallery.svg` SHA-256 `1290d0befffe739c0f353e84e18a93f40d37be26332c2f56dedf1e8ee3291805`
- frozen GAL-011 checkpoint SHA-256 `97f88f0349de134c3c430997f998a8534d1a2b62e372ba81970f85e12baf61d1`
- new trained tiny-head checkpoint SHA-256 `5a158ec70292813b3b3d881f544c5bf7135f89825f7d695e17eeacabe699b94d` (in evidence bundle; not Git)
