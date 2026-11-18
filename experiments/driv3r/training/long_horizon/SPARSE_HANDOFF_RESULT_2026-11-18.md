# DRIV3R — SPARSE-HANDOFF-001: single-search event-triggered recovery (2026-11-18)

**Status:** DEVELOPMENT-only synthetic study; published for review in the existing Draft/HOLD DRIV3R PR #1. No merges or native host evidence.

## Experimental question
Can an event-triggered, one-time restricted planner handoff rescue goal-conditioned neural trajectories without the per-action planner cost of a continuous feasibility shield? This is a frozen-checkpoint evaluation (0 new optimizer updates), not a new training run.

## Source and boundaries
- Parent snapshot: `728a1c6131bc8da12165c8b94080a29b28d3da64` (the exact DRIV3R remote PR branch HEAD inspected before editing).
- Two hash-verified existing 769-parameter goal-conditioned tiny Ranker checkpoints, seeds 711 and 712. Checkpoint bytes stay in an external evidence archive, never committed to Git.
- Synthetic World 96×96, host-provided legal action menu, material/protected exclusions, monotone-prefix planner, 64 logical-move external gate. No native CyberSand receipts.
- Four DEVELOPMENT structural families, exactly one seed per family (3000,3100,3200,3300); two explicitly declared synthetic material layouts (original and east_sleeve); forced five-step eastern mistake; two goals; two checkpoint seeds. The 16 physical states were all restricted-planner-solvable, yielding 32 paired model cases per controller.
- Structural HELDOUT (`archipelago`, `staggered_vault`) remains sealed. Separate designed-inspected `rooms` and `funnel` also remain unused.

## Decision arms
| Arm | One-time planner trigger |
|---|---|
| neural | Never |
| planner_once | Immediately; execute cached exact restricted prefix plan |
| late24 | At 24 elapsed continuation decisions |
| stall4 | After four decisions without new best Manhattan-distance progress |
| stall8 | After eight decisions without new best Manhattan-distance progress |
| reserve16 | When at most sixteen logical moves remain |
| hybrid4 | stall4, reserve16, or neural STOP proposal before reaching goal |

When a trigger occurs, the complete prefix plan is calculated once. Its remaining actions are cached and checked against each current legal menu. No planner re-evaluation occurs after commitment. A controller can therefore search **at most once per episode**; this is not a theorem about general replanning correctness.

## Matched DEVELOPMENT outcomes
| Controller | Goals | Planner searches | Executed neural actions | Mean decisions/episode |
|---|---:|---:|---:|---:|
| `neural` | 22/32 | 0 | 1206 | 37.688 |
| `planner_once` | 32/32 | 32 | 0 | 21.000 |
| `late24` | 32/32 | 18 | 742 | 34.875 |
| `stall4` | 32/32 | 32 | 160 | 23.000 |
| `stall8` | 32/32 | 32 | 288 | 24.500 |
| `reserve16` | 22/32 | 12 | 1032 | 33.000 |
| `hybrid4` | 32/32 | 32 | 160 | 23.000 |

### Main interpretation
- Stall-triggered and fixed-24 handoffs rescued every neural failure in this narrow benchmark: 32/32 arrivals compared with neural-only 22/32. Neither establishes out-of-distribution success. Deterministic one-shot planning also attained 32/32 in fewer actions.
- `stall8` retains 288 neural-selected steps across 32 episodes and then consumes one planner search per episode. `late24` retains 742 neural-selected steps and requires only 18 searches (other episodes naturally finish before decision 24). At equal goal completion, the fixed delayed handoff is more planner-sparing but executes longer paths.
- `reserve16` failed to rescue neural early STOP decisions and finished 22/32: a hard time-remaining threshold is a poor substitute for an observable failure cue.
- `hybrid4` and `stall4` were identical on this cohort, so the STOP and reserve add-ons did not provide incremental measured benefit.
- The descriptive four-family cluster bootstrap for `stall8` minus neural is +31.25 percentage points, 95% interval +12.5 to +50.0. This is not an inferential guarantee: only four structural family clusters, correlated goals/checkpoint seeds, and hand-constructed challenge geometry.
- Planner inference uses privileged complete synthetic terrain and recovery history, not a demonstrated native action/observation interface. Failure of an unprivileged model against this oracle is not proof that a deployable planner exists.

## Integrity and performance
- Full replay verified **224** episodes, **7426** original host-legal action receipts (including forced prelude) and **3588** neural decision probability dictionaries. Source SHA-256 and checkpoint SHA-256 are pinned in the evidence ledger.
- Focused regression tests cover no protected-cell edits in the prelude, at-most-once plan caching, deterministic replay, and receipt/probability tamper rejection.
- CPU run: 12.302 seconds, peak RSS 98064 KiB. Median episode choose latency is a local Python statistic and excludes native host overhead; it is not a production latency claim.
- Original six-panel SVG gallery is computed only from the fully replayed action receipts, including the forced prelude. The unclipped viewport is selected from each real trajectory rather than truncating excursions.
- One-shot planner search counts are algorithmic counts, not raw cache hits; the main experiment uses no cross-episode oracle cache.

## Limitations and follow-on
- All worlds use DEVELOPMENT structural families and an eastern corridor forced-error fixture. The east_sleeve overlay is a synthetic obstacle screen, not a new native map or an independent procedural topology family.
- Restricted prefix search handles monotone retreat followed by empty-space painting. It is not a full Markov planner for arbitrary materials, jumps, switches, rewards, or reversible paths.
- Strict host-legal menus and physical 64-move accounting come from the historical synthetic World implementation, not the present CyberSand native ABI. No production repository was changed.
- More diverse source-faithful layouts, moving goals, native read-only receipts, structural HELDOUT and immutable Actions checkpoint-artifact publication remain outstanding.

## Evidence
- `sparse_handoff_lab.py` — executable study and replay verifier.
- `test_sparse_handoff_lab.py` — focused tests.
- `sparse_handoff_gallery.py` — source-backed six-panel SVG generator.
- `SPARSE_HANDOFF_EVIDENCE_2026-11-18.json` — compact comparative ledger, all per-arm action distributions, exact checkpoint/source hashes and memory/latency metrics.
- `SPARSE_HANDOFF_TRAJECTORIES_2026-11-18.svg` — actual executed paths, not synthetic illustrations.
- Externally retained ZIP: all per-episode trace receipts and checkpoint binaries with SHA256 manifest; binaries are not Git commits.
