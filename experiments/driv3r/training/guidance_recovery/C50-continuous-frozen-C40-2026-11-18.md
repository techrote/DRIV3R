# C50 — frozen C40 mid-trajectory replication with carried host state and RNG

**18 November 2026 — RESEARCH HOLD. Not promoted, not native gameplay.**

Repository: `techrote/DRIV3R`; only Draft/HOLD PR #1. Own directory: `experiments/driv3r/training/guidance_recovery/**`. No CyberSand/Neurounner/DRIV3R-main writes. No new branches, PRs, workflows, Actions runs, paid GPU or API usage.

## Question and controlled difference

C49 found an exploratory nine-goal mid-trajectory advantage of frozen C40-PAIR over C34-RESET in a suffix-only experiment. Its suffix reset the random stream and local tick to zero, so that evidence did **not** qualify uninterrupted recovery.

C50 reruns the **frozen** C40-PAIR neural policy; it does not train a new head. Each independent synthetic-host world executes an actual stochastic, host-legal baseline prefix through global tick 9 or 19. The host then changes or reissues an unvisited goal and issues a matched cue. The compared controllers receive identical world snapshots, prefix hashes, initial post-cue random-generator states and goal events. They run to the **same global tick-78 limit**; the RNG state and host state continue across the cue. A correction controller is instantiated when the cue arrives, so its internal phase begins at the cue (not before).

Arms: NONE 0, BACKTRACK weak 0.35, BACKTRACK strong 1.0, CONTINUE 1.0. Models: C18-A0, C34-RESET and unmodified frozen C40-PAIR. C40's stored checkpoint SHA-256 is `5d9d7c6602e3be68ec7ad3a593e1c06e709b4804974107ef572c30a548766184`, matching C49's committed predecessor receipt. Synthetic host, *not* the canonical native CyberSand ABI.

## Held-out evidence

Generated 96 graph identities across DFS tree/braid, randomized Prim/Prim-braid, Wilson and Kruskal families; after rejecting 92 terminal-prefix instances, 100 valid post-prefix scenarios remained on **62 distinct graphs** (42 in-family, 20 Wilson/Kruskal). Each valid scenario yielded four arms and three policy outcomes: **1,200 executed policy episodes** in total, with **300 guided episodes per policy** and 100 matched unguided trajectories per policy. Two cue ticks for a graph are correlated; graph-cluster uncertainty samples the entire graph.

| Controller | Goals / 300 guided | In-family goals | Wilson/Kruskal goals | Condition groups below RESET |
|---|---:|---:|---:|---:|
| C18-A0 | 77 | 65 | 12 | 8 |
| C34-RESET | 83 | 69 | 14 | 0 |
| **C40-PAIR (frozen)** | **97** | **82** | **15** | **1** |

Paired graph-level **C40 minus RESET = +14 completed-goal events**, from nine favorable and two unfavorable graph clusters. Unadjusted descriptive 95% cluster-bootstrap interval **[+4,+26]**. C40 minus A0 = +20 events, interval [+1,+40]. This is an *exploratory replication signal*, not confirmatory significance: C40 was selected after many C40–C49 experiments, and multiple outcomes and families have been examined. Importantly, **13 of the 14 additional goals arise in the in-family cohort**. OOD improvement is just +1 across 20 Wilson/Kruskal graph identities and does not establish transfer.

BREAKTRACK (weak): in-family C40 31 vs RESET 24; OOD 7 vs 6. BACKTRACK (strong): in-family 30 vs 24; OOD 6 vs 6. CONTINUE: exact goal-event ties on both cohorts. All 100 unguided movement-action and painting hashes matched across all three policies. No executed host action rejections or protected-set mutations. Two raw illegal top-choice proposals were filtered for RESET and C40. Active-guidance stylistic equivalence and native physical gameplay remain **unproven**; one condition-group regression prevents promotion.

## Reproducibility and custody

The local CPU execution used numpy, torch and safetensors with fixed source, seeds, matched RNG streams and a prior-source archive. All four targeted local tests passed, including independent manual RNG-continuity execution and **18/18 replays** of selected synthetic-host paths against original action, ink and path receipts. A nine-panel SVG gallery was reconstructed on 3 graph identities showing a C40 win, a C40 loss and OOD parity; these are executed paths, not native screenshots.

SHA-256 source receipts:
- `c50_continuous_recovery.py`: `97fd4077a6771714515f612fabd2fda365ddb6f4f08cb96079d008331e0d7e2f`
- `test_c50_continuous_recovery.py`: `3b099fb043eeee41063ba36924b7375bdbfc64879b550b4593c3757d3651ab9b`
- `c50_gallery.py`: `f3f82aa7eca0e202d165d62167c8179073ecd089fa923412f69005647cccb8f3`
- `C50-summary.json`: `cc9faaaf7a7f53a4137f9c6b724f01bd650d7350144c218690c2551c447124f3`
- Full per-action online ledger: `43752c6465d06cb775a54a42cea1d74be274c66866018ef87c9b63c7991982ee`
- Nine-panel SVG: `f23e7d00e4be2264eda63f9b90c8dbfda016c5d2f71e39117de9263e344d8c2a`
- Replayed selected-trace ledger: `eb81edc6e40646e684c447669c0ed97ad5fd5659e822b6dc7ed889d89a022d4a`

Source-matched predecessor: the C49 local archive's `c49_midtrajectory_correction.py` SHA-256 `965c55ca177f86a2464c0d8801caae13228338ec9ccd363a5cc564f90971d1ca` matches its GitHub committed evidence receipt.

Run (from the locally preserved C49 dependency archive and the C50 source archive):

```sh
export PYTHONPATH=source:prior/source:prior/previous/C45-source
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python source/c50_continuous_recovery.py prior output_c50 --reps 4 --trace-count 18
python -m unittest source/test_c50_continuous_recovery.py -v
python source/c50_gallery.py
```

The dependency archive includes an externally retained C40 checkpoint; **repository-only reproduction and immutable GitHub Actions checkpoint custody are not yet achieved**. There is no claim that source import alone reproduces the experiment. Complete C50 trajectories and checkpoints remain outside Git history.

**Disposition: HOLD.** A stronger in-family recovery signal survives state/RNG-continuous synthetic execution, but independent Wilson/Kruskal transfer, active personality, native CyberSand validation and zero condition regressions do not pass. Next: larger genuinely blind graph-family replication before any promotion; avoid training a new selector on these selected results.
