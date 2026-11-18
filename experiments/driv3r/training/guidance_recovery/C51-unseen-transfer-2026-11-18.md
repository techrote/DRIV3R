# DRIV3R C51 — frozen C40 independent Wilson/Kruskal replication

**18 November 2026 · Research HOLD · synthetic host only · no model promotion**

**Question:** Does C50's +14 frozen-C40 continuous mid-trajectory recovery gain transfer reliably to new, independently seeded Wilson/Kruskal graph identities? The C50 signal was almost wholly in familiar graph families. C51 tests OOD families only, with no training, checkpoint edits, candidate selection, or tuning on these C51 outcomes.

## Procedure and source custody

- C40 checkpoint frozen at SHA-256 `5d9d7c6602e3be68ec7ad3a593e1c06e709b4804974107ef572c30a548766184`. C51 executes the original C50 synthetic-host `prefix` and `suffix` functions, with same controller instantiation at cue, persistent host world, RNG state and shared global 78-tick clock.
- Graph seed `510118` was chosen independently from C40–C50 evaluation seeds; 12 replicas each of 2 generators × 2 cases × 2 personality values × 2 switch flags produced 96 graph identities after excluding switch episodes. The C50 graph hashes and the new graph hashes are disjoint.
- 64 graph identities supported at least one eligible nonterminal correction; the rest were excluded by the already-defined `prefix`/`resolved_goal` criteria. The 64 accepted graph clusters yielded 89 usable cue states at t=9/t=19, 356 matched three-policy/arm rows (89 no-guidance + 267 guided per policy), and **1,068 executed synthetic-host episodes**. Do not mistake this for 1,068 independent graphs.
- Four matched cue arms: no guidance, backtrack at 0.35, backtrack at 1.0, and continue at 1.0. Policies: C18-A0, C34-RESET and frozen C40-PAIR. Exactly the same host state, cue, candidate legality, RNG state and tick budget were used in each counterfactual arm.
- The primary inference samples entire graph clusters, not cue/arm rows. The graph bootstrap is descriptive; the additional graph-level sign-flip randomization is exploratory and unadjusted for prior selection. The strict pass gate requires improvement on *both* unseen families, bootstrap lower bound above zero, no coarse/fine condition regressions, exact unguided hashes, and at least 60 eligible graphs.

## Outcome: no reproducible out-of-family gain

| Policy | Guided goals / 267 | Wilson goals | Kruskal goals |
|---|---:|---:|---:|
| C18-A0 | 31 | 18 | 13 |
| C34-RESET | 42 | 16 | 26 |
| **Frozen C40-PAIR** | **43** | **16** | **27** |

**C40 vs RESET: +1 goal from only one favorable graph**, out of 64 clusters; 63 clusters had zero difference. No negative graph clusters occurred, but the descriptive graph-cluster 95% interval is **[0, +3]**, not strictly positive. The exploratory two-sided sign-flip p-value is 1.0 because only one graph has a nonzero paired difference. On Wilson the delta was **zero across 31 graphs**; on Kruskal it was **+1 across 33 graphs**. The strict promotion criterion therefore failed in two independent ways (cross-family consistency and positive bootstrap lower bound). C40 vs the unguided C18-A0 baseline was +12 goals but this does not establish an advantage over the recovery controller.

There were **no coarse or personality-stratified condition regressions** in this sample, but that must not be interpreted as proof of reliable nonregression. All 89 matched no-guidance movement and painting hashes per policy were exact, zero synthetic host actions were rejected, and protected sets remained invariant. Across guided signed painting actions, preferred-turn share was 54.0% for C18-A0, 57.2% for RESET, and 55.0% for C40. This is only a proxy, not demonstration of preserved whole-trajectory guided personality.

Five validation tests passed, including byte-for-byte checkpoint/source SHA verification, cohort graph disjointness against C50, ledger consistency, legal/unguided parity, and **12/12 selected full path replays**. Nine additional diagram panels were reconstructed and SHA-checked against their executed action/ink receipts (one favorable graph, one parity success graph, one parity failure graph × three policies). No depicted route was planned or imagined.

## Reproduction and limitations

The retained archive contains exact C51 Python source, source/test/gallery scripts, summary, all 356 per-arm rows, selected paths, replay receipts, compact SVG, test logs, hashes and the frozen C50/C49 prerequisite archive. Unpack with Python 3 and CPU `torch`, NumPy and safetensors installed, then run:

```sh
export PYTHONPATH=.:prereq50/source:prior/source:prior/previous/C45-source
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python c51_unseen_replication.py prior output_c51 --reps 12
python -m unittest -v test_c51_unseen_replication
python c51_gallery.py
```

All 96 candidate graphs and accepted 64 identities were generated once under the same fixed seed; repetition count was increased from an initial small smoke run to 12 based on eligible-sample size, not on unseen outcome comparisons. There is only **one independent C51 seed cohort**; no claimed multi-seed statistical confirmation. The attrition makes these results conditional on having an eligible correction opportunity. The host remains synthetic and uses a limited 7×7 maze model, not the native CyberSand action ABI or Workbench. The C40 checkpoint was previously selected after many experiments, the descriptive intervals do not correct for model-selection and repeated tests, and the current model lacks robust native/held-out personality qualification. Immutable GitHub Actions checkpoint custody is still missing.

**Disposition: HOLD.** C51 does not replicate a reliable C40 OOD transfer advantage. Do not promote C40, change native gameplay, merge PR #1 or open another research PR. Future research should widen genuinely distinct world dynamics and verify host-native observation/action semantics before another model-optimization sweep.
