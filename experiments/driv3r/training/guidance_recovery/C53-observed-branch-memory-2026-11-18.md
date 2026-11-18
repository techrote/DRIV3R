# C53 — observed episodic branch-memory recovery · 18 November 2026

**Repository:** `techrote/DRIV3R` Draft/HOLD PR #1, branch `research/import-consolidated-worm-labs` only. **Research disposition: REJECT / HOLD.** No native CyberSand integration or promotion.

## Experimental question

C52 learned a weak local recovery-action classifier but failed to improve Wilson/Kruskal navigation. C53 tests whether an **eight-feature, host-observed episodic branch ledger**, appended to C52's sixteen present-local/cue features, improves recovery of the same action-level head. Memory features summarize visited positions and previously traversed edges, local revisit density, backtrack-target recency, and witnessed junctions. Memory is reconstructed exclusively from the **executed host event/history ledger and currently legal adjacent menu**, with no distant map lookup. This is a deliberately limited compressed memory, not a map or learned recurrent planner.

Frozen C18-A0, C34-RESET, C40-PAIR and prior C52-ALIGNED remain exact-source controls. The historical `d_common` fixture is not a canonical CyberSand ABI.

## Frozen design, controls, and bounded optimizer

- Training families: DFS/Prim (`tree`, `braid`, `prim`, `prim_braid`); blind and online: independently seeded Wilson and Kruskal. World identities are geometry-hash distinct across 32 training, 16 blind and 64 online graphs, and excluded from C51 and C52 online/training/blind worlds. Seed and deterministic retry receipts are retained.
- Training: 1,118 host-visited, supervised decision states (490 positive). Labels are a **private full-geometry BFS shortest-next-step oracle** used only in training and blind classification. The oracle is not available to inference. Samples are strongly correlated within 32 graphs.
- Three 24-input, 16→8→1 MLP heads underwent **288 real CPU AdamW updates each**, with the same initial parameters and training order. `C53-MEM` sees aligned 16 local + 8 memory features; `C53-LOCAL` receives identical aligned labels but zeroed memory inputs; `C53-SHUFFLED` receives graph-shuffled labels. `C53-MASKED` is the fitted memory head with memory features substituted by training means at inference. A random correction head is also included.
- Inference: applies a *bounded* learned BACKTRACK-logit delta only while an active host-issued BACKTRACK cue and legal backtrack action are present. All other choices are sampled from the host-masked menu; no mandatory action scripts.
- 576 held-out blind classification states, 276 online graph/cue records and **207 guided episodes per policy**; 1,380 independently executed online suffixes, with additional equivalent no/continue results exactly reused and verified. Two cue ticks (9, 19) and none/weak/strong/continue arms use the same host-executed prefix and RNG state.

## Blind classification: offline signal is not gameplay success

| Head | Accuracy | Brier (lower is better) |
|---|---:|---:|
| C53-MEM | 65.97% | **0.22283** |
| C53-LOCAL | 62.50% | 0.23140 |
| C53-SHUFFLED | 37.33% | 0.29717 |
| C53-MASKED | 57.64% | 0.26773 |
| Frozen C52-PREV | **68.40%** | 0.22920 |

Memory improves *its matched LOCAL ablation* on the held-out action-level surrogate (accuracy +3.47 pp; Brier -0.00857) but does not beat frozen C52-PREV's accuracy. These are correlated observations across only 16 graphs; do not attribute significance to the point estimates.

## Online goal completion and negative findings

**207 matched guided episodes per policy** over 64 held-out graph topologies:

| Controller | Goals | Wilson | Kruskal | Condition regressions vs RESET |
|---|---:|---:|---:|---:|
| C18-A0 | 35 | 23 | 12 | 8 |
| C34-RESET | **51** | 27 | 24 | 0 |
| Frozen C40-PAIR | **56** | 32 | 24 | 1 |
| Frozen C52-PREV | 39 | 25 | 14 | 8 |
| **C53-MEM** | **39** | 24 | 15 | **7** |
| C53-LOCAL | 40 | 26 | 14 | 6 |
| C53-MASKED | 37 | 24 | 13 | 7 |
| C53-SHUFFLED | 33 | 21 | 12 | 9 |
| C53-RANDOM | 37 | 23 | 14 | — |

`C53-MEM − RESET = −12` goal events across 64 graph clusters (seven unfavorable, one favorable; descriptive graph-cluster bootstrap interval **−27 to +3**). `C53-MEM − C53-LOCAL = −1` (one favorable and one unfavorable graph; descriptive interval **−6 to +3**). `C53-MEM − SHUFFLED = +6` (four favorable and one unfavorable; descriptive interval **0 to +14**): this is a weak negative-control contrast, not a demonstration of general usefulness. The frozen C40 advantage remains condition-regressive and is not a production acceptance result.

**All strict promotion gates do not pass.** While C53-MEM changed action distributions and blind labels substantially, adding observed branch history to a single-action BACKTRACK classifier did not improve held-out full-goal recovery over a local-only matched model or the non-neural controller. The mechanism likely needs explicit branch-option planning and reliable long-horizon credit assignment, rather than more local-classifier capacity.

## Safety, personality, replay and custody

- Zero executed synthetic-host action rejections or protected-cell modifications across all recorded models. Every one of **69** no-guidance prefix/continuation action and ink hashes was identical across all nine policies.
- The 18-panel gallery reconstructs **three matched host worlds** across six controllers. All 18 selected post-cue paths were independently rerun with matching action SHA, painting SHA and normalized full paths.
- Seven regression tests pass, including two independent CPU refits reproducing all three checkpoint SHA-256 hashes; exact source and C52 frozen checkpoint/teacher receipts were checked.
- **No whole-trajectory guided personality acceptance.** A coarse preference-versus-opposition ink-turn count exists, but a matching unguided hash is not evidence that guided personality survives.
- Research is a fixed 15×15 synthetic grid. No native CyberSand validation, dynamic material reactions, real Workbench actor acceptance or canonical ABI qualification. Graph-cluster intervals are descriptive only, not corrected for the large C40–C53 research sequence. **An initial 24-graph online preview (3 reps/family/case/personality) was inspected before increasing to 64 graphs (8 reps); this is an adaptive sample enlargement, not a pre-registered confirmatory test.** The 64-graph result includes the 24-graph pilot; pilot logs and negative results are preserved. Low number of successful episodes and recurrent graph families limit inference.
- Checkpoints are preserved in the local SHA-256-manifested research ZIP but **have not been placed into immutable GitHub Actions artifacts**; do not commit binaries or claim artifact custody.

## Reproduction and source identity

Use the versioned `c53_episodic_memory.py`, `test_c53_episodic_memory.py` and `c53_gallery.py` under the owned laboratory path. Frozen prerequisites are the source-matched C52 archive and its C51/C50/C49/C47/C42 dependencies. `C53-summary.json` contains training curves, cohort identities, graph generation retries, model hashes and promotion-gate values. `C53-executed-evidence.zip.b64` contains full training examples, blind examples, all 276 online records, the 18 replays, normalization, and logs, with a `C53-extract.py` verifier. `C53-gallery.svg` is constructed only from the replayed paths. Full checkpoint binaries remain outside Git.

**HOLD:** Do not merge or promote PR #1. Next experiment should test an explicit *observed branch ledger with multi-step option selection and delayed feedback*, while retaining the same RESET/LOCAL/SHUFFLED controls and an independent unseen-family goal-completion gate. Avoid further optimization solely for a locally scored BFS surrogate.
