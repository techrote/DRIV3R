# DRIV3R C54 — delayed-goal, host-observed branch option (2026-11-18)

**Research-only / Draft-HOLD / not promotable.** Owner-scoped repository: `techrote/DRIV3R`, consolidated research PR #1, directory `experiments/driv3r/training/guidance_recovery/**`. No CyberSand/Neurounner changes. No native CyberSand validation or canonical ABI claim.

## Scientific question and preregistration limitations

After C53's observed-history classifier failed long-horizon evaluation, can a **multi-step option built only from host-observed, previously encountered branch choices**, trained to activate using **paired full-goal success differences**, improve unseen-topology correction without suppressing worm personality?

This is exploratory. An initial 16-graph out-of-family pilot produced zero completed goals across all policies. We then added a clearly separated held-out *familiar-family* cohort to distinguish simulator ceiling effects from genuine OOD failure; this was an adaptive choice made after inspecting the pilot. Do not treat confidence intervals as prospective confirmatory statistics. An attempted larger 96-training-graph refit finished model creation but its evaluation was interrupted; those partial outputs are not part of the claims below.

## Intervention and host safety

The `ObservedLedger` records only the fresh host-legal adjacent action menu at the cell where the worm *actually stands*, plus the actual executed action sequence. It does not inspect distant protected geometry or reconstruct unobserved side streets retrospectively from the world's hidden maze. On a backtrack cue it builds a breadth-first route through previously observed adjacent edges to an observed-but-unvisited frontier, then softly biases the current *legally offered* movement action by +3 logits for at most 18 ticks. On reaching the frontier, cue cancellation, goal change, exhausted budget or missing host-legal next step, the option terminates. All actions remain stochastic and are independently validated by the host executor; no teleportation, painting permission change or deterministic action script.

The learned gate is a small CPU neural head (16→12→1) trained on **actual paired full-goal outcome differences** between the option route and the frozen `C34-RESET` controller. Its only observations are the issued cue, recent executed history and branch ledger, legal local options, and goal bearing/Manhattan distance. A graph-shuffled counterpart, untrained stochastic random gate, unconditional route ablation, RESET, and no-guidance baseline are separate controls. Model gate threshold is fixed at logit > 0, with no OOD threshold tuning.

## Exact local qualification and results

- **Training:** 32 generated, distinct familiar-family topologies (tree, braid, Prim, braided Prim). 23 contributed 68 matched full-goal intervention comparisons after early-terminal filtering. Only **2/68** had different goal success (2 option wins; 0 option losses). Each neural head underwent **72 CPU AdamW updates across 24 epochs**. Graph-level label shuffle changed four training labels. Both checkpoint SHA-256 values are in `C54-evidence-receipt.json`.
- **Blind:** eight independently generated Wilson/Kruskal graphs, one different goal outcome across the paired comparisons; in that outcome the route *lost* to RESET (4 versus 5 goals overall).
- **Online:** 32 independently generated held-out graphs, 16 unseen Wilson/Kruskal and 16 fresh familiar-family topologies. **76 matched guided suffixes per policy**, 38 matched no-cue suffixes; six policy arms. All 12 selected rendered paths were separately reconstructed and matched the original action/paint hashes.

| Policy | Guided goal events / 76 | Condition regressions vs RESET | Multi-step option activity |
|---|---:|---:|---:|
| C18-A0 | 16 |  | 0 |
| **C34-RESET** | **19** | **0** | 0 |
| C54-ROUTE (ungated) | 18 | 1 | 524 option-biased turns |
| **C54-ALIGNED** | **19** | **0** | **0** |
| C54-SHUFFLED | 19 | 0 | 0 |
| C54-RANDOM | 16 | 1 | 325 option-biased turns |

In the **16 entirely unseen Wilson/Kruskal graphs**, every controller achieved **zero** completed goals. The 19 RESET goals were all among the 16 fresh *familiar-family* graphs; the aligned and shuffled heads merely reproduced RESET without intervening. The unconditional option changed behavior (524 biased turns) but cost one goal, with one unfavorable graph; no reliable benefit. ALIGNED's paired goal difference versus RESET is **0** with descriptive graph-cluster bootstrap interval **[0,0]** because it abstained everywhere. This degenerate interval is **not** evidence of equivalence or safety under intervention.

The gate did **not** learn a useful decision boundary: it started zero of the 76 eligible guided episodes. Its shuffled-label counterpart also abstained everywhere. The training set's two rare discriminative goal outcomes are insufficient to support a reliable selective controller. Although no-guidance action and painting hashes matched across all six policies, **guided personality is not qualified**: the ungated option increased preferred-turn and opposed-turn counts compared with RESET, and style is not validated at the whole-trajectory level.

## Safety, counterfactual controls, and limitations

All generated paths used the synthetic host's ordinary `World.apply` authority, produced **zero host rejections and zero protected-cell changes**, and preserved all **38 matched no-cue** movement and ink hashes per policy. Each option transition was checked against the current host-legal menu. The 12 gallery paths were reconstructed with their original seeds, graph identity, prefix state, RNG continuation, checkpoint and action/ink hashes. This is re-execution of the same implementation, not an independent native CyberSand run.

Important limitations: small conditional sample size; adaptively extended mixed online cohort; extremely sparse full-goal supervision; all-out-of-family zero-success ceiling; 15×15 historical synthetic fixture; categorical family overlap not equal to game transfer; option routing examines observed-but-unvisited frontiers and may favour local traps; graph-cluster bootstrap exploratory; style/freedom measured inadequately; no native CyberSand ABI acceptance; historical `d_common` remains a fixture; no immutable Actions checkpoint artifact. No external paid API/GPU resources were used.

## Reproduction and artifact custody

Readable sources: `c54_delayed_branch_option.py`, `test_c54_delayed_branch_option.py`, `c54_gallery.py`. Evidence: `C54-executed-ledger.zlib.b64` and `C54-extract-ledger.py` (8 individually SHA-256-checked source-matched JSON/log files), `C54-gallery.svg`, `C54-evidence-receipt.json`. The full local owner-evaluation ZIP additionally retains both `.safetensors` weights, frozen dependencies, optimizer losses and interrupted-run receipts. **Checkpoints are not committed to Git**. GitHub Actions artifact custody is **outstanding**, not falsely claimed.

Use the manifest-attested frozen C53 prerequisite archive from the previous run to construct the same local paths. With Python 3, NumPy, CPU PyTorch and safetensors available, from that unpacked root:

```bash
PYTHONPATH=.:c47/previous/C45-source:c47/source:c50/source:source_and_evidence \
  python -m unittest -v test_c54_delayed_branch_option
PYTHONPATH=.:c47/previous/C45-source:c47/source:c50/source:source_and_evidence \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python c54_delayed_branch_option.py \
  c47/previous final_small --train-reps 2 --blind-reps 1 \
  --online-reps 2 --online-in-family-reps 1
python C54-extract-ledger.py C54-executed-ledger.zlib.b64 extracted-evidence
```

The full working directory, baseline prerequisites, exact filenames and artifact hashes are in the separate local ZIP. Deterministic rerun may depend on the exact CPU torch/numpy/safetensors versions in that retained environment. The C53 research archive is an original historical source artifact, not a native acceptance claim.

**Decision: REJECT / HOLD C54-ALIGNED**. Do not promote, merge, or deploy. The next useful model experiment needs substantially more *causally discriminative completed-goal* branch-recovery examples on viable unseen graphs, with an effective non-degenerate success baseline, strong personality metrics, and strict independent held-out controls. Avoid another local-label classifier or further easy-family optimisation without resolving this data deficiency.
