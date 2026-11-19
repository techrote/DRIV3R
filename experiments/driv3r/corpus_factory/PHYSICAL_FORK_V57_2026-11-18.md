# DRIV3R Corpus Factory v57 — Host-replayed physical counterfactual forks

Date: 2026-11-18. **DATA / EVALUATOR ONLY; synthetic graph host, not native CyberSand.**

## Motivation and method

v56's opposite PAINT command consequences were inferred from the two-law symbolic mapping. v57 replaces that inference with **actual fork execution in the original pinned v53 synthetic host**, from the same decision prefix and precisely the same world/arm. For each of 112 pairs (224 material-decision references), reconstruct the world with v46 `candidate(split, 700 + world_index)` and original v53 arm, execute the source-labelled decisions using v52/v51 host commands, and fork at the protected material decision:

- **Correct branch:** execute the original oracle action and require the physical paint result matches the target surface and advances the mission stage.
- **Wrong branch:** execute the opposite *legal* PAINT command from a separately reconstructed identical prefix; require physical result differs from the target, command accepted, and mission stage does not advance.
- **Recovery branch:** on nonterminal incorrect paint, execute bounded observable-state oracle steps (ERASE, corrected paint, STOP), requiring terminal success. On terminal failure record no recovery.

No answer-labelled counterfactuals, physical forks, or future outcomes are exposed to a model. All `*_EVALUATOR_ONLY.jsonl` shards remain evaluator-only. v56 TRAIN data stays unchanged; this version does not train an SLM or change canonical game behaviour.

## Exact verified results

| Split | Material pairs | Correct mission advances | Wrong branches | Wrong terminal failures | Wrong nonterminal bad paints | Oracle recoveries | Equal 32-step history pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 128 | 128 | 64 | 64 | 64 | 0 |
| DEV | 24 | 48 | 48 | 24 | 24 | 24 | 0 |
| HELDOUT | 24 | 48 | 48 | 24 | 24 | 24 | 3 |
| **TOTAL** | **112** | **224/224** | **224** | **112** | **112** | **112/112** | **3** |

**Important interpretation:** all 112 A-surface wrong paints were terminal failures under the *synthetic* protected paint hook, whereas all 112 B-surface wrong paints were nonterminal mistakes; they did not advance the mission and were recoverable by the deterministic ERASE → corrected PAINT → STOP sequence. This exposes a meaningful A/B asymmetry in the synthetic host's protection semantics. The original v56 blanket phrase “wrong-command failure inferred” should be narrowed: v57 now *replays* the negative branch, but a wrong B command is not an irreversible failure. These are host-coded outcomes, not learned policy behaviour.

## Integrity and reproducibility

- Complete v56 parent source manifest SHA-256: `b7f4b9f0d76416378c0d38713b4c60df90c149c2ef29533c73cb0e030ec9690e`. v56 shard identities are pinned in `V57_DEFAULT_MANIFEST.json`.
- Nine exact v53/ancestor host module SHA-256 values and the v57 generator SHA-256 are in the pinned manifest. Local files' Git blob SHAs were checked against current DRIV3R shared branch.
- Python independent auditor rechecked all 112 pair IDs, 224 source action labels, legal opposites, physical target/outcome fields, provenance, terminal/recovery classifications, all shard hashes and split geometry isolation (56 unique worlds). **PASS.**
- Two full regenerations in separate output directories were byte-identical for every output shard and manifest. **PASS.**
- A deliberately altered source shard was rejected by the generator's pinned SHA gate. **PASS.**

## Scientific boundaries

The 224 correct actions are teacher-forced host replays, **not autonomous SLM successes**. The 112 recoveries use a deterministic observational oracle, not a learned recovery policy. SLM weights and CPU inference latency/RSS were not measured. Only three HELDOUT material pairs have an identical 32-step trailing action-observation history; TRAIN has none. These data should supplement, not replace, broad whole-episode v54 training.

No paid API or GPU, no out-of-lane code changes, no CyberSand or Neurounner write, no main modifications and no expensive push-triggered training.

## Consumer handoff

Consumer trainers should keep v54 full-episode TRAIN with optional v55/v56 causal weighting; use v57's evaluator-only forks to distinguish irreversible protected damage from recoverable but progress-blocking paints. Prefer pinned SLM training and autonomous held-out evaluation with memory ablations, coverage and terminal success, plus CPU p50/p95 latency and peak RSS. No HELDOUT optimization.

**Administrative change seen at read time:** DRIV3R PR #1 was already merged by an external actor on 2026-11-18 22:54:12 UTC; this lane did not merge it or modify main. The shared research branch remains separately writable and was the sole intended mutation target. No replacement PR/branch/issue was created.
