# DRIV3R Corpus Factory v58 — post-merge off-policy physical repair

**2026-11-19 — DATA-ONLY — synthetic v53 graph host, NOT native CyberSand.**

## Research question

Can a policy be trained/evaluated on physically observed recovery *after* an off-policy mistake without accidentally learning the deliberately wrong paint as a target? v56 established matched material-memory decisions; v57 executed counterfactual paint forks and exposed a concrete A/B protection asymmetry (A terminal, B recoverable). v58 retains this negative result and makes the recoverable B branch into a stream-format supervised repair challenge.

## Method

1. Read the **pinned** v56 corpus (manifest SHA-256 `b7f4b9f0d76416378c0d38713b4c60df90c149c2ef29533c73cb0e030ec9690e`), reject any mutated input shard.
2. For each of 224 paired decision references, reconstruct its original source world/arm in the original pinned v53 host, replay *all* preceding source actions, and check each streaming observation and legal command against v56 source.
3. At the protected goal, execute the opposite (legally offered) `PAINT_A`/`PAINT_B` command, require wrong physical material and zero mission progress. Preserve the source decision input but **mask the forced-wrong action's training loss to zero**.
4. For nonterminal surface-B mistakes, observe the **actual host's** post-mistake state/event delta and execute `ERASE`, corrected paint and `STOP` in the host. Emit a full sequential policy episode preserving its complete source prefix and three observed recovery decisions. All source prelude and material observations are retained as event streams, not re-expanded historical logs.
5. For terminal surface-A mistakes, preserve only evaluator-side failure certificates, never fabricate a recovery trajectory. Independently fork the nonterminal post-error state for a `STOP`-only negative control.

The source observations and executed actions are model-visible only in ordered streams; the `*_EVALUATOR_ONLY` provenance and fork certificates are not model-visible. TRAIN and evaluation splits follow the original 56 source geometry groups. None of the HELDOUT source material is suitable for training or hyperparameter selection.

## Exact measured outcome

| Synthetic split | Wrong accepted | Terminal A | Recoverable B | Completed recovery trajectories | Recovery decisions (including original full prefix) | Naive STOP successes |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 128 | 64 | 64 | 64 | 1,834 | 0 |
| DEV | 48 | 24 | 24 | 24 | 748 | 0 |
| HELDOUT | 48 | 24 | 24 | 24 | 1,060 | 0 |
| **Total** | **224** | **112** | **112** | **112/112** | **3,642** | **0/112** |

- One intentionally wrong paint per complete B demonstration is **loss-masked**; `ERASE`, corrected paint, `STOP` are labelled with actual host observation/action receipts. The first two recovery actions receive focus weight 6, `STOP` weight 1.
- The synthetic host's 112 recoveries are deterministic oracle runs, not learned-agent successes; 112 fatal A errors have no recovery, as expected.
- 56 geometry hashes retain their original TRAIN/DEV/HELDOUT partitions. No topology-crossing or duplicate fork references.
- Independent Python audit checks all 224 fork references, 112 complete recovery episodes, source prefix/weight integrity, wrong/correct material-result receipts, legal actions, zero-weight wrong interventions, evaluator-only separation and all shard SHA-256s. **PASS.**
- Two complete regenerations gave byte-identical JSONL and manifest, and a modified source input was rejected at the pinned hash gate. **PASS.**

## Reproduction (CPU, no paid API)

Use the `source` scripts from this programme PR plus the exact v56 corpus release matching its pinned manifest:

```bash
node experiments/driv3r/corpus_factory/offpolicy_repair_v58.js /path/to/v56corpus /tmp/v58corpus
python experiments/driv3r/corpus_factory/test_offpolicy_repair_v58.py --source /path/to/v56corpus --output /tmp/v58corpus --modules experiments/driv3r/corpus_factory
```

A source-smoke check without the archived v56 dataset is **not** evidence of the complete benchmark replay. All six generated dataset files and `manifest.json` should be retained in an immutable opt-in Actions artifact in a subsequent consumer run, not silently committed as large replicated histories.

## Consumer priority and limits

Add v58 **TRAIN only** as an auxiliary short off-policy-recovery curriculum alongside broad v54 TRAIN complete episodes and v56 material-memory weighting. The forced intervention should advance the model's recurrent history but contribute zero target loss; after that, score whether the model erases bad paint, selects the physically correct tool and terminates. Preserve paired surface-A severe-error controls and compare with STOP-only baseline. Do not train on DEV/HELDOUT or the fork certificates.

Actual named LFM350/Sprocket model-weight update, neural CPU p50/p95 and peak RSS, autonomous heldout synthetic rollouts, native CyberSand validation and immutable Actions publication **have not occurred** in this data-only lane. No synthetic success above should be construed as real-game behaviour. Next science priority for consumer trainers is actual pinned model/adapter training, not more surrogate controllers.

## Programme ownership

DRIV3R main, CyberSand and Neurounner are read-only for this lane. This post-merge programme lives in one new Draft/HOLD corpus-factory PR on current main; PR #1 is merged and PR #2 belongs to a different trainer lane. No paid API/GPU, force-push, auto-merge or routine-push expensive training.
