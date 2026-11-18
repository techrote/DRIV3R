# Corpus Factory v12 — full-path synthetic replay and trainer-ready challenge

17 November 2026. Data-only increment extending the **source-pinned v11** delayed shutter and material-law corpus. Repo: techrote/DRIV3R existing Draft/HOLD PR #1.

## Problem and intervention

v11 contained one labelled decision after a host-scripted >32-position prelude. That is useful for causal memory classification, but not for meaningful model-controlled downstream trajectories. v12 reconstructs the same material grids, protected cells, host-legal cardinal menus, hidden gate states and dynamic material costs from the v11 world provenance, then replays **every oracle movement** to the goal. It exports full episode initial states plus seven sampled and labeled sequential decision states per episode, with paired last32 and zero-history controls. A minimal exported `createHost/observe/apply` API enables consumers to replace the oracle with their actual pinned SLM without touching CyberSand.

## Executed in-process validation

- Pinned v11 source Git blob: `be98237e63992da967b9a4ce309c549c11e67285`.
- Default two-style-seed dataset: TRAIN 128 episodes / 896 labelled decisions; DEV 48 / 336; sealed HELDOUT 48 / 336. 56 world groups in disjoint grammar/topology partitions.
- 15520 synthetic host-applied oracle movements across 224 episodes; every reference route reached the goal with independently recomputed material cost; zero protected entries. Invalid diagonal action rejected without moving in every episode.
- Two full in-process JavaScript executions produced byte-identical data structures. Measured combined generation time: 10312 ms in this V8 executor (not representative of full Node disk IO).
- At the initial junction, each pair has the same last32 model observation and opposite optimal move; fully causal history is needed. Downstream decision samples are correlated and should not be treated as independent worlds.

## Limitations / host boundary

**External synthetic host, NOT native CyberSand**, and **oracle-controlled**, not a measured learned policy. The generated decision rows are supervised data; only `model_input` is allowed into a model. `privileged_host`, `expert`, `controls` and IDs are evaluation-only metadata; using them as features leaks the answer. Source-matched Node CLI tests and actual generated shard SHA-256 checks still need independent execution; no immutable Actions corpus artifact was published. Painting motif remains in v11 because there is no validated brush/cursor application law in the v12 synthetic replay host.

## Consumer command

```sh
node experiments/driv3r/corpus_factory/test_replay_rollouts_v12.js
node experiments/driv3r/corpus_factory/replay_rollouts_v12.js /new/output/dir 2
```

CLI creates separate TRAIN/DEV/HELDOUT episode and decision JSONL files and a manifest with their actual byte counts and SHA-256 digests; refuses to overwrite an existing output directory. Do not train on DEV/HELDOUT. No paid GPU/API costs, PR or workflow proliferation, or writes outside the corpus factory.
