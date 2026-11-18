# DRIV3R Corpus Factory v35 — trainer-safe v34 recovery export

**18 November 2026 · synthetic DRIV3R host only · not CyberSand-native**

## Purpose

v34 has 68,000 labelled decisions across 1,536 synthetic host-accepted recovery episodes, but its decision records mix the policy-facing bounded observation with teacher-only `oracle_action_costs`, `episode_id` (including explicit `arm`), `applied_action`, reward/regret, and intervention metadata. Feeding entire records to an SLM would leak supervision and invalidate the held-out memory result. v35 produces an explicitly safe, standalone per-step consumer format and paired DEV/HELDOUT first-action memory challenges. **It does not add new worlds or train a model.**

## Run from an exact v34 dataset

```
node experiments/driv3r/corpus_factory/compound_intervention_v34.js /tmp/v34-shards
node experiments/driv3r/corpus_factory/trainer_safe_export_v35.js /tmp/v34-shards /tmp/v35-safe
node experiments/driv3r/corpus_factory/test_trainer_safe_export_v35.js /tmp/v34-shards /tmp/v35-safe /tmp/v35-safe-repeat
```

The v35 generator verifies every v34 shard byte count and SHA-256 and the predecessor generator/dependency hashes before exporting. It refuses to overwrite an existing destination. Generated shards belong in artifacts, **not Git**. The pinned default source dataset is identified by SHA-256 in the v35 manifest. The first script requires the v34 dependencies to be available in the same lane directory.

Use **only `row.input` as model input**. `row.action` is the supervised action label; `row.weight` is the sample loss weight, with zero-weight duplicate or injected-error teaching rows excluded. Never concatenate the entire JSON row into the model prompt. No episode ID, family ID, split marker, oracle costs, regret, applied action or intervention is present in the model input. The memory challenge's `historyless_input` is an **evaluation-only** explicit ablation; it is not a training target. The complete v34 episode receipts remain separately useful for genuine closed-loop evaluation.

## Qualified default output

| Split | Episodes in parent | Safe decisions | Weight 1 | Weight 0 | Four-way clean families |
|---|---:|---:|---:|---:|---:|
| TRAIN | 768 | 23,008 | 17,440 | 5,568 | 64 |
| DEV | 384 | 19,136 | 19,008 | 128 | 32 |
| HELDOUT | 384 | 25,856 | 25,728 | 128 | 32 |
| **Total** | **1,536** | **68,000** | **62,176** | **5,824** | **128** |

Separate DEV and HELDOUT memory challenge shards each contain 128 first-action examples arranged into 32 counterfactual groups. Within each group, **identical memoryless model input requires all four different legal actions** (`MOVE_W`, `MOVE_E`, `MOVE_S`, `BACKTRACK`); the maximum deterministic single-action accuracy per group is 25%. Full bounded inputs have distinct durable paint receipts. This is a mathematical property of the synthetic labels, **not a measured SLM score**.

Key audit results:

- Zero contradictory **full bounded-model-input** labels (note distinction from complete host observation).
- Zero cross-split full bounded-input and geometry overlaps.
- 5,568 duplicated bounded TRAIN inputs retain weight zero; 256 accepted bad-PAINT teaching rows are included in this count and have weight zero. DEV and HELDOUT each have 128 zero-weight injected rows; they are never trainer material.
- DEV and HELDOUT contain repeated observations (5,344 and 7,584 bounded duplicates) with consistent full-input labels; do not mistake the decision count for independent test contexts.
- Two independent complete exports are byte-identical across manifest and five JSONL shards; independent tests verify all exported rows, SHA-256, split isolation and every four-way memory alias.
- Local generation: 4.83 s, ~144 MiB peak RSS. The first independent full test/two-generation run took 14.56 s and reached ~1.46 GiB RSS due to test-side in-memory collection; **not model inference RSS**.

## Scientific interpretation and negative results

This release remedies a **consumer leakage risk** and delivers usable TRAIN/DEV/HELDOUT samples, not a new reasoning or game-law challenge. Synthetic topology and scripted old-history interventions are inherited from v34. A symbolic full-history interpreter can still solve four-way memory; no named SLM weight update, LoRA, native CyberSand game integration, real policy closed-loop rollout, inference latency or model RSS is claimed. Immutable GitHub Actions shard publication is still outstanding; local exported data is subject to the pinned hashes in `V35_DEFAULT_MANIFEST.json`.

**Next consumer experiment:** a pinned LFM350 or Sprocket-500M train-only adapter update on `TRAIN_policy.jsonl` with `weight=1`, no held-out ingestion, followed by autonomous host-legal HELDOUT trajectories on clean/bad-navigation/compound-error cohorts; compare full durable vs deliberately truncated memory and report action legality, terminal completion, residual paint errors, excess steps, latency p50/p95 and RSS. Keep native CyberSand acceptance separate.
