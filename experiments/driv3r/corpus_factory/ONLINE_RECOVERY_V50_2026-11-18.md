# DRIV3R Corpus Factory v50 — physical-feedback recovery after unsignalled contextual law drift

**18 November 2026 · DATA ONLY · synthetic graph/material host, NOT native CyberSand · Draft/HOLD PR #1**

## Question / hypothesis

Prior v49 demonstrated historical physical calibration of two independently mapped material surfaces, but tool laws remained static for the rollout and the per-surface last-calibration rule always succeeded. V50 tests **genuinely surprising physically wrong paints**: a synthetic host silently changes A and/or B material-command mapping on first arrival at an objective, *after* the last known calibration. The demonstrator teacher must make the initially reasonable old-law command, observe actual wrong material, physically erase, update its belief using the accepted paint outcome, then produce the correct material and complete both goals. No privileged flip, hidden law, or expected paint result is in the model input.

This is a deliberately causal recovery corpus, not a policy trainer or evidence of root-SLM ability. The benchmark remains soluble by an event interpreter and does not establish real CyberSand dynamics.

## Design and execution

- **80 source-world-disjoint graph worlds**, using v46 candidates at indices 200–247 TRAIN, 200–215 DEV and 200–215 HELDOUT, plus split-specific source seeds. Cryptographic geometry comparisons against attached v44–v49 releases found **zero overlaps** (including across splits). Cross-version provenance remains explicit for consumers.
- **12 episode arms per world**: static normal, drifting surface A, drifting surface B, drifting A+B, static reversed A/B laws, reversed-law A/B drift, mission retarget, gate closed, and two obsolete-probe controls. Every differing arm has host-accepted legal movement/paint/erase actions.
- The host changes physical mapping **without an announcement at the objective**, after the old host-executed PAINT_A/ERASE calibration. The model-visible observation at the very first drift-affected paint is byte-identical to the matching static arm, including its legal menu and all historical receipts. Consequently both issue the same initial action and *only physical host feedback* makes subsequent recovery possible.
- After each issued paint, the host publishes `[at, PAINT_RESULT, issued command, actual material, surface, location]`. It does not publish a `TOOL_LAW` or a drift flag. An incorrect material is visibly present in the painted-cell observation, must be ERASEd and then repainted. No retroactive oracle annotation is supplied to the trainer input.
- Mission history remains old and superseded, remote obstructions remain delayed, local map knowledge is partial; wrong-material paint from the prelude is distinguished from new drift-induced failures.
- Scenario/arm/drift/outcome metadata are **evaluator-only**, and are not present in `policy.input`. `policy.action` and `policy.weight` are supervised labels, never inputs. Repeated exact TRAIN inputs and TRAIN movement states which differ only by irrelevant calibration history are loss-masked at weight zero; DEV and HELDOUT are unsampled.

## Qualified full corpus

| Split | Source worlds | Terminal synthetic episodes | Supervised decisions | Wrong physical paints | Drift-recovery erases |
|:--|--:|--:|--:|--:|--:|
| TRAIN | 48 | 576 | 12,708 | 288 | 288 |
| DEV | 16 | 192 | 5,665 | 96 | 96 |
| HELDOUT | 16 | 192 | 6,982 | 96 | 96 |
| **TOTAL** | **80** | **960** | **25,355** | **480** | **480** |

TRAIN had **3,839 exact duplicate records and 2,806 additional near-duplicate movement records** loss-masked (6,645 total zero-weight decisions). Across all splits, there were **2,400 physical paint decisions** including 480 incorrect attempts and two successful objective paints per episode.

Two full dataset generations were **byte-for-byte identical** across six JSONL shards and the manifest. Independent label validation used a separately implemented observable-history action validator; an independently written replay dispatcher checked every recorded input, host-accepted action, terminal completion and physical paint outcome across **25,355 decisions** and **960 episodes**. No contradictory exact-input labels or detected cross-split topology leaks. Full checks used Node 22.16.0 on CPU. Generation completed locally in ~18–19 seconds wall with peak measured RSS 377–560 MB; replay audit ~19 seconds and ~1.06 GB peak RSS. These are **corpus generator/auditor figures**, NOT model inference latency or model RSS.

## Explicit negative control (closed-loop, HELDOUT only)

A frozen **stale-calibration symbolic control** ignores all physical outcomes after the scripted initial A/B calibration, even when a host paint visibly contradicts the original law. It successfully completes **112/192** HELDOUT episodes, exactly the seven non-drift arms per world, but gets stuck in paint/erase cycles for **80/80 drifted episodes** within 256 policy decisions (9,423 wrong attempts). By contrast the oracle using actual physical feedback completes **192/192**, including all 80 drifted episodes. This discriminates *online physical feedback recovery* from memorised static calibration, but **is not neural model improvement**.

## Files, reproduction and scientific controls

- `online_recovery_v50.js`: deterministic generator, host-action case construction, strict causal checks; run `node online_recovery_v50.js /tmp/NEW_CORPUS`.
- `test_online_recovery_v50.js`: independent observable-only label checks, full exact-input host replay; run `node test_online_recovery_v50.js /tmp/NEW_CORPUS`.
- `benchmark_controls_v50.js`: HELDOUT stale-calibration control; run `node benchmark_controls_v50.js`.
- `V50_DEFAULT_MANIFEST.json`: pinned full-generation JSONL shard SHA-256 and source/parent SHA-256. Generation requires only Node standard library and pre-existing v44–v49 parent source files. Every output path must be non-existent before generation.
- `LATEST_HANDOFF.json`: portable consumer handoff; producer scope restricted to `experiments/driv3r/corpus_factory/**`.

**Retained locally for transport:** all six TRAIN/DEV/HELDOUT JSONL shards and full manifest, with SHA-256 values in the pinned manifest. No checkpoint binary is committed to Git. The JSONL shards are **not yet retained as immutable GitHub Actions artifacts**; local ZIP handoff is not a substitute for that provenance. No new GitHub Actions workflow is introduced, and routine push must not trigger model training.

## Negative findings / boundaries

1. The host is a **synthetic graph/material adapter**. Unsigned law flips are not validated against native CyberSand mechanics; physical observed-material recovery is only synthetic-host-executed.
2. A few well-written rules can still solve all of v50. Neither root LiquidAI LFM350 nor Sprocket weights were updated or evaluated here; no neural inference p50/p95 latency, neural RSS, GPU, paid API, or native CyberSand closed loop.
3. Early graph survey and calibrations are scripted host-accepted actions, not learned active exploration. Teacher's first post-drift paint is naturally wrong, and its later correction uses observable receipts. The benchmark does not test anticipation of undetectable changes.
4. Real SLM consumption requires sequential closed-loop inference; one-step supervised scores or running the reference oracle do not demonstrate learned recovery. The repeated A/B binary mapping and explicit physical receipts remain an interpretable shortcut.
5. LFM350 and Sprocket lane handoffs still report missing fresh pinned model execution environment: no local root model cache and no available opt-in workflow-dispatch path. Do not substitute NumPy/MLP surrogate training or mislabel source smoke as root model training.

## Suggested consumer study

On a pinned named SLM and split-isolated geometry: train on v49 alone versus v49+v50 TRAIN; select via DEV, then run genuine autonomous HELDOUT rollouts against the synthetic host with blind event and surface ablations. Report per-arm terminal success, wrong paint attempts, recovery erases, intervention regret, raw invalid action preference versus host rejection, first/second material correctness, branching/retracing and CPU p50/p95 inference/RSS. Include original untuned model, v49-only and stale-calibration deterministic controls. Verify exact model/source/checkpoint/tokenizer hashes and retain checkpoints and output trajectories in immutable Actions artifacts. Native CyberSand acceptance must remain a separate read-only-to-this-worker owner-run gate.