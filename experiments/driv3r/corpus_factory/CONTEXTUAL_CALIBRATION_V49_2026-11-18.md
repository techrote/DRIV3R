# DRIV3R Corpus Factory v49 — dual-surface empirical tool calibration

**Date:** 2026-11-18  
**Authority:** `experiments/driv3r/corpus_factory/**` only, Draft/HOLD PR #1, shared research branch.  
**Type:** versioned *synthetic-host* TRAIN/DEV/HELDOUT dataset, **not native CyberSand**, no policy training.

## Hypothesis and discrimination

V48's global last-`PAINT_RESULT` heuristic can solve the painting part of its task. V49 introduces **two distinct material surfaces (A/B)** in each world, with independently reversible command-to-physical-material mappings. At two different host cells, each mapping is actually sampled via an accepted `PAINT_A`, observing material A or B, and an accepted `ERASE`. A first, obsolete sample of each surface precedes the live sample. In the usual arm, the **latest global result comes from surface B even though the first goal is on surface A**; for the reversed-order control, the live A and B samples are performed in the other order.

At the first painting decision, two counterfactuals have identical host state, local legal actions, discovered survey, mission and other history, except for the live **A** calibration result. After removing that A receipt, the two policy inputs are byte-identical, but one requires `PAINT_A`, the other `PAINT_B`. An analogous matched causal pair exists for the second goal on B. The wrong command paints the wrong physical material and cannot satisfy the mission without erase/recovery. Two separate stale-probe controls alter **obsolete** physical measurements, but preserve the policy action; another control changes receipt order while preserving both correct actions. Mission-retargeting and remote obstruction variants preserve the previous v46-v48 challenge dimensions.

This is a harder *associative event-retrieval* benchmark than v48, but it is **not** proof of semantic reasoning: a tiny symbolic `latest receipt by surface` lookup still solves it. Surface labels are synthetic, calibrated by scripted host-accepted actions. Do not present teacher success as learned-game performance.

## Method and provenance

- Generator: `contextual_probe_v49.js` (source SHA-256 pinned in `V49_DEFAULT_MANIFEST.json`). Parents `empirical_tool_v48.js`, `mission_memory_v46.js`, `frontier_memory_v45.js`, `unseen_topology_v44.js` have exact source SHA-256 pinned in the manifest. Source identities independently verified against DRIV3R branch blobs before implementation.
- Scenario base: source world ids `v46.candidate(split,index)`; 48 TRAIN, 16 DEV, 16 HELDOUT, with disjoint geometry across splits and **the same** original world identities as v46/v47/v48. When combining curricula, group across releases by `source_split`, `source_world_index`, `source_seed` and `geometry_sha256`. Do not mix a v46-v49 source-family HELDOUT world into a consumer TRAIN set.
- Nine arms per world: baseline; change current A mapping; change current B mapping; change both; close remote gate; retarget historical mission; corrupt obsolete A result; corrupt obsolete B result; reverse live-probe order. Every arm runs an independent host-accepted trajectory and terminal validation.
- Inputs contain only currently observed local surface, legal action menu, explored graph, painted cells, short recent accepted-action window, and historical mission/gate/physical calibration receipts. Inputs omit hidden tool-law mapping, target, stage, oracle, seed and evaluation metadata. Consumer may provide `policy.input` alone; `action` and `weight` are supervised targets; `*_challenge_EVALUATOR_ONLY.jsonl` may not enter the training context.
- TRAIN movement labels are **loss-masked to weight=0** when their observation is a near duplicate of an earlier movement after stripping purely calibration-result receipts. All complete raw traces remain available for replay, but irrelevant physical variants do not dominate the navigation training signal. Every material decision remains supervised and every DEV/HELDOUT sample retains weight 1.

## Reproduction (CPU, no external model, no network)

```sh
node experiments/driv3r/corpus_factory/contextual_probe_v49.js /tmp/v49-A
node experiments/driv3r/corpus_factory/contextual_probe_v49.js /tmp/v49-B
node experiments/driv3r/corpus_factory/test_contextual_probe_v49.js /tmp/v49-A /tmp/v49-B
```

Requires checked-in parent generator/test sources, particularly `test_mission_memory_v46.js`. The test verifies all shard SHA-256, source SHA-256, terminal host replay, physical PAINT/ERASE evidence, no privileged input leaks, no cross-split geometry/input collision, independent observer-only oracle labels, strict causal and negative controls, reproducible bytes and TRAIN near-duplicate weights.

## Qualified complete release

| Split | Worlds | Synthetic successes | Policy records | TRAIN near-duplicate nav weight=0 |
|---|---:|---:|---:|---:|
| TRAIN | 48 | 432/432 | 9,009 | 4,337 |
| DEV | 16 | 144/144 | 3,516 | 0 |
| HELDOUT | 16 | 144/144 | 4,571 | 0 |
| **Total** | **80** | **720/720** | **17,096** | **4,337** |

The full independent audit passed **17,096/17,096** policy decisions; **2,880** host-accepted calibration `PAINT_A` actions and **2,880** matching `ERASE` actions; **80** causal A pairs, **80** causal B pairs, **160** obsolete-probe negative controls, **80** receipt-order controls, plus 80 mission and 80 remote-gate causal pairs. No cross-split geometry collision, policy-input leak or contradictory label was found. Two full releases had identical per-file SHA-256 and exact bytes.

A deliberately incorrect **global-last-calibration** interpreter chose the wrong paint command **160/1,440** times across all arms/goal decisions, including **160** first-goal decisions; per matched surface-A contrast it cannot be correct on both alternatives (at most 50% pairwise). These are deterministic control results, *not* model-accuracy measurements.

### Performance (local CPU)

Two complete generations took approximately **13–14 seconds each**, with observed process RSS under **520 MiB**. The independent full-host auditor passed with observed RSS approximately **0.8–0.9 GiB**, timing variable with runner load. These are **dataset generation/audit** measurements; **not neural forward latency or model RSS**.

## Scientific negative results / limitations

1. Fixed synthetic binary surface labels A/B, fixed mappings within an episode, and deterministic physically observed results. Last receipt **per surface** remains a trivial symbolic solution, so this is **not** a learned reasoning or game-competence demonstration.
2. The extensive survey and calibration preludes are scripted, although all probe and movement actions are accepted by the synthetic host. There is no learned decision of *when/where* to calibrate, no noisy observation, and no end-to-end autonomous initial exploration policy.
3. Source worlds overlap v46-v48 by design. Cross-release lineage must be grouped to avoid inflated generalisation claims.
4. No named root LFM/Sprocket optimizer step or current-run neural forward, no native CyberSand episode, no real model latency/RSS, and no immutable Actions-shard publication.
5. Calibration material context is synthetic rather than independently checked against CyberSand's native material/painting ABI; native semantics must be validated by an owner-controlled consumer before any game deployment.

## Consumer handoff

Use v49 HELDOUT as a **fresh evaluation stage only if the same source-world family has not entered training in another version**. Evaluate the actual pinned root SLM or adapter, versus original root, v48-trained baseline and deterministic observer-only control: exact legal action rate, physical material-correct first and second paint, terminal completion, recovery, obsolete-probe control stability, global-receipt vs surface-conditioned retrieval, and gate/mission action flips. Run full-history, remove-A, remove-B, stale-only and truncate-to-last-32 ablations on matched pairs. Record raw illegal preferences versus host-rejected actions, CPU p50/p95 forward time, resident model RSS, and exact model/source/checkpoint hashes. Save any weights/adapters only as immutable opt-in Actions artifacts, not Git.
