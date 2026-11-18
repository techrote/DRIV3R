# DRIV3R Corpus Factory v14 — off-policy recovery and changing objectives
17 November 2026 · techrote/DRIV3R consolidated Draft/HOLD PR #1 · DATA ONLY

## Research question
V13 uses a scripted ancient relay prelude followed by a shortest route. Most labels occur on the expert path. Can a source-pinned synthetic corpus expose a real policy trainer to **host-refused actions, bad-branch exploration, in-flight shutter changes, and retargeted goals**, and supply legal, independently replayable expert recoveries rather than additional stationary first-action questions?

## Construction
`recovery_curriculum_v14.js` derives its 26 topology templates, style seeds, paired source states and strictly group-disjoint TRAIN/DEV/HELDOUT splits from immutable v13 generator Git blob `b49afca90c307d8126c8f4c1becba667b7ced6f1`. It does **not** alter the v13 files. Each parent episode produces three interventions:

1. **closed_gate_probe (D6):** scripted agent takes the wrong branch all the way to the shutter, attempts a move rejected by the host, receives a rejected-action receipt, then the oracle must retrace and reach the original goal.
2. **shutter_reversal (D7):** scripted agent approaches the originally open gate, which switches behind a remote shutter-inversion receipt; remaining shortest recovery must reverse and reach the other side.
3. **goal_retarget (D7):** scripted agent approaches an old destination, receives a new, reachable goal on the other branch; the new goal and receipt become visible to the model, but route labels and host state remain privileged.

Each run uses a fresh cloned synthetic v13 host. `createHost` reconstructs the injected prefix and intervention deterministically. `observe/apply` preserve the host-legal synthetic action menu. A BFS distance-to-goal field independently certifies the remaining shortest trajectory and classifies strictly longer *legal* alternatives, distinct from equal-cost ties and illegal actions. Critically, the reported count of "rejected" moves is an explicitly host-refused action, **not** a protected-cell entry.

## Results (styleSeeds=2)

| Partition | Source world groups | Recovery episodes | Labelled decisions | Three modes each |
|---|---:|---:|---:|---:|
| TRAIN | 32 | 384 | 4,992 | 128 |
| DEV | 8 | 96 | 1,248 | 32 |
| HELDOUT | 12 | 144 | 1,872 | 48 |
| **Total** | **52** | **624** | **8,112** | **208** |

- **22,120** accepted scripted intervention moves, **58,544** shortest-path oracle recovery moves, **208** host-refused shutter-probe moves, **zero** accepted illegal oracle moves, **zero** protected entries.
- **8,128** strictly longer *legal* action annotations; zero extra equally optimal alternatives under this grammar. No exact duplicate `model_input` across the 8,112 exported decision rows.
- Every recovery episode reaches its post-intervention goal. All three interventions, both counterfactual sibling branches, and correlated decisions remain within one source world group and one split. Topology templates are split-disjoint.
- `test_recovery_curriculum_v14.js` passed in Node 22.16.0, including replay, retarget, event receipts, rejection without position advance, split integrity, hard-negative legality and deterministic in-process comparison. Two independent Node CLI runs produced byte-identical **all six JSONL shards and manifest**. Generation: **6.45 seconds**, **302–309 MB maximum RSS**. Test: **18.99 seconds**, **423 MB maximum RSS**. These are corpus-generation resources, **not model latency/RSS**.
- `V14_DEFAULT_MANIFEST.json` records exact shard sizes/digests, source and test blob hashes. On-disk output is intentionally untracked; no immutable Actions artifact yet.

## Consumer contract

```bash
node experiments/driv3r/corpus_factory/test_recovery_curriculum_v14.js
node experiments/driv3r/corpus_factory/recovery_curriculum_v14.js /fresh/output/directory 2
```

Only `decision.model_input` and `observe(host)` are model-facing. The episode `parent.privileged_host`, intervention name, expert route, counterfactual IDs, split labels, and decision metadata must never be included in model prompts. Use TRAIN to train, DEV for selection, seal HELDOUT until configuration freeze. Report **per-world-group** recovery completion, action errors, path stretch relative to post-intervention shortest path, p50/p95 inference latency and peak RSS, stratified by intervention and topology. For actual closed-loop evaluation, construct a host from the episode then replace `expert_recovery.actions` with the model's actions; record host-rejected actions separately.

## Scientific limitations

This is a **synthetic, scripted-intervention replay and oracle-labelled corpus**, not native CyberSand gameplay and not a trained-model result. Models control only the recovery after the scripted error/change; the ancient prelude is also scripted. The same coarse corridor geometry underlies v13 and v14; cross-topology generality and difficulty under actual SLM inference remain unknown. Although v13 was constructed to force a last64 alias *at its initial junction*, v14 interventions break that alias; **do not claim long-memory necessity for v14**. Zero equal-cost alternatives reflects the corridor construction, not a general property of policy optimality. Native protected/material/brush acceptance remains out of scope. No weights updated, no paid compute, no physical Windows acceptance.

**Next consumer:** pinned actual LFM/Sprocket SLM/adapter, benchmark end-to-end closed-loop post-error recovery on DEV then sealed HELDOUT with full event history versus last64 and last1. If the SLM merely retraces and cannot replan after a second shutter switch, introduce multiple event interventions in a future **data** revision, not a surrogate policy trainer.
