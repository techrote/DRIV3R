# Corpus Factory v32 — genuinely consequential paint-triggered gate

**Date:** 2026-11-17. **Research scope:** `experiments/driv3r/corpus_factory/**` in `techrote/DRIV3R`, Draft/HOLD PR #1. **Host:** explicitly synthetic; **NOT** native CyberSand. **No policy model or adapter trained.**

## Scientific question and reason for this version

v31 retained a dynamic gate receipt but its gated edge was an equivalent-length alternative and did not establish an actual change in optimal navigation. Does a painting-triggered gate have a **provably consequential** effect on the shortest successful recovery suffix while retaining the challenging four-way delayed-memory counterfactuals?

v32 changes the gate to `(0,0)↔(1,0)` in the orthogonal synthetic graph. This edge is initially **closed**. The first host-accepted paint opens it, and the three-edge `(0,0)→(0,1)→(1,1)→(1,0)` bypass means an unopened gate is traversable around, not a disconnected/infinite-cost world. Preparatory travel to the east arm takes two extra legal steps while closed; the other three arms receive two host-accepted neutral movements so the later public action-budget and event recency are still exactly matched.

Recipes now have four balanced pattern families (alternating A/B, alternating B/A, all A, all B) crossed with two topology variants. This repairs an early prototype collision that inadvertently made every material B; that rejected prototype is **not** included in the qualified corpus.

## Discriminative construction

For every world, four host-accepted prelude paths contaminate a different remote target; all then contaminate the same north target last and execute 160 accepted movements. At the first policy decision:

- Current host observations have the same local view and legal actions across the four cases.
- Their last 64 events, public `memory:false` projection and `lastPaintOnly:true` projection are identical **within each family**.
- Older accepted paint receipts differ by contaminated target and are needed for four different optimal first moves: `MOVE_W`, `MOVE_E`, `MOVE_S`, `BACKTRACK`.
- Choosing any alternative member of the four-way first-action set costs **at least two additional actions** over the exact shortest-successful suffix.
- Exactly one case per family (east-target correction) is **strictly causally sensitive** to the gate. Against a frozen-closed counterfactual, the optimal first move would be `MOVE_S`, while the true open-gate oracle chooses `MOVE_E`. Taking the frozen-closed policy's first move in the actual open-gate host incurs two additional actions. The other three cases have zero such first-action regret: explicit negative controls.

The frozen-closed oracle shares the exact same geometry, painting state, objectives and action semantics; its only intervention is a forced closed value for that edge. The causal comparison does **not** confuse a different survey, different goal, or an illegal host action with a material law effect.

**Caveat:** the learner-controlled phase begins **after** the scripted first paint and gate transition. It does not establish an autonomous model's ability to cause or anticipate the transition. Its value is tested route re-planning conditioned on accepted historical gate receipt and current host-legal observability.

## Full default corpus and evidence

| Partition | World families | Episodes | Decisions | Erasures | Strict gate-causal first actions | Frozen-gate negative controls |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 256 | 7,328 | 512 | 64 | 192 |
| DEV | 32 | 128 | 6,208 | 256 | 32 | 96 |
| HELDOUT | 32 | 128 | 8,448 | 256 | 32 | 96 |
| **Total** | **128** | **512** | **21,984** | **1,024** | **128** | **384** |

All 512 replayed oracle episodes completed the two corrective erasures and all terminal objectives. The regression verifier reconstructed **all 21,984** complete observations from episode headers and per-decision event/position deltas; checked SHA-256 against the generator; separately recomputed suffix labels; applied all actions to the synthetic host, checked exact receipts and legal/terminal completion; recomputed gate counterfactual suffix costs; checked four-way aliases, restricted first-action ceilings (25%), disjoint graph geometry and full-input hashes between TRAIN/DEV/HELDOUT. **No contradictory complete-input labels**, **no exact duplicate TRAIN inputs**, no rejected worlds.

Two independent complete generation runs produced byte-identical six JSONL shards and manifests (`diff -qr`). Generation A: **10.31 s, 406,784 KiB peak RSS**; B: **9.17 s, 408,360 KiB peak RSS** (local Node.js runtime; generator resource use **not** model inference latency or RSS). Source smoke alone is not model competence. Every shard has immutable content SHA-256 recorded in `V32_DEFAULT_MANIFEST.json`; large shards are deliberately not committed to Git.

Reproduction with the three pre-existing pinned lane-local runtime dependencies:

```bash
node experiments/driv3r/corpus_factory/causal_shortcut_v32.js /tmp/driv3r-v32-dataset
node experiments/driv3r/corpus_factory/test_causal_shortcut_v32.js /tmp/driv3r-v32-dataset
```

The output directory must not already exist, to prevent accidental overwrite. Default generation is CPU-only and invokes no remote model. `--quick` on the generator performs one group per split and is only a preflight; use full default for trainer input.

## Data-contract and trainer cautions

- No metadata leak: policy inputs are **only** reconstructed `model_input` or `bounded_model_input`. Never expose `episode_id`, `pair_id`, split, `hidden_interventions`, `first_action`, `gate_causal_control` or oracle metrics to a model.
- Policy receives old paint receipts in its durable event memory. `memory:false` and last-paint-only inputs explicitly remove the old target identity; both have a 25% theoretical first-action ceiling **within each balanced four-way family**. This is a theoretical identification ceiling, **not measured trained-model accuracy**.
- Four recipe patterns are abstract `A/B`, not CyberSand material/chemical laws. The root-east opening gate is an abstract dynamic obstruction, not a proven native simulator effect.
- Simple five-target radial worlds are vulnerable to a symbolic old-receipt parser and graph planner. Overfitting to this grammar is a meaningful risk; future corpora need more heterogeneous partial observability, alternative gate positions, real branch/loop designs, dynamic material law changes and game-like asynchronous events.
- Independent real-model training, fully autonomous held-out recovery, latency/RSS and immutable GitHub Actions shard publication remain outstanding. The LFM/Sprocket workers report unavailable pinned root-model execution/dispatch in their current runtimes, not a reason to replace the SLM with an auxiliary policy.

## Consumer experiment

Pin actual LFM350 or Sprocket root and adapter revision/sha; train only TRAIN, tune DEV and open HELDOUT once. Compare durable full history, last-PAINT-only, last-64 and no-memory controls under **model-controlled complete episodes**. Stratify recovery into 128 gate-sensitive east cases vs 384 gate-negative controls. Report terminal completion, residual wrong paint, first-action choice/regret, total realized suffix regret, host illegality, p50/p95 CPU forward latency, model RSS, exact dataset and checkpoint identities. Do not claim native CyberSand acceptance without owner-run native tests.

**Boundaries:** no paid API/GPU; no training checkpoint committed; no writes outside Corpus Factory; no CyberSand/Neurounner writes, comments or dispatches; no DRIV3R main edit/merge/Ready.
