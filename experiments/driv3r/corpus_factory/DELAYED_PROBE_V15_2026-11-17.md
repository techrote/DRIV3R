# Corpus Factory v15 — delayed remote probe and inversion parity

17 November 2026 · `techrote/DRIV3R` · consolidated Draft/HOLD PR #1 · DATA ONLY

## Research question

V14 recovery data does not preserve a controlled last-64-observation ambiguity after its scripted interventions. Can an **actual host-applied distant probe** (accepted versus host-refused), combined with a later remote shutter inversion and a common long retrace, produce **independently replayable** decision pairs where current local observations and the last 64 positions/events are *identical* but different legal actions are required?

## Construction

`delayed_probe_v15.js` imports the immutable v13 external synthetic host (Git blob `b49afca90c307d8126c8f4c1becba667b7ced6f1`), not CyberSand's native ABI. The v13 boot/toggle events are explicitly filtered from model observations; v15's two causal variables are a host-observed distant WEST/EAST probe receipt (`accepted`/`blocked`) and an inversion receipt (`0`/`1`). Each synthetic episode:

1. Starts from the original v13 source world, traverses to a distant gate on the nominated side, and attempts a legal-or-refused `MOVE_SOUTH`.
2. Returns to the north junction and applies either zero or one remote shutter inversion, leaving a receipt in event history.
3. Executes **126 common, host-applied, neutral stem moves** so the last-64 history contains neither intervention, then returns to the same junction.
4. Supplies a host replay handle and BFS-shortest recovery route, with 9 sampled decision labels, strictly longer legal alternatives, illegal/block alternatives and memory-ablation controls.

For each source world and probe direction, the initial gate state (`WEST`/`EAST`) and inversion count (0/1) form a **four-example counterfactual quartet**. Each quartet has identical current position, goal, local 7×7 material grid, legal menu, and last 64 positions/events; exactly two targets are `MOVE_WEST` and two are `MOVE_EAST`. In a standardized *event-only* projection that keeps the last 64 positions and strips event timestamps, **probe-only** and **inversion-only** each yield two ambiguous pairs with opposite targets. A memory-blind or single-event-only classifier cannot exceed 50% balanced first-choice accuracy on these constructed quartet controls.

**Leakage discovered and corrected:** the first v15 generator failed its own one-event proof. Although the inversion receipt's content was identical across paired probe outcomes, its `step` offset differed because an accepted probe produces two more executed moves than a rejected one. This leaked probe outcome through the supposedly probe-free control. The final `eventControl` projects away receipt timestamps and validates both single-event ablations. The original causal history remains available for the full-history condition. Thus the **event-only control proof must not be extrapolated to all full-input causal features**: older route positions and event timing can legitimately reveal the remote probe result.

## Counts (default styleSeeds=2)

| Split | World groups | Disjoint topologies | Quartets | Episodes | Sampled oracle decisions |
|---|---:|---:|---:|---:|---:|
| TRAIN | 32 | 16 | 64 | 256 | 2,304 |
| DEV | 8 | 4 | 16 | 64 | 576 |
| HELDOUT | 12 | 6 | 24 | 96 | 864 |
| **Total** | **52** | **26** | **104** | **416** | **3,744** |

Observed across the synthetic dataset: **27,936 host-applied oracle movements**, **52,416 neutral washout movements**, **208 deliberately host-refused probe attempts**, **4,160 strictly suboptimal legal-action labels**, **zero accepted illegal oracle moves**, and **zero protected-cell entries**. All 416 oracle recoveries reached their targets. Refusals do not move the host actor.

## Qualification and regeneration

```bash
node experiments/driv3r/corpus_factory/test_delayed_probe_v15.js
node experiments/driv3r/corpus_factory/delayed_probe_v15.js NEW_OUTPUT_DIRECTORY 2
```

The standalone Node.js 22.16 regression passed, checking exact split counts, 104 complete quartets, last64 alias, both timestamp-normalized single-event controls, world/topology split isolation, all 416 oracle trajectory replays, legal-action menus, hard-negative legality, on-disk hashes, and overwrite refusal. Two independent full CLI generations reproduced **byte-identical six JSONL shards and their manifest**. Generation runs took 10.94 and 11.06 seconds (267.7–268.0 MB maximum RSS); regression test took 19.42 seconds (261.3 MB maximum RSS). These are corpus-generation measurements, **not SLM latency or model RSS**.

Source Git blob: `8966b3638d375c907c8a8b520767c24f0efe59f2`.
Test Git blob: `fe1d71f99e10402fb93dc4f385e3ac83ad7b50a5`.
Parent Git blob: `b49afca90c307d8126c8f4c1becba667b7ced6f1`.
`V15_DEFAULT_MANIFEST.json` is the exact default on-disk manifest with six per-shard SHA-256 hashes, row and byte counts. Source and test hashes are recorded above and in `LATEST_HANDOFF.json`. Generated full shards intentionally remain untracked and **are not yet published as immutable GitHub Actions artifacts**.

## Trainer consumer contract and caveats

- **Model-facing**: pass only `decision.model_input` or `observe(host)`. Never pass any `episode.parent`, `expert`, `intervention`, `prefix_receipt`, `quartet_id`, split labels, decision `target`, `controls` or `annotation`.
- `createHost(episode)` reconstructs the fixed scripted prefix; the model then selects *all* post-prefix moves via `apply(host,action)`. Measure world/quartet-grouped goal arrivals, path stretch, accepted/rejected actions, p50/p95 inference latency and peak model RSS.
- Use TRAIN to fit, DEV to choose settings, freeze checkpoint and prompts before opening HELDOUT; all four siblings and both probe directions of a world remain in one split. Use quartets/worlds as statistical units, not 3,744 correlated decisions.
- The target is a **CyberSand-like game policy** but the generator uses a deterministic external v13 material-grid corridor; these data are **not native CyberSand observations, native action legality, painting, material physics, or GPU acceptance**. Probe and inversion histories remain scripted; the post-prefix oracle is BFS, not an SLM, and only the post-prefix episode is suitable for learned closed-loop testing.
- The source v13 geometry still has two long U-shaped corridors. This is a deliberately discriminative memory benchmark, not proof of general navigation or game authoring competence. The quartet is balanced by construction and does not establish real-world 50% performance of any model.

**Next consumer**: a named, source-pinned LFM/Sprocket adapter with matched training budgets evaluating full causal history against last64, probe-only and inversion-only **at every post-prefix decision**, plus a sealed HELDOUT assessment. Next corpus expansion should add cross-geometry event-based recovery without revealing future shutter state in legal actions.
