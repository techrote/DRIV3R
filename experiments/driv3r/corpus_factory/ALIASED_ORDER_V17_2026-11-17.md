# Corpus Factory v17 — alias-controlled ordered-gate transfer

17 November 2026 · DATA lane only · DRIV3R Draft/HOLD PR #1.

## Research question

V16 required noncommutative gate-event order but its first and latest-two instruction controls retained 66.7–83.3% and up to 75% first-action accuracy in some partitions. This follow-up makes those specific cheap controls **exactly non-discriminative** while testing transfer to previously unseen distances between causally relevant events.

All 12 instructions in a sequence share one bag: **one SET_WEST, one SET_EAST, ten TOGGLE**. Every selected sequence begins with two TOGGLE instructions and ends with two TOGGLE instructions. The 12 alternatives within one world share initial/terminal two-instruction projections, final local grid, legal action menu, goal, last 64 position/event history and unordered event inventory, but six require WEST and six require EAST. Event-order reconstruction, rather than recent-event lookup, distinguishes those outcomes.

The 36 selected sequences are split-disjoint: TRAIN has **adjacent** SET instructions, DEV has **one intervening TOGGLE**, and HELDOUT has **2–6 intervening TOGGLEs** (SET index gaps 3–7). This shift is deliberate and not a claim about natural event statistics. Counterfactual siblings share a topology family; world and topology groups are isolated between TRAIN, DEV and HELDOUT.

## Generated dataset

| Partition | World groups | Episodes | Sparse oracle decisions |
| --- | ---: | ---: | ---: |
| TRAIN | 32 | 384 | 3,456 |
| DEV | 8 | 96 | 864 |
| HELDOUT | 12 | 144 | 1,296 |
| **Total** | **52** | **624** | **5,616** |

Replayed: **213,408 scripted prelude moves**, **41,904 post-prefix oracle moves**, **6,240 strictly longer legal hard negatives**. Every expert reaches its goal. In-process checks found no accepted illegal oracle moves or protected entries. Synthetic replay only; native CyberSand observations/actions are **not** verified here. Only post-prefix movement is policy-controllable.

## Discriminative controls: family-conditional majority ceiling

For every 12-case counterfactual world family, event bag / last64 / first instruction / first-two / latest instruction / latest-two are identical between siblings despite six opposite labels. Their theoretical first-action majority ceiling is exactly **6/12 = 50% in every split**, rather than the partly predictive v16 controls.

The remaining stronger cheap controls are not fully eliminated:

| Projection | TRAIN | DEV | HELDOUT |
| --- | ---: | ---: | ---: |
| First three ALPHA instructions | 7/12 | 7/12 | 7/12 |
| Latest three ALPHA instructions | 7/12 | 7/12 | 8/12 |

These ceilings describe information available in deterministic projections, **not neural-model success** or arbitrary inability of a short-context model to solve other tasks. The full ordered history permits deterministic gate-state computation.

## Qualification and source identity

- Parent synthetic host: `relay_memory_v13.js` Git blob `b49afca90c307d8126c8f4c1becba667b7ced6f1`.
- Derived design source: `ordered_relay_v16.js` Git blob `5d64ac0558349f5efe7f40ff866204006a2a1fa7`.
- V17 generator Git blob `014a25750f368f68b3397d0e413f7faca5ea4c67`; standalone test blob `c715626d5e82fae97e92964cea171b32f0d2c191`.
- `node test_aliased_order_v17.js`: **7/7 sections PASS** (including permutation split, legal replay, oracle negatives, controlled alias, deterministic regeneration, on-disk hashes, overwrite refusal).
- Two independent `node aliased_order_v17.js NEW_DIR 2` generations produced byte-identical six JSONL shards and manifest, compared with `diff -qr`.
- Local generation wall time **21.55 / 20.54 s**, peak RSS **512,404 / 506,304 KiB**, Node 22.16.0 Linux; these are corpus-generation, **not SLM inference**, metrics.
- Manifest SHA-256 `cf0a517cf7e77acc0642c979a77c9bf2d78f25ea11dc0743469988cc97f661cb`. Full shards (~114 MB) are generated on demand and omitted from Git. No immutable Actions artifact has been produced.

## Consumer contract

Generate with `node experiments/driv3r/corpus_factory/aliased_order_v17.js /new/output/dir 2`. Fit TRAIN only; select on DEV; seal HELDOUT until the selection is fixed. Feed only `row.model_input` or `observe(host)`; do **not** expose parent/intervention/expert/controls/metadata to the model. The `createHost(ep)`/`observe(host)`/`apply(host,action)` interface supports model-controlled movement after the scripted remote-event prelude. Report world-grouped, 12-way counterfactual first-action success and full autonomous goal completion, path stretch, rejected actions, p50/p95 inference latency and model peak RSS.

The latest LFM handoff identifies frozen GP350-006 (model revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`, adapter SHA-256 `436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c`) as a conservative navigation control, but does not establish ability on order-memory tasks. The Sprocket handoff reports unavailable pinned root weights/dependencies in this executor. **No named SLM was fitted or evaluated here**; such work belongs to trainer lanes.

## Scientific caveats

The remote gate commands and BETA decoys are scripted synthetic environmental events, not legal worm actions; topologies are a narrow U-loop grammar and there are no native sand/paint semantics. First-three/latest-three partially predict outcomes. Samples within a world are correlated; episode rows are not independent replicates. V13 world families also recur across V16 and V17: **do not pool versions as independent splits** without re-grouping by world and topology. No model was trained, no pretrained weights or checkpoint binary was added to Git, and no native CyberSand acceptance is implied.
