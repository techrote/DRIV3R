# DRIV3R Corpus Factory v64 — delayed obsolete measurement and event-time causality

**Date:** 2026-11-19 UTC. **Home:** `techrote/DRIV3R` only. **Owner:** `experiments/driv3r/corpus_factory/**`. **Review:** existing Draft/HOLD programme PR #3, `research/corpus-factory-postmerge-programme`; no merge or main write. **Starting remote head:** `a1cd6e19b2fb0d376c54745813927170529a1aa3`; current main observed at the original merged-import base `85df84666a8bad18f0092b930c08557b8010687f`.

## Why v64

The v63 pivotal-gate task could be solved by taking the **last `EDGE_GATE` bulletin received**. That is not equivalent to tracking the latest physical measurement when transmission is delayed. v64 tests whether a game-policy consumer can disambiguate **measurement time** versus **delivery order** across long streaming action histories. This removes an especially brittle one-message shortcut without changing the established legal action alphabet or passing hidden host state to the model.

## Data generation and causal design

- Reuses exact v63 procedural graph worlds (16 total), split ownership, host-executed partial surveys and route-regret validation. **Not a claim of new world topologies**: v64 is a new *temporal telemetry intervention distribution* on those worlds. Parents are pinned by SHA-256 in the manifest.
- Each group has **16 crossed arms**: pivotal or irrelevant remote gate; old open/closed; revised open/closed; and **fresh-last versus stale-last delivery**. Original physical revision occurs halfway through 48/96/160 legal detour steps for TRAIN/DEV/HELDOUT. An *exact duplicate* of a real old measurement is re-delivered at one-quarter of the detour (fresh-last) or much later (TRAIN 75%, DEV ~83%, HELDOUT 87.5%). Duplicate retains its original `at` timestamp and recorded value; no fabricated physical state change accompanies its later arrival.
- Crucially, the historical delayed measurement is not a newly obtained sensor sample. The host's current `h.closed` changes only at the physical revision; the stale packet merely appends a dated receipt to the stream. The independent replay tests that distinction for every action.
- All 16 current **critical model inputs** (position, legal actions, partial surveyed geometry and most-recent local events) are identical within their world, yet pivotal cases require opposite route actions. Time-sorted receipts are available only from the **previous stream**, never repeated as a durable full ledger.
- Physically validated wrong-choice shortest-path regret **at least 2 legal moves**. Irrelevant-gate controls cannot change the optimal route. All detour actions are host-executed/zero supervised loss; remaining actions are legal and host-completing observable-survey teacher demonstrations.
- TRAIN retains prior v63 noncritical dedup masks and adds **2,246** new repeated, noncritical state-action masks across temporal variants. Counterfactual critical labels remain supervised (weight 1, focus 8), even when their current observations alias. DEV/HELDOUT are evaluator-only.

## Full qualification

| Split | Reused worlds | 16-arm groups | Successful host episodes | Legal decisions | Masked detours | Additional TRAIN dedup |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 8 | 128 | 8,880 | 6,144 | 2,246 |
| DEV | 4 | 4 | 64 | 7,944 | 6,144 | 0 |
| HELDOUT | 4 | 4 | 64 | 12,416 | 10,240 | 0 |
| **Total** | **16** | **16** | **256/256** | **29,240** | **22,528** | **2,246** |

The independent `test_delayed_bulletin_host_v64.js` does **not** import the v64 generator. It reconstructs original source worlds, prelude and physical gate transitions, stale packet provenance, every one of **29,240 pre-action inputs**, legality, action receipts, objective completions, source splits, counterfactual group equivalence and route regret. It separately derives the legal optimal first action using both the full physical graph and a measurement-time-sorted **model-visible partial survey**.

**Deterministic non-neural controls on 128 pivotal critical decisions:** current-only/constant first-action 50%; most-recent *arrival* gate report 75%; highest *measurement time* gate report 100%. The arrival-order heuristic fails **all 32/32** cases where a stale report arrives last after a genuine state reversal. There are 128 irrelevant-gate negative-control episodes. These scores are **symbolic controls, not actual SLM training or inference**.

- Two full generations including all 7 shard/manifest files are **byte-identical**.
- All 7 shard SHA-256 hashes and exact row counts verified; a deliberately altered TRAIN shard is rejected before audit of any receipts.
- No split leakage among the 16 reused v63 geometries. TRAIN 8 vs DEV 4 vs HELDOUT 4 worlds. Within-v64 reuse across variants is deliberate. **The same v63 worlds must not be treated as independent cross-release heldout novelty.**
- All audits are synthetic-host, not native CyberSand. No paid API/GPU, no neural checkpoint, no optimizer steps, no neural latency/RSS and no immutable Actions dataset artifact.

## Reproduce (no paid compute)

```bash
node experiments/driv3r/corpus_factory/delayed_bulletin_v64.js /tmp/driv3r_v64
node experiments/driv3r/corpus_factory/test_delayed_bulletin_host_v64.js /tmp/driv3r_v64
```

Both commands require only Node.js and the committed parent modules under this lane. The committed `V64_DEFAULT_MANIFEST.json` pins all output file hashes and generated-source hashes. No bulk dataset is committed. The exact generated corpus is in the local evidence archive for transfer until an opt-in immutable Actions artifact is published.

## SHA-256 source identity

- `delayed_bulletin_v64.js`: `2f30a346d2a14969b6d28349987b668f7a6fb1f5770cc5c01e07fb0a45b17d20`
- `test_delayed_bulletin_host_v64.js`: `4ffc2772d835de6c20d88139bdca3fe1380f8122dddc92890ad6b572a13d9fff`
- `V64_DEFAULT_MANIFEST.json`: `4ad5eda56adb7797dd5cbb06a011ea5d256ce5224a194bc82961e860916a0999`
- Parent `gate_revision_v63.js`: `a96e1d03d5f01113beb15d719b52de31216cc1f7e5b39b08f53b86cb59b46f8f`

## Limits and consumer experiment

**This is synthetic graph-host data, not native CyberSand or learned model behaviour.** Host-observed telemetry is constructed causally but retains the strong simplifying assumption of timestamped, authentic gate packets. A small symbolic sort-by-measurement-time policy obtains 100%; nothing here shows a named SLM can learn it. Fully scripted, unweighted detours do not demonstrate autonomous exploration. Only four HELDOUT worlds exist, and all are reused from v63.

**Recommended separate trainer-lane experiment:** train a pinned *actual* SLM/adapter on v63/v64 TRAIN-only sequential histories. Compare v63 HELDOUT (no stale delivery) and v64 HELDOUT (stale delivery, longer latency) using full retained history versus last-arrival parser, fixed-age truncated memory, recurrent reset immediately before delayed packet, and measurement-time symbolic oracle. Report *both-correct across counterfactual arm pairs*, autonomous host terminal success, navigation regret, neural CPU p50/p95 inference and RSS with exact model/tokenizer/checkpoint hashes. Seal DEV/HELDOUT and all `*_EVALUATOR_ONLY` certificates against training. The independent long-horizon trainer programme PR #2 remains separately owned; its historical neural DEVELOPMENT replay is not evidence of success on this new benchmark.

**Gate:** `techrote/cybersand` and `techrote/neurounner` read-only. No merging, Ready, auto-merge, main mutation, foreign-lane write, force-push, paid compute or expensive default CI.
