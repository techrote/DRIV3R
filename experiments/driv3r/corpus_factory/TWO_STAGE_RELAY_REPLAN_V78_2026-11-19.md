# DRIV3R Corpus Factory v78 — two-stage physical relay, delayed reports and unannounced recovery

**Lane:** `experiments/driv3r/corpus_factory/**`; Draft/HOLD in DRIV3R programme PR #3. This is a **synthetic 2D grid-graph host**, NOT the canonical native CyberSand observation/action ABI. No neural checkpoint or neural game policy was trained or measured here.

## Scientific target and scenario

v77's single-choice, two-corridor maze remained vulnerable to a trivial source-timestamp parser. v78 adds two sequential junctions, with independently blocked upper/lower branches, branch-specific gate locations, side pockets, long delays between reports and decisions, and an unpredictable *unreported* second-stage gate reversal at the midpoint. The host accepts or rejects each physical movement, tracks local gate visibility and real PAINT_A/STOP completion, and dynamically changes passability while the route is in progress.

* A 200-action zero-loss patrol precedes the first junction. First-stage gate measurements arrive at ticks 12 and 22 (physical measurement at tick 0 or 10), and second-stage measurements at ticks 52 and 62 (measurement at tick 0 or 50). Arrival order is balanced and can be stale.
* First-stage correct lane is derived from the *highest measurement timestamp* present in earlier observed history. Half of arms deliberately enter the wrong branch, meet the **physically blocked** gate and have those mistake actions loss-masked. Only after the obstruction is actually encountered does the supervised recovery route start.
* On entering the midpoint, half of arms experience an **unreported** physical reversal of second-stage gates. The teacher initially follows the previously verified report, physically discovers the new obstruction, backtracks along legal actions and replans to the opposite passage.
* All 32 arms per world exhaustively cross two original gate statuses, two measurement-delivery orders, spontaneous stage-two reversal yes/no and first-stage forced mistake yes/no. **No hidden future event is supplied in the observation**. `events` and true gate outcomes are confined to evaluator-only certificates.

## Provenance and outcome

| Split | Distinct world geometry hashes | Episodes | Audited accepted host actions | Forced wrong-route recoveries | Surprise gate recoveries |
|---|---:|---:|---:|---:|---:|
| TRAIN | 12 | 384 | 100,288 | 192 | 192 |
| DEV | 6 | 192 | 56,064 | 96 | 96 |
| HELDOUT | 6 | 192 | 59,136 | 96 | 96 |
| **Total** | **24** | **768** | **215,488** | **384** | **384** |

All 768 demonstration episodes terminate with a physically accepted PAINT_A followed by STOP. First-junction current-only and last-128-observation maximum among the balanced matched labels is 50%; arrival-order parsing is 75%, and source-time parsing of earlier reports is 100%. This is **a deterministic symbolic control, not trained SLM accuracy**. The second-junction *historical report-intention* label has the same 50/75/100% pattern. Its label is **not** a Bayes-optimal physical action under the balanced 50% unannounced reversal rate: actual remote gate agreement at the midpoint is only 50%, an explicit negative result. Model consumers should score physical terminal completion, legal recovery and realized action cost—not treat S2 intention-label accuracy as optimal decision quality.

Compared with otherwise paired stable controls, physical unannounced reversal requires an extra mean **19.67 TRAIN / 29 DEV / 29 HELDOUT** accepted actions. Forced wrong-first-corridor admission similarly adds **16.67 / 25 / 29** actions. The local observation at both junctions is observationally ambiguous in each of the 24 worlds; a full history resolves intended route reports but not unpredictable later physical flips.

The decision reference index is intentionally compact and exact-history deduplicated. Positive-weight TRAIN junction references: 576 instances → **288 distinct** history/action references (288 duplicates eliminated). DEV 288 → 144, HELDOUT 288 → 144; both withheld split indexes are evaluator-only. Each reference records source episode/step, action, scenario provenance and difficulty without including future gate events. There remain repeated zero-loss prefixes in the complete physical trajectory streams; the compact index prevents repeatedly weighting identical supervised junction samples. The world's width and morphology still correlate with split; they should not be used as proof of out-of-distribution neural generalization.

## Independent verification

* Independently implemented auditor reconstructs all geometry, per-action legality, dynamic gate events, the two asynchronous report streams, local observations, supervision masks and physical goal completion without importing the generator or its planner. Full replay: **24 worlds, 768 episodes, 215,488 accepted actions PASS**.
* All 24 world hashes are unique and split-disjoint; all 32 counterfactual arms per geometry are complete. Both junctions have current-input action conflicts; no conflicting full-observation-history action targets were found in the scored index.
* Two separate full corpus generations were **byte-for-byte equal across all seven generated files**.
* Adversarially changed report evidence was rejected by the independent auditor after the attacker recomputed the relevant stream and certificate hashes, episode hash and manifest checksums. Rejection: physical/observational mismatch at TRAIN episode 0, tick 12.
* Full replay under Node `--max-old-space-size=512`: generator peak RSS **224,084 KB** and auditor peak RSS **226,484 KB**. These are **corpus tooling resource figures, not SLM CPU inference latency/RSS**.

## Consumer recipe (no training in this data lane)

```bash
node experiments/driv3r/corpus_factory/relay_gate_replanning_v78.js /tmp/driv3r-v78
node experiments/driv3r/corpus_factory/test_relay_gate_host_v78.js /tmp/driv3r-v78
node experiments/driv3r/corpus_factory/score_relay_memory_v78.js /tmp/driv3r-v78
node experiments/driv3r/corpus_factory/export_relay_decision_index_v78.js /tmp/driv3r-v78 /tmp/driv3r-v78-index
```

Only `TRAIN_relay_stream.jsonl`, its positive-weight action rows, and `TRAIN_relay_decision_index.jsonl` can be used for training. Every certificate, DEV/HELDOUT file, evaluator score and hidden event must remain outside model inputs and tuning. No 63 MB generated corpus or model checkpoint is committed to Git; retain zipped dataset by immutable, pinned artifact when the appropriate Actions workflow is made available. Minimal CI may run syntax or `--quick` smoke only.

## Limitations and next work

Despite two sequential junctions and genuine closed-loop physical recovery, topology is still a repeated two-branch relay with side spurs, not a varied actual CyberSand map. Its remote telemetry and gate mechanisms are *invented synthetic-host contracts*. The symbolic report parser plus graph search still handles the data, and stage-two flip probability being exactly 50% makes report-first exploration a tie before any physical inspection. v76's zero B-specific DEV intervention contrast and its scorer full-corpus replay gap remain unresolved. **No actual SLM optimization, neural autonomous HELDOUT, native CyberSand test, neural p50/p95 latency, neural RSS or immutable Actions artifact was produced.**

Next corpus increment: varied loop topologies with several physically distinct recovery routes, a calibrated but nondegenerate pre-change fault prior and genuinely policy-discriminative risk, plus a source-verified repair of v76 DEV B contrast. Next consumer experiment (separate trainer lane): actual pinned SLM training on deduplicated TRAIN histories, autonomous HELDOUT goal/route completion, extra action cost, wrong-route recovery, phase-two surprise correction, current-only/last128/arrival-order controls, and neural CPU latency/RSS plus checkpoint/tokenizer hashes.
