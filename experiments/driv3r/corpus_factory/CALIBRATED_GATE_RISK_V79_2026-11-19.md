# DRIV3R Corpus Factory v79 — observable calibration changes ex-ante route choice

**19 November 2026. DATA-only.** Repository `techrote/DRIV3R`, existing Corpus Factory Draft/HOLD programme PR #3. New independent physical-host generator, auditor, two evaluators/index exporters. Only `experiments/driv3r/corpus_factory/**` is writable. No native CyberSand ABI claim; `techrote/cybersand` and `techrote/neurounner` remain untouched.

## Problem and causal contrast

v78's midpoint obstruction flipped in 50% of cases with no observable predictor. Its historical report label described *intention* but provided zero physical Bayes advantage (50% physical lane correctness), despite a nominal 100% source-time report-parser score. This is an invalid success proxy for game-policy learning.

v79 creates physically implemented and independently audited **observed remote actuator trials**. Four earlier trials report the measured `before_open` and `after_open` of a genuinely toggled gate: either one change (25% empirical fault risk) or three changes (75%). No risk-class flag is exposed to the model. After a masked 220-action physical patrol and a first host-executed corridor, the gate either flips unreported at the midpoint or stays, with the **actual future flip outcomes distributed 1/4 or 3/4 across matched counterfactual arms**. At the midpoint, instantaneous observations are identical across pilot histories and future outcomes. The optimal *expected-travel-cost* initial action consequently changes with the old calibration sequence, not with forbidden knowledge of the future.

The agent travels physically until its chosen corridor gate, encounters a real local block in the 25% of cases that do not match its calibrated prior, backtracks using legal MOVE commands, and then paints the goal and STOPs. Crucially, **all 768 episodes reach the goal**; failure to guess the random gate is not mislabelled as policy error. Each source-physical host action is replayed independently. This is an intentionally narrow synthetic stress-test of conditional decisions, not a representative game-level corpus.

## Dataset and supervision

| Split | Distinct geometry hashes | Episodes | Audited accepted commands | Physically blocked chosen lanes | Matched current-observation ambiguity groups |
|---|---:|---:|---:|---:|---:|
| TRAIN | 6 | 384 | 120,512 | 96 | 48 |
| DEV | 3 | 192 | 64,800 | 48 | 24 |
| HELDOUT | 3 | 192 | 70,944 | 48 | 24 |
| **Total** | **12** | **768** | **256,256** | **192** | **96** |

Each world has 64 matched scenarios: S1 open {top,bottom} × S2 prior report {top,bottom} × receipt arrival order {fresh-first,stale-first} × pilot count {1,3} × hidden flip-index {0,1,2,3}. The last variable is evaluator-only. The generated data carry source hash, per-shard digest, geometry provenance, objective certificates and distinct split ownership. All worlds have disjoint widths and source geometry hashes (including disjoint from the v78 worlds); no cross-split world is reused.

Only `TRAIN` positive-weight actions may be used for fitting. The four physical-future branches share the same pre-midpoint observed history; the generator masks redundant S1-route and midpoint-choice supervision in flip-index 1..3 so that four future outcomes do not artificially multiply one training target. The consumer index retains **192 unique TRAIN decision histories** out of 768 TRAIN junction records; **576 repeats are loss-masked**. DEV and HELDOUT indexes are `EVALUATOR_ONLY`, not training targets. Future gate state, flip index, risk-count certificate, oracle costs and full physical outcome certification are absent from agent inputs.

## Independent synthetic-host qualification

1. A standalone audit rederived each world grid and barrier coordinate; independently replayed all 256,256 accepted MOVE/PAINT/STOP commands; checked legal action sets, local visibility, 4 real pilot gate transitions, timestamped and out-of-order remote reports, midpoint physical change, PAINT/STOP completion and the first physical barrier encountered on every recovery.
2. Auditor independently computed both candidate route lengths under stable/flip outcomes and confirmed **ex-ante** cost labels. For all 96 paired groups, the instantaneous midpoint input is byte-identical but 25% versus 75% historical trial evidence changes the target action. For 192 future-outcome groups, different physical outcomes share exactly the same prior history and label, confirming no future clairvoyance.
3. Two complete generator runs and two index exports were byte-identical. A forged physical gate event in HELDOUT was rejected after the offending certificate shard hash and manifest were rehashed. The audit does not rely on the generator's routing algorithm or program imports.
4. Local CPU generation and audit (Node v22.16.0, single process): generation ~4.03 s / peak RSS ~399,200 KiB; independent full replay ~3.07 s / ~233,376 KiB. These are **dataset tooling** resources, not neural inference latency/RSS.

## Honest controls (HELDOUT)

| Deterministic synthetic control | HELDOUT |
|---|---:|
| Pilot-calibrated ex-ante expected-action correctness | 192 / 192 |
| Ignore pilot, always follow latest source-time S2 report: action label correctness | 96 / 192 |
| Pilot-calibrated choice physically traversable at first attempt | 144 / 192 (75%) |
| Always follow latest S2 report physically traversable | 96 / 192 (50%) |
| Mean expected extra actions of ignore-pilot controller | +9.5 |

`+9.5` is the exact paired **expected** path cost difference from all hidden outcomes, not measured neural travel or a model's performance. All four pilot receipts precede the second decision by more than 128 actions. This shows the history can matter, not that a target SLM can learn it.

## Reproduction

```bash
node experiments/driv3r/corpus_factory/calibrated_gate_risk_v79.js /tmp/v79-corpus
node experiments/driv3r/corpus_factory/test_calibrated_gate_host_v79.js /tmp/v79-corpus
node experiments/driv3r/corpus_factory/score_calibrated_risk_v79.js /tmp/v79-corpus
node experiments/driv3r/corpus_factory/export_calibrated_decisions_v79.js /tmp/v79-corpus /tmp/v79-index
```

Pinned files `V79_DEFAULT_MANIFEST.json`, `V79_DECISION_INDEX_MANIFEST.json` and `V79_EVALUATOR_CONTROLS.json` describe the full local qualifying run. Source sha256 for the generator: `8d33957730f459823fc529f874b4a56f444e9afa696e6f8c526bc0effb76dd6a` (must match checked-in source); corpus manifest SHA-256 `8af1a832097d7d67cd8752e82def9d3a7c6b29839d499ee34c1d99428d788e`, decision index manifest SHA-256 `312855380e7beaaf6833b4ac9e828d772c98247775ce68d4deb1e88dd7e459a6`. Bulk trajectories are retained in local research release, NOT committed to Git; ideally publish immutably as opt-in GitHub Actions artifacts when workflow support permits.

## Limitations and handoff

This remains a simple two-corridor graph, four pilot trials, a symbolic timestamp and count rule, and a fully specified synthetic host—not real CyberSand. No SLM weights/adapters trained or evaluated in this lane. All learning, autonomous held-out game-like rollouts and model/tokenizer/checkpoint hash capture belong to the separate trainer. The v76 B-specific DEV contrast failure remains an independent unresolved weakness. Native action ABI compatibility remains unproved. Do not merge/promote Draft/HOLD or spend GPU/API budget.

Recommended separate trainer consumer: train an actual pinned CPU SLM/adapter on broad TRAIN corpora with selected v79 deduplicated decision examples; evaluate sealed v79 HELDOUT choice, full closed-loop physical recovery, held-out action cost/regret, zero-history and last-128 ablations, report-arrival-order error, p50/p95 **neural** inference latency and peak process RSS. A symbolic prior-counter is a control only, not a satisfactory substitute for neural science. Next Factory focus: topologically richer 3+ branch loops and goal-safe dynamic barriers, not another trivial pilot-count demonstration.
