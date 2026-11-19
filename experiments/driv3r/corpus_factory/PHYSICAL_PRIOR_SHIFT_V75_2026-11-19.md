# DRIV3R Corpus Factory v75 — physical prior-shift challenge (19 November 2026)

**Status:** source-matched generation, independent synthetic physical-host replay, byte-identical regeneration and adversarial rehash tamper all PASS locally. No neural model weights trained, no native CyberSand evaluation. Retain Draft/HOLD; commit only in `experiments/driv3r/corpus_factory/**`.

## What changes

v74 was a compact export of v73 and created no new geometries. v75 creates **16 new, split-disjoint worlds** on the pinned v44 host, retaining host-accepted painting, movement, material erosion, counterfactual fault states and the cost-aware physical witnessing decisions of v73. Crucially, it shifts the **observable empirical prior** for future remote abrasion across TRAIN, DEV and HELDOUT.

Each episode physically executes eight local PAINT/ERASE stress samples before remote goals. Erosion removes the actual coupon cell; the synthetic observation receives individual `LOCAL_STRESS_TRIAL` acknowledgements, *not* the privileged empirical risk rate. Coupon reset is physically performed; accepted action counts are padded with legal out-and-back movements so low/high risk histories reach the later intervention with the same host clock. At least 112 subsequent detour actions occur before the recovery decision. The ex-ante teacher weights remote hidden erosion histories by the **number of physically witnessed abrading samples / 8**, weighs fault polarity independently at 50/50, and chooses a minimum expected-cost intervention. The hidden individual future abrasion/fault outcome cannot change the first action label.

The 8-sample empirical prior is an artificial synthetic assumption, not a real Bayesian guarantee or native CyberSand sensor contract. TRAIN rates are 2/8 versus 6/8, DEV rates 3/8 versus 5/8, HELDOUT rates 1/8 versus 7/8. In this way HELDOUT is a **risk-severity extrapolation**, not a native transfer claim.

## Audited dataset

| Split | New geometries | Successful physical episodes | Host-accepted decisions | Risk-contrasting counterfactual groups |
|---|---:|---:|---:|---:|
| TRAIN | 8 | 1,024/1,024 | 176,192 | 24 |
| DEV | 4 | 512/512 | 91,028 | 4 |
| HELDOUT | 4 | 512/512 | 95,108 | 18 |
| **Total** | **16** | **2,048/2,048** | **362,328** | **46** |

All 256 matched intervention groups (128/64/64) have identical **instantaneous observation** across the two witnessed risk rates; **46** have genuinely different optimal first actions. The four-way distribution over hidden future erosion/fault outcomes is complete for each group, and the independent auditor reconstructs every source-time observation and legally accepted action from host state, without importing the producer module. No geometry matches any v73 map. The earlier v74 limitations still apply to its small 4-world TRAIN export; v75 adds new physical examples but does not alter v74 files.

## Evaluation-only controls (not SLM scores)

| Split | Uniform-prior control incorrect choices | Uniform-prior mean expected action regret | Risk-conflicting indistinguishable first-action groups |
|---|---:|---:|---:|
| TRAIN | 24/256 | 0.046875 | 24/128 |
| DEV | 4/128 | 0.009766 | 4/64 |
| HELDOUT | 18/128 | 0.330078 | 18/64 |

Uniform-prior control assumes risk=1/2 and knows the same geometry, material mechanics, fault class and witness tariff; it differs only by ignoring earlier physical stress history. Even this narrow baseline fails on some matched pairs, but **small mean regret on DEV is a negative result**, and most groups are uninformative about prior memory. This corpus should be mixed with v74 and future data; do not portray it as sufficient standalone SLM training diversity.

## Data-handling contract

Only `TRAIN_physical_prior_shift_stream.jsonl` is TRAIN-accessible. `DEV_...` / `HELDOUT_...` streams and **all certificates** are evaluator-only. Evaluation certificates contain hidden erosion sequences, fault flips, selected intervention, oracle expected costs and true terminal outcomes — never include these in model prompts or training examples. Sequence labels are actual host-accepted legal actions with integer focus weights; redundant TRAIN followups are loss-masked as in v73. Check the pinned source SHA-256 and manifest shard hashes before consuming.

### Reproduction

Place v75 generator + independent auditor alongside pinned `unseen_topology_v44.js`, `cross_calibrated_receipt_v70.js`, `physical_rewitness_v72.js` in the Corpus Factory directory.

```bash
node physical_prior_shift_v75.js /path/to/new_v75_corpus
node test_physical_prior_shift_host_v75.js /path/to/new_v75_corpus
```

The complete dataset was regenerated twice with identical manifests and shard hashes, then fully independently replay-audited. One deliberately edited `decision_action` certificate, with the certificate file checksum and manifest updated to match, was correctly **rejected** by independent source reconstruction. The pinned manifest captures all source, scenario and shard hashes. No large binary/data shards are committed to Git; the downloadable reproducible release is a conversation artifact, **not** an immutable GitHub Actions artifact.

## Remaining limitations and next consumer

All sensor, risk and abrasion laws are added synthetic research semantics, not confirmed in CyberSand. v75 is still fundamentally solvable by deterministic rules that count eight historical samples, and the new physically sampled priors are correlated with an assumed remote disturbance model by design. No SLM training, held-out neural closed-loop episode, per-token latency, RSS measurement or immutable Actions artifact. The tutor oracle enumerates small fixed hidden-world sets; its labels do not demonstrate a learned policy.

The separate trainer lane should train **a real pinned CPU SLM** on actual TRAIN inputs with source-time histories, evaluate sealed HELDOUT **extrapolated risk rates**, compare full history to last32, no-stress-history, risk=1/2 and no-witness controls, and measure terminal success, intervention cost/regret, p50/p95 CPU inference latency, and peak RSS. This lane must not edit trainer code or another repository.
