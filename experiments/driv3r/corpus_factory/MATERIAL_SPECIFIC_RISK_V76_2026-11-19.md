# DRIV3R Corpus Factory v76 — independent material-specific risk evidence (19 November 2026)

**Scope:** DATA ONLY. Synthetic host, not native CyberSand. Draft/HOLD, no merge. All new work is in `experiments/driv3r/corpus_factory/**`.

## Mechanism and causal test

v75 used one eight-trial empirical abrasion prior to weight a single remote disturbance distribution. v76 runs **16 actually accepted local stress-coupon tests**, eight each for materials A and B, before painting the two remote objectives. Each material can have a different observed rate. Future remote abrasion independently affects A and B, giving **NONE, A only, B only, BOTH** hidden physical outcomes. Earlier physical coupon trials determine the probability weights for the ex-ante cost-aware choice of NONE, A witness, B witness or BOTH witnesses; the individual future damage and sensor flip cannot affect this policy choice. The choice is made after at least 112 intermediate accepted host movements and sensor/erosion events, with padded action clocks across risk groups. Both material risk counts are absent from the instantaneous observation; only the historical stress-trial event stream can reveal them.

Four paired risk arms (low A/low B, high A/low B, low A/high B, high A/high B) share the same instantaneous input and fault/cost setup and differ in pre-decision physical evidence. Each has four hidden outcome sequences and two sensor-flip states. Source histories include physical PAINT/ERASE, actual erasure, both witness tariff profiles and multichannel reports. Future hidden certificates are **evaluator-only**. No outcome-specific privileged teacher decision is used; the intervention is chosen using expected cost over the complete fixed latent distribution. Actual observed action legality and physical terminal success are replayed by a separately authored auditor.

The 2×2 rate grid is 2/8 versus 6/8 on TRAIN; 3/8 versus 5/8 on DEV; and extrapolated 1/8 versus 7/8 on HELDOUT, independently per material. Each of the eight new worlds belongs to exactly one split. The DEV geometry seed was selected using a fixed oracle-sensitivity curriculum criterion, **not a held-out model score**; this introduces benchmark-selection bias and must be reported, not interpreted as naturally occurring prevalence.

## Verified corpus

| Split | Disjoint worlds | Successful synthetic physical episodes | Independently replayed host-legal actions | Matched 2×2 material-risk groups | Risk-changing intervention groups | Strict uniform-50% baseline mistakes |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 4 | 1,024/1,024 | 202,016 | 256 | 10 | 14 |
| DEV | 2 | 512/512 | 101,304 | 128 | 4 | 8 |
| HELDOUT | 2 | 512/512 | 101,380 | 128 | 6 | 14 |
| TOTAL | 8 | 2,048/2,048 | 404,700 | 512 | 20 | 36 |

A-specific and B-specific paired contrasts across fixed opposite-material risks are TRAIN 8/16, DEV 8/0, HELDOUT 8/8. The **zero B-specific DEV contrast is a negative result**; the benchmark does not guarantee balanced intervention sensitivity. Most samples remain policy-equivalent, so this corpus should be combined with complementary challenges and evaluated with matched-pair metrics rather than aggregate one-step accuracy alone.

The deterministic current-observation-only accuracy ceilings on the first intervention are TRAIN 76.5625%, DEV 68.75%, HELDOUT 78.125%. The 50/50 prior control's **strict positive** mean expected action regret is TRAIN 0.046875, DEV 0.05078125, HELDOUT 0.28857421875 actions per ex-ante cost group. These are evaluator controls, **not trained neural or real-world results**. Most other scenarios admit an easy conservative rule. The full-history symbolic teacher still solves all cases.

## Qualification

* Generated full dataset twice; all six JSONL shard digests, row counts, and the manifest were byte-identical.
* Source-matched separate auditor replayed **404,700** actions, including coupon PAINT/ERASE, counterfactual remote damage, physical recovery, source-time observations, legal movement, material targets, episode hashes, and terminal success. Auditor run: 30.41 seconds, peak RSS 562,804 KB on this ephemeral CPU environment (this is **auditor throughput, not neural model inference**).
* Auditor reconstructed all 512 independent four-risk paired sets, each with eight latent damage/fault arms, and rejected one deliberately modified decision certificate even after its shard SHA-256 and manifest were recalculated (`V76 AUDIT decision mismatch`).
* Zero geometry overlap with independently published v75 (16 worlds) or v73 (8 worlds). These checks are corpus-level, not native host ABI parity checks.
* Inputs do not contain evaluator-only policy costs or incident-count fields. TRAIN repeated recovery suffixes receive loss masking to prevent trivial duplicate overrepresentation.
* Exact source/model contract pins are in `V76_DEFAULT_MANIFEST.json`, including v44 host, v67 erasure, v70 telemetry and v72 witness source SHA-256.

## Data contract and limits

Only `TRAIN_material_specific_risk_stream.jsonl` is permissible for training. Both `DEV_EVALUATOR_ONLY_*` and `HELDOUT_EVALUATOR_ONLY_*` streams and **all** `*_certificates_EVALUATOR_ONLY.jsonl` remain evaluation-only. `score_material_specific_risk_v76.py` is an evaluator-only diagnostic. Reproduce with `node material_specific_risk_v76.js NEW_DIR` and independently verify with `node test_material_specific_risk_host_v76.js NEW_DIR` after installing the four source-pinned local dependencies in the same directory. No generated shard or checkpoint should be committed to Git; immutable Actions artifact publication is outstanding.

No actual target SLM/adapter training occurred in this data lane; no neural HELDOUT completion, per-token CPU inference latency/RSS or native CyberSand runtime qualification was performed. The policy is still solvable by a small symbolic two-counter rule. The physical stress events, remote erosion and sensor inversion remain synthetic research extensions, not established native CyberSand ABI.

Next consumer: pin and train actual CPU SLM on TRAIN v74/v75/v76; seal held-out v76 until evaluation; compare full-history against last32, shuffled/removed A and B physical coupon history, uniform prior, conservative witness/reinspect; report matched four-risk group accuracy, physical completion, expected action regret, p50/p95 CPU inference latency, RSS and full checkpoint/tokenizer hashes. Continue data factory with dynamic obstructions and more balanced material-specific contrasts rather than one more scalar shortcut.
