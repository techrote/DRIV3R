# DRIV3R Corpus Factory v43 — balanced, split-guarded multi-domain export

**18 November 2026. Research-only, Draft/HOLD PR #1. Synthetic host, NOT native CyberSand.**

## Scientific question and response to consumer failures

The last several corpus increments added increasingly discriminative *individual* counterfactuals, but trainers still lacked a pinned **combined TRAIN/DEV/HELDOUT dataset** spanning host-executed material-law changes and delayed dual-interlock memory, with verified provenance, full action-menu legality, and explicit leakage protections. The adjacent Guidance & Recovery C58 evidence also demonstrates that naive synthetic-route training does not establish unseen remote-goal competence. V43 supplies a cross-domain controlled *corpus export*, **not a new scenario generator**, model-training campaign or claim to fix that separate failure.

V43 assembles two already-qualified host-executed source corpora without modifying their learner-visible input objects:

- V38: dynamic material-tool law transitions during corrective ERASE/PAINT and actual closed-loop oracle recovery, with wrong-paint/bad-navigation interventions and four-way historical-memory controls.
- V42: two independent old remote interlock actuator receipts, a blocked iff A AND B junction and paired causal-vs-negative controls, with closed-loop oracle recovery.

Original source manifests SHA-256: V38 `0d0c3c8f821dc9900863d45c9aac916014e4d66e644f99799c491a82f6554f5f`; V42 `fce49195cdfae48ec02a71928a85fecfe398f4e3c93e8b2aefa0fa0ff5af2834`. The generator checks **all parent shard sizes and SHA-256s before writing** and rejects altered/quick-run parents. These parents were originally generated from their corresponding pinned JavaScript source and dependencies. Source origin: `techrote/DRIV3R`, consolidated Draft/HOLD research PR #1, not externally imported hidden experimental data.

## Export, balancing and split isolation

| Split | V38 rows | V42 rows | Combined policy rows | Weight=0 | Distinct geometries |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8,912 | 8,912 | **17,824** | 0 | 16 |
| DEV | 12,096 | 8,736 | **20,832** | 128 | 8 |
| HELDOUT | 16,576 | 12,096 | **28,672** | 128 | 8 |
| **Total** | **37,584** | **29,744** | **67,328** | **256** | **32** |

TRAIN preserves *every* v38 accepted-action PAINT/ERASE/STOP label (2,816 original rare weight-one records), and selects a fixed SHA-256-ranked subset of v38 navigation decisions to exactly match v42's 8,912 weight-one records. This removes 3,632 navigation decisions and excludes 256 already-zero-weight original v38 records. Per-decision selection **does not preserve complete episode sequence ordering**; any sequential trainer must return to original source corpora.

DEV and HELDOUT are untouched, unsampled relative to both source policy shards, including their weight-zero rows. Their evaluator-only data (v38 challenge + memory; v42 challenge) are preserved byte-identically and segregated into six distinct files, never model inputs. Every emitted record is precisely `{input, action, weight}`. `row.input` alone is model-visible; the other fields are labels/loss weights. Each record has a separate `*_provenance_EVALUATOR_ONLY.jsonl` aligned by row number, containing original source row, graph hash, public durable receipt count, and action class. **No provenance, counterfactual answer or teacher metadata is added to input.** The resulting 12 JSONL shards and a manifest are generated outside Git.

## Qualification (local CPU; not a neural evaluation)

Two independent full exports produced byte-identical output shards and manifest. An independent JavaScript auditor verified every **67,328** emitted policy row against the unmodified parent input/action at its exact source-row index; replayed source labels and legality; checked parent manifests and all source/output shard SHA-256/byte sizes; checked train balance and retention of all rare v38 action labels; checked byte-identical evaluator challenge copies (**928 rows**); and enforced split-exclusive full-input SHA-256 and full observed survey topology hashes. **No cross-split survey-graph collision, duplicate input or contradictory label was found** across V38+V42.

Generation after sort-key caching: **6.23 and 6.28 s**, peak process RSS **~1.07 GiB**. Independent audit: **6.10 s**, peak RSS **~1.27 GiB**. These are Node corpus-generation/verification numbers on this worker, not model inference measurements. The memory use is high because this v43 first iteration fully parses the source policy shards and retains their input records; future streaming improvements would be useful if scaled.

Pinned output manifest SHA-256 `0ec96332b8406a77083cea8c626b3d94dac6a7cd3795fef90538e2148bb90b4c`; `V43_DEFAULT_MANIFEST.json` contains its full shard hashes and complete split/action distribution.

## Discriminative claim and **limits**

The combined TRAIN has two distinct causal action tasks: choosing a paint-command token under a changing public law while preserving physical paint output, and choosing a legal route when a remote interlock state must be reconstructed from older memory. DEV/HELDOUT retain the V38 material-law and memory contrasts and the V42 B-ablated positive/negative controls, but **no SLM has attempted them** and no performance gain is asserted. Mixing two symbolic tasks does not eliminate the possibility of a trivial suite-identifying or rule-based algorithm. The observation `v` and presence of the public `law` field may reveal which synthetic task family generated a row.

No new topology family, no new autonomous policy trajectories beyond the underlying source, and no native CyberSand action-observation acceptance are claimed. No GPU or paid API used, no actual LFM/Sprocket weights updated, no model inference latency/RSS measured. Large shards stay in the local evidence archive; they are **not** yet immutable GitHub Actions artifacts.

## Reproduction / consumer handoff

Unpack the *complete* published V38 and V42 corpus artifact archives, expose the original generated shards + `manifest.json` as distinct directories, then run:

```
node composite_curriculum_v43.js PATH_TO_V38_CORPUS PATH_TO_V42_CORPUS NEW_OUTPUT_DIR
node composite_curriculum_v43.js PATH_TO_V38_CORPUS PATH_TO_V42_CORPUS SECOND_OUTPUT_DIR
node test_composite_curriculum_v43.js NEW_OUTPUT_DIR SECOND_OUTPUT_DIR PATH_TO_V38_CORPUS PATH_TO_V42_CORPUS
```

Only use TRAIN for fitting; DEV for candidate selection; keep HELDOUT sealed for a single frozen policy evaluation. Test an actual pinned named SLM/adaptor (separate LFM and Sprocket ownership) against single-domain controls; run autonomous host-legal trajectory continuations in original V38/V42 synthetic hosts, comparing full historical receipts with targeted material-law/B-interlock ablations, measuring terminal success, action regret, residual wrong paint, illegal suggestions and p50/p95 inference latency and RSS. Independently run structurally unfamiliar remote-goal episodes because C58's failure mode is not addressed by this export.

**Custody:** only `experiments/driv3r/corpus_factory/**` may change; shared research branch only, no merge/Ready, no edits/comments to CyberSand or Neurounner, no training triggered by routine push.
